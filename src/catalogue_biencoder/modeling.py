"""
Model components: projections, fusion, and the two-tower reranker.
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TrainConfig


class ProjectionMLP(nn.Module):
    """
    LN -> Linear -> GELU -> Linear -> L2Norm
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


class GatedFusion(nn.Module):
    """
    Learned gate over concatenated [z_txt; z_img].
    e = norm(g * z_txt + (1-g) * z_img)

    This is a principled alternative to manual weights; the model learns modality reliance end-to-end.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, z_txt: torch.Tensor, z_img: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([z_txt, z_img], dim=-1)))
        e = g * z_txt + (1.0 - g) * z_img
        return F.normalize(e, p=2, dim=-1)


class TwoTowerReranker(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        vision_dim: int,
        text_dim: int,
        cfg: TrainConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.vision = vision_encoder
        self.text = text_encoder

        self.proj_img = ProjectionMLP(vision_dim, cfg.proj_hidden_dim, cfg.embed_dim)
        self.proj_txt = ProjectionMLP(text_dim, cfg.proj_hidden_dim, cfg.embed_dim)

        self.fusion = GatedFusion(cfg.embed_dim)

        # Learnable logit scale (CLIP-style). Parameterize as scale, clamp to [1, 100] to avoid blowing up logits.
        # scale = 1/temp; init so scale = 1/temperature_init (e.g. 1/0.07 ≈ 14.3).
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / cfg.temperature_init)))

    def scale(self) -> torch.Tensor:
        return torch.exp(self.logit_scale).clamp(1.0, 100.0)

    def encode_image(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """SigLIP2: vision backbone → pooler_output → our ProjectionMLP. When vision is frozen, no graph is stored (VRAM/speed)."""
        grad_enabled = any(p.requires_grad for p in self.vision.parameters())
        with torch.set_grad_enabled(grad_enabled):
            # Call inner vision_model directly to avoid duplicate attention_mask (Siglip2VisionModel
            # forwards pixel_attention_mask as attention_mask and can pass it again in **kwargs).
            # Return type: BaseModelOutputWithPooling with pooler_output.
            out = self.vision.vision_model(
                pixel_values=batch["pixel_values"],
                attention_mask=batch.get("pixel_attention_mask"),
                spatial_shapes=batch.get("spatial_shapes"),
                return_dict=True,
            )
            h = out.pooler_output  # (B, vision_dim)
        return self.proj_img(h)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """BGE uses the CLS token (index 0) as the sentence embedding. When text is frozen, no graph is stored (VRAM/speed)."""
        grad_enabled = any(p.requires_grad for p in self.text.parameters())
        with torch.set_grad_enabled(grad_enabled):
            out = self.text(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            h = out.last_hidden_state[:, 0, :]
        return self.proj_txt(h)

    def apply_modality_dropout(
        self,
        z_txt: torch.Tensor,
        z_img: torch.Tensor,
        training: bool,
        image_missing: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not training and image_missing is None:
            return z_txt, z_img

        B = z_txt.shape[0]

        # Drop text for a subset of examples (never when image is missing → avoid zero embedding).
        drop_t = torch.zeros(B, dtype=torch.bool, device=z_txt.device)
        if training and self.cfg.drop_text_p > 0:
            drop_t = (torch.rand(B, device=z_txt.device) < self.cfg.drop_text_p)
            if image_missing is not None:
                drop_t = drop_t & (~image_missing.to(z_txt.device))
        z_txt = z_txt * (~drop_t).float().unsqueeze(-1)

        # Drop image for a subset of examples; never drop both modalities (keep at least one).
        drop_i = torch.zeros(B, dtype=torch.bool, device=z_img.device)
        if training and self.cfg.drop_image_p > 0:
            drop_i = (torch.rand(B, device=z_img.device) < self.cfg.drop_image_p)
            drop_i = drop_i & (~drop_t)  # drop image only when text was kept
        z_img = z_img * (~drop_i).float().unsqueeze(-1)

        # Force image to text-only for missing-image samples (no wrong image–text pairing)
        if image_missing is not None and image_missing.any():
            mask = image_missing.to(z_img.device)
            z_img = z_img.clone()
            z_img[mask] = z_txt[mask]

        # Renormalize to keep scales consistent after dropout
        z_txt = F.normalize(z_txt, p=2, dim=-1)
        z_img = F.normalize(z_img, p=2, dim=-1)
        return z_txt, z_img

    def encode_product(
        self,
        batch: Dict[str, torch.Tensor],
        image_missing: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (e_txt, e_img, e_fused) in shared embedding space (after projection + dropout + fusion)."""
        z_img = self.encode_image(batch)
        prod_input_ids = batch["prod_input_ids"]
        prod_attention_mask = batch["prod_attention_mask"]
        z_txt = self.encode_text(prod_input_ids, prod_attention_mask)
        z_txt, z_img = self.apply_modality_dropout(
            z_txt, z_img, training=self.training, image_missing=image_missing
        )
        e_prod = self.fusion(z_txt, z_img)
        return z_txt, z_img, e_prod

    def encode_candidates(self, cand_input_ids, cand_attention_mask) -> torch.Tensor:
        """
        cand_input_ids: (B, C, L)
        returns: (B, C, D)
        """
        B, C, L = cand_input_ids.shape
        flat_ids = cand_input_ids.view(B * C, L)
        flat_mask = cand_attention_mask.view(B * C, L)
        flat_emb = self.encode_text(flat_ids, flat_mask)  # (B*C, D)
        return flat_emb.view(B, C, -1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        e_txt, e_img, e_prod = self.encode_product(
            batch,
            image_missing=batch.get("image_missing"),
        )
        e_cand = self.encode_candidates(
            batch["cand_input_ids"],
            batch["cand_attention_mask"],
        )
        # scores: (B, C). Scale by learned logit_scale (CLIP-style); clamp [1, 100] avoids logit explosion.
        scale = self.scale()
        scores = torch.einsum("bd,bcd->bc", e_prod, e_cand) * scale

        # Mask padded candidates with -inf so they don't affect softmax
        cand_mask = batch["cand_mask"]  # bool (B,C)
        scores = scores.masked_fill(~cand_mask, float("-inf"))

        return {
            "e_prod": e_prod,
            "e_txt": e_txt,
            "e_img": e_img,
            "e_cand": e_cand,
            "scores": scores,
        }
