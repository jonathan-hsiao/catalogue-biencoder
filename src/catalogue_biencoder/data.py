"""
Dataset and data loading utilities.
"""
from typing import Any, Dict, List

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoProcessor

from .config import TrainConfig
from .utils import normalize_category_path, normalize_candidates_and_gt_index


class ProductCatalogueTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.ds = hf_split

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.ds[idx]
        title = row.get("product_title", "") or ""
        desc = row.get("product_description", "") or ""
        image = row.get("product_image", None)
        candidates = row.get("potential_product_categories", []) or []
        gt = row.get("ground_truth_category", "") or ""

        # Stable, deterministic product text formatting (no manual weighting).
        product_text = f"Title: {title}\nDescription: {desc}"

        return {
            "idx": idx,
            "product_text": product_text,
            "image": image,  # PIL Image (HF Image feature)
            "candidates": candidates,
            "gt_category": gt,
        }


class BatchCollator:
    def __init__(
        self,
        vision_processor,
        text_tokenizer,
        cfg: TrainConfig,
        cat2id: Dict[str, int],
        include_export_metadata: bool = False,
    ):
        self.vision_processor = vision_processor
        self.text_tokenizer = text_tokenizer
        self.cfg = cfg
        self.cat2id = cat2id  # Global category -> id (stable across batches) for cross-batch queue
        self.include_export_metadata = include_export_metadata  # True only for embedding export (avoids string overhead in training)
        # Use a neutral gray placeholder for missing images to avoid wrong image–text pairs (label noise).
        # SigLIP2-base expects 256x256; placeholder is (128,128,128) RGB.
        self._placeholder_image = Image.new("RGB", (256, 256), (128, 128, 128))

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # --- images ---
        images = []
        image_missing_flags = []
        for b in batch:
            img = b["image"]
            if img is None:
                img = self._placeholder_image
                image_missing_flags.append(True)
            else:
                # Ensure consistent 3-channel input for the vision processor
                img = img.convert("RGB")
                image_missing_flags.append(False)
            images.append(img)

        image_missing = torch.tensor(image_missing_flags, dtype=torch.bool)

        vision_inputs = self.vision_processor(images=images, return_tensors="pt")
        missing = [k for k in ("pixel_values", "pixel_attention_mask", "spatial_shapes") if k not in vision_inputs]
        if missing:
            raise ValueError(
                f"SigLIP2 processor must return {missing}. "
                "Upgrade transformers / use the SigLIP2 image processor."
            )

        # --- product text ---
        product_texts = [b["product_text"] for b in batch]
        prod_tok = self.text_tokenizer(
            product_texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_text_len_product,
            return_tensors="pt",
        )

        # --- candidates + gt index ---
        cand_texts_padded: List[List[str]] = []
        cand_strings_used: List[List[str]] = []  # normalized category strings actually scored (for export)
        gt_indices: List[int] = []
        cand_masks: List[List[int]] = []

        for b in batch:
            cands, gt_idx = normalize_candidates_and_gt_index(
                b["candidates"], b["gt_category"], max_candidates=self.cfg.max_candidates
            )
            pad_n = self.cfg.max_candidates - len(cands)
            cands_padded = cands + [""] * pad_n
            mask = [1] * (self.cfg.max_candidates - pad_n) + [0] * pad_n

            if self.include_export_metadata:
                cand_strings_used.append(cands_padded)
            # Use true empty for padded slots (masked out); avoids "Category: " token bias if mask ever breaks.
            cands_txt = [f"Category: {c}" if c else "" for c in cands_padded]

            cand_texts_padded.append(cands_txt)
            gt_indices.append(gt_idx)
            cand_masks.append(mask)

        # Tokenize candidates as flat list then reshape
        flat_cands = [s for row in cand_texts_padded for s in row]
        cand_tok_flat = self.text_tokenizer(
            flat_cands,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_text_len_category,
            return_tensors="pt",
        )
        # Reshape to (B, C, L)
        B = len(batch)
        C = self.cfg.max_candidates
        cand_input_ids = cand_tok_flat["input_ids"].view(B, C, -1)
        cand_attention_mask = cand_tok_flat["attention_mask"].view(B, C, -1)

        # GT category ids from global mapping (stable across batches for cross-batch queue)
        gt_category_id = torch.tensor(
            [self.cat2id[normalize_category_path(b["gt_category"])] for b in batch],
            dtype=torch.long,
        )

        batch_out = {
            "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
            "image_missing": image_missing,
            "pixel_values": vision_inputs["pixel_values"],
            "prod_input_ids": prod_tok["input_ids"],
            "prod_attention_mask": prod_tok["attention_mask"],
            "cand_input_ids": cand_input_ids,
            "cand_attention_mask": cand_attention_mask,
            "cand_mask": torch.tensor(cand_masks, dtype=torch.bool),
            "target_idx": torch.tensor(gt_indices, dtype=torch.long),
            "gt_category_id": gt_category_id,
        }
        if self.include_export_metadata:
            batch_out["candidates_used"] = cand_strings_used
            batch_out["raw_candidates"] = [b["candidates"] for b in batch]
            batch_out["raw_gt"] = [b["gt_category"] for b in batch]
        batch_out["pixel_attention_mask"] = vision_inputs["pixel_attention_mask"]
        batch_out["spatial_shapes"] = vision_inputs["spatial_shapes"]
        return batch_out
