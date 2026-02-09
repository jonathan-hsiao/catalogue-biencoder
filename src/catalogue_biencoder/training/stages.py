"""
LoRA application and stage management utilities.
"""
from typing import List, Set, Tuple

import torch
import torch.nn as nn

from transformers import AutoModel, Siglip2VisionModel
from peft import LoraConfig, TaskType, get_peft_model

from ..config import TrainConfig


# PEFT LoRA adapter parameter names (LoraLayer.adapter_layer_names)
_LORA_PARAM_NAMES = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")


def freeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False


def unfreeze_trainable(m: nn.Module) -> None:
    # (Used mainly for non-LoRA heads that we always train.)
    for p in m.parameters():
        p.requires_grad = True


def freeze_lora_params(m: nn.Module) -> None:
    """Set requires_grad=False for PEFT LoRA adapter params (used when LoRA is applied up front but not yet trained)."""
    for n, p in m.named_parameters():
        if any(lora_name in n for lora_name in _LORA_PARAM_NAMES):
            p.requires_grad = False


def freeze_backbone_keep_lora(m: nn.Module) -> None:
    """Freeze base weights but keep PEFT LoRA adapter params trainable."""
    for n, p in m.named_parameters():
        if any(lora_name in n for lora_name in _LORA_PARAM_NAMES):
            p.requires_grad = True
        else:
            p.requires_grad = False


def set_tower_mode(tower: nn.Module, cfg: TrainConfig) -> None:
    """Set train/eval mode for tower."""
    if cfg.tower_eval_during_training:
        tower.eval()
    else:
        if any(p.requires_grad for p in tower.parameters()):
            tower.train()
        else:
            tower.eval()


def build_optimizer(cfg: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    """
    AdamW with decoupled weight decay.
    Best practice: exclude biases and LayerNorm from weight decay.
    """
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n_lower = n.lower()
        if (
            p.ndim <= 1 
            or n_lower.endswith(".bias") 
            or "layernorm" in n_lower 
            or ".ln" in n_lower
            or "lora_" in n_lower
        ):
            no_decay.append(p)
        else:
            decay.append(p)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr
    )


def infer_lora_targets(model: nn.Module, prefer: Tuple[str, ...]) -> List[str]:
    """Collect Linear layer suffix names in the model; return those in prefer that exist (order preserved)."""
    suffixes: Set[str] = set()
    for _n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            suffix = _n.split(".")[-1]
            suffixes.add(suffix)
    return [t for t in prefer if t in suffixes]


def maybe_apply_lora_text(cfg: TrainConfig, text_model: nn.Module) -> nn.Module:
    """BGE (BERT-based) uses query, key, value, dense in attention; auto-discover to avoid wrong names."""
    prefer = ("query", "key", "value", "dense") # could drop dense for faster
    targets = infer_lora_targets(text_model, prefer)
    if not targets:
        raise ValueError(
            f"Could not infer LoRA targets for text model. Tried {prefer}. "
            "Check model architecture (e.g. BERT uses query/key/value/dense, not q_proj/k_proj/...)."
        )
    lora_cfg = LoraConfig(
        r=cfg.text_lora_r,
        lora_alpha=cfg.text_lora_alpha,
        lora_dropout=cfg.text_lora_dropout,
        bias="none",
        target_modules=targets,
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    peft_model = get_peft_model(text_model, lora_cfg)
    _assert_lora_params_added(peft_model, "text")
    return peft_model


def maybe_apply_lora_vision(cfg: TrainConfig, vision_model: nn.Module) -> nn.Module:
    """SigLIP2 ViT uses q_proj, k_proj, v_proj, out_proj; auto-discover to support variants."""
    prefer = ("q_proj", "k_proj", "v_proj", "out_proj", "o_proj")
    targets = infer_lora_targets(vision_model, prefer)
    if not targets:
        raise ValueError(
            f"Could not infer LoRA targets for vision model. Tried {prefer}. "
            "Check model architecture."
        )
    lora_cfg = LoraConfig(
        r=cfg.vision_lora_r,
        lora_alpha=cfg.vision_lora_alpha,
        lora_dropout=cfg.vision_lora_dropout,
        bias="none",
        target_modules=targets,
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    peft_model = get_peft_model(vision_model, lora_cfg)
    _assert_lora_params_added(peft_model, "vision")
    return peft_model


def _assert_lora_params_added(peft_model: nn.Module, tower: str) -> None:
    """Fail fast if PEFT did not add any LoRA parameters."""
    lora_names = [n for n, _ in peft_model.named_parameters() if "lora" in n.lower()]
    if not lora_names:
        raise ValueError(
            f"PEFT applied to {tower} tower but no LoRA parameters found. "
            "Check target_modules match the model's Linear layer names."
        )
