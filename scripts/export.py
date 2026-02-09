#!/usr/bin/env python3
"""
Optional script to export embeddings from a checkpoint/run directory.
"""
import sys
import os
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModel, Siglip2VisionModel
from torch.utils.data import DataLoader
from accelerate import Accelerator

# Add src to path for imports (useful for Colab)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from catalogue_biencoder.config import TrainConfig
from catalogue_biencoder.data import ProductCatalogueTorchDataset, BatchCollator
from catalogue_biencoder.modeling import TwoTowerReranker
from catalogue_biencoder.utils import normalize_category_path
from catalogue_biencoder.exports import export_product_embeddings, export_category_embeddings
from datasets import load_dataset


def load_model_from_checkpoint(checkpoint_dir: str, device: str = "cuda"):
    """
    Load model from checkpoint directory.
    Expects:
      - checkpoint_dir/heads.pt
      - checkpoint_dir/text_adapter/
      - checkpoint_dir/vision_adapter/
      - checkpoint_dir/config.json
    """
    with open(os.path.join(checkpoint_dir, "config.json"), "r") as f:
        cfg_dict = json.load(f)
    cfg = TrainConfig(**cfg_dict)

    # Vision-only load (ignore_mismatched_sizes for patch_embedding when ckpt is siglip v1 layout)
    vision_encoder = Siglip2VisionModel.from_pretrained(
        cfg.vision_ckpt, ignore_mismatched_sizes=True
    )
    text_encoder = AutoModel.from_pretrained(cfg.text_ckpt)

    # Load LoRA adapters if they exist
    text_adapter_path = os.path.join(checkpoint_dir, "text_adapter")
    vision_adapter_path = os.path.join(checkpoint_dir, "vision_adapter")
    
    if os.path.exists(text_adapter_path):
        from peft import PeftModel
        text_encoder = PeftModel.from_pretrained(text_encoder, text_adapter_path)
    if os.path.exists(vision_adapter_path):
        from peft import PeftModel
        vision_encoder = PeftModel.from_pretrained(vision_encoder, vision_adapter_path)

    vdim = vision_encoder.config.hidden_size
    tdim = text_encoder.config.hidden_size

    model = TwoTowerReranker(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        vision_dim=vdim,
        text_dim=tdim,
        cfg=cfg,
    )

    # Load heads
    heads_path = os.path.join(checkpoint_dir, "heads.pt")
    if os.path.exists(heads_path):
        heads_state = torch.load(heads_path, map_location=device)
        model.load_state_dict(heads_state, strict=False)

    model.text_tokenizer = AutoTokenizer.from_pretrained(cfg.text_ckpt)
    model.to(device)
    model.eval()
    return model, cfg


def main():
    """Export embeddings from a checkpoint directory."""
    import argparse
    parser = argparse.ArgumentParser(description="Export embeddings from checkpoint")
    parser.add_argument("checkpoint_dir", type=str, help="Path to checkpoint directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: checkpoint_dir/embeddings)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to export")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(checkpoint_dir, "embeddings")
    os.makedirs(output_dir, exist_ok=True)

    accelerator = Accelerator()
    device = accelerator.device

    # Load model
    model, cfg = load_model_from_checkpoint(checkpoint_dir, device=str(device))
    model = accelerator.prepare(model)

    # Load data
    ds = load_dataset(cfg.hf_dataset_name)
    split_hf = ds[args.split]
    split_ds = ProductCatalogueTorchDataset(split_hf)

    # Build category mapping from both splits (matches training export)
    all_cats_set = set()
    for split_name in [cfg.train_split, cfg.test_split]:
        for cand_list in ds[split_name]["potential_product_categories"]:
            for c in (cand_list or []):
                if (c or "").strip():
                    all_cats_set.add(normalize_category_path(c))
    all_cats_list = sorted(all_cats_set)
    cat2id = {cat: i for i, cat in enumerate(all_cats_list)}

    # Create collator and loader
    vision_processor = AutoProcessor.from_pretrained(cfg.vision_ckpt)
    text_tokenizer = AutoTokenizer.from_pretrained(cfg.text_ckpt)
    collator_export = BatchCollator(vision_processor, text_tokenizer, cfg, cat2id=cat2id, include_export_metadata=True)

    export_loader = DataLoader(
        split_ds,
        batch_size=cfg.embed_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collator_export,
        pin_memory=True,
    )
    export_loader = accelerator.prepare(export_loader)

    # Export
    prefix = os.path.join(output_dir, f"{args.split}_product_emb")
    export_product_embeddings(cfg, accelerator, model, export_loader, prefix)

    # Export category embeddings
    export_category_embeddings(cfg, accelerator, accelerator.unwrap_model(model), all_cats_list, os.path.join(output_dir, "unique_category_emb"))

    if accelerator.is_main_process:
        print(f"Embeddings exported to: {output_dir}")


if __name__ == "__main__":
    main()
