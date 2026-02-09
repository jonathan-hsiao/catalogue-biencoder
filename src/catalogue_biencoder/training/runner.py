"""
Main training runner: orchestrates the full training pipeline.
"""
import os
import json
import math
import time
from dataclasses import asdict
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    Siglip2ImageProcessor,
    Siglip2VisionModel,
    get_linear_schedule_with_warmup,
)
from accelerate import Accelerator

from ..config import TrainConfig
from ..utils import set_seed, normalize_category_path
from ..data import ProductCatalogueTorchDataset, BatchCollator
from ..modeling import TwoTowerReranker
from ..losses import (
    CrossBatchQueue,
    listwise_loss,
    multi_pos_cross_modal_infoNCE_loss,
    contrastive_weight_for_stage,
    compute_ranking_metrics,
)
from ..exports import export_product_embeddings, export_category_embeddings, compute_dataset_stats
from .stages import (
    freeze_module,
    unfreeze_trainable,
    freeze_lora_params,
    freeze_backbone_keep_lora,
    set_tower_mode,
    build_optimizer,
    maybe_apply_lora_text,
    maybe_apply_lora_vision,
)


def save_checkpoint(cfg: TrainConfig, accelerator: Accelerator, model: nn.Module, out_dir: str, tag: str) -> None:
    """Save heads (proj_*, fusion, logit_scale) and PEFT adapters only; avoids full backbone state dicts."""
    ckpt_dir = os.path.join(out_dir, f"checkpoint_{tag}")
    if accelerator.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        # Heads only (projections, fusion, logit_scale)
        full_state = unwrapped.state_dict()
        head_prefixes = ("proj_img.", "proj_txt.", "fusion.")
        heads_state = {
            k: v for k, v in full_state.items()
            if k.startswith(head_prefixes) or k == "logit_scale"
        }
        torch.save(heads_state, os.path.join(ckpt_dir, "heads.pt"))
        # PEFT adapters (lightweight; no full backbones)
        unwrapped.text.save_pretrained(os.path.join(ckpt_dir, "text_adapter"))
        unwrapped.vision.save_pretrained(os.path.join(ckpt_dir, "vision_adapter"))
        with open(os.path.join(ckpt_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2)


@torch.no_grad()
def evaluate(cfg: TrainConfig, accelerator: Accelerator, model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
    model.eval()
    total = 0
    agg = {"acc@1": 0.0, "acc@3": 0.0, "acc@5": 0.0, "mrr": 0.0, "loss": 0.0}

    for batch in tqdm(dataloader, desc="eval", leave=False):
        with accelerator.autocast():
            out = model(batch)
            scores = out["scores"]
            target_idx = batch["target_idx"]

            # Listwise loss only for reporting (contrastive is a regularizer for embeddings)
            loss = listwise_loss(scores, target_idx)

        metrics = compute_ranking_metrics(scores, target_idx, k_list=(1, 3, 5))

        bs = scores.size(0)
        total += bs
        agg["loss"] += loss.item() * bs
        for k in ["acc@1", "acc@3", "acc@5", "mrr"]:
            agg[k] += metrics[k] * bs

    # Reduce across processes so metrics are over the full dataset (not just this process's shard).
    if accelerator.num_processes > 1:
        device = accelerator.device
        total_t = torch.tensor([float(total)], dtype=torch.float64, device=device)
        total_g = accelerator.gather(total_t)
        global_total = total_g.sum().item()
        for k in agg:
            v_t = torch.tensor([agg[k]], dtype=torch.float64, device=device)
            v_g = accelerator.gather(v_t)
            agg[k] = (v_g.sum().item() / global_total) if global_total > 0 else 0.0
    else:
        for k in agg:
            agg[k] /= max(total, 1)
    return agg


def run(cfg: TrainConfig) -> None:
    """
    Main training pipeline: loads data, builds model, trains across stages, evaluates, exports embeddings.
    """
    # ---- setup ----
    if cfg.run_name.strip() == "":
        cfg.run_name = time.strftime("%Y%m%d_%H%M%S")

    run_dir = os.path.join(cfg.output_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)

    set_seed(cfg.seed)

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.grad_accum_steps,
    )
    device = accelerator.device

    if accelerator.is_main_process:
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2)

    # ---- load data (use HF built-in train/test; no custom splitting) ----
    ds = load_dataset(cfg.hf_dataset_name)
    train_hf = ds[cfg.train_split]
    test_hf = ds[cfg.test_split]

    train_ds = ProductCatalogueTorchDataset(train_hf)
    test_ds = ProductCatalogueTorchDataset(test_hf)

    if accelerator.is_main_process:
        dataset_stats = compute_dataset_stats(train_hf, test_hf)
        with open(os.path.join(run_dir, "dataset_stats.json"), "w", encoding="utf-8") as f:
            json.dump(dataset_stats, f, indent=2)
        print("Dataset stats:", dataset_stats)

    # Global category -> id from candidate lists (not ground truth); stable across batches for cross-batch queue
    all_cats_set = set()
    for split_name in [cfg.train_split, cfg.test_split]:
        for cand_list in ds[split_name]["potential_product_categories"]:
            for c in (cand_list or []):
                if (c or "").strip():
                    all_cats_set.add(normalize_category_path(c))
    all_cats_list = sorted(all_cats_set)
    cat2id = {cat: i for i, cat in enumerate(all_cats_list)}

    # ---- tokenizers / processors ----
    # Use Siglip2ImageProcessor so we get pixel_values, pixel_attention_mask, spatial_shapes.
    # AutoProcessor can load Siglip (v1) from the same ckpt, which does not return those keys.
    vision_processor = Siglip2ImageProcessor.from_pretrained(cfg.vision_ckpt)
    # Text: BGE tokenizer
    text_tokenizer = AutoTokenizer.from_pretrained(cfg.text_ckpt)

    collator = BatchCollator(vision_processor, text_tokenizer, cfg, cat2id=cat2id, include_export_metadata=False)
    collator_export = BatchCollator(vision_processor, text_tokenizer, cfg, cat2id=cat2id, include_export_metadata=True)

    train_generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collator,
        pin_memory=True,
        generator=train_generator,
        worker_init_fn=lambda wid: set_seed(cfg.seed + wid),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    # ---- load encoders (SigLIP2 ViT-B + BGE v1.5) ----
    # Vision-only: Hub checkpoint may be siglip (v1) layout (Conv2d patch_embedding); Siglip2 uses Linear.
    # ignore_mismatched_sizes=True reinits that layer and loads the rest. Same result as full-model extract, lower memory.
    vision_encoder = Siglip2VisionModel.from_pretrained(
        cfg.vision_ckpt, ignore_mismatched_sizes=True
    )
    text_encoder = AutoModel.from_pretrained(cfg.text_ckpt)

    # Encoder output dims: Siglip2VisionModel.config.hidden_size, BGE hidden_size
    vdim = vision_encoder.config.hidden_size
    tdim = text_encoder.config.hidden_size

    # Apply LoRA to both towers before building the full model and before accelerator.prepare().
    # Adapters are then frozen until the stage that trains them (avoids mutating model after prepare() in DDP/FSDP).
    text_encoder = maybe_apply_lora_text(cfg, text_encoder)
    vision_encoder = maybe_apply_lora_vision(cfg, vision_encoder)
    freeze_lora_params(text_encoder)
    freeze_lora_params(vision_encoder)

    model = TwoTowerReranker(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        vision_dim=vdim,
        text_dim=tdim,
        cfg=cfg,
    )

    # Attach tokenizer for category export convenience
    model.text_tokenizer = text_tokenizer

    # ---- stage plan ----
    stages = [
        {"name": "stage0_heads_only", "epochs": cfg.stage0_epochs, "lora_text": False, "lora_vision": False},
        {"name": "stage1_text_lora",   "epochs": cfg.stage1_epochs, "lora_text": True,  "lora_vision": False},
        {"name": "stage2_vision_lora", "epochs": cfg.stage2_epochs, "lora_text": True,  "lora_vision": True},
    ]

    # Prepare for accelerator
    model, train_loader, test_loader = accelerator.prepare(model, train_loader, test_loader)

    # Cross-batch queue for contrastive negatives (global embedding structure)
    if cfg.mixed_precision == "bf16":
        queue_dtype = torch.bfloat16
    elif cfg.mixed_precision == "fp16":
        queue_dtype = torch.float16
    else:
        queue_dtype = torch.float32
    queue = CrossBatchQueue(
        max_size=cfg.queue_size,
        embed_dim=cfg.embed_dim,
        device=accelerator.device,
        dtype=queue_dtype,
    )

    # ---- training loop across stages ----
    global_step = 0
    for stage_idx, stage in enumerate(stages):
        if stage["epochs"] <= 0:
            continue

        queue.reset()  # Stale embeddings from previous stage would hurt when embedding geometry changes (e.g. LoRA unfreeze)

        # (Re)configure which params are trainable.
        # Best practice: freeze large backbones by default; only train heads and (optional) LoRA params.
        unwrapped = accelerator.unwrap_model(model)

        # Always train projections + fusion + logit_scale (temperature)
        unfreeze_trainable(unwrapped.proj_img)
        unfreeze_trainable(unwrapped.proj_txt)
        unfreeze_trainable(unwrapped.fusion)
        unwrapped.logit_scale.requires_grad = True

        # Freeze towers: train LoRA adapters only when this stage enables them; otherwise freeze entire tower.
        if stage["lora_text"]:
            freeze_backbone_keep_lora(unwrapped.text)
        else:
            freeze_module(unwrapped.text)

        if stage["lora_vision"]:
            freeze_backbone_keep_lora(unwrapped.vision)
        else:
            freeze_module(unwrapped.vision)

        # Frozen towers in eval() to disable dropout; trainable (LoRA) towers in train().
        set_tower_mode(unwrapped.text, cfg)
        set_tower_mode(unwrapped.vision, cfg)

        # Ensure heads still trainable (outside PEFT wrapper).
        unfreeze_trainable(unwrapped.proj_img)
        unfreeze_trainable(unwrapped.proj_txt)
        unfreeze_trainable(unwrapped.fusion)
        unwrapped.logit_scale.requires_grad = True

        # Build optimizer/scheduler for current stage trainables
        optimizer = build_optimizer(cfg, model)
        num_update_steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum_steps)
        total_steps = stage["epochs"] * num_update_steps_per_epoch
        warmup_steps = int(cfg.warmup_ratio * total_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

        if accelerator.is_main_process:
            print(f"\n=== {stage['name']} ===")
            print(f"epochs={stage['epochs']} | lora_text={stage['lora_text']} | lora_vision={stage['lora_vision']}")
            # Helpful visibility into trainable parameter count
            trainable = sum(p.numel() for p in accelerator.unwrap_model(model).parameters() if p.requires_grad)
            total = sum(p.numel() for p in accelerator.unwrap_model(model).parameters())
            print(f"Trainable params: {trainable:,} / {total:,} ({100.0*trainable/total:.3f}%)")

        # Train epochs: model.train() first, then set_tower_mode (eval for frozen towers, train for LoRA towers).
        model.train()
        set_tower_mode(unwrapped.text, cfg)
        set_tower_mode(unwrapped.vision, cfg)
        log_buffer: List[str] = []
        for epoch in range(stage["epochs"]):
            pbar = tqdm(train_loader, desc=f"{stage['name']} epoch {epoch+1}/{stage['epochs']}", disable=not accelerator.is_main_process)
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(pbar):
                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        out = model(batch)
                        scores = out["scores"]

                        # Primary: listwise reranking over candidate list
                        loss_rank = listwise_loss(scores, batch["target_idx"])

                        # Contrastive regularizer: multi-positive InfoNCE with cross-batch queue (global structure)
                        e_prod = out["e_prod"]   # (B, D)
                        e_cand = out["e_cand"]   # (B, C, D)

                        B = e_prod.size(0)
                        e_gt = e_cand[torch.arange(B, device=e_prod.device), batch["target_idx"]]  # (B, D)

                        q_prod, q_cat, q_id = queue.get()
                        loss_ctr = multi_pos_cross_modal_infoNCE_loss(
                            e_prod=e_prod,
                            e_cat=e_gt,
                            cat_id=batch["gt_category_id"],
                            scale=accelerator.unwrap_model(model).scale(),
                            queue_prod=q_prod,
                            queue_cat=q_cat,
                            queue_id=q_id,
                            normalize_pos=True,
                        )
                        ctr_weight = contrastive_weight_for_stage(cfg, stage_idx)
                        loss = loss_rank + ctr_weight * loss_ctr

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
                            accelerator.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                # Enqueue current batch for cross-batch contrastive (detached; gather for multi-GPU)
                with torch.no_grad():
                    e_prod_d = e_prod.detach()
                    e_gt_d = e_gt.detach()
                    ids_d = batch["gt_category_id"].detach()
                    if accelerator.num_processes > 1:
                        e_prod_d = accelerator.gather(e_prod_d)
                        e_gt_d = accelerator.gather(e_gt_d)
                        ids_d = accelerator.gather(ids_d)
                    queue.enqueue(e_prod_d, e_gt_d, ids_d)

                if accelerator.sync_gradients:
                    global_step += 1
                if accelerator.is_main_process:
                    pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "rank": f"{loss_rank.item():.4f}",
                        "ctr": f"{loss_ctr.item():.4f}",
                        "scale": f"{accelerator.unwrap_model(model).scale().item():.2f}",
                        "step": global_step,
                    })
                    log_line = json.dumps({
                        "step": global_step,
                        "epoch": epoch + 1,
                        "stage": stage["name"],
                        "loss": round(loss.item(), 6),
                        "loss_rank": round(loss_rank.item(), 6),
                        "loss_ctr": round(loss_ctr.item(), 6),
                        "scale": round(accelerator.unwrap_model(model).scale().item(), 4),
                        "ctr_weight": round(ctr_weight, 6),
                        "queue_fill": queue.current_size(),
                    }) + "\n"
                    log_buffer.append(log_line)
                    if len(log_buffer) >= cfg.log_every_n_steps:
                        log_path = os.path.join(run_dir, "training_log.jsonl")
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.writelines(log_buffer)
                        log_buffer.clear()

            # Flush remaining log lines at end of epoch
            if accelerator.is_main_process and log_buffer:
                log_path = os.path.join(run_dir, "training_log.jsonl")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.writelines(log_buffer)
                log_buffer.clear()

            # Optional: evaluate on test at end of epoch (reporting only; default False to avoid peeking at test).
            if cfg.eval_each_epoch:
                metrics = evaluate(cfg, accelerator, model, test_loader)
                if accelerator.is_main_process:
                    print(f"[{stage['name']}] epoch {epoch+1}: {metrics}")

        # Save checkpoint after stage
        save_checkpoint(cfg, accelerator, model, run_dir, tag=stage["name"])

    # ---- final eval ----
    final_metrics = evaluate(cfg, accelerator, model, test_loader)
    if accelerator.is_main_process:
        print("\n=== FINAL TEST METRICS ===")
        print(final_metrics)
        with open(os.path.join(run_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, indent=2)

    # ---- export embeddings for visualization ----
    # Decision point: embeddings should come from the final trained model state for stability.
    # We export both train and test for later interactive plots, clustering, etc.
    export_dir = os.path.join(run_dir, "embeddings")
    if accelerator.is_main_process:
        os.makedirs(export_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Non-shuffled loaders for deterministic export ordering (collator_export includes string metadata for JSONL)
    train_export_loader = DataLoader(
        train_ds,
        batch_size=cfg.embed_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collator_export,
        pin_memory=True,
    )
    test_export_loader = DataLoader(
        test_ds,
        batch_size=cfg.embed_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collator_export,
        pin_memory=True,
    )
    train_export_loader, test_export_loader = accelerator.prepare(train_export_loader, test_export_loader)

    export_product_embeddings(cfg, accelerator, model, train_export_loader, os.path.join(export_dir, "train_product_emb"))
    export_product_embeddings(cfg, accelerator, model, test_export_loader, os.path.join(export_dir, "test_product_emb"))

    # Export embeddings for all unique ground-truth categories (same order as cat2id / all_cats_list)
    export_category_embeddings(cfg, accelerator, accelerator.unwrap_model(model), all_cats_list, os.path.join(export_dir, "unique_category_emb"))

    if accelerator.is_main_process:
        print("\nDone.")
        print(f"Run directory: {run_dir}")
        print(f"Embeddings exported to: {export_dir}")
