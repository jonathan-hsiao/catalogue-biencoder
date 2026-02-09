"""
Embedding export utilities for visualization and analysis.
"""
import os
import json
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator

from .config import TrainConfig
from .modeling import TwoTowerReranker
from .utils import normalize_category_path


@torch.no_grad()
def export_product_embeddings(
    cfg: TrainConfig,
    accelerator: Accelerator,
    model: torch.nn.Module,
    dataloader: DataLoader,
    out_path_prefix: str,
):
    """
    Exports {prefix}.npy (fused), {prefix}_e_txt.npy, {prefix}_e_img.npy (float32) and {prefix}.jsonl (metadata).
    Single-process only (e.g. one GPU); skipped when num_processes > 1.
    """
    if accelerator.num_processes > 1:
        if accelerator.is_main_process:
            print("Embedding export skipped (single-process only).")
        return

    model.eval()
    all_meta: List[Dict[str, Any]] = []

    all_e_txt: List[np.ndarray] = []
    all_e_img: List[np.ndarray] = []
    all_e_fused: List[np.ndarray] = []

    for batch in tqdm(dataloader, desc=f"export {os.path.basename(out_path_prefix)}", leave=False):
        with accelerator.autocast():
            out = model(batch)
        e_txt = out["e_txt"].detach().cpu().float().numpy()
        e_img = out["e_img"].detach().cpu().float().numpy()
        e_fused = out["e_prod"].detach().cpu().float().numpy()
        scores = out["scores"]
        raw_gt = batch["raw_gt"]
        raw_candidates = batch["raw_candidates"]
        candidates_used = batch["candidates_used"]

        for i in range(e_fused.shape[0]):
            pred_idx = scores[i].argmax().item()
            all_meta.append({
                "idx": int(batch["idx"][i].item()),
                "ground_truth_category": normalize_category_path(raw_gt[i]),
                "target_candidate_index": int(batch["target_idx"][i].item()),
                "pred_candidate_index": pred_idx,
                "pred_category": candidates_used[i][pred_idx],
                "score_pred": float(scores[i, pred_idx].item()),
                "candidates_used": candidates_used[i],
                "raw_potential_product_categories": raw_candidates[i],
            })
        all_e_txt.append(e_txt)
        all_e_img.append(e_img)
        all_e_fused.append(e_fused)

    if accelerator.is_main_process:
        with open(out_path_prefix + ".jsonl", "w", encoding="utf-8") as f:
            for m in all_meta:
                f.write(json.dumps(m) + "\n")

        for key, arrays in [("", all_e_fused), ("_e_txt", all_e_txt), ("_e_img", all_e_img)]:
            embs = np.concatenate(arrays, axis=0)
            np.save(out_path_prefix + key + ".npy", embs.astype(np.float32))


@torch.no_grad()
def export_category_embeddings(
    cfg: TrainConfig,
    accelerator: Accelerator,
    model: TwoTowerReranker,
    categories: List[str],
    out_path_prefix: str,
):
    """
    Exports embeddings for unique category paths (useful for later visualization / ANN indexing).
      - {prefix}.npy : (num_categories, D)
      - {prefix}.json: list of category strings in the same row order
    Single-process only (mirrors export_product_embeddings); skipped when num_processes > 1
    to avoid duplicate work and duplicate rows from gather.
    """
    if accelerator.num_processes > 1:
        if accelerator.is_main_process:
            print("Category embedding export skipped (single-process only).")
        return

    model.eval()
    tokenizer = model.text_tokenizer
    batch_size = cfg.embed_batch_size

    all_embs: List[np.ndarray] = []
    for i in tqdm(range(0, len(categories), batch_size), desc="export categories", leave=False):
        chunk = categories[i : i + batch_size]
        texts = [f"Category: {normalize_category_path(c)}" for c in chunk]
        tok = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=cfg.max_text_len_category,
            return_tensors="pt",
        )
        tok = {k: v.to(accelerator.device) for k, v in tok.items()}
        with accelerator.autocast():
            e = model.encode_text(tok["input_ids"], tok["attention_mask"])
        all_embs.append(e.detach().cpu().float().numpy())

    embs = np.concatenate(all_embs, axis=0).astype(np.float32)
    np.save(out_path_prefix + ".npy", embs)
    with open(out_path_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump([normalize_category_path(c) for c in categories], f, indent=2)


def compute_dataset_stats(train_hf: Any, test_hf: Any) -> Dict[str, Any]:
    """Compute dataset stats from HF splits for logging. Uses column access only to avoid decoding images."""
    def _stats(split: Any) -> Dict[str, Any]:
        n = len(split)
        cands = split["potential_product_categories"]
        lengths = [len(c) if c else 0 for c in cands]
        return {
            "n": n,
            "avg_candidate_length": sum(lengths) / n if n else 0.0,
        }
    return {"train": _stats(train_hf), "test": _stats(test_hf)}
