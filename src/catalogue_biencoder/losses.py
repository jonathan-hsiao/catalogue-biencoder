"""
Loss functions, metrics, and cross-batch queue for contrastive learning.
"""
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .config import TrainConfig


class CrossBatchQueue:
    """
    FIFO queue storing (e_prod, e_cat, cat_id) for cross-batch contrastive negatives.
    Stores on the same device for fast matmuls. Use fp16/bf16 to keep memory small.
    """
    def __init__(self, max_size: int, embed_dim: int, device: torch.device, dtype: torch.dtype = torch.float16):
        self.max_size = int(max_size)
        self.embed_dim = int(embed_dim)
        self.device = device
        self.dtype = dtype

        self.prod_q = torch.zeros((self.max_size, self.embed_dim), device=device, dtype=dtype)
        self.cat_q = torch.zeros((self.max_size, self.embed_dim), device=device, dtype=dtype)
        self.id_q = torch.full((self.max_size,), -1, device=device, dtype=torch.long)

        self.ptr = 0
        self.full = False

    @torch.no_grad()
    def enqueue(self, e_prod: torch.Tensor, e_cat: torch.Tensor, cat_id: torch.Tensor) -> None:
        # Expect e_* already detached
        e_prod = e_prod.to(device=self.device, dtype=self.dtype)
        e_cat = e_cat.to(device=self.device, dtype=self.dtype)
        cat_id = cat_id.to(device=self.device, dtype=torch.long)

        n = e_prod.size(0)
        if n == 0:
            return

        # If batch bigger than queue, keep only the last max_size
        if n > self.max_size:
            e_prod = e_prod[-self.max_size:]
            e_cat = e_cat[-self.max_size:]
            cat_id = cat_id[-self.max_size:]
            n = self.max_size

        end = self.ptr + n
        if end <= self.max_size:
            self.prod_q[self.ptr:end] = e_prod
            self.cat_q[self.ptr:end] = e_cat
            self.id_q[self.ptr:end] = cat_id
        else:
            first = self.max_size - self.ptr
            second = end - self.max_size
            self.prod_q[self.ptr:] = e_prod[:first]
            self.cat_q[self.ptr:] = e_cat[:first]
            self.id_q[self.ptr:] = cat_id[:first]
            self.prod_q[:second] = e_prod[first:]
            self.cat_q[:second] = e_cat[first:]
            self.id_q[:second] = cat_id[first:]

        self.ptr = end % self.max_size
        if end >= self.max_size:
            self.full = True

    def is_ready(self) -> bool:
        return self.full or (self.ptr > 0)

    def current_size(self) -> int:
        """Current number of entries in the queue (for logging)."""
        return self.max_size if self.full else self.ptr

    def reset(self) -> None:
        """Clear the queue; call at stage boundaries so stale embeddings from old geometry are not used."""
        self.ptr = 0
        self.full = False
        self.id_q.fill_(-1)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.is_ready():
            return (
                torch.empty((0, self.embed_dim), device=self.device, dtype=self.dtype),
                torch.empty((0, self.embed_dim), device=self.device, dtype=self.dtype),
                torch.empty((0,), device=self.device, dtype=torch.long),
            )
        if self.full:
            return self.prod_q, self.cat_q, self.id_q
        return self.prod_q[:self.ptr], self.cat_q[:self.ptr], self.id_q[:self.ptr]


def listwise_loss(scores: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    # scores: (B, C). Cast to fp32 for CE to avoid edge instability with -inf masks under fp16.
    return F.cross_entropy(scores.float(), target_idx)


def multi_pos_cross_modal_infoNCE_loss(
    e_prod: torch.Tensor,        # (B, D)
    e_cat: torch.Tensor,         # (B, D)  -- GT category embedding per example
    cat_id: torch.Tensor,        # (B,)    -- duplicates allowed
    scale: torch.Tensor,         # scalar logit scale (same as reranking; clamp 1..100)
    queue_prod: torch.Tensor,
    queue_cat: torch.Tensor,
    queue_id: torch.Tensor,
    normalize_pos: bool = True,  # prevents big duplicate-groups dominating
) -> torch.Tensor:
    """
    Symmetric multi-positive InfoNCE with cross-batch queue. 
    Handles duplicate categories by treating same-category entries as positives.
      - p->c uses keys: [batch cats + queued cats]
      - c->p uses keys: [batch prods + queued prods]
    Queue provides additional negatives (and occasional positives if same cat_id repeats).
    """
    device = e_prod.device
    B = e_prod.size(0)
    if B <= 1:
        return torch.zeros((), device=device, dtype=e_prod.dtype)

    # Build key banks
    keys_cat = torch.cat([e_cat, queue_cat.to(device=device, dtype=e_cat.dtype)], dim=0)  # (B+Q, D)
    ids_cat = torch.cat([cat_id, queue_id.to(device=device)], dim=0)  # (B+Q,)

    keys_prod = torch.cat([e_prod, queue_prod.to(device=device, dtype=e_prod.dtype)], dim=0)  # (B+Q, D)
    ids_prod = torch.cat([cat_id, queue_id.to(device=device)], dim=0)  # (B+Q,)

    # ---- p -> c ----
    logits_pc = (e_prod @ keys_cat.t()).float() * scale.float()  # (B, B+Q)
    pos_pc = (cat_id[:, None].to(device) == ids_cat[None, :])  # (B, B+Q)

    log_denom_pc = torch.logsumexp(logits_pc, dim=1)
    logits_pos_pc = logits_pc.masked_fill(~pos_pc, -1e9)
    log_pos_pc = torch.logsumexp(logits_pos_pc, dim=1)

    if normalize_pos:
        pos_count = pos_pc.sum(dim=1).clamp_min(1).float()
        log_pos_pc = log_pos_pc - pos_count.log()

    loss_pc = -(log_pos_pc - log_denom_pc).mean()

    # ---- c -> p ----
    logits_cp = (e_cat @ keys_prod.t()).float() * scale.float()  # (B, B+Q)
    pos_cp = (cat_id[:, None].to(device) == ids_prod[None, :])  # (B, B+Q)

    log_denom_cp = torch.logsumexp(logits_cp, dim=1)
    logits_pos_cp = logits_cp.masked_fill(~pos_cp, -1e9)
    log_pos_cp = torch.logsumexp(logits_pos_cp, dim=1)

    if normalize_pos:
        pos_count = pos_cp.sum(dim=1).clamp_min(1).float()
        log_pos_cp = log_pos_cp - pos_count.log()

    loss_cp = -(log_pos_cp - log_denom_cp).mean()

    return 0.5 * (loss_pc + loss_cp)


def contrastive_weight_for_stage(cfg: TrainConfig, stage_index: int) -> float:
    """
    Return the contrastive loss weight for the given stage.
    """
    w = [cfg.contrastive_weight_stage0, cfg.contrastive_weight_stage1, cfg.contrastive_weight_stage2]
    return w[stage_index] if stage_index < len(w) and w[stage_index] is not None else cfg.contrastive_weight


@torch.no_grad()
def compute_ranking_metrics(scores: torch.Tensor, target_idx: torch.Tensor, k_list=(1, 3, 5)) -> Dict[str, float]:
    """
    scores: (B,C), higher is better
    """
    B, C = scores.shape
    metrics = {}

    # ranks
    sorted_idx = torch.argsort(scores, dim=1, descending=True)  # (B,C)
    # position of target in sorted list
    target = target_idx.view(-1, 1)
    match = (sorted_idx == target).nonzero(as_tuple=False)  # rows: [row, col]
    # By construction, should be exactly one match per row when gt is present in candidates.
    # Still guard against anomalies.
    rank = torch.full((B,), fill_value=C, device=scores.device, dtype=torch.long)
    if match.numel() > 0:
        rank[match[:, 0]] = match[:, 1]

    for k in k_list:
        metrics[f"acc@{k}"] = (rank < k).float().mean().item()

    # MRR
    metrics["mrr"] = (1.0 / (rank.float() + 1.0)).mean().item()
    return metrics
