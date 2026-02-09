"""
Utility functions: seeding and text normalization.
"""
import random
from typing import List, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_category_path(path: str) -> str:
    # Normalize spacing around ">" to reduce duplicate string variants.
    # (No semantic transformation; only formatting.)
    return " > ".join([p.strip() for p in path.split(">")])


def normalize_candidates_and_gt_index(cands: List[str], gt: str, max_candidates: int) -> Tuple[List[str], int]:
    """
    Normalize and dedupe candidates; take first max_candidates; return (window, gt_index).
    Dataset guarantees GT is always in candidates and max length <= 9.
    """
    gt_norm = normalize_category_path(gt)
    norm = [normalize_category_path(c) for c in cands if (c or "").strip()]
    seen: set[str] = set()
    norm = [c for c in norm if c not in seen and not seen.add(c)]
    window = norm[:max_candidates]
    gt_idx = window.index(gt_norm)  # guaranteed in dataset
    return window, gt_idx