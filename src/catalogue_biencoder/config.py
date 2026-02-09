"""
Configuration dataclass for training.
"""
from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Data
    hf_dataset_name: str = "Shopify/product-catalogue"
    train_split: str = "train"
    test_split: str = "test"

    # Model IDs (per spec)
    vision_ckpt: str = "google/siglip2-base-patch16-256"  # SigLIP2 Base (ViT-B scale)
    text_ckpt: str = "BAAI/bge-base-en-v1.5"

    # Tokenization / input
    max_text_len_product: int = 256
    max_text_len_category: int = 128

    # Embedding space
    embed_dim: int = 512
    proj_hidden_dim: int = 1024  # 2-layer MLP hidden dim

    # Modality dropout (robustness)
    drop_text_p: float = 0.05
    drop_image_p: float = 0.02

    # Candidate list handling
    max_candidates: int = 9  # dataset max is 9; pad to this for batching

    # Loss weights
    contrastive_weight: float = 0.10  # small regularizer to improve embedding geometry
    # Optional per-stage contrastive weight (representation first, task alignment later)
    contrastive_weight_stage0: float | None = None  # if set, overrides contrastive_weight for stage0
    contrastive_weight_stage1: float | None = None  # if set, overrides contrastive_weight for stage1
    contrastive_weight_stage2: float | None = None  # if set, overrides contrastive_weight for stage2
    temperature_init: float = 0.07    # initial 1/scale for cosine sims; learned logit_scale (clamp 1..100) for reranking + contrastive

    # Cross-batch queue for global embedding structure (MoCo-style; more negatives for contrastive loss)
    queue_size: int = 2042

    # Optimization
    seed: int = 42
    mixed_precision: str = "fp16"  # "no" | "fp16" | "bf16"
    train_batch_size: int = 16
    eval_batch_size: int = 32
    embed_batch_size: int = 64
    num_workers: int = 2

    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    grad_clip_norm: float | None = 1.0
    grad_accum_steps: int = 1

    # Stage schedule (epochs)
    stage0_epochs: int = 1
    stage1_epochs: int = 1
    stage2_epochs: int = 1

    # LoRA defaults
    text_lora_r: int = 8
    text_lora_alpha: int = 16
    text_lora_dropout: float = 0.05

    vision_lora_r: int = 4
    vision_lora_alpha: int = 8
    vision_lora_dropout: float = 0.05

    tower_eval_during_training: bool = False  # set True for prioritizing embedding stability

    # Eval
    eval_each_epoch: bool = False  # if True, evaluate on test each epoch. False to avoid peeking during training.

    # Logging
    log_every_n_steps: int = 10  # buffer and write training_log.jsonl every N steps (reduces I/O)

    # Output
    output_dir: str = "artifacts/runs_product_catalogue"
    run_name: str = ""  # auto-filled if empty
