# Product Catalogue Bi-Encoder Reranker: Technical Design

## Overview

Two-tower multimodal bi-encoder reranker for product categorization using the [Shopify product catalogue dataset](https://huggingface.co/datasets/Shopify/product-catalogue). Produces stable, reusable product embeddings suitable for visualization, clustering, and downstream tasks.

**Key Design Decisions:**
- Bi-encoder architecture (vs cross-encoder) for reusable embeddings
- Reranking formulation (vs pure classification) for fine-grained category distinctions
- Multi-positive contrastive loss with cross-batch queue for global embedding structure
- LoRA fine-tuning for compute efficiency

## Dataset

**Source:** [Shopify/product-catalogue](https://huggingface.co/datasets/Shopify/product-catalogue) on HuggingFace

**Statistics:**
- Train: ~39K products
- Test: ~10K products
- Unique categories: ~10K hierarchical category paths
- Max candidates per product: 9

**Inputs:**
- Product: image, title, description
- Candidates: list of category text strings (hierarchical paths)
- Ground truth: one category per product

**Reranking Formulation:** Pre-computed candidate lists enable reranking (rank among ~9 candidates) rather than classification (choose from all 10K categories). This:
- Forces fine-grained category distinctions in embedding space
- Reduces compute (score 9 vs 10K)
- Naturally encodes relative category relationships

## Architecture

### High-Level Design

**Bi-encoder (two-tower):**
- **Product tower:** Encodes products (multimodal: image + text)
- **Category tower:** Encodes category text strings
- **Scoring:** Dot product similarity in shared 512-d space

**Rationale:** Cross-encoders (concatenate inputs → single transformer) are more accurate but produce task-specific embeddings. Bi-encoders produce reusable embeddings: encode once, then search/cluster without re-encoding.

### Architecture Diagram

![Architecture Diagram](images/architecture_diagram.png "Architecture Diagram")

## Components

### Base Models

**Vision Tower:** SigLIP2 ViT-B (`google/siglip2-base-patch16-256`)
- Trained with contrastive objective -> well-normalized embeddings
- ViT-B scale balances quality and compute
- Output dimension: 1152 (from `vision_encoder.config.hidden_size`)

**Text Tower:** BGE v1.5 base (`BAAI/bge-base-en-v1.5`)
- Designed for retrieval/reranking tasks
- Trained on large-scale contrastive data
- Output dimension: 768 (from `text_encoder.config.hidden_size`)

### Projection Heads

**Architecture:** 2-layer MLP with LayerNorm

```
LayerNorm(in_dim) -> Linear(in_dim, 1024) -> GELU -> Linear(1024, 512) -> L2 Normalize
```

**Components:**
- **LayerNorm:** Stabilizes training when towers are frozen/partially frozen
- **Hidden layer (1024):** Provides extra capacity for richer mappings
- **GELU:** Non-linearity for flexible transformation
- **Output (512):** Shared embedding dimension for both towers
- **L2 Normalize:** Ensures cosine similarity = dot product, keeps embeddings on unit hypersphere

### Gated Fusion

**Purpose:** Combines vision and text embeddings into single product embedding.

**Architecture:**
```
gate = sigmoid(Linear([z_txt; z_img]))  # 512-d vector
e_fused = normalize(gate * z_txt + (1 - gate) * z_img)
```

**Properties:**
- Elementwise gate (512-d) -> per-dimension modality weighting
- Learned -> dynamically learns when to emphasize text vs image
- Modality dropout: 5% text, 2% image (training only) for robustness

**Rationale:** Gated fusion learns dynamic modality weights vs fixed concatenation/averaging. Enables text emphasis for descriptive products, image emphasis for visually distinctive items.

### Learned Logit Scale

**Purpose:** Controls softmax temperature. Larger scale → sharper softmax → stronger gradients. Learned so model adapts sharpness to embedding quality.

**Parameter:** `logit_scale` (initialized as `log(1/0.07) ≈ 14.3`, clamped [1, 100])

**Usage:** `scores = (e_prod @ e_cand.T) * scale`

**Rationale:** Borrowed from CLIP. Without scale, cosine similarities in [-1, 1] produce flat softmax → weak learning signal.

## Training

### Training Stages

**Stage 0** (1 epoch): Train projections + fusion only
- Both towers frozen (LoRA adapters applied but frozen)
- Establishes shared embedding space geometry

**Stage 1** (1 epoch): Unfreeze text LoRA
- Text LoRA: r=8, alpha=16
- Targets: linear layers matching (query, key, value, dense)
- Text often more informative for category disambiguation

**Stage 2** (1 epoch): Unfreeze vision LoRA
- Vision LoRA: r=4, alpha=8
- Targets: linear layers matching (q_proj, k_proj, v_proj, out_proj, o_proj)
- Smaller rank: images less directly tied to category semantics

**LoRA Configuration:**
- Targets attention linear layers for minimal parameters
- In BGE, "dense" suffix matches both attention and MLP layers, research suggests this improves performance
- Total trainable params: ~0.1% of model

**Rationale:** Staged training progressively adapts model without overwhelming memory. LoRA enables adaptation with minimal parameter overhead.

### Loss Functions

#### Primary: Listwise Softmax

```
scores = (e_product @ e_candidates.T) * scale  # (B, C)
loss = CrossEntropy(scores, target_idx)
```

Purpose: Rank correct category highest among candidates. Trains decision boundaries among candidate categories.

#### Secondary: Multi-Positive InfoNCE with Cross-Batch Queue

Purpose: Contrastive regularization to shape global embedding geometry.

Key Properties:
- **Multi-positive:** All keys with same `category_id` are positives
- **Cross-batch queue:** Keys = current batch + queue (2042 entries) -> more negatives, global structure
- **Symmetric:** Product->category and category->product, then average
- **Normalized:** Divide by |P(i)| so multi-positive examples don't dominate

Implementation:
```python
log_pos = logsumexp(logits[pos_mask])       # stable
log_denom = logsumexp(logits)
log_pos = log_pos - log(|P(i)|)             # normalize by # positives
loss_pc = -(log_pos_pc - log_denom_pc).mean()
loss_cp = -(log_pos_cp - log_denom_cp).mean()
loss = 0.5 * (loss_pc + loss_cp)
```

Cross-Batch Queue:
- FIFO buffer: (product_emb, category_emb, category_id) tuples
- Size: 2042 (fp16 for memory efficiency)
- Reset at stage boundaries
- Provides: (1) more negatives for sharper contrastive signal, (2) global structure via cross-batch consistency

Loss Weight: `contrastive_weight = 0.10` (regularizer, not primary objective)

Literature Basis:
- Multi-positive: Supervised Contrastive Learning (Khosla et al., 2020)
- Queue: MoCo (He et al., 2020)
- Symmetric: CLIP-style
- Normalization: SupCon

Contrastive Loss Diagram

![Contrastive Loss Diagram](images/contrastive_loss_diagram.png, "Contrastive Loss Diagram")

### Configuration

**Key Hyperparameters:**
- `embed_dim = 512`
- `proj_hidden_dim = 1024`
- `train_batch_size = 16`
- `lr = 2e-4`
- `weight_decay = 0.01`
- `warmup_ratio = 0.05`
- `grad_clip_norm = 1.0`
- `queue_size = 2042`
- `contrastive_weight = 0.10`
- `temperature_init = 0.07`

**Modality Dropout:**
- `drop_text_p = 0.05`
- `drop_image_p = 0.02`

**LoRA:**
- Text: `r=8`, `alpha=16`, `dropout=0.05`
- Vision: `r=4`, `alpha=8`, `dropout=0.05`

## Results

**Metrics** (reranking over ~9 candidates per product):
- Acc@1: 48.0%
- Acc@3: 77.9%
- Acc@5: 89.6%
- MRR: 0.653

**Random Baseline Estimate** (uniform among 9 candidates):
- Acc@1: ~11.1%
- Acc@3: ~33.3%
- Acc@5: ~55.6%
- MRR: ~0.31

**Training:** 6 epochs total (2 + 2 + 2 across stages), free-tier Colab T4

**Embedding Quality:** Good global structure, products cluster by category, neighborhoods are semantically meaningful.

![Viz Demo](images/demo_0.png, "Viz Demo")

## Visualization Pipeline

### Dimensionality Reduction

**Method:** UMAP (Uniform Manifold Approximation and Projection)

**Parameters:**
- `n_neighbors=30`: Balances local vs global structure
- `min_dist=0.05`: Allow more densely packed points
- `metric="cosine"`: Matches ranking similarity (embeddings are L2-normalized)
- `n_components=3`: 3D visualization

**Preprocessing:** PCA to 64-d before UMAP (speeds up nearest-neighbor search, removes noise)

### Web Application

**Stack:**
- React + Next.js (static export)
- Three.js + WebGL for 3D rendering
- Cloudflare Pages (deployment)
- Cloudflare R2 (static assets: embeddings, thumbnails)

**Features:**
- 3D point cloud (~49K products)
- Category stars (centroids)
- Hover tooltips
- Click selection with neighbor highlighting
- Search/filtering
- Misclassification highlighting

**Performance Optimizations:**
- Binary format for embeddings/metadata
- Shader-based filtering
- Lazy metadata loading (`requestIdleCallback`)
- Throttled hover picking

## Implementation Notes

**Code Structure:**
- `src/catalogue_biencoder/modeling.py`: Model components (ProjectionMLP, GatedFusion, TwoTowerReranker)
- `src/catalogue_biencoder/losses.py`: Contrastive loss, cross-batch queue
- `src/catalogue_biencoder/training/`: Training stages, LoRA setup, runner
- `src/catalogue_biencoder/config.py`: Configuration dataclass

**Key Implementation Details:**
- Shared BGE encoder for product text and category text
- Separate projection heads for vision and text
- Queue stores fp16 embeddings for memory efficiency
- LoRA adapters applied upfront, frozen until their stage
- Modality dropout: never drop both modalities (keep at least one)

**Outputs:**
- Checkpoints per stage
- Product embeddings (train/test)
- Category embeddings
- Training logs and metrics
