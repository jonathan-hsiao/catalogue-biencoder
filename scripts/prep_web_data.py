# =========================
# WEB DATA PREP (copy and run in Colab) — MERGED DATASET (train+test) WITH SPLIT FLAGS
# + COARSE CATEGORY STARS (depth=2 or 3) WITH ALL/TRAIN/TEST COUNTS
# =========================
#
# Goal:
# Prepare web data for a web application that displays an interactive visualization of product embeddings.
# Features:
# - 3D visualization of product embeddings with pan/zoom 
# - Category constellations - color by top-level category, show granular category centroids as "stars", draw faint "gravity lines" for selected category 
# - Mouseover tooltip-style details (image, title, brand, description) 
# - Click product point for neighborhood compare 
# - Highlight neighbor points and display neighbor details (title, description, brand, image) in slide-in sidebar, with ability to switch between text space, image space, and fused space 
# - Ability to highlight "Ambiguous products" (defined as products that were categorized incorrectly by the biencoder) 
# Visual polish: 
# - Must be beautiful, "wow" factor 
# - Dark space theme with subtle bloom/glow sprites that look like stars/planets 
# - Smooth camera transitions (fly-to selection, zoom/pan around in 3D space) 
# Tech stack: 
# - webgl, Next.js, React
#
# Assumes your Google Drive has:
# - Thumbnails zips:
#     MyDrive/catalogue_biencoder_runs/images/train_images.zip
#     MyDrive/catalogue_biencoder_runs/images/test_images.zip
# - Text metadata:
#     MyDrive/catalogue_biencoder_runs/raw_text/train_meta.jsonl
#     MyDrive/catalogue_biencoder_runs/raw_text/test_meta.jsonl
# - Embeddings + embedding metadata:
#     MyDrive/catalogue_biencoder_runs/output/embeddings/
#         train_product_emb.npy
#         train_product_emb_e_txt.npy
#         train_product_emb_e_img.npy
#         train_product_emb.jsonl
#         test_product_emb.npy
#         test_product_emb_e_txt.npy
#         test_product_emb_e_img.npy
#         test_product_emb.jsonl
#         unique_category_emb.npy
#         unique_category_emb.json
#
# Produces:
#   MyDrive/catalogue_biencoder_runs/web_export_v1/
#     web_data/v1/... (binaries + sharded json + manifest)
#     thumbs_150px_public.zip  (a single zip containing thumbs/train/*.webp + thumbs/test/*.webp)
#
# Frontend assumptions:
# - Thumbnails served at: /thumbs/{split}/{split_idx}.webp
# - Web data served at:   /web_data/v1/...
#
# View modes:
# - GLOBAL: use all points
# - TRAIN-only / TEST-only: filter by split_id in JS; neighbors are global and filtered in UI
#
# Neighbors:
# - Stores GLOBAL neighbors only (K_GLOBAL). UI filters to visible neighbors by split.
#
# Category stars (COARSE):
# - Uses coarse GT categories at depth=COARSE_DEPTH to compute ONE constellation (GLOBAL centroid positions).
# - Writes counts_all/count_train/count_test per star.
# - Robust to zero-product categories: centroids are computed ONLY for categories with counts>=max(MIN_STAR_COUNT,1)

!pip -q install umap-learn faiss-cpu # optional: scipy pandas numpy tqdm scikit-learn

import os, json, math, zipfile
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.decomposition import PCA
import umap
from scipy.linalg import orthogonal_procrustes
import faiss

# ---- Mount Drive (Colab) ----
from google.colab import drive
drive.mount("/content/drive")

# -------------------------
# Paths in your Drive
# -------------------------
DRIVE_ROOT = "/content/drive/MyDrive/catalogue_biencoder_runs"

THUMBS_ZIP_DIR = os.path.join(DRIVE_ROOT, "images")
TRAIN_THUMBS_ZIP = os.path.join(THUMBS_ZIP_DIR, "train_images.zip")
TEST_THUMBS_ZIP  = os.path.join(THUMBS_ZIP_DIR, "test_images.zip")

TEXT_DIR = os.path.join(DRIVE_ROOT, "raw_text")
TRAIN_META_JSONL = os.path.join(TEXT_DIR, "train_meta.jsonl")
TEST_META_JSONL  = os.path.join(TEXT_DIR, "test_meta.jsonl")

EMB_DIR = os.path.join(DRIVE_ROOT, "output", "embeddings")

# Output
OUT_ROOT = os.path.join(DRIVE_ROOT, "web_export_v1")
WEB_DATA_DIR = os.path.join(OUT_ROOT, "web_data", "v1")
DATA_DIR = os.path.join(WEB_DATA_DIR, "data")
CAT_DIR  = os.path.join(WEB_DATA_DIR, "categories")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CAT_DIR, exist_ok=True)

# -------------------------
# Tunables
# -------------------------
SEED = 42

# Layout
PCA_DIM = 64
UMAP_NEIGHBORS = 30
UMAP_MIN_DIST = 0.05

# Neighbors:
K_UI = 10         # what you show in the UI (after filtering)
K_GLOBAL = 50     # stored global neighbors per point; UI filters to train/test/both

# Coarse category stars
COARSE_DEPTH = 2      # <<<<<< set to 2 or 3
MIN_STAR_COUNT = 10   # minimum number of products for a coarse category to get a star

# Shards
SHARD_SIZE = 5000

# Public thumbs URL base (your Next.js will serve from /public/thumbs/...)
THUMB_BASE_URL = "/thumbs"

# -------------------------
# Helpers
# -------------------------
def read_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)

def assert_l2_norm(X: np.ndarray, name: str = "X", mode: str = "fail", tol: float = 1e-3) -> np.ndarray:
    """
    Checks row-wise L2 normalization (||x|| ~= 1).
    mode:
      - "fail": raise ValueError if any row is not within tol
      - "fix":  renormalize all rows (safe if no zero vectors)
    Returns X (possibly normalized).
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"{name}: expected 2D array, got shape {X.shape}")

    if not np.isfinite(X).all():
        raise ValueError(f"{name}: contains NaN/Inf")

    norms = np.linalg.norm(X, axis=1)
    if (norms == 0).any():
        i = int(np.where(norms == 0)[0][0])
        raise ValueError(f"{name}: zero vector at row {i}")

    # quick stats
    mean, mn, mx = norms.mean(), norms.min(), norms.max()
    frac_ok = float(np.mean(np.abs(norms - 1.0) <= tol))
    print(f"{name}: shape={X.shape} norms mean={mean:.6f} min={mn:.6f} max={mx:.6f} ok={frac_ok:.4f} (tol={tol})")

    ok = np.abs(norms - 1.0) <= tol
    if ok.all():
        return X

    if mode == "fix":
        X = X.astype(np.float32, copy=False)
        X /= norms[:, None]
        # re-check one last time
        norms2 = np.linalg.norm(X, axis=1)
        if not np.all(np.abs(norms2 - 1.0) <= tol * 5):
            raise ValueError(f"{name}: normalization produced unexpected norms")
        print(f"{name}: fixed {int((~ok).sum())} rows")
        return X

    if mode == "fail":
        i = int(np.where(~ok)[0][0])
        raise ValueError(f"{name}: not L2-normalized (example row {i}, norm={norms[i]:.6f})")

    raise ValueError(f"{name}: mode must be 'fail' or 'fix'")

def normalize_path(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    return " > ".join([p.strip() for p in s.split(">")])

def coarsen_path(cat_str: str, depth: int) -> str:
    cat_str = normalize_path(cat_str)
    if not cat_str:
        return ""
    parts = [p.strip() for p in cat_str.split(" > ") if p.strip()]
    if not parts:
        return ""
    d = max(1, int(depth))
    return " > ".join(parts[:min(d, len(parts))])

def top_level(cat_str: str) -> str:
    cat_str = normalize_path(cat_str)
    return cat_str.split(" > ")[0].strip() if cat_str else ""

def ensure_embeddings_aligned(E: np.ndarray, df_with_idx_col: pd.DataFrame, name: str) -> np.ndarray:
    """
    Ensures embeddings are in row == idx order (0..N-1).
    If df idx column is already [0..N-1] in order, returns E unchanged.
    Else, reindexes into an array of shape (max_idx+1, D) with out[idx]=E[row].
    """
    if "idx" not in df_with_idx_col.columns:
        raise ValueError(f"{name}: dataframe must include an 'idx' column")

    idxs = df_with_idx_col["idx"].to_numpy().astype(int)
    if len(idxs) != E.shape[0]:
        raise ValueError(f"{name}: len(idx)={len(idxs)} != embeddings rows={E.shape[0]}")

    if np.array_equal(idxs, np.arange(len(idxs))):
        return E

    N = int(idxs.max()) + 1
    out = np.zeros((N, E.shape[1]), dtype=E.dtype)
    out[idxs] = E
    return out

def assert_contiguous_index(df: pd.DataFrame, expected_n: int, name: str):
    idx = df.index.to_numpy().astype(int)
    if len(idx) != expected_n:
        raise ValueError(f"{name}: expected {expected_n} rows, got {len(idx)}")
    if not np.array_equal(idx, np.arange(expected_n)):
        raise ValueError(
            f"{name}: index is not contiguous 0..{expected_n-1}. "
            f"Example head={idx[:10].tolist()} tail={idx[-10:].tolist()}"
        )

def center_scale(P: np.ndarray) -> np.ndarray:
    P = P - P.mean(axis=0, keepdims=True)
    s = np.max(np.linalg.norm(P, axis=1))
    return (P / (s + 1e-9)).astype(np.float32)

def procrustes_align(P: np.ndarray, ref: np.ndarray) -> np.ndarray:
    R, _ = orthogonal_procrustes(P, ref)
    return (P @ R).astype(np.float32)

def umap3_with_pca(X: np.ndarray, pca_dim=64, seed=42, n_neighbors=30, min_dist=0.05) -> np.ndarray:
    X = X.astype(np.float32)
    if pca_dim and pca_dim < X.shape[1]:
        pca = PCA(n_components=pca_dim, random_state=seed)
        Xr = pca.fit_transform(X).astype(np.float32)
    else:
        Xr = X

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
        transform_seed=seed,
        verbose=False,
        low_memory=True,
    )
    return reducer.fit_transform(Xr).astype(np.float32)

def write_bin(path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.ascontiguousarray(arr)
    with open(path, "wb") as f:
        f.write(arr.tobytes())

def read_bin(path: str, dtype: np.dtype, shape: tuple) -> np.ndarray:
    """
    Read binary file written by write_bin.
    Args:
        path: Path to binary file
        dtype: numpy dtype (e.g., np.float32, np.int32)
        shape: Expected shape tuple (e.g., (N, 3))
    Returns:
        Array with given dtype and shape
    """
    with open(path, "rb") as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=dtype)
    return arr.reshape(shape)

def write_cache_key(path: str, key: dict) -> None:
    """Write cache key JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(key, f, indent=2, sort_keys=True)

def read_cache_key(path: str) -> dict:
    """Read cache key JSON file. Returns None if file doesn't exist."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def cache_key_matches(cache_key_path: str, expected_key: dict) -> bool:
    """Check if cache key exists and matches expected values."""
    cached_key = read_cache_key(cache_key_path)
    if cached_key is None:
        return False
    return cached_key == expected_key

def shard_ordered_json_by_idx(df: pd.DataFrame, out_dir: str, shard_size: int, make_record_fn):
    """
    Writes meta_{shard:03d}.json where each shard is a LIST in index order.
    Frontend can locate by:
      shard = id // shard_size
      offset = id % shard_size
    """
    os.makedirs(out_dir, exist_ok=True)
    df = df.sort_index()
    N = len(df)
    num_shards = math.ceil(N / shard_size)
    for s in tqdm(range(num_shards), desc=f"shard:{os.path.basename(out_dir)}"):
        lo = s * shard_size
        hi = min(N, (s + 1) * shard_size)
        chunk = df.iloc[lo:hi]
        out = [make_record_fn(idx, row) for idx, row in chunk.iterrows()]
        with open(os.path.join(out_dir, f"meta_{s:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)

def build_thumbs_public_zip(train_zip_path: str, test_zip_path: str, out_zip_path: str):
    """
    Creates a single zip with:
      thumbs/train/{idx}.webp
      thumbs/test/{idx}.webp
    without exploding Google Drive into 50k small files.
    """
    def add_zip(zsrc_path: str, split: str, zout: zipfile.ZipFile):
        with zipfile.ZipFile(zsrc_path, "r") as zsrc:
            names = [n for n in zsrc.namelist() if n.lower().endswith(".webp") and not n.endswith("/")]
            if not names:
                raise ValueError(f"No .webp files found in {zsrc_path}")

            for n in tqdm(names, desc=f"pack:{split}", leave=False):
                base = os.path.basename(n)  # e.g. 123.webp
                stem = os.path.splitext(base)[0]
                if not stem.isdigit():
                    continue
                data = zsrc.read(n)
                zout.writestr(f"thumbs/{split}/{stem}.webp", data)

    os.makedirs(os.path.dirname(out_zip_path), exist_ok=True)
    with zipfile.ZipFile(out_zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zout:
        add_zip(train_zip_path, "train", zout)
        add_zip(test_zip_path, "test", zout)
    print("Wrote thumbs zip:", out_zip_path)

def knn_faiss_hnsw_ip(X: np.ndarray, k: int, m: int = 32, ef_search: int = 96, ef_construction: int = 200):
    """
    Approximate kNN using HNSW (inner product).
    Much faster than brute-force IndexFlatIP for N~50k and k~200 on CPU.
    Returns (idx, sims) with self removed.
    """
    X = np.ascontiguousarray(X.astype(np.float32))
    d = X.shape[1]

    index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(X)

    index.hnsw.efSearch = ef_search
    sims, idx = index.search(X, k + 1)  # includes self
    return idx[:, 1:].astype(np.int32), sims[:, 1:].astype(np.float32)

def build_candidate_id_matrix(emb_meta_df: pd.DataFrame, cat2id: dict, max_cands: int = 9) -> np.ndarray:
    N = len(emb_meta_df)
    out = np.full((N, max_cands), -1, dtype=np.int32)

    for row in tqdm(emb_meta_df.itertuples(index=True), total=N, desc="cand_ids", leave=False):
        idx = int(row.Index)
        cands = getattr(row, "candidates_used", None) or []
        for j, c in enumerate(cands[:max_cands]):
            c_normalized = normalize_path(c)
            if c_normalized:  # ← This check is needed!
                out[idx, j] = cat2id.get(c_normalized, -1)

    return out

def compute_margin_batched(E_prod: np.ndarray, cand_ids: np.ndarray, E_cat: np.ndarray, batch: int = 2048) -> np.ndarray:
    """
    margin = top1 - top2 over candidate similarities (dot product) using LEAF category embeddings.
    """
    N, C = cand_ids.shape
    margins = np.zeros((N,), dtype=np.float32)
    for start in tqdm(range(0, N, batch), desc="margin", leave=False):
        end = min(N, start + batch)
        E = E_prod[start:end].astype(np.float32)  # (b, D)
        ids = cand_ids[start:end]                 # (b, C)
        b = end - start

        sims = np.full((b, C), -1e9, dtype=np.float32)
        for j in range(C):
            valid = ids[:, j] >= 0
            if valid.any():
                sims[valid, j] = (E[valid] * E_cat[ids[valid, j]]).sum(axis=1)

        top2 = np.partition(sims, kth=-2, axis=1)[:, -2:]
        top1 = top2.max(axis=1)
        top2v = top2.min(axis=1)
        margins[start:end] = (top1 - top2v).astype(np.float32)
    return margins

def safe_centroids_and_counts(P_xyz: np.ndarray, ids: np.ndarray, num_ids: int, min_count: int):
    """
    Robust centroids: ONLY returns centroids for ids with count >= max(min_count, 1).
    Handles zero-product categories safely (never divides by zero).
    Returns:
      keep_ids (int32), centroids (float32 Nx3), counts (int32)
    """
    ids = ids.astype(np.int32, copy=False)
    valid = (ids >= 0) & (ids < num_ids)
    ids_v = ids[valid]
    P_v = P_xyz[valid].astype(np.float64)

    sums = np.zeros((num_ids, 3), dtype=np.float64)
    counts = np.zeros((num_ids,), dtype=np.int64)

    if len(ids_v) > 0:
        np.add.at(sums, ids_v, P_v)
        np.add.at(counts, ids_v, 1)

    thr = max(int(min_count), 1)
    keep = counts >= thr
    keep_ids = np.where(keep)[0].astype(np.int32)

    if keep_ids.size == 0:
        return keep_ids, np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    centroids = (sums[keep_ids] / counts[keep_ids][:, None]).astype(np.float32)
    # IMPORTANT: Do NOT re-center/re-scale centroids; they must remain in the SAME coordinate system as points.
    return keep_ids, centroids, counts[keep_ids].astype(np.int32)

def counts_per_id(ids: np.ndarray, num_ids: int) -> np.ndarray:
    ids = ids.astype(np.int32, copy=False)
    valid = (ids >= 0) & (ids < num_ids)
    out = np.zeros((num_ids,), dtype=np.int32)
    if valid.any():
        np.add.at(out, ids[valid], 1)
    return out

# -------------------------
# Load inputs
# -------------------------
print("Loading text metadata...")
train_meta = read_jsonl(TRAIN_META_JSONL).set_index("idx").sort_index()
test_meta  = read_jsonl(TEST_META_JSONL).set_index("idx").sort_index()

print("Loading embedding metadata JSONL...")
train_emb_meta = read_jsonl(os.path.join(EMB_DIR, "train_product_emb.jsonl")).set_index("idx").sort_index()
test_emb_meta  = read_jsonl(os.path.join(EMB_DIR, "test_product_emb.jsonl")).set_index("idx").sort_index()

print("Loading embeddings .npy...")
E_train_fused = np.load(os.path.join(EMB_DIR, "train_product_emb.npy")).astype(np.float32)
E_train_txt   = np.load(os.path.join(EMB_DIR, "train_product_emb_e_txt.npy")).astype(np.float32)
E_train_img   = np.load(os.path.join(EMB_DIR, "train_product_emb_e_img.npy")).astype(np.float32)

E_test_fused  = np.load(os.path.join(EMB_DIR, "test_product_emb.npy")).astype(np.float32)
E_test_txt    = np.load(os.path.join(EMB_DIR, "test_product_emb_e_txt.npy")).astype(np.float32)
E_test_img    = np.load(os.path.join(EMB_DIR, "test_product_emb_e_img.npy")).astype(np.float32)

print("Loading LEAF categories (+ embeddings for margin)...")
leaf_cats = json.load(open(os.path.join(EMB_DIR, "unique_category_emb.json"), "r", encoding="utf-8"))
leaf_cats = [normalize_path(c) for c in leaf_cats]
E_leaf_cat = np.load(os.path.join(EMB_DIR, "unique_category_emb.npy")).astype(np.float32)
leaf_cat2id = {c: i for i, c in enumerate(leaf_cats)}

print("Asserting L2 norms...") # UMAP metric "cosine", FAISS metric "inner product", margin uses "dot product"
E_train_fused = assert_l2_norm(E_train_fused, "E_train_fused", mode="fail", tol=1e-3)
E_test_fused  = assert_l2_norm(E_test_fused,  "E_test_fused",  mode="fail", tol=1e-3)
E_train_txt   = assert_l2_norm(E_train_txt,   "E_train_txt",   mode="fail", tol=1e-3)
E_test_txt    = assert_l2_norm(E_test_txt,    "E_test_txt",    mode="fail", tol=1e-3)
E_train_img   = assert_l2_norm(E_train_img,   "E_train_img",   mode="fail", tol=1e-3)
E_test_img    = assert_l2_norm(E_test_img,    "E_test_img",    mode="fail", tol=1e-3)
E_leaf_cat    = assert_l2_norm(E_leaf_cat,    "E_leaf_cat",    mode="fail", tol=1e-3)

# Normalize categories in meta
train_meta["ground_truth_category"] = train_meta["ground_truth_category"].map(normalize_path)
test_meta["ground_truth_category"]  = test_meta["ground_truth_category"].map(normalize_path)

train_emb_meta["pred_category"] = train_emb_meta["pred_category"].map(normalize_path)
test_emb_meta["pred_category"]  = test_emb_meta["pred_category"].map(normalize_path)

# Defensive checks: meta idx should match emb_meta idx (same dataset split)
if not train_meta.index.equals(train_emb_meta.index):
    raise ValueError("train_meta idx set != train_emb_meta idx set (cannot safely join)")
if not test_meta.index.equals(test_emb_meta.index):
    raise ValueError("test_meta idx set != test_emb_meta idx set (cannot safely join)")

N_train = len(train_meta)
N_test  = len(test_meta)
N_all   = N_train + N_test

assert_contiguous_index(train_meta, N_train, "train_meta")
assert_contiguous_index(test_meta,  N_test,  "test_meta")

# -------------------------
# Merge train + test into global id space
# global_id = train_idx for train, N_train + test_idx for test
# -------------------------
split_id  = np.concatenate([np.zeros(N_train, dtype=np.uint8), np.ones(N_test, dtype=np.uint8)])        # 0 train, 1 test
split_idx = np.concatenate([np.arange(N_train, dtype=np.uint32), np.arange(N_test, dtype=np.uint32)])   # for /thumbs/{split}/{split_idx}.webp

# -------------------------
# Build COARSE category vocabulary (from GT only) + per-product coarse ids
# -------------------------
print(f"Building coarse categories at depth={COARSE_DEPTH} ...")
train_gt_coarse = train_meta["ground_truth_category"].map(lambda s: coarsen_path(s, COARSE_DEPTH))
test_gt_coarse  = test_meta["ground_truth_category"].map(lambda s: coarsen_path(s, COARSE_DEPTH))

# Vocabulary from GT only => avoids lots of unused coarse cats
coarse_cats = sorted({c for c in pd.concat([train_gt_coarse, test_gt_coarse]).tolist() if c})
coarse2id = {c: i for i, c in enumerate(coarse_cats)}
M_coarse = len(coarse_cats)
print("Coarse categories:", M_coarse)

train_gt_coarse_id = np.array([coarse2id.get(c, -1) for c in train_gt_coarse], dtype=np.int32)
test_gt_coarse_id  = np.array([coarse2id.get(c, -1) for c in test_gt_coarse],  dtype=np.int32)

# Also coarse preds (for UI comparisons); may be -1 if something weird
train_pred_coarse = train_emb_meta["pred_category"].map(lambda s: coarsen_path(s, COARSE_DEPTH))
test_pred_coarse  = test_emb_meta["pred_category"].map(lambda s: coarsen_path(s, COARSE_DEPTH))
train_pred_coarse_id = np.array([coarse2id.get(c, -1) for c in train_pred_coarse], dtype=np.int32)
test_pred_coarse_id  = np.array([coarse2id.get(c, -1) for c in test_pred_coarse],  dtype=np.int32)

# -------------------------
# Labels: leaf gt/pred ids + correctness + top-level ids (for coloring)
# -------------------------
train_gt_leaf_id   = np.array([leaf_cat2id.get(c, -1) for c in train_meta["ground_truth_category"]], dtype=np.int32)
test_gt_leaf_id    = np.array([leaf_cat2id.get(c, -1) for c in test_meta["ground_truth_category"]], dtype=np.int32)
train_pred_leaf_id = np.array([leaf_cat2id.get(c, -1) for c in train_emb_meta["pred_category"]], dtype=np.int32)
test_pred_leaf_id  = np.array([leaf_cat2id.get(c, -1) for c in test_emb_meta["pred_category"]], dtype=np.int32)

train_correct = (train_gt_leaf_id == train_pred_leaf_id).astype(np.uint8)
test_correct  = (test_gt_leaf_id == test_pred_leaf_id).astype(np.uint8)
train_amb = (train_correct == 0).astype(np.uint8)
test_amb  = (test_correct == 0).astype(np.uint8)

# Top-level mapping (derive from GT categories to avoid unused)
top_levels = sorted({top_level(c) for c in pd.concat([train_meta["ground_truth_category"], test_meta["ground_truth_category"]]).tolist() if c})
top2id = {t: i for i, t in enumerate(top_levels)}

train_gt_top   = np.array([top2id.get(top_level(c), -1) for c in train_meta["ground_truth_category"]], dtype=np.int16)
test_gt_top    = np.array([top2id.get(top_level(c), -1) for c in test_meta["ground_truth_category"]], dtype=np.int16)
train_pred_top = np.array([top2id.get(top_level(c), -1) for c in train_emb_meta["pred_category"]], dtype=np.int16)
test_pred_top  = np.array([top2id.get(top_level(c), -1) for c in test_emb_meta["pred_category"]], dtype=np.int16)

# Merge labels into global arrays
all_gt_leaf_id      = np.concatenate([train_gt_leaf_id,      test_gt_leaf_id]).astype(np.int32)
all_pred_leaf_id    = np.concatenate([train_pred_leaf_id,    test_pred_leaf_id]).astype(np.int32)
all_correct         = np.concatenate([train_correct,         test_correct]).astype(np.uint8)
all_amb             = np.concatenate([train_amb,             test_amb]).astype(np.uint8)

all_gt_top          = np.concatenate([train_gt_top,          test_gt_top]).astype(np.int16)
all_pred_top        = np.concatenate([train_pred_top,        test_pred_top]).astype(np.int16)

all_gt_coarse_id    = np.concatenate([train_gt_coarse_id,    test_gt_coarse_id]).astype(np.int32)
all_pred_coarse_id  = np.concatenate([train_pred_coarse_id,  test_pred_coarse_id]).astype(np.int32)

# -------------------------
# Candidate-based margins (confidence proxy) using LEAF category embeddings
# -------------------------
print("Building candidate id matrices (leaf)...")
train_cand_ids = build_candidate_id_matrix(train_emb_meta, leaf_cat2id, max_cands=9)
test_cand_ids  = build_candidate_id_matrix(test_emb_meta,  leaf_cat2id, max_cands=9)

# -------------------------
# Reindex embeddings by idx (defensive) and merge
# -------------------------
print("Reindexing embeddings to match idx order (defensive)...")
train_idxcol = train_emb_meta.reset_index()[["idx"]]
test_idxcol  = test_emb_meta.reset_index()[["idx"]]

E_train_fused = ensure_embeddings_aligned(E_train_fused, train_idxcol, "E_train_fused")
E_train_txt   = ensure_embeddings_aligned(E_train_txt,   train_idxcol, "E_train_txt")
E_train_img   = ensure_embeddings_aligned(E_train_img,   train_idxcol, "E_train_img")

E_test_fused  = ensure_embeddings_aligned(E_test_fused,  test_idxcol,  "E_test_fused")
E_test_txt    = ensure_embeddings_aligned(E_test_txt,    test_idxcol,  "E_test_txt")
E_test_img    = ensure_embeddings_aligned(E_test_img,    test_idxcol,  "E_test_img")

if E_train_fused.shape[0] != N_train or E_test_fused.shape[0] != N_test:
    raise ValueError(
        f"Embeddings rows do not match meta counts: "
        f"train {E_train_fused.shape[0]} vs {N_train}, test {E_test_fused.shape[0]} vs {N_test}"
    )

E_all_fused = np.vstack([E_train_fused, E_test_fused]).astype(np.float32)
E_all_txt   = np.vstack([E_train_txt,   E_test_txt]).astype(np.float32)
E_all_img   = np.vstack([E_train_img,   E_test_img]).astype(np.float32)

print("Computing margins (fused; leaf candidates)...")
train_margin = compute_margin_batched(E_train_fused.astype(np.float32), train_cand_ids, E_leaf_cat, batch=2048)
test_margin  = compute_margin_batched(E_test_fused.astype(np.float32),  test_cand_ids,  E_leaf_cat, batch=2048)
all_margin   = np.concatenate([train_margin, test_margin]).astype(np.float32)

# -------------------------
# 3D layouts (UMAP 3D on combined train+test per space)
# + align txt/img to fused so morphing looks good
# -------------------------
pos_fused_path = os.path.join(DATA_DIR, "pos_fused.bin")
pos_text_path = os.path.join(DATA_DIR, "pos_text.bin")
pos_image_path = os.path.join(DATA_DIR, "pos_image.bin")
pos_cache_key_path = os.path.join(DATA_DIR, "pos_cache_key.json")

# Cache key for positions: depends on UMAP params + dataset size
pos_cache_key = {
    "pca_dim": int(PCA_DIM),
    "umap_neighbors": int(UMAP_NEIGHBORS),
    "umap_min_dist": float(UMAP_MIN_DIST),
    "seed": int(SEED),
    "n_all": int(N_all),
}

if (os.path.exists(pos_fused_path) and os.path.exists(pos_text_path) and 
    os.path.exists(pos_image_path) and cache_key_matches(pos_cache_key_path, pos_cache_key)):
    print("Loading existing UMAP layouts from disk...")
    P_all_fused = read_bin(pos_fused_path, np.float32, (N_all, 3))
    P_all_txt = read_bin(pos_text_path, np.float32, (N_all, 3))
    P_all_img = read_bin(pos_image_path, np.float32, (N_all, 3))
    print("✓ Loaded existing positions")
else:
    if os.path.exists(pos_fused_path):
        print("Cache key mismatch or missing - recomputing positions (params may have changed)")
    print("Computing fused layout (UMAP 3D on train+test)...")
    P_all_fused = center_scale(
        umap3_with_pca(E_all_fused, pca_dim=PCA_DIM, seed=SEED, n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST)
    )

    print("Computing text layout (UMAP 3D on train+test)...")
    P_all_txt = center_scale(
        umap3_with_pca(E_all_txt, pca_dim=PCA_DIM, seed=SEED, n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST)
    )
    P_all_txt = procrustes_align(P_all_txt, P_all_fused)

    print("Computing image layout (UMAP 3D on train+test)...")
    P_all_img = center_scale(
        umap3_with_pca(E_all_img, pca_dim=PCA_DIM, seed=SEED, n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST)
    )
    P_all_img = procrustes_align(P_all_img, P_all_fused)
    
    # Write cache key after computation
    write_cache_key(pos_cache_key_path, pos_cache_key)

# convenience slices
P_train_fused = P_all_fused[:N_train]
P_test_fused  = P_all_fused[N_train:]

# -------------------------
# Neighbors (GLOBAL)
# -------------------------
neigh_fused_path = os.path.join(DATA_DIR, "neigh_fused_global.bin")
neigh_text_path = os.path.join(DATA_DIR, "neigh_text_global.bin")
neigh_image_path = os.path.join(DATA_DIR, "neigh_image_global.bin")
neigh_cache_key_path = os.path.join(DATA_DIR, "neigh_cache_key.json")

# Cache key for neighbors: depends on K_GLOBAL + dataset size
neigh_cache_key = {
    "k_global": int(K_GLOBAL),
    "n_all": int(N_all),
}

if (os.path.exists(neigh_fused_path) and os.path.exists(neigh_text_path) and 
    os.path.exists(neigh_image_path) and cache_key_matches(neigh_cache_key_path, neigh_cache_key)):
    print(f"Loading existing neighbors from disk (K_GLOBAL={K_GLOBAL})...")
    neigh_all_fused = read_bin(neigh_fused_path, np.int32, (N_all, K_GLOBAL))
    neigh_all_txt = read_bin(neigh_text_path, np.int32, (N_all, K_GLOBAL))
    neigh_all_img = read_bin(neigh_image_path, np.int32, (N_all, K_GLOBAL))
    print("✓ Loaded existing neighbors")
else:
    if os.path.exists(neigh_fused_path):
        print("Cache key mismatch or missing - recomputing neighbors (K_GLOBAL or dataset size may have changed)")
    print(f"Computing GLOBAL neighbors with HNSW (K_GLOBAL={K_GLOBAL})...")
    neigh_all_fused, _ = knn_faiss_hnsw_ip(E_all_fused, K_GLOBAL, m=32, ef_search=96, ef_construction=200)
    neigh_all_txt,   _ = knn_faiss_hnsw_ip(E_all_txt,   K_GLOBAL, m=32, ef_search=96, ef_construction=200)
    neigh_all_img,   _ = knn_faiss_hnsw_ip(E_all_img,   K_GLOBAL, m=32, ef_search=96, ef_construction=200)
    
    # Write cache key after computation
    write_cache_key(neigh_cache_key_path, neigh_cache_key)

# -------------------------
# COARSE CATEGORY “stars” (GLOBAL constellation) + ALL/TRAIN/TEST counts
# Robust to zero-product categories
# -------------------------
print(f"Computing COARSE stars (depth={COARSE_DEPTH})...")
star_ids, star_pos, star_count_all = safe_centroids_and_counts(
    P_all_fused, all_gt_coarse_id, num_ids=M_coarse, min_count=MIN_STAR_COUNT
)

# Counts for train/test for ONLY these stars (positions fixed)
counts_train_full = counts_per_id(train_gt_coarse_id, M_coarse)
counts_test_full  = counts_per_id(test_gt_coarse_id,  M_coarse)
star_count_train  = counts_train_full[star_ids].astype(np.int32)
star_count_test   = counts_test_full[star_ids].astype(np.int32)

num_stars = len(star_ids)
print(f"Stars kept: {num_stars} (min_count={max(MIN_STAR_COUNT,1)})")

# -------------------------
# Write web_data/v1 artifacts (MERGED under data/)
# -------------------------
print("Writing merged binaries under data/ ...")

# positions
write_bin(os.path.join(DATA_DIR, "pos_fused.bin"), P_all_fused.astype(np.float32))
write_bin(os.path.join(DATA_DIR, "pos_text.bin"),  P_all_txt.astype(np.float32))
write_bin(os.path.join(DATA_DIR, "pos_image.bin"), P_all_img.astype(np.float32))

# split mapping
write_bin(os.path.join(DATA_DIR, "split_id.bin"),  split_id.astype(np.uint8))
write_bin(os.path.join(DATA_DIR, "split_idx.bin"), split_idx.astype(np.uint32))

# labels (leaf + top-level + coarse)
write_bin(os.path.join(DATA_DIR, "gt_cat_leaf_id.bin"),   all_gt_leaf_id.astype(np.int32))     # keep signed for -1
write_bin(os.path.join(DATA_DIR, "pred_cat_leaf_id.bin"), all_pred_leaf_id.astype(np.int32))   # keep signed for -1

write_bin(os.path.join(DATA_DIR, "gt_top_id.bin"),   all_gt_top.astype(np.int16))
write_bin(os.path.join(DATA_DIR, "pred_top_id.bin"), all_pred_top.astype(np.int16))

write_bin(os.path.join(DATA_DIR, "gt_coarse_id.bin"),   all_gt_coarse_id.astype(np.int32))     # signed
write_bin(os.path.join(DATA_DIR, "pred_coarse_id.bin"), all_pred_coarse_id.astype(np.int32))   # signed

write_bin(os.path.join(DATA_DIR, "correct.bin"),   all_correct.astype(np.uint8))
write_bin(os.path.join(DATA_DIR, "ambiguous.bin"), all_amb.astype(np.uint8))
write_bin(os.path.join(DATA_DIR, "margin.bin"),    all_margin.astype(np.float32))

# neighbors (global ids)
write_bin(os.path.join(DATA_DIR, "neigh_fused_global.bin"), neigh_all_fused.astype(np.int32))
write_bin(os.path.join(DATA_DIR, "neigh_text_global.bin"),  neigh_all_txt.astype(np.int32))
write_bin(os.path.join(DATA_DIR, "neigh_image_global.bin"), neigh_all_img.astype(np.int32))

# categories metadata (leaf + coarse + top-level)
with open(os.path.join(CAT_DIR, "leaf_categories.json"), "w", encoding="utf-8") as f:
    json.dump(leaf_cats, f, ensure_ascii=False)

with open(os.path.join(CAT_DIR, "coarse_categories.json"), "w", encoding="utf-8") as f:
    json.dump(coarse_cats, f, ensure_ascii=False)

with open(os.path.join(CAT_DIR, "top_levels.json"), "w", encoding="utf-8") as f:
    json.dump(top_levels, f, ensure_ascii=False)

# coarse stars (global constellation)
write_bin(os.path.join(CAT_DIR, "coarse_star_ids.bin"),          star_ids.astype(np.int32))
write_bin(os.path.join(CAT_DIR, "coarse_star_pos_fused.bin"),    star_pos.astype(np.float32))
write_bin(os.path.join(CAT_DIR, "coarse_star_count_all.bin"),    star_count_all.astype(np.int32))
write_bin(os.path.join(CAT_DIR, "coarse_star_count_train.bin"),  star_count_train.astype(np.int32))
write_bin(os.path.join(CAT_DIR, "coarse_star_count_test.bin"),   star_count_test.astype(np.int32))

# -------------------------
# Shard merged meta/details keyed by global id
# -------------------------
print("Building merged meta/details (global id keyed)...")

# Global meta (text)
train_meta_out = train_meta.copy()
train_meta_out["split"] = "train"
train_meta_out["split_idx"] = train_meta_out.index.astype(int)
train_meta_out["global_id"] = train_meta_out.index.astype(int)
train_meta_out["gt_coarse"] = train_meta_out["ground_truth_category"].map(lambda s: coarsen_path(s, COARSE_DEPTH))

test_meta_out = test_meta.copy()
test_meta_out["split"] = "test"
test_meta_out["split_idx"] = test_meta_out.index.astype(int)
test_meta_out["global_id"] = N_train + test_meta_out.index.astype(int)
test_meta_out["gt_coarse"] = test_meta_out["ground_truth_category"].map(lambda s: coarsen_path(s, COARSE_DEPTH))

all_meta = pd.concat(
    [train_meta_out.set_index("global_id"), test_meta_out.set_index("global_id")],
    axis=0
).sort_index()

assert_contiguous_index(all_meta, N_all, "all_meta(global)")

def make_text_meta_record(global_id, row):
    split = row["split"]
    sidx = int(row["split_idx"])
    return {
        "id": int(global_id),
        "split": split,
        "split_idx": sidx,
        "title": row.get("product_title", ""),
        "brand": row.get("ground_truth_brand", ""),
        "desc": row.get("product_description", ""),
        "gt_category": row.get("ground_truth_category", ""),
        "gt_coarse": row.get("gt_coarse", ""),
        "thumb": f"{THUMB_BASE_URL}/{split}/{sidx}.webp",
    }

print("Writing sharded merged meta...")
shard_ordered_json_by_idx(
    all_meta,
    os.path.join(DATA_DIR, "meta"),
    SHARD_SIZE,
    make_text_meta_record
)

# Global details (embedding meta)
train_details_out = train_emb_meta.copy()
train_details_out["split"] = "train"
train_details_out["split_idx"] = train_details_out.index.astype(int)
train_details_out["global_id"] = train_details_out.index.astype(int)
train_details_out["pred_coarse"] = train_details_out["pred_category"].map(lambda s: coarsen_path(s, COARSE_DEPTH))

test_details_out = test_emb_meta.copy()
test_details_out["split"] = "test"
test_details_out["split_idx"] = test_details_out.index.astype(int)
test_details_out["global_id"] = N_train + test_details_out.index.astype(int)
test_details_out["pred_coarse"] = test_details_out["pred_category"].map(lambda s: coarsen_path(s, COARSE_DEPTH))

all_details = pd.concat(
    [train_details_out.set_index("global_id"), test_details_out.set_index("global_id")],
    axis=0
).sort_index()

assert_contiguous_index(all_details, N_all, "all_details(global)")

def make_details_record(global_id, row):
    return {
        "id": int(global_id),
        "split": row.get("split", ""),
        "split_idx": int(row.get("split_idx", -1)),
        "target_candidate_index": int(row.get("target_candidate_index", -1)),
        "pred_candidate_index": int(row.get("pred_candidate_index", -1)),
        "pred_category": (row.get("pred_category", "") or ""),
        "pred_coarse": (row.get("pred_coarse", "") or ""),
        "score_pred": float(row.get("score_pred", 0.0) or 0.0),
        "candidates_used": row.get("candidates_used", []) or [],
        "raw_potential_product_categories": row.get("raw_potential_product_categories", []) or [],
    }

print("Writing sharded merged details...")
shard_ordered_json_by_idx(
    all_details,
    os.path.join(DATA_DIR, "details"),
    SHARD_SIZE,
    make_details_record
)

# -------------------------
# Manifest for frontend
# -------------------------
manifest = {
    "version": "v1",
    "coarse": {
        "depth": int(COARSE_DEPTH),
        "min_star_count": int(max(MIN_STAR_COUNT, 1)),
    },
    "neighbors": {
        "k_ui": int(K_UI),
        "k_global": int(K_GLOBAL),
        "type": "global_only_filter_in_ui",
    },
    "meta_shard_size": int(SHARD_SIZE),
    "splits": {
        "train": {"count": int(N_train), "split_id": 0},
        "test":  {"count": int(N_test),  "split_id": 1},
        "all":   {"count": int(N_all)},
    },
    "formats": {
        "pos": {
            "dtype": "float32",
            "shape": [int(N_all), 3],
            "description": "3D positions (x, y, z) for fused/text/image layouts"
        },
        "mapping": {
            "split_id": {"dtype": "uint8", "shape": [int(N_all)], "description": "0=train, 1=test"},
            "split_idx": {"dtype": "uint32", "shape": [int(N_all)], "description": "Index within split for /thumbs/{split}/{split_idx}.webp"}
        },
        "labels": {
            "gt_cat_leaf_id": {"dtype": "int32", "shape": [int(N_all)], "description": "Ground truth leaf category ID (-1 if unknown)"},
            "pred_cat_leaf_id": {"dtype": "int32", "shape": [int(N_all)], "description": "Predicted leaf category ID (-1 if unknown)"},
            "gt_top_id": {"dtype": "int16", "shape": [int(N_all)], "description": "Ground truth top-level category ID"},
            "pred_top_id": {"dtype": "int16", "shape": [int(N_all)], "description": "Predicted top-level category ID"},
            "gt_coarse_id": {"dtype": "int32", "shape": [int(N_all)], "description": "Ground truth coarse category ID (-1 if unknown)"},
            "pred_coarse_id": {"dtype": "int32", "shape": [int(N_all)], "description": "Predicted coarse category ID (-1 if unknown)"},
            "correct": {"dtype": "uint8", "shape": [int(N_all)], "description": "1 if prediction matches ground truth, 0 otherwise"},
            "ambiguous": {"dtype": "uint8", "shape": [int(N_all)], "description": "1 if prediction is incorrect, 0 otherwise"},
            "margin": {"dtype": "float32", "shape": [int(N_all)], "description": "Confidence margin (top1 - top2) over candidate similarities"}
        },
        "neighbors": {
            "dtype": "int32",
            "shape": [int(N_all), int(K_GLOBAL)],
            "description": "Global neighbor indices (row i contains K_GLOBAL nearest neighbors of point i)"
        },
        "stars": {
            "num_stars": int(num_stars),
            "coarse_ids": {"dtype": "int32", "shape": [int(num_stars)], "description": "Coarse category IDs that have stars"},
            "pos_fused": {"dtype": "float32", "shape": [int(num_stars), 3], "description": "3D star positions in fused layout"},
            "count_all": {"dtype": "int32", "shape": [int(num_stars)], "description": "Total count per star across all splits"},
            "count_train": {"dtype": "int32", "shape": [int(num_stars)], "description": "Count per star in train split"},
            "count_test": {"dtype": "int32", "shape": [int(num_stars)], "description": "Count per star in test split"}
        }
    },
    "paths": {
        "thumbs_base": THUMB_BASE_URL,
        "web_data_base": "/web_data/v1",
        "categories": {
            "leaf_list": "categories/leaf_categories.json",
            "coarse_list": "categories/coarse_categories.json",
            "top_levels": "categories/top_levels.json",
            "stars": {
                "coarse_ids": "categories/coarse_star_ids.bin",
                "pos_fused": "categories/coarse_star_pos_fused.bin",
                "count_all": "categories/coarse_star_count_all.bin",
                "count_train": "categories/coarse_star_count_train.bin",
                "count_test": "categories/coarse_star_count_test.bin",
            },
        },
        "data": {
            "pos": {
                "fused": "data/pos_fused.bin",
                "text":  "data/pos_text.bin",
                "image": "data/pos_image.bin",
            },
            "mapping": {
                "split_id":  "data/split_id.bin",
                "split_idx": "data/split_idx.bin",
            },
            "labels": {
                "gt_cat_leaf_id":   "data/gt_cat_leaf_id.bin",
                "pred_cat_leaf_id": "data/pred_cat_leaf_id.bin",
                "gt_top_id":        "data/gt_top_id.bin",
                "pred_top_id":      "data/pred_top_id.bin",
                "gt_coarse_id":     "data/gt_coarse_id.bin",
                "pred_coarse_id":   "data/pred_coarse_id.bin",
                "correct":          "data/correct.bin",
                "ambiguous":        "data/ambiguous.bin",
                "margin":           "data/margin.bin",
            },
            "neighbors": {
                "fused_global": "data/neigh_fused_global.bin",
                "text_global":  "data/neigh_text_global.bin",
                "image_global": "data/neigh_image_global.bin",
            },
            "meta":    "data/meta/meta_{shard:03d}.json",
            "details": "data/details/meta_{shard:03d}.json",
        },
    },
}

with open(os.path.join(WEB_DATA_DIR, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("Wrote manifest:", os.path.join(WEB_DATA_DIR, "manifest.json"))

# -------------------------
# Create a single "public thumbs" zip (no Drive file explosion)
# -------------------------
THUMBS_PUBLIC_ZIP = os.path.join(OUT_ROOT, "thumbs_150px_public.zip")
print("Packing thumbs into one zip (no Drive explosion)...")
build_thumbs_public_zip(TRAIN_THUMBS_ZIP, TEST_THUMBS_ZIP, THUMBS_PUBLIC_ZIP)

print("\nDONE ✅")
print("Web data folder:", WEB_DATA_DIR)
print("Thumbs zip:", THUMBS_PUBLIC_ZIP)

print("\nNext.js usage:")
print("- Copy web_data/v1 into your app's public/web_data/v1 (or upload to CDN with same path).")
print("- Unzip thumbs_150px_public.zip into public/ so you have public/thumbs/train/{idx}.webp and public/thumbs/test/{idx}.webp (or upload to CDN).")

print("\nFrontend notes:")
print(f"- Load global neighbors (K_GLOBAL={K_GLOBAL}) and filter by split_id to show train-only/test-only neighbors.")
print(f"- Show up to K_UI={K_UI} neighbors after filtering.")
print(f"- Coarse stars: depth={COARSE_DEPTH}. Star positions are GLOBAL centroids; size/alpha via count_all/count_train/count_test.")
