# Running catalogue-biencoder in Google Colab

This guide sets up Colab so you can clone the repo, run training, and save all outputs (checkpoints, logs, embeddings) to Google Drive so they persist after the runtime disconnects.

---

## 1. Mount Drive and set paths

Run this in your first Colab cell:

```python
from google.colab import drive
drive.mount("/content/drive")

# Where to store all run outputs (checkpoints, logs, embeddings)
DRIVE_RUNS = "/content/drive/MyDrive/catalogue_biencoder_runs"
REPO_DIR = "/content/catalogue_biencoder"  # clone here (faster than cloning to Drive)
```

---

## 2. Clone repo and install

```python
import os

# Clone to /content (faster I/O than Drive)
if not os.path.exists(REPO_DIR):
    !git clone https://github.com/jonathan-hsiao/catalogue-biencoder.git {REPO_DIR}
else:
    !cd {REPO_DIR} && git pull

%cd {REPO_DIR}
!pip install -e . -q
```

---

## 3. Run training with outputs on Drive

```python
import sys
sys.path.insert(0, REPO_DIR)

from catalogue_biencoder.config import TrainConfig
from catalogue_biencoder.training.runner import run

cfg = TrainConfig()

# Send all outputs to Drive (persistent)
cfg.output_dir = DRIVE_RUNS
# Optional: set run name so you can find it later
# cfg.run_name = "colab_run_1"
# Optional: more epochs for better embeddings
# cfg.stage0_epochs = 2
# cfg.stage1_epochs = 2
# cfg.stage2_epochs = 2

run(cfg)
```

All run artifacts (e.g. `artifacts/runs_product_catalogue/<run_name>/`) will live under `DRIVE_RUNS` on your Drive.

---

## 4. Optional: create the Drive folder first

```python
os.makedirs(DRIVE_RUNS, exist_ok=True)
```

---

## Summary

| Step | Purpose |
|------|--------|
| Mount Drive | So outputs persist after you disconnect. |
| Clone to `/content/` | Fast disk for git and training; clone once, reuse. |
| `cfg.output_dir = DRIVE_RUNS` | All checkpoints, logs, and embeddings go to Drive. |

After training, find your run in **My Drive → catalogue_biencoder_runs**. On later sessions, run the mount and clone/install cells again, then the training cell; you don’t need to re-clone if you only care about new runs.
