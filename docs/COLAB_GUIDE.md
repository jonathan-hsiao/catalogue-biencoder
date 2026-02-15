# Running catalogue-biencoder in Google Colab

This guide sets up Colab so you can clone the repo, run training, and save all outputs (checkpoints, logs, embeddings) to Google Drive.

---

## 1. Setup (one cell) — copy-paste and run

Run this **entire block** in a single Colab cell:

```python
import os
import sys
import subprocess
from google.colab import drive
from IPython import get_ipython

# Mount Google Drive
drive.mount("/content/drive")

DRIVE_RUNS = "/content/drive/MyDrive/catalogue_biencoder_runs"
REPO_DIR = "/content/catalogue_biencoder"
os.makedirs(DRIVE_RUNS, exist_ok=True)

# Clone (public repo)
if not os.path.exists(REPO_DIR):
    get_ipython().system(f'git clone https://github.com/jonathan-hsiao/catalogue-biencoder.git {REPO_DIR}')
else:
    get_ipython().system(f'cd {REPO_DIR} && git pull')

# Install with the same Python as the notebook
subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "-q"], cwd=REPO_DIR, check=True)

# Use package under src/; avoid repo root being seen as catalogue_biencoder
REPO_SRC = os.path.join(REPO_DIR, "src")
sys.path = [p for p in sys.path if p not in ("/content", REPO_DIR)]
sys.path.insert(0, REPO_SRC)

# Clear cached import if this cell was run before (no-op on fresh runtime)
for key in list(sys.modules.keys()):
    if key == "catalogue_biencoder" or key.startswith("catalogue_biencoder."):
        del sys.modules[key]

from catalogue_biencoder.config import TrainConfig
from catalogue_biencoder.training.runner import run
```

If private repo, use the same cell but clone with a token: add `from google.colab import userdata`, `token = userdata.get("GITHUB_PAT")`, and use  
`get_ipython().system(f'git clone https://x-access-token:{token}@github.com/jonathan-hsiao/catalogue-biencoder.git {REPO_DIR}')` in the clone branch.

---

## 2. Run training (next cell)

In the **next** cell, configure and run. Outputs go to Drive (`DRIVE_RUNS`).

```python
cfg = TrainConfig(
    output_dir=DRIVE_RUNS,
    stage0_epochs=2,
    stage1_epochs=2,
    stage2_epochs=2,
)
run(cfg)
```

Optional: set `cfg.run_name = "colab_run_1"` (or any name) so you can find the run folder easily under `DRIVE_RUNS`.

---

## 3. After training

After training, find your run in **My Drive → catalogue_biencoder_runs**. On a new session, run the setup cell again, then the training cell.
