# 🧠 BrainTumorVLM — Google Colab Guide (Step-by-Step)

> **Copy-paste each cell into Google Colab.** Run them in order, top to bottom.
> **Total time:** ~2-4 hours (50 epochs on GPU) or ~20 min (5 epochs quick test)

---

## BEFORE YOU START

1. **Push all files to your GitHub repo** (from your local BrainTumorVLM folder)
2. **Get Kaggle API key**: Kaggle → Settings → API → Create New Token → downloads `kaggle.json`
3. **Open Google Colab** → Runtime → Change runtime type → **GPU (T4)** → Save

---

## STEP 1 — Check GPU

```python
# CELL 1 — Verify GPU is available
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("⚠️  NO GPU! Go to Runtime → Change runtime type → GPU → Save → Restart")
```

---

## STEP 2 — Clone Repository

```python
# CELL 2 — Clone your GitHub repo
import os

# ⬇️ CHANGE THIS to your GitHub repo URL
REPO_URL = "https://github.com/YOUR_USERNAME/BrainTumorVLM.git"

if os.path.exists("/content/BrainTumorVLM"):
    os.chdir("/content/BrainTumorVLM")
    !git pull origin main
    print("✅ Updated existing repo")
else:
    !git clone {REPO_URL} /content/BrainTumorVLM
    os.chdir("/content/BrainTumorVLM")
    print("✅ Cloned repo")

!ls -la
```

---

## STEP 3 — Install Dependencies

```python
# CELL 3 — Install requirements
%cd /content/BrainTumorVLM
!pip install -q -r requirements.txt
print("✅ Dependencies installed")
```

---

## STEP 4 — Download FLAIR BraTS 2020 (from Kaggle)

```python
# CELL 4 — Upload kaggle.json and download FLAIR dataset
import os

# Upload kaggle.json
from google.colab import files
print("📤 Upload your kaggle.json file:")
uploaded = files.upload()

# Setup Kaggle
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download dataset
!kaggle datasets download -d hussainnasirkhan/flair-brats2020 -p /content/

# Unzip
!unzip -q -o /content/flair-brats2020.zip -d /content/FLAIR_BRATS2020_split

# Verify structure
print("\n📂 Dataset structure:")
!ls /content/FLAIR_BRATS2020_split/
!echo "--- Train images ---"
!ls /content/FLAIR_BRATS2020_split/train/images/ | head -5
!echo "--- Train masks ---"
!ls /content/FLAIR_BRATS2020_split/train/masks/ | head -5
!echo "--- Val images ---"
!ls /content/FLAIR_BRATS2020_split/val/images/ | head -5

print(f"\n✅ Train images: {len(os.listdir('/content/FLAIR_BRATS2020_split/train/images'))}")
print(f"✅ Val images: {len(os.listdir('/content/FLAIR_BRATS2020_split/val/images'))}")
```

---

## STEP 5 — Download TextBraTS (from Google Drive)

```python
# CELL 5 — Download and extract TextBraTS
import os

!pip install -q gdown

# Download from Google Drive
!gdown 17YKI4nwPW8qMKlg9k53dVax7F_1JCk9B -O /content/TextBRats.zip

# Extract
!unzip -q -o /content/TextBRats.zip -d /content/

# Find the extracted directory
import subprocess
result = subprocess.run(['find', '/content/', '-type', 'd', '-name', 'TextBraTSData', '-maxdepth', '3'],
                       capture_output=True, text=True)
dirs = [d.strip() for d in result.stdout.strip().split('\n') if d.strip()]
TEXT_DIR = dirs[0] if dirs else "/content/TextBraTSData"
print(f"📂 TextBraTS directory: {TEXT_DIR}")

# Check contents
if os.path.isdir(TEXT_DIR):
    patient_dirs = sorted([d for d in os.listdir(TEXT_DIR) if d.startswith("BraTS")])
    print(f"✅ Found {len(patient_dirs)} patient folders")
    if patient_dirs:
        sample = os.path.join(TEXT_DIR, patient_dirs[0])
        print(f"   Sample folder ({patient_dirs[0]}): {os.listdir(sample)}")
else:
    print("⚠️  TextBraTSData not found. Will use placeholder text reports.")
    TEXT_DIR = "/content/TextBraTSData"
```

---

## STEP 6 — Prepare Text Data

```python
# CELL 6 — Build text CSV from TextBraTS
import os
os.chdir('/content/BrainTumorVLM')

# Use the TEXT_DIR found in Step 5
# If Step 5 failed, try common paths:
import subprocess
result = subprocess.run(['find', '/content/', '-type', 'd', '-name', 'TextBraTSData', '-maxdepth', '3'],
                       capture_output=True, text=True)
dirs = [d.strip() for d in result.stdout.strip().split('\n') if d.strip()]
TEXT_DIR = dirs[0] if dirs else "/content/TextBraTSData"

!python prepare_data.py \
    --text_brats_dir "{TEXT_DIR}" \
    --hussain_dir /content/FLAIR_BRATS2020_split \
    --output_csv /content/data/text_brats.csv

# Verify
import pandas as pd
df = pd.read_csv('/content/data/text_brats.csv')
print(f"\n✅ Text CSV: {len(df)} patients")
print(f"Sample:\n{df.head(3).to_string()}")
```

---

## STEP 7 — Smoke Test

```python
# CELL 7 — Verify everything works
import os
os.chdir('/content/BrainTumorVLM')

!python smoke_test.py \
    --hussain_dir /content/FLAIR_BRATS2020_split \
    --text_csv /content/data/text_brats.csv
```

You should see all `[PASS] ✓` messages. If any fail, check the error and fix before continuing.

---

## STEP 8 — Train VLM Model (with text)

```python
# CELL 8 — Train VLM model
import os
os.chdir('/content/BrainTumorVLM')

# For a full run: --epochs 50 (takes ~2-3 hours on T4 GPU)
# For a quick test: --epochs 5 (takes ~15 min)

!python train.py \
    --hussain_dir /content/FLAIR_BRATS2020_split \
    --text_csv /content/data/text_brats.csv \
    --checkpoint_dir /content/checkpoints \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4
```

---

## STEP 9 — Train Baseline Model (no text, for ablation)

```python
# CELL 9 — Train baseline model (without text)
import os
os.chdir('/content/BrainTumorVLM')

!python train.py \
    --hussain_dir /content/FLAIR_BRATS2020_split \
    --text_csv /content/data/text_brats.csv \
    --checkpoint_dir /content/checkpoints \
    --no_text \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4
```

---

## STEP 10 — Evaluate VLM Model

```python
# CELL 10 — Evaluate with all metrics
import os
os.chdir('/content/BrainTumorVLM')

!python evaluate.py \
    --hussain_dir /content/FLAIR_BRATS2020_split \
    --text_csv /content/data/text_brats.csv \
    --checkpoint /content/checkpoints/best_vlm.pth \
    --output_dir /content/outputs
```

---

## STEP 11 — Generate All Visualizations

```python
# CELL 11A — Training curves + Segmentation results + Failure cases
import os
os.chdir('/content/BrainTumorVLM')

!python visualize.py \
    --hussain_dir /content/FLAIR_BRATS2020_split \
    --text_csv /content/data/text_brats.csv \
    --checkpoint /content/checkpoints/best_vlm.pth \
    --history /content/checkpoints/history_vlm.json \
    --eval_results /content/outputs/eval_results_vlm.json \
    --output_dir /content/outputs
```

```python
# CELL 11B — Display all generated plots
from IPython.display import display, Image as IPImage
import os

output_dir = "/content/outputs"
for fname in sorted(os.listdir(output_dir)):
    if fname.endswith(".png"):
        print(f"\n📊 {fname}")
        display(IPImage(filename=os.path.join(output_dir, fname), width=800))
```

---

## STEP 12 — Grad-CAM Attention Maps

```python
# CELL 12 — Attention visualization
import os
os.chdir('/content/BrainTumorVLM')

!python attention_viz.py \
    --hussain_dir /content/FLAIR_BRATS2020_split \
    --text_csv /content/data/text_brats.csv \
    --checkpoint /content/checkpoints/best_vlm.pth \
    --output_dir /content/outputs \
    --n_samples 6
```

```python
# Display Grad-CAM
from IPython.display import display, Image as IPImage
display(IPImage(filename="/content/outputs/gradcam_attention.png", width=800))
```

---

## STEP 13 — t-SNE Embedding Visualization

```python
# CELL 13 — t-SNE of image and text embeddings
import os
os.chdir('/content/BrainTumorVLM')

!python embedding_viz.py \
    --hussain_dir /content/FLAIR_BRATS2020_split \
    --text_csv /content/data/text_brats.csv \
    --checkpoint /content/checkpoints/best_vlm.pth \
    --output_dir /content/outputs
```

```python
# Display t-SNE
from IPython.display import display, Image as IPImage
display(IPImage(filename="/content/outputs/tsne_embeddings.png", width=800))
```

---

## STEP 14 — Ablation Study (VLM vs Baseline)

```python
# CELL 14 — Ablation study
import os
os.chdir('/content/BrainTumorVLM')

!python ablation.py \
    --hussain_dir /content/FLAIR_BRATS2020_split \
    --text_csv /content/data/text_brats.csv \
    --vlm_checkpoint /content/checkpoints/best_vlm.pth \
    --baseline_checkpoint /content/checkpoints/best_baseline.pth \
    --output_dir /content/outputs
```

```python
# Display ablation chart
from IPython.display import display, Image as IPImage
display(IPImage(filename="/content/outputs/ablation_study.png", width=800))
```

---

## STEP 15 — Display ALL Results Summary

```python
# CELL 15 — Final summary of all results
import json
import os

print("=" * 70)
print("  📋 COMPLETE RESULTS SUMMARY")
print("=" * 70)

# VLM eval results
vlm_path = "/content/outputs/eval_results_vlm.json"
if os.path.isfile(vlm_path):
    with open(vlm_path) as f:
        vlm = json.load(f)
    print("\n🔬 VLM Model (with text):")
    for k, v in vlm.items():
        if k == "per_sample_dice":
            continue
        if isinstance(v, float):
            print(f"   {k:20s}: {v:.4f}")
        else:
            print(f"   {k:20s}: {v}")

# Ablation results
abl_path = "/content/outputs/ablation_results.json"
if os.path.isfile(abl_path):
    with open(abl_path) as f:
        abl = json.load(f)
    print("\n📊 Ablation Study:")
    print(f"   {'Metric':<15} {'VLM':>10} {'Baseline':>10} {'Δ':>10}")
    print(f"   {'-'*45}")
    for m in ["dice", "iou", "precision", "recall"]:
        v = abl["vlm"].get(m, 0)
        b = abl["baseline"].get(m, 0)
        d = v - b
        print(f"   {m:<15} {v:>10.4f} {b:>10.4f} {d:>+10.4f}")

# List all output files
print(f"\n📁 Output files in /content/outputs/:")
if os.path.isdir("/content/outputs"):
    for f in sorted(os.listdir("/content/outputs")):
        size = os.path.getsize(os.path.join("/content/outputs", f))
        print(f"   {f:40s} ({size/1024:.1f} KB)")

print("\n" + "=" * 70)
print("  ✅ PROJECT COMPLETE!")
print("=" * 70)
```

---

## STEP 16 — Download All Outputs

```python
# CELL 16 — Zip and download all outputs
import shutil

# Zip outputs
shutil.make_archive("/content/BrainTumorVLM_outputs", "zip", "/content/outputs")

# Zip checkpoints
shutil.make_archive("/content/BrainTumorVLM_checkpoints", "zip", "/content/checkpoints")

# Download
from google.colab import files
files.download("/content/BrainTumorVLM_outputs.zip")
# files.download("/content/BrainTumorVLM_checkpoints.zip")  # Uncomment if needed

print("✅ Download started!")
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No GPU detected` | Runtime → Change runtime type → GPU → Save → Restart |
| `kaggle.json not found` | Re-upload kaggle.json in Step 4 |
| `FileNotFoundError: images` | Check path: `!ls /content/FLAIR_BRATS2020_split/train/images/ \| head` |
| `No text extracted` | Check: `!find /content -name "TextBraTSData" -type d` |
| `CUDA out of memory` | Reduce batch_size to 4: `--batch_size 4` |
| `Cannot make 2D from shape` | Dataset has unusual .npy format — file an issue |
| `Module not found` | Re-run Step 3 (install requirements) |
| `gdown fails` | Upload TextBRats.zip manually: Files panel → Upload |
