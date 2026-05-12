# 🧠 BrainTumorVLM — Text-Guided Brain Tumor Segmentation

**Vision Language Model for Brain Tumor Segmentation using BraTS 2020 FLAIR + Radiology Reports**

## Architecture

```
Input: FLAIR MRI (1×224×224) + Radiology Report (text)
  ↓                                    ↓
ResNet-34 Encoder              Frozen CLIP ViT-B/32
  ↓ (skip connections)                 ↓
  ↓                            Text Embedding (512)
  ↓                                    ↓
Bottleneck (512, 7×7) ← FiLM Modulation (γ·x + β)
  ↓
U-Net Decoder (with skip connections + FiLM at multiple scales)
  ↓
Segmentation Mask (1×224×224)
```

## Files

| File | Description |
|------|-------------|
| `config.py` | Central configuration (hyperparameters, paths) |
| `prepare_data.py` | Build text CSV from TextBraTS dataset |
| `dataset.py` | PyTorch Dataset (handles all .npy shapes) |
| `model.py` | VLM model (ResNet-34 + CLIP + FiLM + U-Net) |
| `train.py` | Training loop (VLM + baseline modes) |
| `evaluate.py` | All metrics (Dice, IoU, Hausdorff, P, R, ROUGE, BLEU) |
| `visualize.py` | Training curves, segmentation results, failure cases |
| `attention_viz.py` | Grad-CAM attention maps |
| `embedding_viz.py` | t-SNE embedding visualization |
| `ablation.py` | VLM vs baseline comparison |
| `smoke_test.py` | Quick sanity check |
| `requirements.txt` | Dependencies |
| `BrainTumorVLM_Colab_Guide.md` | Step-by-step Colab instructions |

## Quick Start

See `BrainTumorVLM_Colab_Guide.md` for complete Google Colab instructions.

## Datasets

- **FLAIR BraTS 2020**: [Kaggle](https://www.kaggle.com/datasets/hussainnasirkhan/flair-brats2020)
- **Text BraTS 2020**: [Google Drive](https://drive.google.com/file/d/17YKI4nwPW8qMKlg9k53dVax7F_1JCk9B/view)
