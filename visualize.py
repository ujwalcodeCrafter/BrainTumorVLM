"""
visualize.py — All required plots and visualizations.
Training curves, segmentation results, dice distribution, failure cases.

Usage:
    python visualize.py --hussain_dir /content/FLAIR_BRATS2020_split \
                        --text_csv /content/data/text_brats.csv \
                        --checkpoint /content/checkpoints/best_vlm.pth \
                        --history /content/checkpoints/history_vlm.json \
                        --output_dir /content/outputs
"""

import os
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader

from config import *
from dataset import BraTSDataset, get_val_transforms
from model import VLMSegModel
from evaluate import dice_score


def plot_training_curves(history_path, output_dir):
    """Plot Dice, IoU, and Loss vs Epoch."""
    with open(history_path) as f:
        hist = json.load(f)

    epochs = range(1, len(hist["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training Performance", fontsize=16, fontweight="bold")

    # Loss
    axes[0].plot(epochs, hist["train_loss"], "b-", label="Train", linewidth=2)
    axes[0].plot(epochs, hist["val_loss"], "r-", label="Val", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss vs Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Dice
    axes[1].plot(epochs, hist["train_dice"], "b-", label="Train", linewidth=2)
    axes[1].plot(epochs, hist["val_dice"], "r-", label="Val", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Score")
    axes[1].set_title("Dice Score vs Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # IoU
    axes[2].plot(epochs, hist["train_iou"], "b-", label="Train", linewidth=2)
    axes[2].plot(epochs, hist["val_iou"], "r-", label="Val", linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("IoU")
    axes[2].set_title("IoU vs Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path}")


def plot_segmentation_results(model, val_ds, device, output_dir, n_samples=8):
    """Plot: Input MRI | Ground Truth | Prediction | Overlay."""
    model.eval()

    # Find slices WITH tumor
    tumor_indices = []
    for i in range(len(val_ds)):
        _, mask, _ = val_ds[i]
        if mask.sum() > 0:
            tumor_indices.append(i)
        if len(tumor_indices) >= n_samples:
            break

    # If not enough tumor slices, add non-tumor ones
    if len(tumor_indices) < n_samples:
        for i in range(len(val_ds)):
            if i not in tumor_indices:
                tumor_indices.append(i)
            if len(tumor_indices) >= n_samples:
                break

    n = min(n_samples, len(tumor_indices))
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle("Segmentation Results", fontsize=16, fontweight="bold", y=1.02)
    col_titles = ["Input MRI", "Ground Truth", "Predicted Mask", "Overlay"]

    for row, idx in enumerate(tumor_indices[:n]):
        img_t, mask_t, text = val_ds[idx]

        with torch.no_grad():
            out = model(img_t.unsqueeze(0).to(device), texts=[text])
        pred = (torch.sigmoid(out["seg_logits"]) > 0.5).float()

        img_np = img_t[0].numpy()
        gt_np = mask_t[0].numpy()
        pred_np = pred[0, 0].cpu().numpy()

        # Input MRI
        axes[row, 0].imshow(img_np, cmap="gray")
        axes[row, 0].set_title(col_titles[0] if row == 0 else "")
        axes[row, 0].axis("off")

        # Ground Truth
        axes[row, 1].imshow(gt_np, cmap="Reds", vmin=0, vmax=1)
        axes[row, 1].set_title(col_titles[1] if row == 0 else "")
        axes[row, 1].axis("off")

        # Predicted Mask
        axes[row, 2].imshow(pred_np, cmap="Blues", vmin=0, vmax=1)
        axes[row, 2].set_title(col_titles[2] if row == 0 else "")
        axes[row, 2].axis("off")

        # Overlay: MRI + GT (red) + Pred (blue)
        overlay = np.stack([img_np] * 3, axis=-1)
        overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-8)
        overlay[gt_np > 0.5, 0] = 1.0   # GT in red
        overlay[gt_np > 0.5, 1] *= 0.3
        overlay[pred_np > 0.5, 2] = 1.0  # Pred in blue
        axes[row, 3].imshow(overlay)
        d = dice_score(pred_np, gt_np)
        d_str = f"{d:.3f}" if not np.isnan(d) else "N/A"
        axes[row, 3].set_title(f"Dice: {d_str}" if row == 0 else f"D={d_str}")
        axes[row, 3].axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "segmentation_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path}")


def plot_dice_distribution(eval_results_path, output_dir):
    """Plot histogram of per-sample Dice scores."""
    with open(eval_results_path) as f:
        results = json.load(f)

    dice_scores = results.get("per_sample_dice", [])
    if not dice_scores:
        print("⚠️  No per-sample dice scores found, skipping distribution plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(dice_scores, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(dice_scores), color="red", linestyle="--",
               linewidth=2, label=f"Mean: {np.mean(dice_scores):.4f}")
    ax.axvline(np.median(dice_scores), color="orange", linestyle="--",
               linewidth=2, label=f"Median: {np.median(dice_scores):.4f}")
    ax.set_xlabel("Dice Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Dice Score Distribution (Tumor Slices Only)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "dice_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path}")


def plot_failure_cases(model, val_ds, device, output_dir, n_cases=4):
    """Show worst-performing tumor slices with error maps."""
    model.eval()

    # Collect dice scores for tumor slices
    sample_data = []
    for i in range(len(val_ds)):
        img_t, mask_t, text = val_ds[i]
        if mask_t.sum() == 0:
            continue

        with torch.no_grad():
            out = model(img_t.unsqueeze(0).to(device), texts=[text])
        pred = (torch.sigmoid(out["seg_logits"]) > 0.5).float()

        p = pred[0, 0].cpu().numpy()
        g = mask_t[0].numpy()
        d = dice_score(p, g)

        if not np.isnan(d):
            sample_data.append((d, img_t[0].numpy(), g, p, i))

    if not sample_data:
        print("⚠️  No tumor slices for failure analysis")
        return

    # Sort by dice (ascending = worst first)
    sample_data.sort(key=lambda x: x[0])
    worst = sample_data[:n_cases]

    fig, axes = plt.subplots(len(worst), 4, figsize=(16, 4 * len(worst)))
    if len(worst) == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle("Failure Cases (Lowest Dice Scores)", fontsize=16, fontweight="bold")

    for row, (d, img, gt, pred, idx) in enumerate(worst):
        error_map = np.abs(gt - pred)

        axes[row, 0].imshow(img, cmap="gray")
        axes[row, 0].set_title("Input MRI" if row == 0 else "")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt, cmap="Reds")
        axes[row, 1].set_title("Ground Truth" if row == 0 else "")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred, cmap="Blues")
        axes[row, 2].set_title("Prediction" if row == 0 else "")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(error_map, cmap="hot")
        axes[row, 3].set_title(f"Error Map (Dice={d:.3f})" if row == 0
                                else f"Dice={d:.3f}")
        axes[row, 3].axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "failure_cases.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hussain_dir", type=str, default=DEFAULT_HUSSAIN_DIR)
    parser.add_argument("--text_csv", type=str, default=DEFAULT_TEXT_CSV)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--history", type=str, default=None)
    parser.add_argument("--eval_results", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no_text", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "baseline" if args.no_text else "vlm"

    # Defaults
    if args.checkpoint is None:
        args.checkpoint = os.path.join(DEFAULT_CHECKPOINT_DIR, f"best_{mode}.pth")
    if args.history is None:
        args.history = os.path.join(DEFAULT_CHECKPOINT_DIR, f"history_{mode}.json")
    if args.eval_results is None:
        args.eval_results = os.path.join(DEFAULT_OUTPUT_DIR, f"eval_results_{mode}.json")

    # 1. Training curves
    if os.path.isfile(args.history):
        plot_training_curves(args.history, args.output_dir)
    else:
        print(f"⚠️  History not found: {args.history}")

    # 2. Load model for segmentation visualizations
    if os.path.isfile(args.checkpoint):
        model = VLMSegModel(no_text=args.no_text).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        val_ds = BraTSDataset(
            args.hussain_dir, split="val", text_csv=args.text_csv,
            transform=get_val_transforms(), image_size=IMG_SIZE,
        )

        # Segmentation results
        plot_segmentation_results(model, val_ds, device, args.output_dir)

        # Failure cases
        plot_failure_cases(model, val_ds, device, args.output_dir)
    else:
        print(f"⚠️  Checkpoint not found: {args.checkpoint}")

    # 3. Dice distribution
    if os.path.isfile(args.eval_results):
        plot_dice_distribution(args.eval_results, args.output_dir)
    else:
        print(f"⚠️  Eval results not found: {args.eval_results}")

    print(f"\n🎨 All visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
