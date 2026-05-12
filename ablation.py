"""
ablation.py — Ablation study comparing model WITH text vs WITHOUT text.
Generates comparison bar charts and summary table.

Usage:
    python ablation.py --hussain_dir /content/FLAIR_BRATS2020_split \
                       --text_csv /content/data/text_brats.csv \
                       --vlm_checkpoint /content/checkpoints/best_vlm.pth \
                       --baseline_checkpoint /content/checkpoints/best_baseline.pth \
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
from torch.utils.data import DataLoader

from config import *
from dataset import BraTSDataset, get_val_transforms
from model import VLMSegModel
from evaluate import dice_score, iou_score, hausdorff_distance, precision_score, recall_score


def evaluate_model_quick(model, val_ds, device, no_text):
    """Quick evaluation returning all metrics."""
    model.eval()
    loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    all_dice, all_iou, all_hd, all_prec, all_rec = [], [], [], [], []

    with torch.no_grad():
        for images, masks, texts in loader:
            images = images.to(device)

            if no_text:
                out = model(images, texts=None)
            else:
                out = model(images, texts=texts)

            preds = (torch.sigmoid(out["seg_logits"]) > 0.5).float()

            for j in range(preds.size(0)):
                p = preds[j, 0].cpu().numpy()
                g = masks[j, 0].numpy()

                d = dice_score(p, g)
                if not np.isnan(d):
                    all_dice.append(d)
                    all_iou.append(iou_score(p, g))
                    hd = hausdorff_distance(p, g)
                    all_hd.append(hd if not np.isnan(hd) else 0.0)
                    all_prec.append(precision_score(p, g))
                    all_rec.append(recall_score(p, g))

    return {
        "dice": np.mean(all_dice) if all_dice else 0.0,
        "iou": np.mean(all_iou) if all_iou else 0.0,
        "hausdorff": np.mean(all_hd) if all_hd else 0.0,
        "precision": np.mean(all_prec) if all_prec else 0.0,
        "recall": np.mean(all_rec) if all_rec else 0.0,
        "n_tumor_slices": len(all_dice),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hussain_dir", type=str, default=DEFAULT_HUSSAIN_DIR)
    parser.add_argument("--text_csv", type=str, default=DEFAULT_TEXT_CSV)
    parser.add_argument("--vlm_checkpoint", type=str, default=None)
    parser.add_argument("--baseline_checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.vlm_checkpoint is None:
        args.vlm_checkpoint = os.path.join(DEFAULT_CHECKPOINT_DIR, "best_vlm.pth")
    if args.baseline_checkpoint is None:
        args.baseline_checkpoint = os.path.join(DEFAULT_CHECKPOINT_DIR, "best_baseline.pth")

    # Dataset
    val_ds = BraTSDataset(
        args.hussain_dir, split="val", text_csv=args.text_csv,
        transform=get_val_transforms(), image_size=IMG_SIZE,
    )

    # ── Evaluate VLM (with text) ──
    print("📊 Evaluating VLM model (with text)...")
    vlm_model = VLMSegModel(no_text=False).to(device)
    ckpt = torch.load(args.vlm_checkpoint, map_location=device, weights_only=False)
    vlm_model.load_state_dict(ckpt["model_state_dict"])
    vlm_metrics = evaluate_model_quick(vlm_model, val_ds, device, no_text=False)
    del vlm_model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # ── Evaluate Baseline (without text) ──
    print("📊 Evaluating Baseline model (no text)...")
    base_model = VLMSegModel(no_text=True).to(device)
    ckpt = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
    base_model.load_state_dict(ckpt["model_state_dict"])
    base_metrics = evaluate_model_quick(base_model, val_ds, device, no_text=True)
    del base_model

    # ── Print comparison ──
    print(f"\n{'='*60}")
    print(f"  ABLATION STUDY RESULTS")
    print(f"{'='*60}")
    print(f"  {'Metric':<15} {'VLM (text)':>12} {'Baseline':>12} {'Δ Improvement':>15}")
    print(f"  {'-'*54}")
    for metric in ["dice", "iou", "precision", "recall"]:
        v = vlm_metrics[metric]
        b = base_metrics[metric]
        delta = v - b
        sign = "+" if delta >= 0 else ""
        print(f"  {metric:<15} {v:>12.4f} {b:>12.4f} {sign}{delta:>14.4f}")
    # Hausdorff (lower is better)
    v = vlm_metrics["hausdorff"]
    b = base_metrics["hausdorff"]
    delta = b - v  # reverse: lower HD is better
    sign = "+" if delta >= 0 else ""
    print(f"  {'hausdorff':<15} {v:>12.2f} {b:>12.2f} {sign}{delta:>14.2f} (↓ better)")
    print(f"{'='*60}")

    # ── Bar chart ──
    metrics_names = ["Dice", "IoU", "Precision", "Recall"]
    vlm_vals = [vlm_metrics[m.lower()] for m in metrics_names]
    base_vals = [base_metrics[m.lower()] for m in metrics_names]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    bars1 = axes[0].bar(x - width/2, vlm_vals, width, label="VLM (with text)",
                        color="steelblue", edgecolor="white")
    bars2 = axes[0].bar(x + width/2, base_vals, width, label="Baseline (no text)",
                        color="coral", edgecolor="white")

    axes[0].set_ylabel("Score", fontsize=12)
    axes[0].set_title("Ablation: VLM vs Baseline", fontsize=14, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_names, fontsize=11)
    axes[0].legend(fontsize=11)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar in bars1:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{bar.get_height():.3f}", ha="center", fontsize=9)
    for bar in bars2:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{bar.get_height():.3f}", ha="center", fontsize=9)

    # Improvement delta chart
    deltas = [v - b for v, b in zip(vlm_vals, base_vals)]
    colors = ["green" if d >= 0 else "red" for d in deltas]
    axes[1].bar(metrics_names, deltas, color=colors, edgecolor="white", alpha=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Improvement (VLM - Baseline)", fontsize=12)
    axes[1].set_title("Text Contribution (Δ)", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")
    for i, d in enumerate(deltas):
        axes[1].text(i, d + 0.005 * (1 if d >= 0 else -1),
                     f"{d:+.4f}", ha="center", fontsize=10)

    plt.tight_layout()
    path = os.path.join(args.output_dir, "ablation_study.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path}")

    # Save results JSON
    results = {
        "vlm": vlm_metrics,
        "baseline": base_metrics,
        "improvement": {m: vlm_metrics[m] - base_metrics[m]
                        for m in ["dice", "iou", "precision", "recall"]},
    }
    json_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"💾 Saved: {json_path}")


if __name__ == "__main__":
    main()
