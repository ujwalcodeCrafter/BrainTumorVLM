"""
evaluate.py — Evaluation with all required metrics.
Dice, IoU, Hausdorff Distance, Precision, Recall, ROUGE, BLEU.

Usage:
    python evaluate.py --hussain_dir /content/FLAIR_BRATS2020_split \
                       --text_csv /content/data/text_brats.csv \
                       --checkpoint /content/checkpoints/best_vlm.pth \
                       --output_dir /content/outputs
"""

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff

from config import *
from dataset import BraTSDataset, get_val_transforms
from model import VLMSegModel


# ═══════════════════════════════════════════════════════════════
# Segmentation Metrics (per-sample, tumor-only)
# ═══════════════════════════════════════════════════════════════

def dice_score(pred, gt):
    """Dice. Returns NaN if GT is empty."""
    if gt.sum() == 0 and pred.sum() == 0:
        return float("nan")  # both empty — skip
    if gt.sum() == 0:
        return 0.0  # GT empty but predicted something
    intersection = (pred * gt).sum()
    return float(2.0 * intersection / (pred.sum() + gt.sum() + 1e-8))


def iou_score(pred, gt):
    """IoU. Returns NaN if GT is empty."""
    if gt.sum() == 0 and pred.sum() == 0:
        return float("nan")
    if gt.sum() == 0:
        return 0.0
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    return float(intersection / (union + 1e-8))


def hausdorff_distance(pred, gt):
    """Hausdorff distance between predicted and GT boundaries."""
    if gt.sum() == 0 or pred.sum() == 0:
        return float("nan")

    pred_pts = np.argwhere(pred > 0)
    gt_pts = np.argwhere(gt > 0)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("nan")

    d1 = directed_hausdorff(pred_pts, gt_pts)[0]
    d2 = directed_hausdorff(gt_pts, pred_pts)[0]
    return float(max(d1, d2))


def precision_score(pred, gt):
    """Precision = TP / (TP + FP)."""
    if gt.sum() == 0 and pred.sum() == 0:
        return float("nan")
    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    if tp + fp == 0:
        return 0.0
    return float(tp / (tp + fp))


def recall_score(pred, gt):
    """Recall = TP / (TP + FN)."""
    if gt.sum() == 0:
        return float("nan")
    tp = (pred * gt).sum()
    fn = ((1 - pred) * gt).sum()
    if tp + fn == 0:
        return 0.0
    return float(tp / (tp + fn))


# ═══════════════════════════════════════════════════════════════
# Text Metrics — generate description from mask, compare with GT
# ═══════════════════════════════════════════════════════════════

def mask_to_text(mask_2d):
    """Generate a simple radiology-style description from a binary mask."""
    h, w = mask_2d.shape
    tumor_pixels = mask_2d.sum()
    total_pixels = h * w

    if tumor_pixels == 0:
        return "No significant abnormality detected on FLAIR sequence."

    ratio = tumor_pixels / total_pixels
    # Determine size descriptor
    if ratio < 0.02:
        size = "small"
    elif ratio < 0.08:
        size = "moderate"
    else:
        size = "large"

    # Determine location
    ys, xs = np.where(mask_2d > 0)
    cy, cx = ys.mean(), xs.mean()

    lr = "left" if cx < w / 2 else "right"
    tb = "anterior" if cy < h / 2 else "posterior"

    if cy < h * 0.33:
        region = "frontal"
    elif cy < h * 0.66:
        region = "parietal"
    else:
        region = "occipital"

    desc = (
        f"FLAIR MRI demonstrates a {size} hyperintense lesion in the "
        f"{lr} {region} lobe, {tb} region. "
        f"The lesion occupies approximately {ratio*100:.1f}% of the visible brain area."
    )

    if ratio > 0.05:
        desc += " Surrounding edema is noted."
    if ratio > 0.10:
        desc += " Mass effect with mild midline shift."

    return desc


def compute_text_metrics(pred_texts, gt_texts):
    """Compute ROUGE and BLEU scores between generated and GT descriptions."""
    results = {}

    # ROUGE
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1, r2, rl = [], [], []
        for pred_t, gt_t in zip(pred_texts, gt_texts):
            scores = scorer.score(gt_t, pred_t)
            r1.append(scores["rouge1"].fmeasure)
            r2.append(scores["rouge2"].fmeasure)
            rl.append(scores["rougeL"].fmeasure)
        results["rouge1"] = float(np.mean(r1))
        results["rouge2"] = float(np.mean(r2))
        results["rougeL"] = float(np.mean(rl))
    except ImportError:
        results["rouge1"] = results["rouge2"] = results["rougeL"] = 0.0
        print("⚠️  rouge-score not installed, skipping ROUGE")

    # BLEU
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smooth = SmoothingFunction().method1
        bleu_scores = []
        for pred_t, gt_t in zip(pred_texts, gt_texts):
            ref = nltk.word_tokenize(gt_t.lower())
            hyp = nltk.word_tokenize(pred_t.lower())
            if len(hyp) == 0:
                bleu_scores.append(0.0)
            else:
                bleu_scores.append(sentence_bleu([ref], hyp, smoothing_function=smooth))
        results["bleu"] = float(np.mean(bleu_scores))
    except ImportError:
        results["bleu"] = 0.0
        print("⚠️  nltk not installed, skipping BLEU")

    return results


# ═══════════════════════════════════════════════════════════════
# Main Evaluation
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hussain_dir", type=str, default=DEFAULT_HUSSAIN_DIR)
    parser.add_argument("--text_csv", type=str, default=DEFAULT_TEXT_CSV)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no_text", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Default checkpoint path
    if args.checkpoint is None:
        mode = "baseline" if args.no_text else "vlm"
        args.checkpoint = os.path.join(DEFAULT_CHECKPOINT_DIR, f"best_{mode}.pth")

    # ── Load model ──
    model = VLMSegModel(no_text=args.no_text).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"✅ Loaded checkpoint: {args.checkpoint}")
    print(f"   Epoch: {ckpt.get('epoch', '?')}, Val Dice: {ckpt.get('val_dice', '?')}")

    # ── Data ──
    val_ds = BraTSDataset(
        args.hussain_dir, split="val", text_csv=args.text_csv,
        transform=get_val_transforms(), image_size=IMG_SIZE,
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # ── Evaluate ──
    all_dice, all_iou, all_hd, all_prec, all_rec = [], [], [], [], []
    pred_texts, gt_texts = [], []
    total_slices = 0
    tumor_slices = 0

    print(f"\n📊 Evaluating on {len(val_ds)} samples...")

    for images, masks, texts in tqdm(val_loader, desc="Eval"):
        images = images.to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                if args.no_text:
                    out = model(images, texts=None)
                else:
                    out = model(images, texts=texts)

        preds = (torch.sigmoid(out["seg_logits"]) > 0.5).float()

        # Per-sample metrics
        for j in range(preds.size(0)):
            p = preds[j, 0].cpu().numpy()
            g = masks[j, 0].numpy()
            total_slices += 1

            d = dice_score(p, g)
            if not np.isnan(d):
                tumor_slices += 1
                all_dice.append(d)
                all_iou.append(iou_score(p, g))
                all_hd.append(hausdorff_distance(p, g))
                all_prec.append(precision_score(p, g))
                all_rec.append(recall_score(p, g))

                # Text generation for ROUGE/BLEU
                pred_texts.append(mask_to_text(p))
                gt_texts.append(texts[j])

    # ── Aggregate ──
    # Filter NaN from hausdorff
    valid_hd = [h for h in all_hd if not np.isnan(h)]

    results = {
        "num_total_slices": total_slices,
        "num_tumor_slices": tumor_slices,
        "dice_mean": float(np.mean(all_dice)) if all_dice else 0.0,
        "dice_std": float(np.std(all_dice)) if all_dice else 0.0,
        "iou_mean": float(np.mean(all_iou)) if all_iou else 0.0,
        "iou_std": float(np.std(all_iou)) if all_iou else 0.0,
        "hausdorff_mean": float(np.mean(valid_hd)) if valid_hd else 0.0,
        "hausdorff_std": float(np.std(valid_hd)) if valid_hd else 0.0,
        "precision_mean": float(np.mean(all_prec)) if all_prec else 0.0,
        "recall_mean": float(np.mean(all_rec)) if all_rec else 0.0,
    }

    # Text metrics
    if pred_texts:
        text_metrics = compute_text_metrics(pred_texts, gt_texts)
        results.update(text_metrics)
    else:
        results.update({"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "bleu": 0.0})

    # Also save per-sample dice for visualizations
    results["per_sample_dice"] = all_dice

    # ── Print ──
    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS ({total_slices} total, {tumor_slices} tumor slices)")
    print(f"{'='*60}")
    for k, v in results.items():
        if k == "per_sample_dice":
            continue
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")

    # ── Save ──
    mode_name = "baseline" if args.no_text else "vlm"
    out_path = os.path.join(args.output_dir, f"eval_results_{mode_name}.json")

    # Convert for JSON serialization
    save_results = {k: v for k, v in results.items()}
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)

    print(f"\n💾 Results saved to: {out_path}")


if __name__ == "__main__":
    main()
