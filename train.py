"""
train.py — Training loop for VLM Brain Tumor Segmentation.
Supports both VLM (with text) and baseline (no text) training.

Usage:
    # VLM mode (with text):
    python train.py --hussain_dir /content/FLAIR_BRATS2020_split \
                    --text_csv /content/data/text_brats.csv \
                    --epochs 50 --batch_size 8

    # Baseline mode (no text, for ablation):
    python train.py --hussain_dir /content/FLAIR_BRATS2020_split \
                    --text_csv /content/data/text_brats.csv \
                    --no_text --epochs 50 --batch_size 8
"""

import os
import sys
import json
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from dataset import BraTSDataset, get_train_transforms, get_val_transforms
from model import VLMSegModel, DiceBCELoss


# ═══════════════════════════════════════════════════════════════
# Metrics (tumor-only, to avoid inflated scores from empty masks)
# ═══════════════════════════════════════════════════════════════

def compute_dice(pred, target, smooth=1e-6):
    """Compute Dice for a batch. Only counts slices with tumor in GT."""
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    # Per-sample
    gt_sums = target_flat.sum(dim=1)
    has_tumor = gt_sums > 0

    if has_tumor.sum() == 0:
        return torch.tensor(float("nan"))

    intersection = (pred_flat[has_tumor] * target_flat[has_tumor]).sum(dim=1)
    pred_sums = pred_flat[has_tumor].sum(dim=1)
    gt_sums_t = gt_sums[has_tumor]

    dice = (2.0 * intersection + smooth) / (pred_sums + gt_sums_t + smooth)
    return dice.mean()


def compute_iou(pred, target, smooth=1e-6):
    """Compute IoU for a batch. Only counts slices with tumor in GT."""
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    gt_sums = target_flat.sum(dim=1)
    has_tumor = gt_sums > 0

    if has_tumor.sum() == 0:
        return torch.tensor(float("nan"))

    intersection = (pred_flat[has_tumor] * target_flat[has_tumor]).sum(dim=1)
    union = pred_flat[has_tumor].sum(dim=1) + gt_sums[has_tumor] - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


# ═══════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, no_text):
    model.train()
    running_loss = 0.0
    dice_vals = []
    iou_vals = []

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, masks, texts in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            if no_text:
                out = model(images, texts=None)
            else:
                out = model(images, texts=texts)
            loss = criterion(out["seg_logits"], masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        # Metrics (on sigmoid predictions)
        with torch.no_grad():
            pred_binary = (torch.sigmoid(out["seg_logits"]) > 0.5).float()
            d = compute_dice(pred_binary, masks)
            i = compute_iou(pred_binary, masks)
            if not torch.isnan(d):
                dice_vals.append(d.item())
            if not torch.isnan(i):
                iou_vals.append(i.item())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    n = len(loader.dataset)
    avg_loss = running_loss / n
    avg_dice = np.mean(dice_vals) if dice_vals else 0.0
    avg_iou = np.mean(iou_vals) if iou_vals else 0.0

    return avg_loss, avg_dice, avg_iou


@torch.no_grad()
def validate(model, loader, criterion, device, no_text):
    model.eval()
    running_loss = 0.0
    dice_vals = []
    iou_vals = []

    for images, masks, texts in loader:
        images = images.to(device)
        masks = masks.to(device)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            if no_text:
                out = model(images, texts=None)
            else:
                out = model(images, texts=texts)
            loss = criterion(out["seg_logits"], masks)

        running_loss += loss.item() * images.size(0)

        pred_binary = (torch.sigmoid(out["seg_logits"]) > 0.5).float()
        d = compute_dice(pred_binary, masks)
        i = compute_iou(pred_binary, masks)
        if not torch.isnan(d):
            dice_vals.append(d.item())
        if not torch.isnan(i):
            iou_vals.append(i.item())

    n = len(loader.dataset)
    avg_loss = running_loss / n
    avg_dice = np.mean(dice_vals) if dice_vals else 0.0
    avg_iou = np.mean(iou_vals) if iou_vals else 0.0

    return avg_loss, avg_dice, avg_iou


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hussain_dir", type=str, default=DEFAULT_HUSSAIN_DIR)
    parser.add_argument("--text_csv", type=str, default=DEFAULT_TEXT_CSV)
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--no_text", action="store_true", help="Baseline mode (no text)")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ──
    train_ds = BraTSDataset(
        args.hussain_dir, split="train", text_csv=args.text_csv,
        transform=get_train_transforms(), image_size=IMG_SIZE,
    )
    val_ds = BraTSDataset(
        args.hussain_dir, split="val", text_csv=args.text_csv,
        transform=get_val_transforms(), image_size=IMG_SIZE,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # ── Model ──
    mode_name = "baseline" if args.no_text else "vlm"
    print(f"\n[INFO] Training in {mode_name.upper()} mode")

    model = VLMSegModel(no_text=args.no_text).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── Training History ──
    history = {
        "train_loss": [], "train_dice": [], "train_iou": [],
        "val_loss": [], "val_dice": [], "val_iou": [],
    }
    best_val_dice = 0.0

    print(f"\n{'='*60}")
    print(f"  Training: {args.epochs} epochs, batch_size={args.batch_size}")
    print(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_dice, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, args.no_text
        )
        val_loss, val_dice, val_iou = validate(
            model, val_loader, criterion, device, args.no_text
        )
        scheduler.step()

        elapsed = time.time() - t0

        # Record history
        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["train_iou"].append(train_iou)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Dice: {train_dice:.4f} IoU: {train_iou:.4f} | "
            f"Val Loss: {val_loss:.4f} Dice: {val_dice:.4f} IoU: {val_iou:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Save best checkpoint
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            ckpt_path = os.path.join(
                args.checkpoint_dir, f"best_{mode_name}.pth"
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
                "val_iou": val_iou,
            }, ckpt_path)
            print(f"  ✅ Saved best model (Dice: {val_dice:.4f}) → {ckpt_path}")

    # ── Save final checkpoint ──
    final_path = os.path.join(args.checkpoint_dir, f"final_{mode_name}.pth")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "val_dice": val_dice,
    }, final_path)

    # ── Save history ──
    hist_path = os.path.join(args.checkpoint_dir, f"history_{mode_name}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training complete! Best val Dice: {best_val_dice:.4f}")
    print(f"  Checkpoints: {args.checkpoint_dir}")
    print(f"  History:     {hist_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
