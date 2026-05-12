"""
smoke_test.py — Quick sanity check that everything works.
Tests: dataset loading, model forward pass, loss computation.

Usage:
    python smoke_test.py --hussain_dir /content/FLAIR_BRATS2020_split \
                         --text_csv /content/data/text_brats.csv
"""

import os
import sys
import time
import argparse
import traceback

import torch
import numpy as np
from torch.utils.data import DataLoader

from config import *


def check(name, fn):
    """Run a test function, print pass/fail."""
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        print(f"  [PASS] {name:<25s} ✓  ({elapsed:.2f}s)")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [FAIL] {name:<25s} ✗  ({elapsed:.2f}s)")
        print(f"         Error: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hussain_dir", type=str, default=DEFAULT_HUSSAIN_DIR)
    parser.add_argument("--text_csv", type=str, default=DEFAULT_TEXT_CSV)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  SMOKE TEST — BrainTumorVLM")
    print("=" * 60)
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    results = []

    # ── Test 1: Dataset loading ──
    train_ds = None
    val_ds = None

    def test_dataset():
        nonlocal train_ds, val_ds
        from dataset import BraTSDataset, get_train_transforms, get_val_transforms

        train_ds = BraTSDataset(
            args.hussain_dir, split="train", text_csv=args.text_csv,
            transform=get_train_transforms(), image_size=IMG_SIZE,
        )
        val_ds = BraTSDataset(
            args.hussain_dir, split="val", text_csv=args.text_csv,
            transform=get_val_transforms(), image_size=IMG_SIZE,
        )

        # Load one sample
        img, mask, text = train_ds[0]
        assert img.shape == (1, IMG_SIZE, IMG_SIZE), f"Image shape: {img.shape}"
        assert mask.shape == (1, IMG_SIZE, IMG_SIZE), f"Mask shape: {mask.shape}"
        assert isinstance(text, str), f"Text type: {type(text)}"

    results.append(check("Dataset loading", test_dataset))

    # ── Test 2: DataLoader ──
    def test_dataloader():
        loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
        batch = next(iter(loader))
        imgs, masks, texts = batch
        assert imgs.shape == (2, 1, IMG_SIZE, IMG_SIZE), f"Batch img: {imgs.shape}"
        assert masks.shape == (2, 1, IMG_SIZE, IMG_SIZE), f"Batch mask: {masks.shape}"
        assert len(texts) == 2

    results.append(check("DataLoader", test_dataloader))

    # ── Test 3: Model forward (VLM) ──
    def test_model_vlm():
        from model import VLMSegModel
        model = VLMSegModel(no_text=False).to(device)

        dummy_img = torch.randn(2, 1, IMG_SIZE, IMG_SIZE).to(device)
        dummy_texts = ["Test report one.", "Test report two."]

        with torch.no_grad():
            out = model(dummy_img, texts=dummy_texts)

        expected = (2, 1, IMG_SIZE, IMG_SIZE)
        actual = out["seg_logits"].shape
        assert actual == expected, f"Expected {expected}, got {actual}"
        assert "img_embed" in out
        assert "text_embed" in out
        assert out["img_embed"].shape == (2, CLIP_EMBED_DIM)
        assert out["text_embed"].shape == (2, CLIP_EMBED_DIM)

    results.append(check("Model VLM forward", test_model_vlm))

    # ── Test 4: Model forward (baseline) ──
    def test_model_baseline():
        from model import VLMSegModel
        model = VLMSegModel(no_text=True).to(device)

        dummy_img = torch.randn(2, 1, IMG_SIZE, IMG_SIZE).to(device)

        with torch.no_grad():
            out = model(dummy_img, texts=None)

        expected = (2, 1, IMG_SIZE, IMG_SIZE)
        actual = out["seg_logits"].shape
        assert actual == expected, f"Expected {expected}, got {actual}"

    results.append(check("Model Baseline forward", test_model_baseline))

    # ── Test 5: Loss computation ──
    def test_loss():
        from model import DiceBCELoss
        criterion = DiceBCELoss()

        logits = torch.randn(2, 1, IMG_SIZE, IMG_SIZE)
        targets = torch.randint(0, 2, (2, 1, IMG_SIZE, IMG_SIZE)).float()

        loss = criterion(logits, targets)
        assert loss.dim() == 0, "Loss should be scalar"
        assert not torch.isnan(loss), "Loss is NaN"
        assert loss.item() > 0, "Loss should be positive"

    results.append(check("Loss computation", test_loss))

    # ── Summary ──
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} passed")
    if passed == total:
        print("  ✅ All tests passed! Ready to train.")
    else:
        print("  ❌ Some tests failed. Check errors above.")
    print(f"{'='*60}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
