"""
dataset.py — PyTorch Dataset for BraTS 2020 FLAIR + Text.
Handles the Hussain .npy format directly. Maps image index → patient → text.
Robust loading for ALL .npy shapes: 2D, 3D, 4D volumes.
"""

import os
import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
except ImportError:
    A = None

from config import IMG_SIZE, SLICES_PER_PATIENT


# ═══════════════════════════════════════════════════════════════
# .npy loader — handles ALL possible shapes from BraTS datasets
# ═══════════════════════════════════════════════════════════════

def load_npy_as_2d(path, is_mask=False):
    """
    Load a .npy file and return a 2D (H, W) float32 array.

    Handles ALL possible shapes from BraTS datasets:
        2D: (H, W)
        3D: (H, W, C), (C, H, W), or (H, W, D) volume
        4D: (H, W, D, C) or (D, H, W, C) — full 3D volume with channels
    """
    arr = np.load(path, allow_pickle=True)

    # Handle object arrays (sometimes np.save wraps dicts)
    if arr.dtype == object:
        item = arr.item()
        if isinstance(item, np.ndarray):
            arr = item
        elif isinstance(item, dict):
            for key in ["data", "image", "mask", "vol", "flair"]:
                if key in item:
                    arr = np.array(item[key])
                    break
            else:
                arr = np.array(list(item.values())[0])

    arr = np.squeeze(arr).astype(np.float32)

    # ── 4D: (H, W, D, C) or (D, H, W, C) etc. ──
    if arr.ndim == 4:
        # Find the channel axis (smallest dim ≤ 4)
        if arr.shape[0] <= 4:       # (C, D, H, W) or (C, H, W, D)
            if is_mask:
                arr = (arr.max(axis=0) > 0).astype(np.float32)
            else:
                arr = arr[0]
            # Now 3D — will be handled below
        elif arr.shape[3] <= 4:     # (H, W, D, C) or (D, H, W, C)
            mid = arr.shape[2] // 2 if arr.shape[2] > arr.shape[0] else arr.shape[0] // 2
            if arr.shape[2] > 4:    # (H, W, D, C) — take middle depth
                arr = arr[:, :, mid, :]
            else:                   # (D, H, W, C) — take middle depth
                arr = arr[mid]
            # Now 3D
        else:
            # All dims > 4, treat dim 0 as depth
            arr = arr[arr.shape[0] // 2]
            # Now 3D

    # ── 3D: could be (C, H, W), (H, W, C), or (H, W, D) volume ──
    if arr.ndim == 3:
        if arr.shape[0] <= 4 and arr.shape[1] > 4 and arr.shape[2] > 4:
            # (C, H, W) — channel first
            if is_mask:
                arr = (arr.max(axis=0) > 0).astype(np.float32)
            else:
                arr = arr[0]
        elif arr.shape[2] <= 4 and arr.shape[0] > 4 and arr.shape[1] > 4:
            # (H, W, C) — channel last
            if is_mask:
                arr = (arr.max(axis=2) > 0).astype(np.float32)
            else:
                arr = arr[:, :, 0]
        else:
            # (H, W, D) volume — take middle slice along last axis
            mid = arr.shape[2] // 2
            arr = arr[:, :, mid]

    # ── 1D: reshape to square ──
    if arr.ndim == 1:
        s = int(np.sqrt(arr.size))
        if s * s == arr.size:
            arr = arr.reshape(s, s)
        else:
            arr = arr[: s * s].reshape(s, s)

    # ── Ensure 2D ──
    if arr.ndim != 2:
        raise ValueError(
            f"Cannot make 2D from shape {np.load(path, allow_pickle=True).shape} "
            f"in {path}"
        )

    # ── Binarize masks ──
    if is_mask:
        arr = (arr > 0).astype(np.float32)

    return arr.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# Augmentation pipeline
# ═══════════════════════════════════════════════════════════════

def get_train_transforms(image_size=IMG_SIZE):
    if A is None:
        return None

    transforms_list = [
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
    ]

    # Use Affine instead of deprecated ShiftScaleRotate
    try:
        transforms_list.append(
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.06, 0.06),
                     rotate=(-15, 15), p=0.4)
        )
    except Exception:
        pass

    transforms_list.extend([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    ])

    try:
        transforms_list.append(A.GaussNoise(p=0.2))
    except Exception:
        pass

    return A.Compose(transforms_list)


def get_val_transforms(image_size=IMG_SIZE):
    if A is None:
        return None
    return A.Compose([A.Resize(image_size, image_size)])


# ═══════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════

class BraTSDataset(Dataset):
    """
    Dataset for Hussain FLAIR BraTS2020 .npy files + TextBraTS text.

    Directory layout (Hussain):
        hussain_dir/
            train/
                images/  image_0.npy, image_1.npy, ...
                masks/   mask_0.npy,  mask_1.npy,  ...
            val/
                images/  ...
                masks/   ...

    Patient mapping:
        patient_id = slice_index // SLICES_PER_PATIENT
        BraTS20_Training_001 → patient 0
        BraTS20_Training_002 → patient 1
        ...
    """

    def __init__(self, hussain_dir, split="train", text_csv=None,
                 transform=None, image_size=IMG_SIZE):
        super().__init__()
        self.image_size = image_size
        self.transform = transform

        # ── Locate image and mask files ──
        img_dir = os.path.join(hussain_dir, split, "images")
        mask_dir = os.path.join(hussain_dir, split, "masks")

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not os.path.isdir(mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        # Sort numerically: image_0.npy, image_1.npy, ...
        def sort_key(fname):
            try:
                return int(fname.split("_")[-1].split(".")[0])
            except ValueError:
                return fname

        img_files = sorted(
            [f for f in os.listdir(img_dir) if f.endswith(".npy")],
            key=sort_key
        )
        mask_files = sorted(
            [f for f in os.listdir(mask_dir) if f.endswith(".npy")],
            key=sort_key
        )

        assert len(img_files) == len(mask_files), (
            f"Mismatch: {len(img_files)} images vs {len(mask_files)} masks in {split}"
        )

        self.image_files = [os.path.join(img_dir, f) for f in img_files]
        self.mask_files = [os.path.join(mask_dir, f) for f in mask_files]

        # ── Load text CSV → patient_id → text ──
        self.patient_texts = {}
        if text_csv and os.path.isfile(text_csv):
            with open(text_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pid = int(row["patient_id"])
                    self.patient_texts[pid] = row["text"]

        # ── Map each slice index to a patient ──
        self.patient_ids = []
        for i in range(len(self.image_files)):
            pid = i // SLICES_PER_PATIENT
            self.patient_ids.append(pid)

        # Count unique patients
        unique_patients = set(self.patient_ids)
        print(f"[INFO] {split} dataset: {len(self.image_files)} samples, "
              f"{len(unique_patients)} patients")

    def __len__(self):
        return len(self.image_files)

    def get_text_for_index(self, idx):
        """Get the radiology report text for a given slice index."""
        pid = self.patient_ids[idx]
        if pid in self.patient_texts:
            return self.patient_texts[pid]
        # Fallback: generic description
        return "FLAIR MRI brain scan showing potential abnormality."

    def __getitem__(self, idx):
        # ── Load image and mask ──
        img = load_npy_as_2d(self.image_files[idx], is_mask=False)
        mask = load_npy_as_2d(self.mask_files[idx], is_mask=True)

        # ── Resize with cv2 ──
        img = cv2.resize(img, (self.image_size, self.image_size),
                         interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size),
                          interpolation=cv2.INTER_NEAREST)

        # ── Normalize image to [0, 1] ──
        img_min, img_max = img.min(), img.max()
        if img_max - img_min > 1e-8:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

        # ── Binarize mask ──
        mask = (mask > 0.5).astype(np.float32)

        # ── Apply augmentations ──
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # ── Convert to tensors: (1, H, W) ──
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)     # (1, H, W)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)   # (1, H, W)

        # ── Get text ──
        text = self.get_text_for_index(idx)

        return img_tensor, mask_tensor, text
