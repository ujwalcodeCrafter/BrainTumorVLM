"""
prepare_data.py — Build text CSV from TextBraTS dataset.
Reads text files from TextBraTSData folders and creates patient→text mapping.
Generates placeholder reports when real text is unavailable.

Usage:
    python prepare_data.py --text_brats_dir /content/TextBraTSData \
                           --hussain_dir /content/FLAIR_BRATS2020_split \
                           --output_csv /content/data/text_brats.csv
"""

import os
import sys
import csv
import glob
import argparse
import numpy as np


# ── Placeholder radiology reports (used when TextBraTS text is unavailable) ──
PLACEHOLDER_REPORTS = [
    "FLAIR MRI demonstrates hyperintense signal in the right temporal lobe with "
    "surrounding vasogenic edema. The lesion measures approximately 4.2 x 3.8 cm "
    "and shows irregular borders suggesting high-grade glioma. Midline shift of "
    "3mm is noted. The ventricles appear mildly compressed on the right side.",

    "Axial FLAIR sequence reveals a large heterogeneous mass in the left frontal "
    "lobe extending into the parietal region. Significant peritumoral edema is "
    "present with mass effect on the lateral ventricle. The tumor shows areas of "
    "necrosis centrally. No hemorrhage identified.",

    "FLAIR imaging shows a well-circumscribed hyperintense lesion in the right "
    "parietal lobe measuring 3.5 x 2.9 cm. Mild surrounding edema is noted. The "
    "lesion appears to involve the white matter predominantly. No significant "
    "midline shift. Findings are consistent with a low-grade glioma.",

    "Multiple areas of FLAIR hyperintensity are observed in the left temporal and "
    "insular regions. The dominant lesion is approximately 5.1 x 4.3 cm with "
    "extensive peritumoral edema causing compression of the left lateral ventricle. "
    "Enhancement pattern suggests glioblastoma multiforme.",

    "FLAIR sequence demonstrates a small hyperintense focus in the right occipital "
    "lobe measuring 2.1 x 1.8 cm. Minimal surrounding edema. The lesion margins "
    "are relatively well-defined. No mass effect or midline shift. Differential "
    "includes low-grade glioma or metastatic disease.",
]


def read_text_from_patient_folder(patient_dir):
    """
    Read text content from a TextBraTS patient folder.
    Tries: .txt files first, then files without extension (Windows 'Text Document').
    Skips .npy files (those are CLIP embeddings, not readable text).
    """
    patient_name = os.path.basename(patient_dir)

    # Strategy 1: Look for .txt files
    txt_files = glob.glob(os.path.join(patient_dir, "*.txt"))
    for tf in sorted(txt_files):
        try:
            with open(tf, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
                if len(content) > 10:
                    return content
        except Exception:
            continue

    # Strategy 2: Look for files WITHOUT an extension (Windows "Text Document")
    for fname in sorted(os.listdir(patient_dir)):
        fpath = os.path.join(patient_dir, fname)
        if os.path.isfile(fpath) and not fname.endswith(".npy"):
            # No extension or unknown extension — try reading as text
            if "." not in fname or fname.endswith("_text"):
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read().strip()
                        if len(content) > 10:
                            return content
                except Exception:
                    continue

    # Strategy 3: Try reading any non-npy file
    for fname in sorted(os.listdir(patient_dir)):
        fpath = os.path.join(patient_dir, fname)
        if os.path.isfile(fpath) and not fname.endswith(".npy"):
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
                    if len(content) > 10:
                        return content
            except Exception:
                continue

    return None


def count_patients_in_hussain(hussain_dir):
    """Count number of patients based on image files in train+val."""
    total_slices = 0
    for split in ["train", "val"]:
        img_dir = os.path.join(hussain_dir, split, "images")
        if os.path.isdir(img_dir):
            total_slices += len([f for f in os.listdir(img_dir) if f.endswith(".npy")])

    from config import SLICES_PER_PATIENT
    n_patients = max(1, (total_slices + SLICES_PER_PATIENT - 1) // SLICES_PER_PATIENT)
    return n_patients, total_slices


def main():
    parser = argparse.ArgumentParser(description="Build text CSV from TextBraTS")
    parser.add_argument("--text_brats_dir", type=str, default="/content/TextBraTSData")
    parser.add_argument("--hussain_dir", type=str, default="/content/FLAIR_BRATS2020_split")
    parser.add_argument("--output_csv", type=str, default="/content/data/text_brats.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # ── Count patients from image data ──
    n_patients, total_slices = count_patients_in_hussain(args.hussain_dir)
    print(f"📊 Hussain dataset: {total_slices} total slices → ~{n_patients} patients")

    # ── Read TextBraTS text files ──
    patient_texts = {}
    extracted = 0

    if os.path.isdir(args.text_brats_dir):
        patient_dirs = sorted(glob.glob(
            os.path.join(args.text_brats_dir, "BraTS20_Training_*")
        ))
        print(f"📂 Found {len(patient_dirs)} patient folders in TextBraTS")

        for pdir in patient_dirs:
            pname = os.path.basename(pdir)
            try:
                pnum = int(pname.split("_")[-1])  # e.g., 001 → 1
            except ValueError:
                continue

            text = read_text_from_patient_folder(pdir)
            if text:
                # Patient index is 0-based: BraTS20_Training_001 → patient 0
                patient_texts[pnum - 1] = text
                extracted += 1

        print(f"✅ Extracted text from {extracted}/{len(patient_dirs)} patients")
    else:
        print(f"⚠️  TextBraTS directory not found: {args.text_brats_dir}")

    # ── Fill missing patients with placeholders ──
    for pid in range(n_patients):
        if pid not in patient_texts:
            patient_texts[pid] = PLACEHOLDER_REPORTS[pid % len(PLACEHOLDER_REPORTS)]

    # ── Write CSV ──
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "text"])
        for pid in sorted(patient_texts.keys()):
            writer.writerow([pid, patient_texts[pid]])

    print(f"\n✅ Text CSV: {args.output_csv}  ({len(patient_texts)} entries)")

    # Show sample
    for pid in sorted(patient_texts.keys())[:3]:
        text_preview = patient_texts[pid][:100] + "..."
        print(f"   Patient {pid}: {text_preview}")


if __name__ == "__main__":
    main()
