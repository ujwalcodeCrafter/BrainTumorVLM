"""
embedding_viz.py — t-SNE visualization of image and text embeddings.
Demonstrates alignment in embedding space.

Usage:
    python embedding_viz.py --hussain_dir /content/FLAIR_BRATS2020_split \
                            --text_csv /content/data/text_brats.csv \
                            --checkpoint /content/checkpoints/best_vlm.pth \
                            --output_dir /content/outputs
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from config import *
from dataset import BraTSDataset, get_val_transforms
from model import VLMSegModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hussain_dir", type=str, default=DEFAULT_HUSSAIN_DIR)
    parser.add_argument("--text_csv", type=str, default=DEFAULT_TEXT_CSV)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_samples", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint is None:
        args.checkpoint = os.path.join(DEFAULT_CHECKPOINT_DIR, "best_vlm.pth")

    # Load model
    model = VLMSegModel(no_text=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Dataset
    val_ds = BraTSDataset(
        args.hussain_dir, split="val", text_csv=args.text_csv,
        transform=get_val_transforms(), image_size=IMG_SIZE,
    )
    loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    # Collect embeddings
    img_embeds = []
    txt_embeds = []
    has_tumor = []

    n_collected = 0
    with torch.no_grad():
        for images, masks, texts in loader:
            if n_collected >= args.max_samples:
                break

            images = images.to(device)
            out = model(images, texts=texts)

            if "img_embed" in out and "text_embed" in out:
                img_embeds.append(out["img_embed"].cpu().numpy())
                txt_embeds.append(out["text_embed"].cpu().numpy())

                # Track which slices have tumor
                for j in range(masks.size(0)):
                    has_tumor.append(masks[j].sum().item() > 0)

                n_collected += images.size(0)

    if not img_embeds:
        print("⚠️  No embeddings collected. Model may not have text encoder.")
        return

    img_embeds = np.concatenate(img_embeds, axis=0)[:args.max_samples]
    txt_embeds = np.concatenate(txt_embeds, axis=0)[:args.max_samples]
    has_tumor = has_tumor[:args.max_samples]

    print(f"📊 Collected {len(img_embeds)} image + {len(txt_embeds)} text embeddings")

    # ── t-SNE on combined embeddings ──
    combined = np.concatenate([img_embeds, txt_embeds], axis=0)
    n_img = len(img_embeds)

    perplexity = min(30, len(combined) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                n_iter=1000, learning_rate="auto", init="pca")
    coords = tsne.fit_transform(combined)

    img_coords = coords[:n_img]
    txt_coords = coords[n_img:]

    # ── Plot 1: Image vs Text embeddings ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Color by modality
    axes[0].scatter(img_coords[:, 0], img_coords[:, 1],
                    c="steelblue", alpha=0.6, s=30, label="Image", edgecolors="white",
                    linewidth=0.3)
    axes[0].scatter(txt_coords[:, 0], txt_coords[:, 1],
                    c="coral", alpha=0.6, s=30, label="Text", marker="^",
                    edgecolors="white", linewidth=0.3)
    axes[0].set_title("t-SNE: Image vs Text Embeddings", fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.2)

    # Color by tumor presence
    tumor_arr = np.array(has_tumor)
    colors_img = np.where(tumor_arr, "red", "green")
    axes[1].scatter(img_coords[:, 0], img_coords[:, 1],
                    c=colors_img, alpha=0.6, s=30, edgecolors="white", linewidth=0.3)
    # Legend
    axes[1].scatter([], [], c="red", s=50, label="Tumor present")
    axes[1].scatter([], [], c="green", s=50, label="No tumor")
    axes[1].set_title("t-SNE: Image Embeddings by Tumor Presence", fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.2)

    fig.suptitle("Embedding Space Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(args.output_dir, "tsne_embeddings.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path}")


if __name__ == "__main__":
    main()
