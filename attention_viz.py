"""
attention_viz.py — Grad-CAM attention map visualization.
Shows where the model focuses when making segmentation decisions.

Usage:
    python attention_viz.py --hussain_dir /content/FLAIR_BRATS2020_split \
                            --text_csv /content/data/text_brats.csv \
                            --checkpoint /content/checkpoints/best_vlm.pth \
                            --output_dir /content/outputs
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import *
from dataset import BraTSDataset, get_val_transforms
from model import VLMSegModel


class GradCAM:
    """Grad-CAM for the last encoder layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, texts=None):
        """Generate Grad-CAM heatmap."""
        self.model.eval()

        # Temporarily enable gradients for the model
        for p in self.model.parameters():
            p.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor, texts=texts)
        logits = output["seg_logits"]

        # Backward on mean of positive predictions
        self.model.zero_grad()
        target = torch.sigmoid(logits)
        target.mean().backward()

        # Restore frozen parameters
        if hasattr(self.model, "text_encoder"):
            for p in self.model.text_encoder.parameters():
                p.requires_grad_(False)

        if self.gradients is None or self.activations is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam = F.relu(cam)

        # Resize to input size
        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                            mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()

        # Normalize
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hussain_dir", type=str, default=DEFAULT_HUSSAIN_DIR)
    parser.add_argument("--text_csv", type=str, default=DEFAULT_TEXT_CSV)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n_samples", type=int, default=6)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint is None:
        args.checkpoint = os.path.join(DEFAULT_CHECKPOINT_DIR, "best_vlm.pth")

    # Load model
    model = VLMSegModel(no_text=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Target: last layer of encoder
    target_layer = model.encoder.layer4[-1]
    grad_cam = GradCAM(model, target_layer)

    # Dataset
    val_ds = BraTSDataset(
        args.hussain_dir, split="val", text_csv=args.text_csv,
        transform=get_val_transforms(), image_size=IMG_SIZE,
    )

    # Find tumor slices
    tumor_indices = []
    for i in range(len(val_ds)):
        _, mask, _ = val_ds[i]
        if mask.sum() > 0:
            tumor_indices.append(i)
        if len(tumor_indices) >= args.n_samples:
            break

    # Add non-tumor if needed
    while len(tumor_indices) < args.n_samples and len(tumor_indices) < len(val_ds):
        for i in range(len(val_ds)):
            if i not in tumor_indices:
                tumor_indices.append(i)
                break

    n = min(args.n_samples, len(tumor_indices))

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle("Grad-CAM Attention Maps", fontsize=16, fontweight="bold")

    for row, idx in enumerate(tumor_indices[:n]):
        img_t, mask_t, text = val_ds[idx]
        img_gpu = img_t.unsqueeze(0).to(device)

        # Generate Grad-CAM
        cam = grad_cam.generate(img_gpu, texts=[text])

        # Get prediction
        with torch.no_grad():
            out = model(img_gpu, texts=[text])
        pred = (torch.sigmoid(out["seg_logits"]) > 0.5).float()

        img_np = img_t[0].numpy()
        gt_np = mask_t[0].numpy()
        pred_np = pred[0, 0].cpu().numpy()

        # Input
        axes[row, 0].imshow(img_np, cmap="gray")
        axes[row, 0].set_title("Input MRI" if row == 0 else "")
        axes[row, 0].axis("off")

        # Ground Truth
        axes[row, 1].imshow(gt_np, cmap="Reds", vmin=0, vmax=1)
        axes[row, 1].set_title("Ground Truth" if row == 0 else "")
        axes[row, 1].axis("off")

        # Grad-CAM heatmap overlay
        axes[row, 2].imshow(img_np, cmap="gray")
        axes[row, 2].imshow(cam, cmap="jet", alpha=0.5)
        axes[row, 2].set_title("Grad-CAM" if row == 0 else "")
        axes[row, 2].axis("off")

        # Prediction
        axes[row, 3].imshow(pred_np, cmap="Blues", vmin=0, vmax=1)
        axes[row, 3].set_title("Prediction" if row == 0 else "")
        axes[row, 3].axis("off")

    plt.tight_layout()
    path = os.path.join(args.output_dir, "gradcam_attention.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path}")


if __name__ == "__main__":
    main()
