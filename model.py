"""
model.py — VLM Segmentation Model.
Architecture: ResNet-34 encoder + Frozen CLIP text encoder + FiLM fusion + U-Net decoder.
Supports both VLM mode (with text) and baseline mode (no text, for ablation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

try:
    import open_clip
except ImportError:
    open_clip = None

from config import CLIP_MODEL_NAME, CLIP_PRETRAINED, CLIP_EMBED_DIM, IMG_SIZE


# ═══════════════════════════════════════════════════════════════
# FiLM Layer — Feature-wise Linear Modulation
# ═══════════════════════════════════════════════════════════════

class FiLMLayer(nn.Module):
    """Modulate visual features using text embeddings: out = (1 + γ) * x + β"""

    def __init__(self, text_dim, feature_dim):
        super().__init__()
        self.gamma_proj = nn.Sequential(
            nn.Linear(text_dim, feature_dim),
            nn.Tanh(),
        )
        self.beta_proj = nn.Linear(text_dim, feature_dim)

    def forward(self, features, text_embed):
        """
        features:   (B, C, H, W)
        text_embed: (B, text_dim)
        """
        gamma = self.gamma_proj(text_embed).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.beta_proj(text_embed).unsqueeze(-1).unsqueeze(-1)
        return features * (1.0 + gamma) + beta


# ═══════════════════════════════════════════════════════════════
# Decoder Block
# ═══════════════════════════════════════════════════════════════

class DecoderBlock(nn.Module):
    """Upsample + concatenate skip + double conv."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Match spatial dimensions of skip connection
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ═══════════════════════════════════════════════════════════════
# CLIP Text Encoder (Frozen)
# ═══════════════════════════════════════════════════════════════

class CLIPTextEncoder(nn.Module):
    """Frozen CLIP text encoder producing (B, 512) embeddings."""

    def __init__(self):
        super().__init__()
        if open_clip is None:
            raise ImportError("Install open-clip-torch: pip install open-clip-torch")

        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
        )
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

        # Freeze all CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.clip_model.eval()

    @torch.no_grad()
    def forward(self, texts, device):
        """
        texts:  list of strings, length B
        device: torch.device
        Returns: (B, 512) text embeddings, L2-normalized
        """
        tokens = self.tokenizer(texts).to(device)
        text_features = self.clip_model.encode_text(tokens)
        text_features = F.normalize(text_features.float(), dim=-1)
        return text_features


# ═══════════════════════════════════════════════════════════════
# ResNet-34 Encoder with skip connections
# ═══════════════════════════════════════════════════════════════

class ResNetEncoder(nn.Module):
    """
    ResNet-34 encoder modified for 1-channel input.
    Returns bottleneck + 4 skip connection feature maps.
    
    For 224×224 input:
        skip0: (B,  64, 112, 112)  — after conv1+bn+relu (before maxpool)
        skip1: (B,  64,  56,  56)  — after layer1
        skip2: (B, 128,  28,  28)  — after layer2
        skip3: (B, 256,  14,  14)  — after layer3
        bottleneck: (B, 512, 7, 7) — after layer4
    """

    def __init__(self):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # Modify first conv for 1-channel input (average pretrained RGB weights)
        old_conv = resnet.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1   # 64 ch
        self.layer2 = resnet.layer2   # 128 ch
        self.layer3 = resnet.layer3   # 256 ch
        self.layer4 = resnet.layer4   # 512 ch

    def forward(self, x):
        # x: (B, 1, 224, 224)
        x0 = self.relu(self.bn1(self.conv1(x)))  # (B, 64, 112, 112)
        x1 = self.layer1(self.maxpool(x0))        # (B, 64, 56, 56)
        x2 = self.layer2(x1)                      # (B, 128, 28, 28)
        x3 = self.layer3(x2)                      # (B, 256, 14, 14)
        x4 = self.layer4(x3)                      # (B, 512, 7, 7)

        return x4, [x0, x1, x2, x3]


# ═══════════════════════════════════════════════════════════════
# Full VLM Segmentation Model
# ═══════════════════════════════════════════════════════════════

class VLMSegModel(nn.Module):
    """
    Text-Guided Brain Tumor Segmentation Model.

    Pipeline:
        1. ResNet-34 encodes MRI image → bottleneck + skip features
        2. CLIP encodes radiology text → text embedding
        3. FiLM modulates bottleneck features with text info
        4. U-Net decoder with skip connections → segmentation mask

    Set no_text=True for ablation (baseline without text input).
    """

    def __init__(self, no_text=False):
        super().__init__()
        self.no_text = no_text

        # Image encoder
        self.encoder = ResNetEncoder()

        # Text encoder (frozen CLIP)
        if not no_text:
            self.text_encoder = CLIPTextEncoder()

        # FiLM fusion layers — modulate at bottleneck and decoder levels
        if not no_text:
            self.film_bottleneck = FiLMLayer(CLIP_EMBED_DIM, 512)
            self.film_dec3 = FiLMLayer(CLIP_EMBED_DIM, 256)
            self.film_dec2 = FiLMLayer(CLIP_EMBED_DIM, 128)

        # Decoder: bottleneck(512, 7×7) → up to (1, 224×224)
        self.dec4 = DecoderBlock(512, 256, 256)   # 7→14,  cat skip3(256)
        self.dec3 = DecoderBlock(256, 128, 128)   # 14→28, cat skip2(128)
        self.dec2 = DecoderBlock(128, 64, 64)     # 28→56, cat skip1(64)
        self.dec1 = DecoderBlock(64, 64, 64)      # 56→112, cat skip0(64)

        # Final upsample 112→224 + conv
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

        # Text projection for embedding extraction
        if not no_text:
            self.img_proj = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, CLIP_EMBED_DIM),
            )

    def forward(self, images, texts=None):
        """
        images: (B, 1, H, W) FLAIR MRI
        texts:  list[str] of length B, or None

        Returns dict with:
            seg_logits: (B, 1, H, W) raw logits
            img_embed:  (B, 512) image embedding (if text mode)
            text_embed: (B, 512) text embedding (if text mode)
        """
        B = images.shape[0]
        input_size = images.shape[2:]  # (H, W)
        device = images.device

        # 1. Encode image
        bottleneck, skips = self.encoder(images)
        # bottleneck: (B, 512, 7, 7), skips: [skip0, skip1, skip2, skip3]

        result = {}

        # 2. Encode text and apply FiLM fusion
        if not self.no_text and texts is not None:
            text_embed = self.text_encoder(texts, device)   # (B, 512)
            img_embed = self.img_proj(bottleneck)            # (B, 512)
            result["text_embed"] = text_embed
            result["img_embed"] = F.normalize(img_embed, dim=-1)

            # FiLM modulation at bottleneck
            bottleneck = self.film_bottleneck(bottleneck, text_embed)

        # 3. Decode with skip connections
        x = self.dec4(bottleneck, skips[3])   # (B, 256, 14, 14)
        if not self.no_text and texts is not None:
            x = self.film_dec3(x, text_embed)
        x = self.dec3(x, skips[2])            # (B, 128, 28, 28)
        if not self.no_text and texts is not None:
            x = self.film_dec2(x, text_embed)
        x = self.dec2(x, skips[1])            # (B, 64, 56, 56)
        x = self.dec1(x, skips[0])            # (B, 64, 112, 112)

        # 4. Final upsample to full resolution
        x = self.final_up(x)                  # (B, 32, 224, 224)
        seg_logits = self.final_conv(x)       # (B, 1, 224, 224)

        # Safety: ensure output matches input spatial dimensions
        if seg_logits.shape[2:] != input_size:
            seg_logits = F.interpolate(seg_logits, size=input_size,
                                       mode="bilinear", align_corners=False)

        result["seg_logits"] = seg_logits
        return result


# ═══════════════════════════════════════════════════════════════
# Loss Function
# ═══════════════════════════════════════════════════════════════

class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss for binary segmentation."""

    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        logits:  (B, 1, H, W) raw logits
        targets: (B, 1, H, W) binary masks
        """
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        pflat = probs.view(-1)
        tflat = targets.view(-1)
        intersection = (pflat * tflat).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            pflat.sum() + tflat.sum() + self.smooth
        )

        return self.dice_weight * dice_loss + self.bce_weight * bce_loss
