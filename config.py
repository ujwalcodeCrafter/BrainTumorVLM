"""
config.py — Central configuration for BrainTumorVLM project.
All hyperparameters, paths, and settings in one place.
"""

# ── Image Settings ──
IMG_SIZE = 224
IN_CHANNELS = 1
NUM_CLASSES = 1

# ── Model Settings ──
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
CLIP_EMBED_DIM = 512
ENCODER_NAME = "resnet34"

# ── Training Settings ──
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 2

# ── Data Settings ──
SLICES_PER_PATIENT = 155

# ── Default Paths (for Colab) ──
DEFAULT_HUSSAIN_DIR = "/content/FLAIR_BRATS2020_split"
DEFAULT_TEXT_CSV = "/content/data/text_brats.csv"
DEFAULT_CHECKPOINT_DIR = "/content/checkpoints"
DEFAULT_OUTPUT_DIR = "/content/outputs"
