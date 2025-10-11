"""
Configuration file for DiagXNet-Lite project
"""

from pathlib import Path
import torch

# ═══════════════════════════════════════════════════════════════════════════════
# PROJECT PATHS
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "chexpert_small"
CHEXPERT_ROOT = DATA_ROOT / "CheXpert-v1.0-small"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# CheXpert Labels (14 pathological observations)
CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum", 
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion", 
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis", 
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices"
]

# Label handling for uncertain (-1) labels
UNCERTAINTY_POLICY = {
    "ignore": -1,      # Ignore uncertain labels during training
    "positive": 1,     # Treat uncertain as positive
    "negative": 0,     # Treat uncertain as negative
    "uignore": "U-Ignore",  # U-Ignore policy from CheXpert paper
    "uzeros": "U-Zeros"     # U-Zeros policy from CheXpert paper
}

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Image preprocessing
IMAGE_SIZE = 224
RESIZE_SIZE = 256
MEAN = [0.485, 0.456, 0.406]  # ImageNet means
STD = [0.229, 0.224, 0.225]   # ImageNet stds

# For grayscale (single channel)
GRAYSCALE_MEAN = [0.485]
GRAYSCALE_STD = [0.229]

# Model architectures to evaluate
MODEL_CONFIGS = {
    "densenet121": {
        "arch": "densenet121",
        "pretrained": True,
        "num_classes": len(CHEXPERT_LABELS),
        "input_channels": 1
    },
    "resnet50": {
        "arch": "resnet50", 
        "pretrained": True,
        "num_classes": len(CHEXPERT_LABELS),
        "input_channels": 1
    },
    "efficientnet_b0": {
        "arch": "efficientnet_b0",
        "pretrained": True, 
        "num_classes": len(CHEXPERT_LABELS),
        "input_channels": 1
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

TRAINING_CONFIG = {
    "batch_size": 32,
    "num_epochs": 50,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "optimizer": "adam",
    "scheduler": "cosine",
    "early_stopping_patience": 10,
    "validation_split": 0.2,
}

# ═══════════════════════════════════════════════════════════════════════════════
# DEVICE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

def get_device():
    """Get the best available device"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

DEVICE = get_device()

# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════════════

METRICS = [
    "auc_roc",
    "auc_pr", 
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "specificity",
    "sensitivity"
]

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

LOGGING_CONFIG = {
    "log_level": "INFO",
    "log_file": RESULTS_DIR / "training.log",
    "tensorboard_dir": RESULTS_DIR / "tensorboard",
    "checkpoint_dir": MODELS_DIR / "checkpoints",
    "save_frequency": 5  # Save checkpoint every N epochs
}
