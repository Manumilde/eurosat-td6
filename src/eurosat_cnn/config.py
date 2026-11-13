"""Configuración global del TP3 (EuroSAT RGB)."""
from pathlib import Path
import os

# --- Rutas -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Se puede sobreescribir con la variable de entorno EUROSAT_DIR
# (ej. en Kaggle: /kaggle/input/eurosat-dataset/EuroSAT).
DATA_DIR = Path(os.environ.get("EUROSAT_DIR", PROJECT_ROOT / "data" / "raw"))
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# --- Dataset ---------------------------------------------------------------
CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 64  # resolución nativa de EuroSAT (64x64 px, 10 m/px)

# Normalización de ImageNet (obligatoria para usar pesos preentrenados)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Split si el dataset no trae CSVs de partición (train/val/test)
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# --- Entrenamiento ---------------------------------------------------------
RANDOM_STATE = 42
BATCH_SIZE = 128
NUM_WORKERS = 2
