"""Carga de EuroSAT RGB, splits, transforms (data augmentation) y DataLoaders.

Soporta dos layouts del dataset:

1. El de Kaggle (apollo2506/eurosat-dataset): carpeta ``EuroSAT/`` con una
   subcarpeta por clase y, opcionalmente, ``train.csv``, ``validation.csv`` y
   ``test.csv`` con la columna ``Filename`` (ruta relativa a la carpeta raíz).
   Si los CSVs están, se usa esa partición oficial.
2. Cualquier carpeta estilo ImageFolder (una subcarpeta por clase). En ese
   caso se hace un split estratificado train/val/test reproducible.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from . import config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Localización del dataset y armado de los splits
# ---------------------------------------------------------------------------
def find_dataset_root(base_dir: Path | str = config.DATA_DIR) -> Path:
    """Busca recursivamente la carpeta que contiene las 10 subcarpetas de clase.

    Ignora la versión multiespectral (``EuroSATallBands``) si está presente.
    """
    base_dir = Path(base_dir)
    candidates = [base_dir, *[p for p in base_dir.rglob("*") if p.is_dir()]]
    for cand in candidates:
        if "allbands" in str(cand).lower():
            continue
        subdirs = {p.name for p in cand.iterdir() if p.is_dir()}
        if set(config.CLASSES).issubset(subdirs):
            return cand
    raise FileNotFoundError(
        f"No encontré las carpetas de clases de EuroSAT dentro de {base_dir}. "
        "Descargá el dataset de Kaggle y apuntá EUROSAT_DIR a la carpeta 'EuroSAT'."
    )


def _scan_folder(root: Path) -> pd.DataFrame:
    rows = []
    for label, cls in enumerate(config.CLASSES):
        for p in sorted((root / cls).iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                rows.append({"path": str(p), "label": label, "class_name": cls})
    return pd.DataFrame(rows)


def _read_split_csv(root: Path, csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    # La clase se deriva de la carpeta del archivo (más robusto que confiar en
    # el label numérico del CSV, cuyo orden podría diferir de config.CLASSES).
    paths = df["Filename"].astype(str)
    out = pd.DataFrame({"path": [str(root / f) for f in paths]})
    out["class_name"] = [Path(f).parent.name for f in paths]
    out["label"] = out["class_name"].map({c: i for i, c in enumerate(config.CLASSES)})
    return out[["path", "label", "class_name"]]


def load_splits(
    base_dir: Path | str = config.DATA_DIR,
    val_size: float = config.VAL_SIZE,
    test_size: float = config.TEST_SIZE,
    seed: int = config.RANDOM_STATE,
    max_per_class: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Devuelve {'train','val','test'} -> DataFrame(path, label, class_name)."""
    root = find_dataset_root(base_dir)
    csvs = {"train": root / "train.csv", "val": root / "validation.csv", "test": root / "test.csv"}

    if all(p.exists() for p in csvs.values()):
        splits = {k: _read_split_csv(root, p) for k, p in csvs.items()}
        source = "CSVs oficiales del dataset de Kaggle"
    else:
        df = _scan_folder(root)
        train_val, test = train_test_split(
            df, test_size=test_size, stratify=df["label"], random_state=seed
        )
        train, val = train_test_split(
            train_val,
            test_size=val_size / (1 - test_size),
            stratify=train_val["label"],
            random_state=seed,
        )
        splits = {"train": train, "val": val, "test": test}
        source = "split estratificado propio"

    if max_per_class is not None:  # útil para iterar rápido / smoke tests
        splits = {
            k: pd.concat(
                [g.sample(min(len(g), max_per_class), random_state=seed)
                 for _, g in v.groupby("label")]
            )
            for k, v in splits.items()
        }

    splits = {k: v.reset_index(drop=True) for k, v in splits.items()}
    print(f"Dataset en {root} ({source})")
    for k, v in splits.items():
        print(f"  {k:<5}: {len(v):>6} imágenes")
    return splits


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class EuroSATDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------
class RandomRot90:
    """Rotación aleatoria en múltiplos de 90° (0, 90, 180, 270).

    Las imágenes satelitales son tomadas en vista cenital (nadir): no existe un
    "arriba" canónico, por lo que una rotación de 90° genera una imagen
    perfectamente plausible y sin bordes vacíos (a diferencia de rotaciones
    arbitrarias, que requieren rellenar esquinas).
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        k = random.randint(0, 3)
        return img if k == 0 else img.rotate(90 * k)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def compute_mean_std(df: pd.DataFrame, n_samples: int = 2000, seed: int = 42):
    """Media y desvío por canal (RGB) estimados sobre una muestra de train."""
    sample = df.sample(min(n_samples, len(df)), random_state=seed)
    to_tensor = T.ToTensor()
    s, s2, n = torch.zeros(3), torch.zeros(3), 0
    for p in sample["path"]:
        x = to_tensor(Image.open(p).convert("RGB"))
        s += x.sum(dim=(1, 2))
        s2 += (x**2).sum(dim=(1, 2))
        n += x.shape[1] * x.shape[2]
    mean = s / n
    std = (s2 / n - mean**2).sqrt()
    return tuple(mean.tolist()), tuple(std.tolist())


def build_transforms(
    mean,
    std,
    img_size: int = config.IMG_SIZE,
    augment: bool = True,
    color_jitter: float = 0.1,
):
    """Transforms de train (con augmentation) y de evaluación (sin augmentation).

    Augmentations elegidas (invarianzas geométricas de imágenes cenitales):
      * flips horizontales y verticales,
      * rotaciones de 90°,
      * un leve RandomResizedCrop (escala 0.8-1) para robustez a pequeños
        desplazamientos/cambios de escala del recorte,
      * un ColorJitter suave: simula variaciones de iluminación, estación y
        atmósfera entre adquisiciones. Se mantiene chico a propósito porque el
        color es una señal muy informativa (agua, vegetación, cultivos).
    """
    resize = [T.Resize(img_size, antialias=True)] if img_size != config.IMG_SIZE else []
    normalize = [T.ToTensor(), T.Normalize(mean, std)]

    eval_tf = T.Compose(resize + normalize)
    if not augment:
        return eval_tf, eval_tf

    train_tf = T.Compose(
        [
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            RandomRot90(),
            T.ColorJitter(
                brightness=color_jitter, contrast=color_jitter, saturation=color_jitter
            ),
            *normalize,
        ]
    )
    return train_tf, eval_tf


def make_dataloaders(
    splits: dict[str, pd.DataFrame],
    train_tf,
    eval_tf,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    seed: int = config.RANDOM_STATE,
) -> dict[str, DataLoader]:
    g = torch.Generator().manual_seed(seed)
    pin = torch.cuda.is_available()
    loaders = {
        "train": DataLoader(
            EuroSATDataset(splits["train"], train_tf),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin,
            generator=g,
            persistent_workers=num_workers > 0,
        )
    }
    for k in ("val", "test"):
        loaders[k] = DataLoader(
            EuroSATDataset(splits[k], eval_tf),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=num_workers > 0,
        )
    return loaders


def denormalize(x: torch.Tensor, mean, std) -> np.ndarray:
    """Tensor normalizado (C,H,W) -> imagen HWC en [0,1] para graficar."""
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (x.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()
