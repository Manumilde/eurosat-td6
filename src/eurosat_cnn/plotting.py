"""Gráficos de curvas de entrenamiento y comparación entre modelos."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .data import denormalize


def plot_history(history: pd.DataFrame, title: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].plot(history["epoch"], history["train_loss"], "o-", label="train")
    axes[0].plot(history["epoch"], history["val_loss"], "o-", label="validación")
    axes[0].set_title("Loss (Cross-Entropy)")
    axes[1].plot(history["epoch"], history["train_acc"], "o-", label="train")
    axes[1].plot(history["epoch"], history["val_acc"], "o-", label="validación")
    best = history["val_acc"].idxmax()
    axes[1].scatter(history.loc[best, "epoch"], history.loc[best, "val_acc"], s=150,
                    facecolors="none", edgecolors="red", lw=2, label="mejor val", zorder=5)
    axes[1].set_title("Accuracy")
    for ax in axes:
        ax.set_xlabel("época")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_comparison(histories: dict[str, pd.DataFrame]):
    """Val accuracy / val loss vs época y vs tiempo de cómputo acumulado."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for name, h in histories.items():
        axes[0].plot(h["epoch"], h["val_acc"], "o-", label=name, ms=4)
        axes[1].plot(h["epoch"], h["val_loss"], "o-", label=name, ms=4)
        axes[2].plot(h["elapsed_s"] / 60, h["val_acc"], "o-", label=name, ms=4)
    axes[0].set(title="Accuracy de validación por época", xlabel="época", ylabel="val acc")
    axes[1].set(title="Loss de validación por época", xlabel="época", ylabel="val loss")
    axes[2].set(title="Accuracy de validación vs tiempo de cómputo",
                xlabel="minutos acumulados", ylabel="val acc")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    return fig


def epochs_to_reach(history: pd.DataFrame, threshold: float):
    hit = history[history["val_acc"] >= threshold]
    return int(hit["epoch"].iloc[0]) if len(hit) else np.nan


def show_batch_by_class(dataset, labels, mean, std, n_per_class: int = 6,
                        classes=config.CLASSES, seed: int = 0, title: str | None = None):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    fig, axes = plt.subplots(len(classes), n_per_class,
                             figsize=(1.6 * n_per_class, 1.7 * len(classes)))
    for r, c in enumerate(classes):
        idx = np.where(labels == r)[0]
        pick = rng.choice(idx, min(n_per_class, len(idx)), replace=False)
        for k in range(n_per_class):
            ax = axes[r, k]
            ax.axis("off")
            if k < len(pick):
                img, _ = dataset[pick[k]]
                ax.imshow(denormalize(img, mean, std))
        axes[r, 0].text(-0.1, 0.5, c, transform=axes[r, 0].transAxes, ha="right",
                        va="center", fontsize=9)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    return fig


def show_augmentations(dataset_aug, idx: int, mean, std, n: int = 8, title: str = ""):
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.3))
    for k in range(n):
        img, _ = dataset_aug[idx]
        axes[k].imshow(denormalize(img, mean, std))
        axes[k].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    return fig
