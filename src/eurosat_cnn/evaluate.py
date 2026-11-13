"""Evaluación en test: métricas, matriz de confusión y análisis de errores."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from . import config
from .data import denormalize


@torch.no_grad()
def predict(model, loader, device):
    """Devuelve (y_true, y_pred, probs) sobre todo el loader."""
    model.eval()
    ys, probs = [], []
    for x, y in loader:
        logits = model(x.to(device))
        probs.append(torch.softmax(logits, 1).cpu())
        ys.append(y)
    probs = torch.cat(probs).numpy()
    y_true = torch.cat(ys).numpy()
    return y_true, probs.argmax(1), probs


def metrics_summary(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }


def report_df(y_true, y_pred, classes=config.CLASSES) -> pd.DataFrame:
    rep = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True, digits=4, zero_division=0
    )
    return pd.DataFrame(rep).T


def plot_confusion_matrix(y_true, y_pred, classes=config.CLASSES, normalize=True, title=None):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes,
                yticklabels=classes, ax=axes[0], cbar=False)
    axes[0].set_title("Matriz de confusión (conteos)")
    cm_norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes,
                yticklabels=classes, ax=axes[1], cbar=False, vmin=0, vmax=1)
    axes[1].set_title("Matriz de confusión normalizada por fila (recall)")
    for ax in axes:
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.tick_params(axis="x", rotation=45)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    return fig, cm


def top_confusions(cm: np.ndarray, classes=config.CLASSES, k: int = 8) -> pd.DataFrame:
    """Pares (real -> predicho) con más errores, fuera de la diagonal."""
    rows = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                rows.append(
                    {
                        "real": classes[i],
                        "predicho": classes[j],
                        "n_errores": int(cm[i, j]),
                        "pct_de_la_clase_real": cm[i, j] / cm[i].sum(),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["real", "predicho", "n_errores", "pct_de_la_clase_real"])
    return pd.DataFrame(rows).sort_values("n_errores", ascending=False).head(k).reset_index(drop=True)


def plot_misclassified(dataset, y_true, y_pred, probs, mean, std, pairs=None,
                       n_per_pair: int = 6, classes=config.CLASSES):
    """Grilla de ejemplos mal clasificados para los pares de confusión dados."""
    if pairs is None:
        pairs = []
    pairs = list(pairs)
    if not pairs:
        return None
    fig, axes = plt.subplots(len(pairs), n_per_pair, figsize=(2.2 * n_per_pair, 2.5 * len(pairs)),
                             squeeze=False)
    for r, (real, pred) in enumerate(pairs):
        ri, pi = classes.index(real), classes.index(pred)
        idx = np.where((y_true == ri) & (y_pred == pi))[0]
        # los errores "más seguros" primero (más interesantes)
        idx = idx[np.argsort(-probs[idx, pi])][:n_per_pair]
        for c in range(n_per_pair):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(idx):
                img, _ = dataset[idx[c]]
                ax.imshow(denormalize(img, mean, std))
                ax.set_title(f"p={probs[idx[c], pi]:.2f}", fontsize=8)
        axes[r, 0].text(-0.15, 0.5, f"{real}\n→ {pred}", transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=9)
    fig.suptitle("Errores más confiados por par (real → predicho)")
    plt.tight_layout()
    return fig
