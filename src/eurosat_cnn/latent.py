"""Extracción del espacio latente (features previas a la capa de clasificación)
y proyección con PCA / t-SNE / UMAP."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config
from .models import get_classifier_layer


@torch.no_grad()
def extract_features(model, loader, device):
    """Captura, con un forward hook, el INPUT de la última capa lineal, es
    decir, la representación justo antes de la decisión final. Funciona igual
    para la CNN custom (embedding de 128-d) y para ResNet18 (512-d tras el
    average pooling)."""
    feats, labels = [], []
    layer = get_classifier_layer(model)
    handle = layer.register_forward_hook(lambda m, inp, out: feats.append(inp[0].detach().cpu()))
    model.eval()
    try:
        for x, y in loader:
            model(x.to(device))
            labels.append(y)
    finally:
        handle.remove()
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def raw_pixels(loader, max_items: int | None = None):
    """Píxeles crudos aplanados (baseline: ¿qué tan separables son las clases
    SIN la red?)."""
    xs, ys, n = [], [], 0
    for x, y in loader:
        xs.append(x.flatten(1))
        ys.append(y)
        n += len(y)
        if max_items and n >= max_items:
            break
    X, Y = torch.cat(xs).numpy(), torch.cat(ys).numpy()
    return (X[:max_items], Y[:max_items]) if max_items else (X, Y)


def project(features: np.ndarray, method: str = "tsne", n_components: int = 2,
            seed: int = config.RANDOM_STATE) -> np.ndarray:
    X = StandardScaler().fit_transform(features)
    if method == "pca":
        return PCA(n_components=n_components, random_state=seed).fit_transform(X)
    if method == "tsne":
        # PCA a 50-d antes de t-SNE: práctica estándar para reducir ruido y costo
        if X.shape[1] > 50:
            X = PCA(n_components=50, random_state=seed).fit_transform(X)
        perplexity = min(30, max(5, (len(X) - 1) // 3))
        return TSNE(n_components=n_components, perplexity=perplexity, init="pca",
                    random_state=seed).fit_transform(X)
    if method == "umap":
        import umap  # umap-learn

        return umap.UMAP(n_components=n_components, n_neighbors=15, min_dist=0.1,
                         random_state=seed).fit_transform(X)
    raise ValueError(method)


def separability_scores(features: np.ndarray, labels: np.ndarray, emb2d: dict | None = None,
                        sample: int = 5000, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Silhouette score (con las etiquetas reales como clusters): ~1 clusters
    compactos y separados, ~0 solapados, <0 mezclados."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(labels), min(sample, len(labels)), replace=False)
    rows = [{"espacio": f"latente ({features.shape[1]}-d)",
             "silhouette": silhouette_score(StandardScaler().fit_transform(features)[idx], labels[idx])}]
    for name, e in (emb2d or {}).items():
        rows.append({"espacio": name, "silhouette": silhouette_score(e[idx], labels[idx])})
    return pd.DataFrame(rows)


def plot_embeddings(embs: dict, labels: np.ndarray, classes=config.CLASSES, title=None):
    n = len(embs)
    fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 6.5), squeeze=False)
    palette = sns.color_palette("tab10", len(classes))
    for ax, (name, e) in zip(axes[0], embs.items()):
        for i, c in enumerate(classes):
            m = labels == i
            ax.scatter(e[m, 0], e[m, 1], s=4, alpha=0.6, color=palette[i], label=c)
            if m.any():  # etiqueta en el centroide de cada clase
                cx, cy = np.median(e[m, 0]), np.median(e[m, 1])
                ax.text(cx, cy, c, fontsize=8, weight="bold", ha="center",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0))
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, -1].legend(markerscale=4, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_embedding_3d(emb: np.ndarray, labels: np.ndarray, classes=config.CLASSES, title=None):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    palette = sns.color_palette("tab10", len(classes))
    for i, c in enumerate(classes):
        m = labels == i
        ax.scatter(emb[m, 0], emb[m, 1], emb[m, 2], s=3, alpha=0.6, color=palette[i], label=c)
    ax.legend(markerscale=4, fontsize=7, loc="upper left")
    ax.set_title(title or "Proyección 3D")
    plt.tight_layout()
    return fig
