"""Modelos: CNN custom desde cero y ResNet18 preentrenada (transfer learning)."""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from . import config


# ---------------------------------------------------------------------------
# CNN custom
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """[Conv3x3 -> BN -> ReLU] x 2 -> MaxPool 2x2 (-> Dropout2d opcional)."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class EuroSATCNN(nn.Module):
    """CNN tipo VGG compacta para imágenes de 64x64.

    Resolución espacial: 64 -> 32 -> 16 -> 8 -> 4, canales 32 -> 64 -> 128 -> 256.
    Luego Global Average Pooling (256-d), una capa densa de ``embed_dim``
    (el "espacio latente" que se analiza con t-SNE/UMAP) y la capa de
    clasificación final.

    Decisiones de diseño:
      * kernels 3x3 apilados: mismo campo receptivo que kernels grandes con
        menos parámetros y más no-linealidades;
      * BatchNorm: estabiliza la escala de las activaciones y del gradiente
        (ayuda directamente contra vanishing/exploding gradients);
      * Global Average Pooling en vez de flatten: reduce drásticamente los
        parámetros de la parte densa y aporta invarianza a la posición, útil
        porque en EuroSAT la clase depende de la textura global del recorte;
      * Dropout: regularización (dataset chico: ~27k imágenes).
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        embed_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        blocks, in_ch = [], 3
        for i, ch in enumerate(channels):
            blocks.append(ConvBlock(in_ch, ch, dropout=0.1 if i >= 2 else 0.0))
            in_ch = ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        # He/Kaiming init: pensada para ReLU, mantiene la varianza de las
        # activaciones a través de las capas.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward_features(self, x):
        """Embedding justo antes de la capa de clasificación final."""
        return self.embedding(self.pool(self.features(x)))

    def forward(self, x):
        return self.classifier(self.forward_features(x))


# ---------------------------------------------------------------------------
# Transfer learning: ResNet18
# ---------------------------------------------------------------------------
def build_resnet18(
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = True,
    trainable: str = "all",
) -> nn.Module:
    """ResNet18 de torchvision con la cabeza adaptada a EuroSAT.

    trainable:
      * ``"head"``   -> solo se entrena la nueva capa ``fc`` (feature extraction).
      * ``"layer4"`` -> se descongelan el último bloque residual + ``fc``.
      * ``"all"``    -> fine-tuning completo de toda la red.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if trainable != "all":
        unfrozen = {"head": ["fc"], "layer4": ["layer4", "fc"]}[trainable]
        for name, p in model.named_parameters():
            p.requires_grad = any(name.startswith(u) for u in unfrozen)
    return model


def resnet_param_groups(model: nn.Module, lr_backbone: float, lr_head: float):
    """Learning rates discriminativos: más chico para el backbone preentrenado
    (no "romper" los filtros aprendidos en ImageNet) y más grande para la
    cabeza nueva, que arranca desde cero."""
    head, backbone = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if name.startswith("fc") else backbone).append(p)
    groups = [{"params": head, "lr": lr_head}]
    if backbone:
        groups.append({"params": backbone, "lr": lr_backbone})
    return groups


def get_classifier_layer(model: nn.Module) -> nn.Linear:
    """Última capa lineal (la de clasificación) de cualquiera de los modelos."""
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        raise ValueError("El modelo no tiene capas lineales")
    return last
