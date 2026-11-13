"""Monitoreo del flujo de gradientes (vanishing / exploding gradients).

Después de cada ``loss.backward()`` (y antes de ``optimizer.step()``) se
registra la norma L2 del gradiente de los pesos de cada capa. Comparar esas
normas entre capas tempranas y tardías permite detectar:

  * Vanishing gradients: las capas cercanas a la entrada reciben gradientes
    órdenes de magnitud más chicos que las cercanas a la salida (o ~0), por lo
    que prácticamente no aprenden.
  * Exploding gradients: normas que crecen sin control (o NaN/Inf), lo que
    desestabiliza el entrenamiento (la loss "salta" o diverge).
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn


class GradientMonitor:
    def __init__(
        self,
        model: nn.Module,
        max_epochs: int = 3,
        log_every: int = 10,
        include_bn: bool = False,
    ):
        self.max_epochs = max_epochs
        self.log_every = log_every
        # Solo pesos de conv/linear (los de BN tienen otra escala y ensucian la
        # comparación entre capas). El orden de named_parameters es de
        # entrada -> salida, que es el orden que queremos graficar.
        self.params = [
            (name, p)
            for name, p in model.named_parameters()
            if p.requires_grad
            and name.endswith("weight")
            and (include_bn or p.dim() > 1)
        ]
        self.layer_order = [n for n, _ in self.params]
        self.records: list[dict] = []
        self._step = 0

    def active(self, epoch: int) -> bool:
        return epoch < self.max_epochs

    @torch.no_grad()
    def log(self, epoch: int) -> None:
        """Llamar después de backward(). ``epoch`` es 0-indexed."""
        if not self.active(epoch):
            return
        if self._step % self.log_every == 0:
            total_sq = 0.0
            for name, p in self.params:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                norm = g.norm(2).item()
                total_sq += norm**2
                self.records.append(
                    {
                        "step": self._step,
                        "epoch": epoch,
                        "layer": name,
                        "grad_norm": norm,
                        # norma relativa a la de los pesos: proxy del tamaño
                        # del update que recibe la capa
                        "grad_to_weight": norm / (p.detach().norm(2).item() + 1e-12),
                    }
                )
            self.records.append(
                {
                    "step": self._step,
                    "epoch": epoch,
                    "layer": "__total__",
                    "grad_norm": math.sqrt(total_sq),
                    "grad_to_weight": np.nan,
                }
            )
        self._step += 1

    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def summary(self) -> pd.DataFrame:
        """Estadísticas por capa (ordenadas de entrada a salida)."""
        df = self.to_dataframe()
        df = df[df["layer"] != "__total__"]
        s = (
            df.groupby("layer")
            .agg(
                grad_norm_mean=("grad_norm", "mean"),
                grad_norm_min=("grad_norm", "min"),
                grad_norm_max=("grad_norm", "max"),
                grad_to_weight_mean=("grad_to_weight", "mean"),
            )
            .reindex(self.layer_order)
            .dropna(how="all")
        )
        return s

    def diagnose(
        self, vanish_thr: float = 1e-6, explode_thr: float = 1e2, ratio_thr: float = 1e3
    ) -> dict:
        """Chequeos automáticos simples sobre las normas registradas."""
        df = self.to_dataframe()
        layers = df[df["layer"] != "__total__"]
        s = self.summary()
        ratio = s["grad_norm_mean"].max() / max(s["grad_norm_mean"].min(), 1e-30)
        first, last = s.index[0], s.index[-1]
        out = {
            "hay_nan_o_inf": bool(~np.isfinite(layers["grad_norm"]).all()),
            "capas_con_grad_casi_nulo (<%.0e)" % vanish_thr: s.index[
                s["grad_norm_mean"] < vanish_thr
            ].tolist(),
            "max_norma_registrada": float(layers["grad_norm"].max()),
            "explosion (>%g)" % explode_thr: bool(layers["grad_norm"].max() > explode_thr),
            "ratio_max/min_entre_capas": float(ratio),
            "desbalance_fuerte (>%g)" % ratio_thr: bool(ratio > ratio_thr),
            f"norma_media_primera_capa ({first})": float(s.loc[first, "grad_norm_mean"]),
            f"norma_media_ultima_capa ({last})": float(s.loc[last, "grad_norm_mean"]),
        }
        return out

    # ------------------------------------------------------------------
    def plot(self, title: str = "Flujo de gradientes", max_lines: int = 12):
        df = self.to_dataframe()
        layers_df = df[df["layer"] != "__total__"]
        total = df[df["layer"] == "__total__"]
        s = self.summary()

        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        # (a) evolución de la norma por capa a lo largo de los steps
        order = list(s.index)
        if len(order) > max_lines:  # submuestreo uniforme de capas para legibilidad
            idx = np.linspace(0, len(order) - 1, max_lines).round().astype(int)
            order = [order[i] for i in idx]
        cmap = plt.get_cmap("viridis")
        for i, layer in enumerate(order):
            d = layers_df[layers_df["layer"] == layer]
            axes[0].plot(d["step"], d["grad_norm"], color=cmap(i / max(len(order) - 1, 1)),
                         label=layer, lw=1.2)
        axes[0].set_yscale("log")
        axes[0].set_xlabel("step (batch)")
        axes[0].set_ylabel("‖∇W‖₂ (log)")
        axes[0].set_title("Norma del gradiente por capa (oscuro = entrada)")
        axes[0].legend(fontsize=7, ncol=2)

        # (b) norma media por capa, ordenada por profundidad
        x = np.arange(len(s))
        axes[1].bar(x, s["grad_norm_mean"], color="#4C72B0")
        axes[1].errorbar(
            x,
            s["grad_norm_mean"],
            yerr=[s["grad_norm_mean"] - s["grad_norm_min"], s["grad_norm_max"] - s["grad_norm_mean"]],
            fmt="none",
            ecolor="gray",
            lw=0.8,
        )
        axes[1].set_yscale("log")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([l.replace(".weight", "") for l in s.index], rotation=90, fontsize=7)
        axes[1].set_title("Norma media (min-max) por capa: entrada → salida")

        # (c) norma total
        axes[2].plot(total["step"], total["grad_norm"], color="#C44E52")
        axes[2].set_yscale("log")
        axes[2].set_xlabel("step (batch)")
        axes[2].set_title("Norma global del gradiente")

        for ax in axes[[0, 2]]:
            for e in sorted(layers_df["epoch"].unique())[1:]:
                st = layers_df[layers_df["epoch"] == e]["step"].min()
                ax.axvline(st, color="k", ls=":", lw=0.8)

        fig.suptitle(title)
        plt.tight_layout()
        return fig
