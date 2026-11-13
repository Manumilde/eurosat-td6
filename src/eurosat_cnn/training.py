"""Loop de entrenamiento / validación con registro de métricas y tiempos."""
from __future__ import annotations

import copy
import time

import pandas as pd
import torch
from torch import nn

from .gradients import GradientMonitor


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch,
                    scheduler=None, grad_monitor: GradientMonitor | None = None,
                    max_grad_norm: float | None = None):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if grad_monitor is not None:
            grad_monitor.log(epoch)
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None:  # schedulers por batch (OneCycleLR)
            scheduler.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == y).sum().item()
        n += bs
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n


def fit(
    model: nn.Module,
    loaders: dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    scheduler=None,
    criterion: nn.Module | None = None,
    grad_monitor: GradientMonitor | None = None,
    patience: int | None = None,
    max_grad_norm: float | None = None,
    name: str = "model",
):
    """Entrena ``epochs`` épocas guardando el mejor estado según val accuracy.

    Devuelve (history DataFrame, best_state_dict). Se reporta el tiempo de
    cada época separando entrenamiento y validación (comparación de cómputo).
    """
    criterion = criterion or nn.CrossEntropyLoss()
    model.to(device)
    history, best_acc, best_state, bad_epochs = [], -1.0, None, 0
    t_start = time.perf_counter()

    for epoch in range(epochs):
        _sync(device)
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, epoch,
            scheduler=scheduler, grad_monitor=grad_monitor, max_grad_norm=max_grad_norm,
        )
        _sync(device)
        t_train = time.perf_counter() - t0
        va_loss, va_acc = evaluate(model, loaders["val"], criterion, device)
        _sync(device)
        t_epoch = time.perf_counter() - t0

        lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": va_acc,
                "lr": lr,
                "train_time_s": t_train,
                "epoch_time_s": t_epoch,
                "elapsed_s": time.perf_counter() - t_start,
            }
        )
        print(
            f"[{name}] epoch {epoch + 1:>2}/{epochs} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
            f"lr {lr:.2e} | {t_epoch:.1f}s"
        )

        if va_acc > best_acc:
            best_acc, bad_epochs = va_acc, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if patience is not None and bad_epochs >= patience:
                print(f"[{name}] Early stopping en la época {epoch + 1}")
                break

    model.load_state_dict(best_state)
    print(f"[{name}] Mejor val acc: {best_acc:.4f}")
    return pd.DataFrame(history), best_state
