"""In-memory evaluation for binary in-hospital mortality."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from torch import nn

from src.utils import move_to_device

Batch = Mapping[str, Any]
ForwardAdapter = Callable[[nn.Module, Batch], torch.Tensor]
MetricResult = dict[str, float | int | str | None]


def predict_logits(
    model: nn.Module,
    loader: Iterable[Batch],
    forward_fn: ForwardAdapter,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect CPU labels and finite logits without applying a sigmoid."""
    model.to(device)
    model.eval()
    resolved_device = torch.device(device)
    collected_labels: list[torch.Tensor] = []
    collected_logits: list[torch.Tensor] = []

    with torch.no_grad():
        for raw_batch in loader:
            batch = move_to_device(
                raw_batch,
                device,
                non_blocking=resolved_device.type == "cuda",
            )
            labels = batch["label"].to(dtype=torch.float32).reshape(-1)
            logits = forward_fn(model, batch)
            if logits.shape != labels.shape:
                raise ValueError(
                    "Forward adapter must return one logit per label; "
                    f"got logits {tuple(logits.shape)} and labels {tuple(labels.shape)}."
                )
            finite_logits = torch.isfinite(logits).all()
            finite_labels = torch.isfinite(labels).all()
            if logits.device.type == "cuda" and hasattr(torch, "_assert_async"):
                torch._assert_async(
                    finite_logits,
                    "Model produced non-finite evaluation logits.",
                )
                torch._assert_async(
                    finite_labels,
                    "Evaluation labels contain non-finite values.",
                )
            else:
                if not finite_logits:
                    raise FloatingPointError(
                        "Model produced non-finite evaluation logits."
                    )
                if not finite_labels:
                    raise ValueError("Evaluation labels contain non-finite values.")
            collected_labels.append(labels.detach())
            collected_logits.append(logits.detach())

    if not collected_labels:
        raise ValueError("Evaluation loader produced no samples.")
    return (
        torch.cat(collected_labels).cpu(),
        torch.cat(collected_logits).cpu(),
    )


def compute_mortality_metrics(
    labels: torch.Tensor,
    logits: torch.Tensor,
) -> MetricResult:
    """Compute AUROC, AUPRC, and Brier score from logits.

    Sigmoid is intentionally confined to evaluation. If only one label class
    is present, AUROC and AUPRC are returned as NaN with an explanatory note;
    Brier score remains defined.
    """
    if labels.shape != logits.shape or labels.ndim != 1:
        raise ValueError("labels and logits must be matching 1-D tensors.")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Cannot evaluate non-finite logits.")

    probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    raw_labels = labels.detach().cpu().numpy()
    if not np.all((raw_labels == 0) | (raw_labels == 1)):
        raise ValueError("Mortality labels must be binary 0/1 values.")
    label_array = raw_labels.astype(np.int64)
    unique_labels = np.unique(label_array)

    note: str | None = None
    if unique_labels.size < 2:
        auroc = float("nan")
        auprc = float("nan")
        note = (
            "AUROC and AUPRC are undefined because this evaluation split "
            "contains only one label class."
        )
    else:
        auroc = float(roc_auc_score(label_array, probabilities))
        auprc = float(average_precision_score(label_array, probabilities))

    return {
        "auroc": auroc,
        "auprc": auprc,
        "brier_score": float(brier_score_loss(label_array, probabilities)),
        "sample_count": int(label_array.size),
        "note": note,
    }


def evaluate(
    model: nn.Module,
    loader: Iterable[Batch],
    forward_fn: ForwardAdapter,
    device: str | torch.device,
) -> MetricResult:
    """Evaluate one split entirely in memory and return no accuracy metric."""
    labels, logits = predict_logits(model, loader, forward_fn, device)
    return compute_mortality_metrics(labels, logits)


def evaluate_split(
    model: nn.Module,
    loader: Iterable[Batch],
    forward_fn: ForwardAdapter,
    device: str | torch.device,
) -> MetricResult:
    """Backward-compatible name for ``evaluate``."""
    return evaluate(model, loader, forward_fn, device)
