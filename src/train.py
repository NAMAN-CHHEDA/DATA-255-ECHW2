"""Shared forward adapters and one-epoch training step for all models."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import nn

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from src.config import GRAD_CLIP_NORM  # noqa: E402
from src.utils import ElapsedTimer, move_to_device  # noqa: E402

Batch = Mapping[str, Any]
ForwardAdapter = Callable[[nn.Module, Batch], torch.Tensor]


def forward_gru(model: nn.Module, batch: Batch) -> torch.Tensor:
    """Adapt a padded irregular batch to the hourly GRU final logit."""
    if "hourly_values" in batch and "hourly_mask" in batch:
        return model(batch["hourly_values"], batch["hourly_mask"])

    from src.models.gru_baseline import prepare_hourly_gru_inputs

    hourly_values, hourly_mask = prepare_hourly_gru_inputs(
        times=batch["times"],
        values=batch["values"],
        mask=batch["mask"],
        valid_time_mask=batch["valid_time_mask"],
    )
    return model(hourly_values, hourly_mask)


def forward_neural_ode(model: nn.Module, batch: Batch) -> torch.Tensor:
    """Use the shared MLP/KAN Neural ODE final risk at 48 hours."""
    return model.final_risk_logits(
        values=batch["values"],
        mask=batch["mask"],
        valid_time_mask=batch["valid_time_mask"],
        times=batch["times"],
        delta_t=batch["delta_t"],
        final_time=48.0,
    )


def forward_liquid(model: nn.Module, batch: Batch) -> torch.Tensor:
    """Use the LTC/CfC logit at each patient's final valid timestep."""
    return model.forward_final(
        values=batch["values"],
        mask=batch["mask"],
        delta_t=batch["delta_t"],
        valid_time_mask=batch["valid_time_mask"],
        times=batch["times"],
    )


FORWARD_ADAPTERS: dict[str, ForwardAdapter] = {
    "gru_baseline": forward_gru,
    "mlp_neural_ode": forward_neural_ode,
    "kan_neural_ode": forward_neural_ode,
    "ltc": forward_liquid,
    "cfc": forward_liquid,
}


def get_forward_adapter(model_key: str) -> ForwardAdapter:
    """Return the registered model-specific final-logit adapter."""
    try:
        return FORWARD_ADAPTERS[model_key]
    except KeyError as exc:
        known = ", ".join(sorted(FORWARD_ADAPTERS))
        raise ValueError(
            f"No forward adapter registered for {model_key!r}; expected one of {known}."
        ) from exc


def _validate_logits(logits: torch.Tensor, batch_size: int) -> None:
    """Require one finite, unsquashed logit per patient."""
    if logits.shape != (batch_size,):
        raise ValueError(
            f"Forward adapter must return logits shaped ({batch_size},), "
            f"got {tuple(logits.shape)}."
        )
    finite = torch.isfinite(logits).all()
    if logits.device.type == "cuda" and hasattr(torch, "_assert_async"):
        torch._assert_async(finite, "Model produced non-finite logits.")
    elif not finite:
        raise FloatingPointError("Model produced non-finite logits.")


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    forward_fn: ForwardAdapter,
    device: str | torch.device,
    progress_every: int = 0,
    progress_label: str = "",
    time_limit_seconds: float | None = None,
) -> tuple[float, float]:
    """Train for one epoch and return sample-mean BCE loss and wall seconds.

    With ``progress_every > 0`` a line with the batch index and elapsed
    seconds is printed every that many batches (no CUDA synchronization).
    ``time_limit_seconds`` raises ``TimeoutError("FIRST_EPOCH_TIMEOUT ...")``
    if the epoch runs longer (used by the unattended driver's KAN guard).
    """
    model.to(device)
    model.train()
    resolved_device = torch.device(device)
    total_loss = torch.zeros((), dtype=torch.float64, device=resolved_device)
    sample_count = 0

    try:
        total_batches: int | str = len(loader)  # type: ignore[arg-type]
    except TypeError:
        total_batches = "?"

    epoch_start = time.perf_counter()

    def timer_elapsed() -> float:
        return time.perf_counter() - epoch_start

    with ElapsedTimer() as timer:
        for batch_index, raw_batch in enumerate(loader, start=1):
            batch = move_to_device(
                raw_batch,
                device,
                non_blocking=resolved_device.type == "cuda",
            )
            labels = batch["label"].to(dtype=torch.float32)
            if labels.ndim != 1:
                labels = labels.reshape(-1)

            optimizer.zero_grad(set_to_none=True)
            ode_func = getattr(model, "ode_func", None)
            if ode_func is not None:
                ode_func.nfe = 0  # per-batch NFE counter (diagnostic only)
            logits = forward_fn(model, batch)
            _validate_logits(logits, labels.numel())
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            finite_loss = torch.isfinite(loss)
            if loss.device.type == "cuda" and hasattr(torch, "_assert_async"):
                torch._assert_async(
                    finite_loss,
                    "Training loss became non-finite.",
                )
            elif not finite_loss:
                raise FloatingPointError("Training loss became non-finite.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            if progress_every > 0 and batch_index % progress_every == 0:
                line = (
                    f"{progress_label} batch {batch_index}/{total_batches} | "
                    f"elapsed={timer_elapsed():.1f}s"
                )
                if ode_func is not None:
                    forward_nfe = int(getattr(model, "last_forward_nfe", 0))
                    max_state = getattr(model, "last_max_abs_state", None)
                    # The only extra sync: reading max|h| on progress batches.
                    max_text = (
                        "n/a" if max_state is None else f"{float(max_state):.3g}"
                    )
                    line += (
                        f" | fwd_nfe={forward_nfe}"
                        f" | bwd_nfe={ode_func.nfe - forward_nfe}"
                        f" | max|h|={max_text}"
                    )
                print(line, flush=True)

            if (
                time_limit_seconds is not None
                and timer_elapsed() > time_limit_seconds
            ):
                raise TimeoutError(
                    f"FIRST_EPOCH_TIMEOUT: epoch exceeded {time_limit_seconds:.0f}s "
                    f"after {batch_index} batches."
                )

            batch_size = labels.numel()
            total_loss += loss.detach().to(dtype=torch.float64) * batch_size
            sample_count += batch_size

        if sample_count == 0:
            raise ValueError("Training loader produced no samples.")
        # One synchronization per epoch supplies the logged mean and makes the
        # reported training-phase wall time include queued CUDA work.
        mean_loss = float((total_loss / sample_count).item())
    return mean_loss, timer.seconds
