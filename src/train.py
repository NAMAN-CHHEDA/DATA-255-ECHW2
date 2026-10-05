"""Training entry point.

The public command is ``run_experiment``. It checks configuration and stops
before optimization. ``fit`` is the function to implement later.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import nn

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from src.config import (  # noqa: E402
    BATCH_SIZE,
    EARLY_STOPPING_METRIC,
    EARLY_STOPPING_MODE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    RAW_DATA_DIR,
    SEED,
    WEIGHT_DECAY,
    checkpoint_path,
    format_config,
    processed_path,
    resolve_device,
    validate_config,
)
from src.utils import (  # noqa: E402
    ElapsedTimer,
    directory_has_data,
    move_to_device,
    set_seed,
)

Batch = Mapping[str, Any]
ForwardAdapter = Callable[[nn.Module, Batch], torch.Tensor]


def forward_gru(model: nn.Module, batch: Batch) -> torch.Tensor:
    """Adapt a padded irregular batch to the hourly GRU final logit."""
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
        final_time=48.0,
    )


def forward_liquid(model: nn.Module, batch: Batch) -> torch.Tensor:
    """Use the LTC/CfC logit at each patient's final valid timestep."""
    return model.forward_final(
        values=batch["values"],
        mask=batch["mask"],
        delta_t=batch["delta_t"],
        valid_time_mask=batch["valid_time_mask"],
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
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Model produced non-finite logits.")


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    forward_fn: ForwardAdapter,
    device: str | torch.device,
) -> tuple[float, float]:
    """Train for one epoch and return sample-mean BCE loss and wall seconds."""
    model.to(device)
    model.train()
    total_loss = 0.0
    sample_count = 0

    with ElapsedTimer() as timer:
        for raw_batch in loader:
            batch = move_to_device(raw_batch, device)
            labels = batch["label"].to(dtype=torch.float32)
            if labels.ndim != 1:
                labels = labels.reshape(-1)

            optimizer.zero_grad(set_to_none=True)
            logits = forward_fn(model, batch)
            _validate_logits(logits, labels.numel())
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss became non-finite.")
            loss.backward()
            optimizer.step()

            batch_size = labels.numel()
            total_loss += float(loss.detach().item()) * batch_size
            sample_count += batch_size

    if sample_count == 0:
        raise ValueError("Training loader produced no samples.")
    return total_loss / sample_count, timer.seconds

# String targets only. Importing a class also imports torch, so that happens
# inside ``load_model_class`` when training is implemented.
EXPERIMENTS: dict[str, dict[str, str | int]] = {
    "gru_baseline": {
        "number": 1,
        "title": "Discrete-time GRU baseline",
        "qualname": "src.models.gru_baseline:GRUBaseline",
    },
    "mlp_neural_ode": {
        "number": 2,
        "title": "MLP Neural ODE",
        "qualname": "src.models.mlp_neural_ode:MLPNeuralODE",
    },
    "kan_neural_ode": {
        "number": 3,
        "title": "KAN Neural ODE",
        "qualname": "src.models.kan_neural_ode:KANNeuralODE",
    },
    "liquid_model": {
        "number": 4,
        "title": "Liquid time-constant model",
        "qualname": "src.models.liquid_model:LiquidModel",
    },
}


def load_model_class(model_key: str) -> type:
    """Import the ``nn.Module`` subclass registered for ``model_key``.

    TODO: call this from ``fit`` after the processed cache supplies
    ``input_dim`` and ``static_dim``. Construct the module with values from
    ``src.config``, then move it to the resolved device.
    """
    spec = EXPERIMENTS[model_key]
    qualname = str(spec["qualname"])
    module_name, class_name = qualname.split(":")
    module = importlib.import_module(module_name)
    model_cls = getattr(module, class_name)
    return model_cls


def fit(model_key: str, device: str) -> None:
    """Train one model with early stopping.

    TODO:
    - load train and validation stays from the processed cache
    - build loaders with ``BATCH_SIZE`` and the collate function
    - optimize with Adam (``LEARNING_RATE``, ``WEIGHT_DECAY``)
    - use binary cross-entropy with logits
    - if a positive-class weight is used, estimate it on the training split only
    - score the validation split with ``EARLY_STOPPING_METRIC``
    - stop after ``EARLY_STOPPING_PATIENCE`` epochs without improvement
      (``EARLY_STOPPING_MODE``)
    - keep at most ``EPOCHS`` epochs
    - clip gradients with ``GRAD_CLIP_NORM``
    - write a checkpoint to ``checkpoint_path(model_key)`` only after real updates
    - append ``RUN_LOG.txt`` with values this process printed

    The test split is not used for fitting or early stopping.
    """
    raise NotImplementedError(
        "TODO: training loop is not implemented. "
        f"model={model_key}, device={device}, epochs={EPOCHS}, "
        f"lr={LEARNING_RATE}, batch_size={BATCH_SIZE}, "
        f"weight_decay={WEIGHT_DECAY}, "
        f"monitor={EARLY_STOPPING_METRIC} ({EARLY_STOPPING_MODE}), "
        f"patience={EARLY_STOPPING_PATIENCE}. "
        "No parameters were updated."
    )


def run_experiment(model_key: str, dry_run: bool = False) -> int:
    """Validate config, print it, and stop before training.

    A dry run exits 0 after printing, including when torch is not installed.
    A full run exits 1 until raw data, the processed cache, and ``fit`` exist.
    Neither path writes metrics or checkpoints.
    """
    if model_key not in EXPERIMENTS:
        known = ", ".join(sorted(EXPERIMENTS))
        print(f"Unknown model '{model_key}'. Known models: {known}.", file=sys.stderr)
        return 2

    validate_config()
    set_seed(SEED)

    spec = EXPERIMENTS[model_key]
    try:
        device = resolve_device()
    except ImportError:
        if not dry_run:
            print(
                "PyTorch is not installed. Install requirements before training.",
                file=sys.stderr,
            )
            return 1
        device = "not resolved (torch is not installed)"
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Experiment {spec['number']}: {spec['title']}")
    print(f"Registered class: {spec['qualname']}")
    print(format_config(device))
    print(f"Intended checkpoint: {checkpoint_path(model_key)}")

    if dry_run:
        print("Dry run only. No data was loaded and no training was started.")
        return 0

    if not directory_has_data(RAW_DATA_DIR):
        print("Stopping before training.")
        print(f"Raw data is not in {RAW_DATA_DIR}.")
        print("This starter does not download PhysioNet files.")
        print("No model was fit and no metric was computed.")
        return 1

    cache = processed_path()
    if not cache.is_file():
        print("Stopping before training.")
        print(f"Processed cache was not found at {cache}.")
        print("Preprocessing is not implemented, so training did not start.")
        print("No model was fit and no metric was computed.")
        return 1

    print("Stopping before training.")
    print("The training loop is not implemented.")
    print("No model was fit and no metric was computed.")
    return 1
