"""Shared helpers that do not train a model or write experimental results."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import random
import sys
import time
import tracemalloc
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, TypeVar

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from src.config import (  # noqa: E402
    CHECKPOINTS_DIR,
    EARLY_STOPPING_PATIENCE,
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SEED,
    TABLES_DIR,
)

T = TypeVar("T")


def as_cpu_rng_state(state: Any) -> Any:
    """Return the detached CPU byte tensor required by PyTorch RNG setters."""
    import torch

    if not isinstance(state, torch.Tensor):
        raise TypeError("RNG state must be a torch.Tensor.")
    return state.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch when those libraries are installed.

    Missing optional libraries are skipped. This does not generate data.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed a data-loader worker deterministically from PyTorch's worker seed."""
    del worker_id
    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def move_to_device(
    value: T,
    device: str | Any,
    non_blocking: bool = False,
) -> T:
    """Recursively move tensors while preserving non-tensor batch metadata."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)  # type: ignore[return-value]
    if isinstance(value, dict):
        return {
            key: move_to_device(item, device, non_blocking)
            for key, item in value.items()
        }  # type: ignore[return-value]
    if isinstance(value, list):
        return [
            move_to_device(item, device, non_blocking) for item in value
        ]  # type: ignore[return-value]
    if isinstance(value, tuple):
        return tuple(
            move_to_device(item, device, non_blocking) for item in value
        )  # type: ignore[return-value]
    return value


def count_trainable_parameters(model: Any) -> int:
    """Count scalar parameters that participate in gradient updates."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def recompute_delta_t(
    times: Any,
    mask: Any,
    valid_time_mask: Any,
) -> Any:
    """Recompute per-variable elapsed times for a padded irregular batch.

    Each valid cell stores hours since that variable's previous observation.
    Before its first observation, elapsed time is measured from admission.
    Padded cells remain zero.
    """
    import torch

    if times.ndim != 2 or mask.ndim != 3:
        raise ValueError("times must be (B, T) and mask must be (B, T, D).")
    if mask.shape[:2] != times.shape or valid_time_mask.shape != times.shape:
        raise ValueError("times, mask, and valid_time_mask must share B and T.")
    valid = valid_time_mask.to(device=times.device, dtype=torch.bool)
    result = torch.zeros_like(mask, dtype=times.dtype, device=times.device)
    batch_size, _, n_vars = mask.shape

    for patient_index in range(batch_size):
        last_observed: list[Any | None] = [None] * n_vars
        valid_indices = torch.nonzero(valid[patient_index], as_tuple=False).flatten()
        for time_index in valid_indices.tolist():
            current_time = times[patient_index, time_index]
            for variable_index in range(n_vars):
                previous = last_observed[variable_index]
                result[patient_index, time_index, variable_index] = (
                    current_time if previous is None else current_time - previous
                )
                if mask[patient_index, time_index, variable_index] > 0:
                    last_observed[variable_index] = current_time
    return result.to(dtype=mask.dtype)


def package_versions(package_names: list[str]) -> dict[str, str]:
    """Return installed package versions without importing those packages."""
    versions: dict[str, str] = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def write_json(path: Path, payload: Any) -> None:
    """Atomically write JSON to an explicitly supplied output path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=True)
        handle.write("\n")
    temporary.replace(path)


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    """Atomically write CSV rows to an explicitly supplied output path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_text(path: Path, content: str) -> None:
    """Atomically write text to an explicitly supplied output path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


class ElapsedTimer:
    """Small reusable wall-clock timer based on ``time.perf_counter``."""

    def __init__(self) -> None:
        self._started_at: float | None = None
        self.seconds = 0.0

    def __enter__(self) -> "ElapsedTimer":
        self._started_at = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        if self._started_at is not None:
            self.seconds = time.perf_counter() - self._started_at


@dataclass
class PeakMemoryTracker:
    """Best-effort peak-memory context manager.

    CUDA reports allocator peak bytes via ``torch.cuda.max_memory_allocated``.
    CPU reports only Python-tracked allocations from ``tracemalloc``; this is
    not process RSS or total system RAM and excludes most native tensor memory.
    """

    device: str | Any
    peak_bytes: int = 0
    label: str = ""
    _owns_tracemalloc: bool = False

    def __enter__(self) -> "PeakMemoryTracker":
        import torch

        resolved = torch.device(self.device)
        if resolved.type == "cuda":
            torch.cuda.reset_peak_memory_stats(resolved)
            self.label = "CUDA allocator peak memory"
        else:
            self._owns_tracemalloc = not tracemalloc.is_tracing()
            if self._owns_tracemalloc:
                tracemalloc.start()
            else:
                tracemalloc.reset_peak()
            self.label = "Python-tracked peak memory (not total RAM)"
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        import torch

        resolved = torch.device(self.device)
        if resolved.type == "cuda":
            torch.cuda.synchronize(resolved)
            self.peak_bytes = int(torch.cuda.max_memory_allocated(resolved))
        else:
            _, peak = tracemalloc.get_traced_memory()
            self.peak_bytes = int(peak)
            if self._owns_tracemalloc:
                tracemalloc.stop()


class EarlyStopping:
    """Maximize validation AUPRC with optional in-memory or file restoration."""

    def __init__(
        self,
        patience: int = EARLY_STOPPING_PATIENCE,
        min_delta: float = 0.0,
    ) -> None:
        if patience < 1:
            raise ValueError("patience must be at least 1.")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative.")
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float("-inf")
        self.bad_epochs = 0
        self.should_stop = False
        self.best_state_dict: dict[str, Any] | None = None

    def step(
        self,
        validation_auprc: float,
        model: Any | None = None,
        checkpoint_path: Path | str | None = None,
    ) -> bool:
        """Update state and return whether this score is a new best.

        A checkpoint is written only when ``checkpoint_path`` is explicitly
        supplied. With a model and no path, the best state is retained only in
        memory.
        """
        import math
        import torch

        improved = math.isfinite(validation_auprc) and (
            validation_auprc > self.best_score + self.min_delta
        )
        if improved:
            self.best_score = float(validation_auprc)
            self.bad_epochs = 0
            self.should_stop = False
            if model is not None:
                self.best_state_dict = deepcopy(model.state_dict())
            if checkpoint_path is not None:
                if model is None:
                    raise ValueError(
                        "model is required when checkpoint_path is supplied."
                    )
                path = Path(checkpoint_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), path)
        else:
            self.bad_epochs += 1
            self.should_stop = self.bad_epochs >= self.patience
        return improved

    def load_best(
        self,
        model: Any,
        checkpoint_path: Path | str | None = None,
        map_location: str | Any = "cpu",
    ) -> None:
        """Load the best state from an explicit path or the in-memory copy."""
        import torch

        if checkpoint_path is not None:
            state_dict = torch.load(
                Path(checkpoint_path),
                map_location=map_location,
                weights_only=True,
            )
        elif self.best_state_dict is not None:
            state_dict = self.best_state_dict
        else:
            raise RuntimeError("No best model state is available to load.")
        model.load_state_dict(state_dict)


def ensure_output_dirs() -> None:
    """Create output directories if they are missing. Does not write results."""
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        CHECKPOINTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def directory_has_data(directory: Path) -> bool:
    """Return True when ``directory`` contains something other than ``.gitkeep``."""
    if not directory.is_dir():
        return False
    for path in directory.iterdir():
        if path.name == ".gitkeep":
            continue
        return True
    return False
