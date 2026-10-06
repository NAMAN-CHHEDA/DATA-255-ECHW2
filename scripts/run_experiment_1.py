"""Experiment 1: headline comparison of four mortality-risk models."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PRIMARY_MODEL_NAMES = {
    "gru_baseline": "GRU",
    "mlp_neural_ode": "MLP-ODE",
    "kan_neural_ode": "KAN-ODE",
    "ltc": "LTC",
}
VARIANT_MODEL_NAME = "CfC"
# Models whose epochs are slow enough to warrant in-epoch batch progress lines.
SLOW_MODEL_KEYS = frozenset({"mlp_neural_ode", "kan_neural_ode", "ltc", "cfc"})
PROGRESS_EVERY_BATCHES = 10
RESULT_FIELDS = [
    "model_key",
    "model",
    "auroc",
    "auprc",
    "brier_score",
    "sample_count",
    "trainable_parameters",
    "mean_train_seconds_per_epoch",
    "mean_train_val_seconds_per_epoch",
    "peak_memory_bytes",
    "peak_memory_type",
    "early_stopped",
    "epochs_completed",
    "best_epoch",
    "best_validation_auprc",
    "max_epochs",
    "min_epochs",
    "patience",
    "min_delta",
    "checkpoint_interval",
    "device",
    "elapsed_seconds",
    "time_capped",
    "max_run_minutes",
    "kan_hidden_width",
    "kan_grid",
    "kan_spline_order",
    "kan_grid_range",
    "mlp_target_parameters",
    "parameter_gap",
    "parameter_gap_pct",
]


class _Tee:
    """Duplicate a text stream into a log file."""

    def __init__(self, stream: Any, log_handle: Any) -> None:
        self._stream = stream
        self._log = log_handle

    def write(self, text: str) -> int:
        self._stream.write(text)
        self._log.write(text)
        return len(text)

    def flush(self) -> None:
        self._stream.flush()
        self._log.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


@contextlib.contextmanager
def tee_output_to_log(log_path: Path):
    """Mirror stdout and stderr to ``log_path`` (append mode, line-flushed)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(
            f"\n===== run started {datetime.now(timezone.utc).isoformat()} "
            f"argv={sys.argv[1:]} =====\n"
        )
        original = (sys.stdout, sys.stderr)
        sys.stdout = _Tee(original[0], handle)
        sys.stderr = _Tee(original[1], handle)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = original


@dataclass(frozen=True)
class TrainingSettings:
    """Effective Experiment 1 loop controls."""

    max_epochs: int
    min_epochs: int
    patience: int
    min_delta: float
    checkpoint_every: int
    log_every_epoch: bool


def default_training_settings() -> TrainingSettings:
    """Return settings matching the original behavior where applicable."""
    from src.config import EARLY_STOPPING_PATIENCE, EPOCHS

    return TrainingSettings(
        max_epochs=EPOCHS,
        min_epochs=1,
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=0.0,
        checkpoint_every=5,
        log_every_epoch=False,
    )


def validate_training_settings(settings: TrainingSettings) -> None:
    """Reject unsafe or nonsensical loop controls before creating outputs."""
    if settings.max_epochs < 1:
        raise ValueError("max_epochs must be at least 1.")
    if not 1 <= settings.min_epochs <= settings.max_epochs:
        raise ValueError("min_epochs must be between 1 and max_epochs.")
    if settings.patience < 1:
        raise ValueError("patience must be at least 1.")
    if settings.min_delta < 0.0:
        raise ValueError("min_delta must be non-negative.")
    if settings.checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least 1.")


def build_parser() -> argparse.ArgumentParser:
    """Build a CLI that cannot train accidentally."""
    defaults = default_training_settings()
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 1: train GRU, MLP-ODE, KAN-ODE, and LTC with one "
            "protocol and report held-out test metrics."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate cache/configuration and print the plan without writing files.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Explicitly execute training and create Experiment 1 outputs.",
    )
    parser.add_argument(
        "--include-cfc",
        action="store_true",
        help=(
            "Also train CfC and write it to separate variant files "
            "(never a headline row; see scripts/time_ltc_vs_cfc.py)."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[*PRIMARY_MODEL_NAMES, "cfc"],
        default=None,
        help=(
            "Train only these models. Rows already saved for the other models "
            "are kept in the headline table (so the table can still have all "
            "four rows). Default: all four."
        ),
    )
    parser.add_argument(
        "--kan-grid-range",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        default=None,
        help="Override the KAN spline grid range (default -1 1; runaway fallback -3 3).",
    )
    parser.add_argument(
        "--first-epoch-timeout",
        type=float,
        default=None,
        help=(
            "Abort (exit code 3) if a model's first epoch in this process "
            "exceeds this many seconds. Used by scripts/run_all.py for KAN."
        ),
    )
    parser.add_argument(
        "--max-run-minutes",
        type=float,
        default=None,
        help=(
            "Wall-clock cap per model (default config.DEFAULT_MAX_RUN_MINUTES = 20; "
            "0 disables). After the current epoch the run stops, restores the "
            "best checkpoint, tests, and is marked time_capped."
        ),
    )
    parser.add_argument("--max-epochs", type=int, default=defaults.max_epochs)
    parser.add_argument("--min-epochs", type=int, default=defaults.min_epochs)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--min-delta", type=float, default=defaults.min_delta)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=defaults.checkpoint_every,
    )
    parser.add_argument(
        "--log-every-epoch",
        action="store_true",
        help="Print validation progress after every epoch.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume incomplete models from latest.pt and skip models with "
            "already-saved results."
        ),
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> TrainingSettings:
    """Construct and validate loop settings from parsed CLI arguments."""
    settings = TrainingSettings(
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        checkpoint_every=args.checkpoint_every,
        log_every_epoch=args.log_every_epoch,
    )
    validate_training_settings(settings)
    return settings


def validate_part_a_cache(cache_path: Path) -> dict[str, Any]:
    """Validate the saved Part A schema and exact split membership."""
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Part A cache is absent: {cache_path}. "
            "Run `python -m src.data_preprocessing` first."
        )
    with cache_path.open("rb") as handle:
        payload = pickle.load(handle)
    required = {"records", "split_record_ids", "variables", "seed"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"Unsupported Part A cache schema: {cache_path}")
    for split in ("train", "validation", "test"):
        records = payload["records"].get(split)
        saved_ids = payload["split_record_ids"].get(split)
        if not isinstance(records, list) or not records:
            raise ValueError(f"Part A cache has no usable {split} records.")
        record_ids = [str(record["record_id"]) for record in records]
        if record_ids != [str(record_id) for record_id in saved_ids]:
            raise ValueError(
                f"Part A {split} records do not match saved split RecordIDs."
            )
    return payload


def planned_model_keys(include_cfc: bool) -> list[str]:
    """Return primary models plus the optional separate CfC variant."""
    keys = list(PRIMARY_MODEL_NAMES)
    if include_cfc:
        keys.append("cfc")
    return keys


def validate_dry_run(
    cache_path: Path,
    output_dir: Path,
    include_cfc: bool = False,
    settings: TrainingSettings | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Validate an Experiment 1 plan without creating ``output_dir``."""
    from src.config import SEED, validate_config

    effective = settings or default_training_settings()
    validate_training_settings(effective)
    validate_config()
    payload = validate_part_a_cache(cache_path)
    return {
        "seed": SEED,
        "models": planned_model_keys(include_cfc),
        "split_counts": {
            split: len(payload["records"][split])
            for split in ("train", "validation", "test")
        },
        "settings": asdict(effective),
        "resume": resume,
        "output_dir": str(output_dir),
    }


def collate_for_experiment(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Reuse the Part B collator and add recomputed padded ``delta_t``."""
    from src.dataset import collate_stays
    from src.utils import recompute_delta_t

    collated = collate_stays(batch)
    collated["delta_t"] = recompute_delta_t(
        collated["times"],
        collated["mask"],
        collated["valid_time_mask"],
    )
    return collated


def build_model(model_key: str) -> Any:
    """Construct one configured model without changing its architecture."""
    if model_key == "gru_baseline":
        from src.models.gru_baseline import GRUBaseline

        return GRUBaseline()
    if model_key == "mlp_neural_ode":
        from src.models.mlp_neural_ode import MLPNeuralODE

        return MLPNeuralODE()
    if model_key == "kan_neural_ode":
        # Capacity-matched to the MLP-ODE by direct search (src/matching.py).
        from src.config import ODE_HIDDEN_DIM
        from src.matching import build_matched_kan

        return build_matched_kan(ODE_HIDDEN_DIM)
    if model_key in {"ltc", "cfc"}:
        from src.models.liquid_model import LiquidRiskModel

        return LiquidRiskModel(cell_type=model_key)
    raise ValueError(f"Unknown Experiment 1 model key: {model_key!r}.")


def materialize_experiment_datasets(
    cache_path: Path,
) -> tuple[dict[str, Any], float]:
    """Build all deterministic CPU tensors once without writing any files."""
    from src.dataset import materialize_records

    started = time.perf_counter()
    payload = validate_part_a_cache(cache_path)
    datasets = {
        split: materialize_records(payload["records"][split], split)
        for split in ("train", "validation", "test")
    }
    return datasets, time.perf_counter() - started


def build_loaders(
    generator: Any | None = None,
    materialized_splits: dict[str, Any] | None = None,
    model_key: str | None = None,
    pin_memory: bool = False,
) -> tuple[Any, Any, Any]:
    """Build deterministic loaders over legacy or materialized split views."""
    import torch
    from torch.utils.data import DataLoader

    from src.config import BATCH_SIZE, NUM_WORKERS, SEED, eval_batch_size
    from src.dataset import load_split, materialized_collator
    from src.utils import seed_worker

    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(SEED)
    common = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "worker_init_fn": seed_worker,
        "pin_memory": pin_memory,
    }
    if NUM_WORKERS > 0:
        common["persistent_workers"] = True
    if materialized_splits is None:
        datasets = {
            split: load_split(split)
            for split in ("train", "validation", "test")
        }
        common["collate_fn"] = collate_for_experiment
    else:
        if model_key is None:
            raise ValueError("model_key is required for materialized loaders.")
        datasets = materialized_splits
        common["collate_fn"] = materialized_collator(model_key)
    train_loader = DataLoader(
        datasets["train"],
        shuffle=True,
        generator=generator,
        **common,
    )
    # Larger evaluation batches only for models whose outputs are
    # batch-invariant (see config.EVAL_BATCH_INVARIANT_MODELS).
    eval_common = {**common, "batch_size": eval_batch_size(model_key)}
    validation_loader = DataLoader(
        datasets["validation"],
        shuffle=False,
        **eval_common,
    )
    test_loader = DataLoader(
        datasets["test"],
        shuffle=False,
        **eval_common,
    )
    return train_loader, validation_loader, test_loader


def configuration_snapshot() -> dict[str, Any]:
    """Return the repository protocol values needed to reproduce a run."""
    from src import config

    keys = [
        "SEED",
        "TRAIN_RATIO",
        "VAL_RATIO",
        "TEST_RATIO",
        "BATCH_SIZE",
        "LEARNING_RATE",
        "WEIGHT_DECAY",
        "EPOCHS",
        "EARLY_STOPPING_PATIENCE",
        "N_VARS",
        "GRU_HIDDEN_SIZE",
        "ODE_HIDDEN_DIM",
        "ODE_WIDTH",
        "ODE_SOLVER",
        "ODE_RTOL",
        "ODE_ATOL",
        "KAN_GRID",
        "KAN_SPLINE_ORDER",
        "KAN_HIDDEN_WIDTH_MULTIPLIER",
        "LIQUID_UNITS",
        "LIQUID_ODE_UNFOLDS",
        "LIQUID_TIMESPAN_MIN",
        "LIQUID_TIMESPAN_MAX",
        "LIQUID_BATCH_MODE",
        "KAN_GRID_RANGE",
    ]
    return {
        key: (
            list(value) if isinstance(value := getattr(config, key), tuple) else value
        )
        for key in keys
    }


def effective_configuration(
    settings: TrainingSettings,
    device: str,
) -> dict[str, Any]:
    """Combine fixed repository values with effective CLI loop controls."""
    return {
        **configuration_snapshot(),
        "EARLY_STOPPING_METRIC": "validation_auprc",
        "MAX_EPOCHS": settings.max_epochs,
        "MIN_EPOCHS": settings.min_epochs,
        "PATIENCE": settings.patience,
        "MIN_DELTA": settings.min_delta,
        "CHECKPOINT_INTERVAL": settings.checkpoint_every,
        "LOG_EVERY_EPOCH": settings.log_every_epoch,
        "DEVICE": device,
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Save a checkpoint through a same-directory atomic replacement."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint_payload(
    checkpoint_path: Path,
    map_location: str | Any = "cpu",
) -> Mapping[str, Any]:
    """Load a rich Experiment 1 checkpoint."""
    import torch

    payload = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(payload, Mapping) or "model_state" not in payload:
        raise ValueError(f"Unsupported Experiment 1 checkpoint: {checkpoint_path}")
    return payload


def load_model_state(
    checkpoint_path: Path,
    map_location: str | Any = "cpu",
) -> Mapping[str, Any]:
    """Return model weights from either rich or legacy Experiment 1 files."""
    import torch

    payload = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=True,
    )
    if isinstance(payload, Mapping) and "model_state" in payload:
        return payload["model_state"]
    if not isinstance(payload, Mapping):
        raise ValueError(f"Unsupported model checkpoint: {checkpoint_path}")
    return payload


def _checkpoint_payload(
    *,
    model: Any,
    optimizer: Any,
    model_key: str,
    epoch: int,
    best_epoch: int,
    best_validation_auprc: float,
    patience_counter: int,
    effective_config: dict[str, Any],
    train_generator: Any,
    epoch_seconds: list[float],
    train_seconds: list[float],
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Build the complete resumable checkpoint contract."""
    from src.config import SEED

    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation_auprc": best_validation_auprc,
        "patience_counter": patience_counter,
        "seed": SEED,
        "effective_configuration": effective_config,
        "model_name": model_key,
        "train_generator_state": train_generator.get_state(),
        "epoch_seconds": list(epoch_seconds),
        "train_seconds": list(train_seconds),
        "elapsed_seconds": elapsed_seconds,
    }


def _save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    """Atomically save and immediately report a checkpoint."""
    _atomic_torch_save(payload, path)
    print(f"Checkpoint saved: {path}", flush=True)


def _gpu_peak_bytes(device: str) -> int | None:
    """Return current CUDA allocator peak, or ``None`` on non-CUDA devices."""
    import torch

    resolved = torch.device(device)
    if resolved.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(resolved))


def _format_peak_memory(peak_bytes: int | None) -> str:
    """Format a concise per-epoch GPU peak field."""
    if peak_bytes is None:
        return "n/a"
    return f"{peak_bytes / (1024**2):.1f} MiB"


def _validate_resume_payload(
    payload: Mapping[str, Any],
    model_key: str,
    effective_config: dict[str, Any],
) -> None:
    """Prevent accidental cross-model or cross-protocol resume."""
    from src.config import SEED

    if payload.get("model_name") != model_key:
        raise ValueError(
            f"Checkpoint model {payload.get('model_name')!r} does not match "
            f"requested model {model_key!r}."
        )
    if payload.get("seed") != SEED:
        raise ValueError("Checkpoint seed does not match the configured seed.")
    if payload.get("effective_configuration") != effective_config:
        raise ValueError(
            f"Cannot safely resume {model_key}: effective configuration changed."
        )


def train_and_test_model(
    model_key: str,
    device: str,
    checkpoint_dir: Path,
    settings: TrainingSettings,
    model_number: int = 1,
    total_models: int = 1,
    resume: bool = False,
    materialized_splits: dict[str, Any] | None = None,
    model_factory: Callable[[], Any] | None = None,
    run_name: str | None = None,
    display_name: str | None = None,
    extra_config: dict[str, Any] | None = None,
    first_epoch_timeout: float | None = None,
    max_run_minutes: float | None = None,
) -> dict[str, Any]:
    """Train on validation AUPRC, restore the best state, and test once.

    ``model_key`` selects the forward adapter and collator. The optional
    ``model_factory``/``run_name``/``display_name``/``extra_config`` let
    Experiments 2 and 4 reuse this exact routine for custom-capacity models
    and thinned data; ``extra_config`` joins the resume-compatibility check.
    """
    import torch

    from src.config import LEARNING_RATE, SEED, WEIGHT_DECAY
    from src.evaluate import evaluate
    from src.train import get_forward_adapter, train_one_epoch
    from src.utils import (
        PeakMemoryTracker,
        as_cpu_rng_state,
        count_trainable_parameters,
        set_seed,
    )

    validate_training_settings(settings)
    run_name = run_name or model_key
    kan_match: dict[str, Any] | None = None
    if model_factory is None and model_key == "kan_neural_ode":
        from src.config import ODE_HIDDEN_DIM
        from src.matching import format_match, match_kan_to_mlp

        kan_match = match_kan_to_mlp(ODE_HIDDEN_DIM)
        print(format_match(kan_match), flush=True)
        extra_config = {
            **(extra_config or {}),
            "EXP1_KAN_WIDTH": kan_match["chosen"]["hidden_width"],
            "EXP1_KAN_GRID": kan_match["chosen"]["grid"],
        }
    set_seed(SEED)
    model = (
        build_model(model_key) if model_factory is None else model_factory()
    ).to(device)
    parameter_count = count_trainable_parameters(model)
    forward_fn = get_forward_adapter(model_key)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(SEED)
    effective_config = {
        **effective_configuration(settings, device),
        **(extra_config or {}),
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    latest_path = checkpoint_dir / "latest.pt"

    start_epoch = 1
    best_epoch = 0
    best_validation_auprc = float("-inf")
    patience_counter = 0
    epoch_seconds: list[float] = []
    train_seconds: list[float] = []
    prior_elapsed_seconds = 0.0

    if resume and latest_path.is_file():
        payload = load_checkpoint_payload(latest_path, map_location=device)
        _validate_resume_payload(payload, run_name, effective_config)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload["epoch"]) + 1
        best_epoch = int(payload["best_epoch"])
        best_validation_auprc = float(payload["best_validation_auprc"])
        patience_counter = int(payload["patience_counter"])
        epoch_seconds = [float(value) for value in payload.get("epoch_seconds", [])]
        train_seconds = [float(value) for value in payload.get("train_seconds", [])]
        prior_elapsed_seconds = float(payload.get("elapsed_seconds", 0.0))
        generator_state = payload.get("train_generator_state")
        if generator_state is not None:
            train_generator.set_state(as_cpu_rng_state(generator_state))
        print(
            f"Resumed {run_name} from epoch {start_epoch - 1}: {latest_path}",
            flush=True,
        )

    if materialized_splits is None:
        train_loader, validation_loader, test_loader = build_loaders(
            train_generator
        )
    else:
        train_loader, validation_loader, test_loader = build_loaders(
            generator=train_generator,
            materialized_splits=materialized_splits,
            model_key=model_key,
            pin_memory=torch.device(device).type == "cuda",
        )
    print(f"Model {model_number}/{total_models}: {run_name}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Trainable parameters: {parameter_count:,}", flush=True)
    print(
        "Training settings: "
        f"max_epochs={settings.max_epochs}, min_epochs={settings.min_epochs}, "
        f"monitor=validation AUPRC, patience={settings.patience}, "
        f"min_delta={settings.min_delta:g}, "
        f"checkpoint_every={settings.checkpoint_every}",
        flush=True,
    )

    model_started = time.perf_counter()
    epochs_completed = start_epoch - 1
    early_stopped = (
        epochs_completed >= settings.min_epochs
        and patience_counter >= settings.patience
        and epochs_completed < settings.max_epochs
    )
    cap_seconds = (max_run_minutes or 0.0) * 60.0
    time_capped = bool(
        cap_seconds and epochs_completed >= 1 and prior_elapsed_seconds > cap_seconds
    )

    with PeakMemoryTracker(device) as memory:
        epochs_to_run = (
            ()
            if early_stopped or time_capped
            else range(start_epoch, settings.max_epochs + 1)
        )
        for epoch in epochs_to_run:
            epoch_started = time.perf_counter()
            training_loss, training_seconds = train_one_epoch(
                model,
                train_loader,
                optimizer,
                forward_fn,
                device,
                progress_every=(
                    PROGRESS_EVERY_BATCHES if model_key in SLOW_MODEL_KEYS else 0
                ),
                progress_label=f"{run_name} epoch {epoch}",
                time_limit_seconds=(
                    first_epoch_timeout if epoch == start_epoch else None
                ),
            )
            validation_started = time.perf_counter()
            validation_metrics = evaluate(
                model,
                validation_loader,
                forward_fn,
                device,
            )
            validation_seconds = time.perf_counter() - validation_started
            validation_auroc = float(validation_metrics["auroc"])
            validation_auprc = float(validation_metrics["auprc"])
            improvement = validation_auprc - best_validation_auprc
            improved = (
                math.isfinite(validation_auprc)
                and validation_auprc > best_validation_auprc
                and improvement >= settings.min_delta
            )
            if improved:
                best_validation_auprc = validation_auprc
                best_epoch = epoch
                patience_counter = 0
                print(
                    f"Validation AUPRC improved for {run_name}: "
                    f"{validation_auprc:.6f} at epoch {epoch}.",
                    flush=True,
                )
            else:
                patience_counter += 1

            epoch_duration = time.perf_counter() - epoch_started
            epoch_seconds.append(epoch_duration)
            train_seconds.append(training_seconds)
            epochs_completed = epoch
            total_duration = (
                prior_elapsed_seconds + time.perf_counter() - model_started
            )
            payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                model_key=run_name,
                epoch=epoch,
                best_epoch=best_epoch,
                best_validation_auprc=best_validation_auprc,
                patience_counter=patience_counter,
                effective_config=effective_config,
                train_generator=train_generator,
                epoch_seconds=epoch_seconds,
                train_seconds=train_seconds,
                elapsed_seconds=total_duration,
            )
            if improved:
                _save_checkpoint(payload, best_path)
            if epoch % settings.checkpoint_every == 0:
                periodic_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
                _save_checkpoint(payload, periodic_path)
            _save_checkpoint(payload, latest_path)

            if settings.log_every_epoch:
                print(
                    f"{run_name} | epoch {epoch}/{settings.max_epochs} | "
                    f"train_loss={training_loss:.6f} | "
                    f"val_AUROC={validation_auroc:.6f} | "
                    f"val_AUPRC={validation_auprc:.6f} | "
                    f"best_AUPRC={best_validation_auprc:.6f} | "
                    f"best_epoch={best_epoch} | "
                    f"patience={patience_counter}/{settings.patience} | "
                    f"train_time={training_seconds:.2f}s | "
                    f"val_time={validation_seconds:.2f}s | "
                    f"epoch_time={epoch_duration:.2f}s | "
                    f"model_time={total_duration:.2f}s | "
                    f"gpu_peak={_format_peak_memory(_gpu_peak_bytes(device))}",
                    flush=True,
                )

            if (
                epoch >= settings.min_epochs
                and patience_counter >= settings.patience
            ):
                early_stopped = True
                print(
                    f"Early stopping activated for {run_name} at epoch {epoch}: "
                    f"validation AUPRC did not improve by at least "
                    f"{settings.min_delta:g} for {patience_counter} epochs.",
                    flush=True,
                )
                break

            if cap_seconds and total_duration > cap_seconds:
                time_capped = True
                print(
                    f"Time cap reached for {run_name} after epoch {epoch} "
                    f"({total_duration / 60:.1f} min > {max_run_minutes:g} min); "
                    "stopping, restoring the best checkpoint, and testing.",
                    flush=True,
                )
                break

        if best_epoch == 0 or not math.isfinite(best_validation_auprc):
            raise RuntimeError(
                f"{run_name} never produced a finite validation AUPRC."
            )
        if not best_path.is_file():
            raise FileNotFoundError(f"Best checkpoint is missing: {best_path}")
        best_payload = load_checkpoint_payload(best_path, map_location=device)
        model.load_state_dict(best_payload["model_state"])
        print(
            f"Best checkpoint restored for {run_name}: epoch {best_epoch}, "
            f"validation AUPRC={best_validation_auprc:.6f}.",
            flush=True,
        )
        print(f"Test evaluation begins for {run_name}.", flush=True)
        test_metrics = evaluate(model, test_loader, forward_fn, device)

    elapsed_seconds = prior_elapsed_seconds + time.perf_counter() - model_started
    kan_row_fields: dict[str, Any] = {}
    if kan_match is not None:
        from src import config as _config

        chosen = kan_match["chosen"]
        kan_row_fields = {
            "kan_hidden_width": chosen["hidden_width"],
            "kan_grid": chosen["grid"],
            "kan_spline_order": kan_match["kan_spline_order"],
            "kan_grid_range": list(_config.KAN_GRID_RANGE),
            "mlp_target_parameters": kan_match["mlp_parameters"],
            "parameter_gap": chosen["gap"],
            "parameter_gap_pct": chosen["gap_pct"],
        }
    if display_name is None:
        display_name = (
            VARIANT_MODEL_NAME
            if model_key == "cfc"
            else PRIMARY_MODEL_NAMES[model_key]
        )
    return {
        "model_key": run_name,
        "model": display_name,
        "auroc": float(test_metrics["auroc"]),
        "auprc": float(test_metrics["auprc"]),
        "brier_score": float(test_metrics["brier_score"]),
        "sample_count": int(test_metrics["sample_count"]),
        "trainable_parameters": parameter_count,
        "mean_train_seconds_per_epoch": sum(train_seconds) / len(train_seconds),
        "mean_train_val_seconds_per_epoch": sum(epoch_seconds)
        / len(epoch_seconds),
        "peak_memory_bytes": memory.peak_bytes,
        "peak_memory_type": memory.label,
        "early_stopped": early_stopped,
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "best_validation_auprc": best_validation_auprc,
        "max_epochs": settings.max_epochs,
        "min_epochs": settings.min_epochs,
        "patience": settings.patience,
        "min_delta": settings.min_delta,
        "checkpoint_interval": settings.checkpoint_every,
        "device": device,
        "elapsed_seconds": elapsed_seconds,
        "time_capped": time_capped,
        "max_run_minutes": max_run_minutes,
        **kan_row_fields,
    }


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    """Read prior atomic results for resume, returning no rows if absent."""
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not all(
        isinstance(row, dict) for row in payload
    ):
        raise ValueError(f"Result file does not contain a row list: {path}")
    return payload


def _row_model_key(row: dict[str, Any]) -> str:
    """Resolve model keys from new rows and legacy display-name-only rows."""
    if "model_key" in row:
        return str(row["model_key"])
    by_display = {value: key for key, value in PRIMARY_MODEL_NAMES.items()}
    by_display[VARIANT_MODEL_NAME] = "cfc"
    try:
        return by_display[str(row["model"])]
    except KeyError as exc:
        raise ValueError(f"Unrecognized completed model row: {row}") from exc


def _load_completed_rows(output_dir: Path) -> list[dict[str, Any]]:
    """Load already-completed model results for a resume run."""
    rows = _read_json_rows(output_dir / "headline_results.json")
    rows.extend(_read_json_rows(output_dir / "cfc_variant_results.json"))
    keys = [_row_model_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Saved Experiment 1 results contain duplicate models.")
    return rows


def _write_results(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically refresh CSV and JSON immediately after model completion."""
    from src.utils import write_csv, write_json

    headline_rows = [row for row in rows if _row_model_key(row) != "cfc"]
    cfc_rows = [row for row in rows if _row_model_key(row) == "cfc"]
    write_json(output_dir / "headline_results.json", headline_rows)
    headline_csv_rows = [
        {field: row.get(field) for field in RESULT_FIELDS}
        for row in headline_rows
    ]
    write_csv(
        output_dir / "headline_results.csv",
        headline_csv_rows,
        RESULT_FIELDS,
    )
    if cfc_rows:
        write_json(output_dir / "cfc_variant_results.json", cfc_rows)
        cfc_csv_rows = [
            {field: row.get(field) for field in RESULT_FIELDS}
            for row in cfc_rows
        ]
        write_csv(
            output_dir / "cfc_variant_results.csv",
            cfc_csv_rows,
            RESULT_FIELDS,
        )


def _metadata(
    *,
    device: str,
    settings: TrainingSettings,
    model_keys: list[str],
    rows: list[dict[str, Any]],
    resume: bool,
) -> dict[str, Any]:
    """Build run metadata, including completed per-model protocol outcomes."""
    import torch

    from src.config import SEED
    from src.utils import package_versions

    device_metadata: dict[str, Any] = {"resolved": device}
    if device == "cuda":
        device_metadata["name"] = torch.cuda.get_device_name(0)
    return {
        "experiment": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "configuration": effective_configuration(settings, device),
        "package_versions": package_versions(
            [
                "torch",
                "numpy",
                "scikit-learn",
                "torchdiffeq",
                "pykan",
                "ncps",
                "matplotlib",
            ]
        ),
        "device": device_metadata,
        "models": model_keys,
        "resume": resume,
        "completed_models": rows,
        "test_metrics_used_only_after_best_checkpoint_restoration": True,
        "early_stopping_uses_test_metrics": False,
    }


def _print_experiment_summary(rows: list[dict[str, Any]]) -> None:
    """Print one final concise result line per completed model."""
    print("Experiment 1 summary", flush=True)
    for row in rows:
        print(
            f"{_row_model_key(row)} | epochs={row['epochs_completed']} | "
            f"best_epoch={row['best_epoch']} | "
            f"best_val_AUPRC={float(row['best_validation_auprc']):.6f} | "
            f"test_AUROC={float(row['auroc']):.6f} | "
            f"test_AUPRC={float(row['auprc']):.6f}"
            + (" | time_capped" if row.get("time_capped") else ""),
            flush=True,
        )


def run_experiment(
    include_cfc: bool = False,
    settings: TrainingSettings | None = None,
    resume: bool = False,
    models: list[str] | None = None,
    first_epoch_timeout: float | None = None,
    max_run_minutes: float | None = None,
) -> list[dict[str, Any]]:
    """Execute Experiment 1 and persist each model before continuing."""
    from src.config import EXPERIMENT_1_DIR, PART_A_CACHE_PATH, resolve_device
    from src.utils import write_json

    effective = settings or default_training_settings()
    validate_dry_run(
        PART_A_CACHE_PATH,
        EXPERIMENT_1_DIR,
        include_cfc,
        effective,
        resume,
    )
    device = resolve_device()
    output_dir = EXPERIMENT_1_DIR
    include_cfc = include_cfc or bool(models and "cfc" in models)
    model_keys = planned_model_keys(include_cfc)
    selected = set(model_keys if models is None else models)

    materialized_splits, materialization_seconds = materialize_experiment_datasets(
        PART_A_CACHE_PATH
    )
    print(
        "One-time dataset materialization: "
        f"{materialization_seconds:.2f}s (in memory; no files created).",
        flush=True,
    )

    # No Experiment 1 output path is created until --run reaches this function
    # and all read-only preflight validation above has succeeded.
    output_dir.mkdir(parents=True, exist_ok=True)
    # A partial --models run must keep the other models' saved rows, so it
    # always loads them; a full run without --resume starts from scratch.
    keep_existing = resume or models is not None
    rows = _load_completed_rows(output_dir) if keep_existing else []
    rows = [row for row in rows if _row_model_key(row) in model_keys]
    if not resume:
        # Selected models are retrained from scratch, so drop their old rows.
        rows = [row for row in rows if _row_model_key(row) not in selected]
    completed = {_row_model_key(row) for row in rows}

    for model_number, model_key in enumerate(model_keys, start=1):
        model_checkpoint_dir = output_dir / "checkpoints" / model_key
        if model_key in completed:
            if not (model_checkpoint_dir / "best.pt").is_file():
                raise FileNotFoundError(
                    f"Completed model {model_key} is missing best.pt."
                )
            print(
                f"Resume: completed model {model_key} will not be retrained.",
                flush=True,
            )
        elif model_key not in selected:
            print(
                f"Skipping {model_key}: not selected by --models and no saved "
                "result exists.",
                flush=True,
            )
        else:
            row = train_and_test_model(
                model_key=model_key,
                device=device,
                checkpoint_dir=model_checkpoint_dir,
                settings=effective,
                model_number=model_number,
                total_models=len(model_keys),
                resume=resume,
                materialized_splits=materialized_splits,
                first_epoch_timeout=first_epoch_timeout,
                max_run_minutes=max_run_minutes,
            )
            rows.append(row)
            completed.add(model_key)
            _write_results(output_dir, rows)
            write_json(
                output_dir / "run_metadata.json",
                _metadata(
                    device=device,
                    settings=effective,
                    model_keys=model_keys,
                    rows=rows,
                    resume=resume,
                ),
            )
            print(
                f"Model completed and results saved: {model_key}.",
                flush=True,
            )

        if model_number < len(model_keys):
            print(
                f"Switching to next model: {model_keys[model_number]}.",
                flush=True,
            )

    ordered_rows = sorted(rows, key=lambda row: model_keys.index(_row_model_key(row)))
    headline_count = sum(1 for row in ordered_rows if _row_model_key(row) != "cfc")
    if headline_count != len(PRIMARY_MODEL_NAMES):
        print(
            f"WARNING: headline table has {headline_count} rows, expected "
            f"{len(PRIMARY_MODEL_NAMES)}; missing: "
            + ", ".join(
                key
                for key in PRIMARY_MODEL_NAMES
                if key not in {_row_model_key(row) for row in ordered_rows}
            ),
            flush=True,
        )
    _write_results(output_dir, ordered_rows)
    write_json(
        output_dir / "run_metadata.json",
        _metadata(
            device=device,
            settings=effective,
            model_keys=model_keys,
            rows=ordered_rows,
            resume=resume,
        ),
    )
    _print_experiment_summary(ordered_rows)
    return ordered_rows


def _print_effective_settings(plan: dict[str, Any]) -> None:
    """Display dry-run loop controls without writing output files."""
    settings = plan["settings"]
    print("Effective settings:", flush=True)
    print(f"  max_epochs: {settings['max_epochs']}", flush=True)
    print(f"  min_epochs: {settings['min_epochs']}", flush=True)
    print("  early_stopping_metric: validation AUPRC", flush=True)
    print(f"  patience: {settings['patience']}", flush=True)
    print(f"  min_delta: {settings['min_delta']}", flush=True)
    print(
        f"  checkpoint_every: {settings['checkpoint_every']}",
        flush=True,
    )
    print(f"  log_every_epoch: {settings['log_every_epoch']}", flush=True)
    print(f"  resume: {plan['resume']}", flush=True)


def resolve_max_run_minutes(value: float | None) -> float | None:
    """CLI value -> cap in minutes (None/0 means no cap; default from config)."""
    from src.config import DEFAULT_MAX_RUN_MINUTES

    minutes = DEFAULT_MAX_RUN_MINUTES if value is None else value
    return minutes if minutes and minutes > 0 else None


def apply_kan_grid_range(value: list[float] | tuple[float, float] | None) -> None:
    """Override ``config.KAN_GRID_RANGE`` for this process (CLI flag helper)."""
    if value is None:
        return
    from src import config

    config.KAN_GRID_RANGE = (float(value[0]), float(value[1]))
    config.validate_config()


def main(argv: list[str] | None = None) -> int:
    """Validate a plan or explicitly execute Experiment 1."""
    args = build_parser().parse_args(argv)
    settings = settings_from_args(args)
    apply_kan_grid_range(args.kan_grid_range)
    from src.config import EXPERIMENT_1_DIR, PART_A_CACHE_PATH

    if args.dry_run:
        plan = validate_dry_run(
            PART_A_CACHE_PATH,
            EXPERIMENT_1_DIR,
            include_cfc=args.include_cfc,
            settings=settings,
            resume=args.resume,
        )
        print("Experiment 1 dry run", flush=True)
        print(f"Part A cache: {PART_A_CACHE_PATH}", flush=True)
        print(f"Planned models: {', '.join(plan['models'])}", flush=True)
        print(f"Split counts: {plan['split_counts']}", flush=True)
        _print_effective_settings(plan)
        print(f"Planned output directory: {EXPERIMENT_1_DIR}", flush=True)
        print("No directories or files were created.", flush=True)
        return 0

    with tee_output_to_log(EXPERIMENT_1_DIR / "train.log"):
        try:
            run_experiment(
                include_cfc=args.include_cfc,
                settings=settings,
                resume=args.resume,
                models=args.models,
                first_epoch_timeout=args.first_epoch_timeout,
                max_run_minutes=resolve_max_run_minutes(args.max_run_minutes),
            )
        except TimeoutError as error:
            print(str(error), flush=True)
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
