"""Experiment 1: headline comparison of four mortality-risk models."""

from __future__ import annotations

import argparse
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
RESULT_FIELDS = [
    "model",
    "auroc",
    "auprc",
    "brier_score",
    "sample_count",
    "trainable_parameters",
    "mean_seconds_per_completed_epoch",
    "peak_memory_bytes",
    "peak_memory_type",
    "best_validation_auprc",
    "epochs_completed",
]


def build_parser() -> argparse.ArgumentParser:
    """Build a CLI that cannot train accidentally."""
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
        help="Also train CfC and write its result to a separate variant CSV.",
    )
    return parser


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
) -> dict[str, Any]:
    """Validate an Experiment 1 plan without creating ``output_dir``."""
    from src.config import SEED, validate_config

    validate_config()
    payload = validate_part_a_cache(cache_path)
    return {
        "seed": SEED,
        "models": planned_model_keys(include_cfc),
        "split_counts": {
            split: len(payload["records"][split])
            for split in ("train", "validation", "test")
        },
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
        from src.models.kan_neural_ode import KANNeuralODE

        return KANNeuralODE()
    if model_key in {"ltc", "cfc"}:
        from src.models.liquid_model import LiquidRiskModel

        return LiquidRiskModel(cell_type=model_key)
    raise ValueError(f"Unknown Experiment 1 model key: {model_key!r}.")


def build_loaders() -> tuple[Any, Any, Any]:
    """Build deterministic loaders over the exact saved Part A splits."""
    import torch
    from torch.utils.data import DataLoader

    from src.config import BATCH_SIZE, NUM_WORKERS, SEED
    from src.dataset import load_split
    from src.utils import seed_worker

    generator = torch.Generator()
    generator.manual_seed(SEED)
    common = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "collate_fn": collate_for_experiment,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        load_split("train"),
        shuffle=True,
        generator=generator,
        **common,
    )
    validation_loader = DataLoader(
        load_split("validation"),
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        load_split("test"),
        shuffle=False,
        **common,
    )
    return train_loader, validation_loader, test_loader


def train_and_test_model(
    model_key: str,
    device: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Train one model, select by validation AUPRC, then test once."""
    import math

    import torch

    from src.config import (
        EARLY_STOPPING_PATIENCE,
        EPOCHS,
        LEARNING_RATE,
        SEED,
        WEIGHT_DECAY,
    )
    from src.evaluate import evaluate
    from src.train import get_forward_adapter, train_one_epoch
    from src.utils import (
        EarlyStopping,
        PeakMemoryTracker,
        count_trainable_parameters,
        set_seed,
    )

    set_seed(SEED)
    model = build_model(model_key).to(device)
    forward_fn = get_forward_adapter(model_key)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    epoch_seconds: list[float] = []
    train_loader, validation_loader, test_loader = build_loaders()

    with PeakMemoryTracker(device) as memory:
        for _ in range(EPOCHS):
            _, seconds = train_one_epoch(
                model,
                train_loader,
                optimizer,
                forward_fn,
                device,
            )
            epoch_seconds.append(seconds)
            validation_metrics = evaluate(
                model,
                validation_loader,
                forward_fn,
                device,
            )
            validation_auprc = float(validation_metrics["auprc"])
            early_stopping.step(
                validation_auprc,
                model=model,
                checkpoint_path=checkpoint_path,
            )
            if early_stopping.should_stop:
                break

        if not math.isfinite(early_stopping.best_score):
            raise RuntimeError(
                f"{model_key} never produced a finite validation AUPRC."
            )
        early_stopping.load_best(
            model,
            checkpoint_path=checkpoint_path,
            map_location=device,
        )
        test_metrics = evaluate(model, test_loader, forward_fn, device)

    display_name = (
        VARIANT_MODEL_NAME
        if model_key == "cfc"
        else PRIMARY_MODEL_NAMES[model_key]
    )
    return {
        "model": display_name,
        "auroc": float(test_metrics["auroc"]),
        "auprc": float(test_metrics["auprc"]),
        "brier_score": float(test_metrics["brier_score"]),
        "sample_count": int(test_metrics["sample_count"]),
        "trainable_parameters": count_trainable_parameters(model),
        "mean_seconds_per_completed_epoch": sum(epoch_seconds) / len(epoch_seconds),
        "peak_memory_bytes": memory.peak_bytes,
        "peak_memory_type": memory.label,
        "best_validation_auprc": early_stopping.best_score,
        "epochs_completed": len(epoch_seconds),
    }


def configuration_snapshot() -> dict[str, Any]:
    """Return the protocol values needed to reproduce a real run."""
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
    ]
    return {key: getattr(config, key) for key in keys}


def run_experiment(include_cfc: bool = False) -> list[dict[str, Any]]:
    """Execute Experiment 1 and write only its explicitly defined outputs."""
    import torch

    from src.config import EXPERIMENT_1_DIR, PART_A_CACHE_PATH, SEED, resolve_device
    from src.utils import package_versions, write_csv, write_json

    validate_dry_run(PART_A_CACHE_PATH, EXPERIMENT_1_DIR, include_cfc)
    device = resolve_device()
    output_dir = EXPERIMENT_1_DIR
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for model_key in planned_model_keys(include_cfc):
        checkpoint_path = checkpoint_dir / f"{model_key}_best.pt"
        rows.append(train_and_test_model(model_key, device, checkpoint_path))

    headline_rows = [row for row in rows if row["model"] != VARIANT_MODEL_NAME]
    write_csv(
        output_dir / "headline_results.csv",
        headline_rows,
        RESULT_FIELDS,
    )
    write_json(output_dir / "headline_results.json", headline_rows)
    if include_cfc:
        cfc_rows = [row for row in rows if row["model"] == VARIANT_MODEL_NAME]
        write_csv(
            output_dir / "cfc_variant_results.csv",
            cfc_rows,
            RESULT_FIELDS,
        )

    device_metadata: dict[str, Any] = {"resolved": device}
    if device == "cuda":
        device_metadata["name"] = torch.cuda.get_device_name(0)
    metadata = {
        "experiment": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "configuration": configuration_snapshot(),
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
        "models": planned_model_keys(include_cfc),
        "test_metrics_used_only_after_model_selection": True,
    }
    write_json(output_dir / "run_metadata.json", metadata)
    return rows


def main(argv: list[str] | None = None) -> int:
    """Validate a plan or explicitly execute Experiment 1."""
    args = build_parser().parse_args(argv)
    from src.config import EXPERIMENT_1_DIR, PART_A_CACHE_PATH

    if args.dry_run:
        plan = validate_dry_run(
            PART_A_CACHE_PATH,
            EXPERIMENT_1_DIR,
            include_cfc=args.include_cfc,
        )
        print("Experiment 1 dry run")
        print(f"Part A cache: {PART_A_CACHE_PATH}")
        print(f"Planned models: {', '.join(plan['models'])}")
        print(f"Planned output directory: {EXPERIMENT_1_DIR}")
        print("No directories or files were created.")
        return 0

    run_experiment(include_cfc=args.include_cfc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
