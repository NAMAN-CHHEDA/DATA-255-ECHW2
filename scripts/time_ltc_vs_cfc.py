"""Part D timing: wall-clock cost of LTC versus CfC for a few epochs.

Trains each cell type from scratch for a fixed small number of epochs under
the Experiment 1 protocol (same data, seed, optimizer, clipping) and writes a
separate timing file. It does not touch the Experiment 1 headline table.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiment_1 import (  # noqa: E402
    build_loaders,
    build_model,
    materialize_experiment_datasets,
)

FIELDS = [
    "model",
    "trainable_parameters",
    "epochs",
    "mean_train_seconds_per_epoch",
    "mean_train_val_seconds_per_epoch",
    "total_seconds",
    "final_train_loss",
    "validation_auprc_last_epoch",
    "device",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Print the plan only.")
    mode.add_argument("--run", action="store_true", help="Train and write timings.")
    parser.add_argument("--epochs", type=int, default=3)
    return parser


def time_cell(
    model_key: str,
    epochs: int,
    device: str,
    materialized: dict[str, Any],
) -> dict[str, Any]:
    import torch

    from src.config import LEARNING_RATE, SEED, WEIGHT_DECAY
    from src.evaluate import evaluate
    from src.train import get_forward_adapter, train_one_epoch
    from src.utils import count_trainable_parameters, set_seed

    set_seed(SEED)
    model = build_model(model_key).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    generator = torch.Generator()
    generator.manual_seed(SEED)
    train_loader, validation_loader, _ = build_loaders(
        generator=generator,
        materialized_splits=materialized,
        model_key=model_key,
        pin_memory=torch.device(device).type == "cuda",
    )
    forward_fn = get_forward_adapter(model_key)
    train_seconds: list[float] = []
    epoch_seconds: list[float] = []
    loss = float("nan")
    auprc = float("nan")
    started = time.perf_counter()
    for _ in range(epochs):
        epoch_started = time.perf_counter()
        loss, seconds = train_one_epoch(
            model, train_loader, optimizer, forward_fn, device
        )
        auprc = float(evaluate(model, validation_loader, forward_fn, device)["auprc"])
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        train_seconds.append(seconds)
        epoch_seconds.append(time.perf_counter() - epoch_started)
        print(
            f"{model_key} epoch | train={seconds:.2f}s | "
            f"train+val={epoch_seconds[-1]:.2f}s | loss={loss:.4f} | "
            f"val_AUPRC={auprc:.4f}",
            flush=True,
        )
    return {
        "model": model_key.upper(),
        "trainable_parameters": count_trainable_parameters(model),
        "epochs": epochs,
        "mean_train_seconds_per_epoch": sum(train_seconds) / epochs,
        "mean_train_val_seconds_per_epoch": sum(epoch_seconds) / epochs,
        "total_seconds": time.perf_counter() - started,
        "final_train_loss": loss,
        "validation_auprc_last_epoch": auprc,
        "device": device,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    from src.config import EXPERIMENT_1_DIR, PART_A_CACHE_PATH, resolve_device

    if args.dry_run:
        print(f"Would time ltc and cfc for {args.epochs} epochs each.")
        print(f"Output directory: {EXPERIMENT_1_DIR}")
        return 0

    from src.utils import write_csv, write_json

    device = resolve_device()
    materialized, _ = materialize_experiment_datasets(PART_A_CACHE_PATH)
    rows = [
        time_cell(key, args.epochs, device, materialized) for key in ("ltc", "cfc")
    ]
    write_csv(EXPERIMENT_1_DIR / "ltc_vs_cfc_timing.csv", rows, FIELDS)
    write_json(EXPERIMENT_1_DIR / "ltc_vs_cfc_timing.json", rows)
    ratio = rows[0]["mean_train_seconds_per_epoch"] / rows[1]["mean_train_seconds_per_epoch"]
    print(f"LTC / CfC train-time ratio: {ratio:.2f}x", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
