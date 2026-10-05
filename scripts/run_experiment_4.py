"""Experiment 4: MLP-ODE versus KAN-ODE parameter efficiency."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiment_1 import (  # noqa: E402
    build_loaders,
    configuration_snapshot,
    validate_part_a_cache,
)

RESULT_FIELDS = [
    "family",
    "target_hidden_dim",
    "trainable_parameters",
    "test_auroc",
    "test_auprc",
    "test_brier_score",
    "sample_count",
    "best_validation_auprc",
    "epochs_completed",
    "mean_seconds_per_completed_epoch",
    "selected_kan_hidden_width",
    "selected_kan_width_multiplier",
    "selected_kan_grid",
    "parameter_difference_from_mlp",
    "absolute_parameter_difference_from_mlp",
    "checkpoint",
]


def build_parser() -> argparse.ArgumentParser:
    """Build an explicit dry-run/run CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 4: compare MLP-ODE and closest-count KAN-ODE "
            "configurations at hidden dimensions 8, 20, and 40."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate cache and candidate configuration without writing files.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Explicitly train models and create Experiment 4 outputs.",
    )
    return parser


def select_kan_candidate(
    target_parameter_count: int,
    hidden_dim: int,
    width_multipliers: tuple[int, ...],
    grid_sizes: tuple[int, ...],
    model_factory: Callable[..., Any] | None = None,
) -> tuple[dict[str, int], list[dict[str, int]]]:
    """Measure every valid candidate and deterministically choose the closest."""
    from src.config import SEED
    from src.utils import count_trainable_parameters, set_seed

    if target_parameter_count < 1 or hidden_dim < 1:
        raise ValueError("target_parameter_count and hidden_dim must be positive.")
    if not width_multipliers or not grid_sizes:
        raise ValueError("KAN candidate width and grid sets must be non-empty.")
    if model_factory is None:
        from src.models.kan_neural_ode import KANNeuralODE

        model_factory = KANNeuralODE

    candidates: list[dict[str, int]] = []
    for width_multiplier in sorted(set(width_multipliers)):
        for grid in sorted(set(grid_sizes)):
            if width_multiplier < 1 or grid < 1:
                raise ValueError("KAN candidate widths and grids must be positive.")
            set_seed(SEED)
            model = model_factory(
                hidden_dim=hidden_dim,
                width_multiplier=width_multiplier,
                grid=grid,
            )
            parameter_count = count_trainable_parameters(model)
            hidden_width = width_multiplier * hidden_dim + 1
            candidates.append(
                {
                    "hidden_dim": hidden_dim,
                    "width_multiplier": width_multiplier,
                    "hidden_width": hidden_width,
                    "grid": grid,
                    "parameter_count": parameter_count,
                    "parameter_difference": parameter_count
                    - target_parameter_count,
                    "absolute_parameter_difference": abs(
                        parameter_count - target_parameter_count
                    ),
                }
            )

    selected = min(
        candidates,
        key=lambda candidate: (
            candidate["absolute_parameter_difference"],
            candidate["hidden_width"],
            candidate["grid"],
            candidate["width_multiplier"],
        ),
    )
    return selected, candidates


def validate_dry_run(
    cache_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate Experiment 4 inputs/configuration without creating outputs."""
    from src.config import (
        KAN_CANDIDATE_GRIDS,
        KAN_CANDIDATE_WIDTH_MULTIPLIERS,
        PARAMETER_EFFICIENCY_HIDDEN_DIMS,
        validate_config,
    )

    validate_config()
    payload = validate_part_a_cache(cache_path)
    return {
        "hidden_dimensions": list(PARAMETER_EFFICIENCY_HIDDEN_DIMS),
        "kan_width_multipliers": list(KAN_CANDIDATE_WIDTH_MULTIPLIERS),
        "kan_grid_sizes": list(KAN_CANDIDATE_GRIDS),
        "split_counts": {
            split: len(payload["records"][split])
            for split in ("train", "validation", "test")
        },
        "output_dir": str(output_dir),
    }


def train_custom_model(
    model: Any,
    adapter_key: str,
    family: str,
    target_hidden_dim: int,
    checkpoint_path: Path,
    device: str,
    target_mlp_parameter_count: int,
    selected_kan: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Train/select/test one custom-capacity ODE model."""
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
    from src.utils import EarlyStopping, count_trainable_parameters, set_seed

    set_seed(SEED)
    model = model.to(device)
    parameter_count = count_trainable_parameters(model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    forward_fn = get_forward_adapter(adapter_key)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    train_loader, validation_loader, test_loader = build_loaders()
    epoch_seconds: list[float] = []

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
        early_stopping.step(
            float(validation_metrics["auprc"]),
            model=model,
            checkpoint_path=checkpoint_path,
        )
        if early_stopping.should_stop:
            break

    if not math.isfinite(early_stopping.best_score):
        raise RuntimeError(f"{family} never produced a finite validation AUPRC.")
    early_stopping.load_best(
        model,
        checkpoint_path=checkpoint_path,
        map_location=device,
    )
    test_metrics = evaluate(model, test_loader, forward_fn, device)
    difference = parameter_count - target_mlp_parameter_count

    return {
        "family": family,
        "target_hidden_dim": target_hidden_dim,
        "trainable_parameters": parameter_count,
        "test_auroc": float(test_metrics["auroc"]),
        "test_auprc": float(test_metrics["auprc"]),
        "test_brier_score": float(test_metrics["brier_score"]),
        "sample_count": int(test_metrics["sample_count"]),
        "best_validation_auprc": early_stopping.best_score,
        "epochs_completed": len(epoch_seconds),
        "mean_seconds_per_completed_epoch": sum(epoch_seconds)
        / len(epoch_seconds),
        "selected_kan_hidden_width": (
            "" if selected_kan is None else selected_kan["hidden_width"]
        ),
        "selected_kan_width_multiplier": (
            "" if selected_kan is None else selected_kan["width_multiplier"]
        ),
        "selected_kan_grid": (
            "" if selected_kan is None else selected_kan["grid"]
        ),
        "parameter_difference_from_mlp": difference,
        "absolute_parameter_difference_from_mlp": abs(difference),
        "checkpoint": str(checkpoint_path),
    }


def save_parameter_plot(rows: list[dict[str, Any]], path: Path) -> None:
    """Plot test AUROC against actual measured trainable parameter counts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    styles = {
        "MLP-ODE": {"marker": "o", "linestyle": "-", "label": "MLP-ODE"},
        "KAN-ODE": {"marker": "s", "linestyle": "--", "label": "KAN-ODE"},
    }
    for family, style in styles.items():
        family_rows = sorted(
            (row for row in rows if row["family"] == family),
            key=lambda row: row["trainable_parameters"],
        )
        axis.plot(
            [row["trainable_parameters"] for row in family_rows],
            [row["test_auroc"] for row in family_rows],
            **style,
        )
    axis.set_xlabel("Actual trainable parameter count")
    axis.set_ylabel("Test AUROC")
    axis.set_title("MLP-ODE vs KAN-ODE parameter efficiency")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_experiment() -> list[dict[str, Any]]:
    """Execute Experiment 4 using the shared Experiment 1 protocol."""
    from src.config import (
        EXPERIMENT_4_DIR,
        KAN_CANDIDATE_GRIDS,
        KAN_CANDIDATE_WIDTH_MULTIPLIERS,
        ODE_WIDTH,
        PARAMETER_EFFICIENCY_HIDDEN_DIMS,
        PART_A_CACHE_PATH,
        SEED,
        resolve_device,
    )
    from src.models.kan_neural_ode import KANNeuralODE
    from src.models.mlp_neural_ode import MLPNeuralODE
    from src.utils import (
        count_trainable_parameters,
        package_versions,
        set_seed,
        write_csv,
        write_json,
    )

    validate_dry_run(PART_A_CACHE_PATH, EXPERIMENT_4_DIR)
    device = resolve_device()
    checkpoint_dir = EXPERIMENT_4_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    selection_metadata: list[dict[str, Any]] = []

    for hidden_dim in PARAMETER_EFFICIENCY_HIDDEN_DIMS:
        set_seed(SEED)
        mlp_model = MLPNeuralODE(hidden_dim=hidden_dim, width=ODE_WIDTH)
        mlp_parameter_count = count_trainable_parameters(mlp_model)
        selected, candidates = select_kan_candidate(
            target_parameter_count=mlp_parameter_count,
            hidden_dim=hidden_dim,
            width_multipliers=KAN_CANDIDATE_WIDTH_MULTIPLIERS,
            grid_sizes=KAN_CANDIDATE_GRIDS,
        )
        selection_metadata.append(
            {
                "target_hidden_dim": hidden_dim,
                "mlp_parameter_count": mlp_parameter_count,
                "selected": selected,
                "all_candidates": candidates,
            }
        )

        mlp_checkpoint = (
            checkpoint_dir / f"mlp_ode_hidden{hidden_dim}_best.pt"
        )
        rows.append(
            train_custom_model(
                model=mlp_model,
                adapter_key="mlp_neural_ode",
                family="MLP-ODE",
                target_hidden_dim=hidden_dim,
                checkpoint_path=mlp_checkpoint,
                device=device,
                target_mlp_parameter_count=mlp_parameter_count,
            )
        )

        set_seed(SEED)
        kan_model = KANNeuralODE(
            hidden_dim=hidden_dim,
            width_multiplier=selected["width_multiplier"],
            grid=selected["grid"],
        )
        kan_checkpoint = checkpoint_dir / (
            f"kan_ode_hidden{hidden_dim}_width"
            f"{selected['hidden_width']}_grid{selected['grid']}_best.pt"
        )
        rows.append(
            train_custom_model(
                model=kan_model,
                adapter_key="kan_neural_ode",
                family="KAN-ODE",
                target_hidden_dim=hidden_dim,
                checkpoint_path=kan_checkpoint,
                device=device,
                target_mlp_parameter_count=mlp_parameter_count,
                selected_kan=selected,
            )
        )

    write_csv(
        EXPERIMENT_4_DIR / "parameter_efficiency_results.csv",
        rows,
        RESULT_FIELDS,
    )
    save_parameter_plot(
        rows,
        EXPERIMENT_4_DIR / "parameter_efficiency_auroc_vs_params.png",
    )
    metadata = {
        "experiment": 4,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "device": device,
        "configuration": configuration_snapshot(),
        "target_hidden_dimensions": list(PARAMETER_EFFICIENCY_HIDDEN_DIMS),
        "candidate_width_multipliers": list(
            KAN_CANDIDATE_WIDTH_MULTIPLIERS
        ),
        "candidate_grid_sizes": list(KAN_CANDIDATE_GRIDS),
        "candidate_selection": selection_metadata,
        "parameter_counts_exactly_matched": all(
            selection["selected"]["absolute_parameter_difference"] == 0
            for selection in selection_metadata
        ),
        "package_versions": package_versions(
            [
                "torch",
                "numpy",
                "scikit-learn",
                "torchdiffeq",
                "pykan",
                "matplotlib",
            ]
        ),
    }
    write_json(EXPERIMENT_4_DIR / "run_metadata.json", metadata)
    return rows


def main(argv: list[str] | None = None) -> int:
    """Validate or explicitly execute Experiment 4."""
    args = build_parser().parse_args(argv)
    from src.config import EXPERIMENT_4_DIR, PART_A_CACHE_PATH

    if args.dry_run:
        plan = validate_dry_run(PART_A_CACHE_PATH, EXPERIMENT_4_DIR)
        print("Experiment 4 dry run")
        print(
            "Target hidden dimensions: "
            + ", ".join(str(value) for value in plan["hidden_dimensions"])
        )
        print(
            "KAN width multipliers: "
            + ", ".join(str(value) for value in plan["kan_width_multipliers"])
        )
        print(
            "KAN grid sizes: "
            + ", ".join(str(value) for value in plan["kan_grid_sizes"])
        )
        print(f"Planned output directory: {EXPERIMENT_4_DIR}")
        print("No directories or files were created.")
        return 0

    run_experiment()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
