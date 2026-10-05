"""Experiment 3: post-training interpretation of the KAN ODE vector field."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiment_1 import (  # noqa: E402
    configuration_snapshot,
    validate_part_a_cache,
)


def build_parser() -> argparse.ArgumentParser:
    """Build an explicitly protected interpretation CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 3: inspect the trained Experiment 1 KAN vector field "
            "using PyKAN activations and feature attribution."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate cache/checkpoint/capabilities without writing files.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Explicitly create Experiment 3 interpretation artifacts.",
    )
    return parser


def kan_input_labels(hidden_dim: int) -> list[str]:
    """Truthful labels for ``[hidden_state, scalar_time]`` KAN inputs."""
    if hidden_dim < 1:
        raise ValueError("hidden_dim must be positive.")
    return [
        *[f"latent_state_{index}" for index in range(hidden_dim)],
        "time_hours",
    ]


def require_pykan_capabilities(network: Any) -> str:
    """Fail clearly when real activation/attribution APIs are unavailable."""
    try:
        version = importlib.metadata.version("pykan")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "PyKAN is not installed; Experiment 3 requires its attribution API."
        ) from exc
    required_methods = [
        "get_act",
        "attribute",
        "plot",
        "prune",
        "auto_symbolic",
        "symbolic_formula",
    ]
    missing = [
        method for method in required_methods if not callable(getattr(network, method, None))
    ]
    if not hasattr(type(network), "feature_score"):
        missing.append("feature_score")
    if missing:
        raise RuntimeError(
            f"PyKAN {version} lacks required interpretability capability: "
            + ", ".join(missing)
        )
    return version


def validate_dry_run(
    cache_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    validate_checkpoint_state: bool = True,
    validate_capabilities: bool = True,
) -> dict[str, Any]:
    """Validate Experiment 3 without creating ``output_dir``."""
    from src.config import ODE_HIDDEN_DIM, validate_config

    validate_config()
    payload = validate_part_a_cache(cache_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Experiment 1 KAN checkpoint is absent: {checkpoint_path}"
        )

    pykan_version: str | None = None
    if validate_checkpoint_state or validate_capabilities:
        from src.models.kan_neural_ode import KANNeuralODE

        model = KANNeuralODE()
        if validate_checkpoint_state:
            state_dict = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state_dict)
        if validate_capabilities:
            pykan_version = require_pykan_capabilities(model.ode_func.network)

    return {
        "checkpoint": str(checkpoint_path),
        "test_patients": len(payload["records"]["test"]),
        "input_labels": kan_input_labels(ODE_HIDDEN_DIM),
        "pykan_version": pykan_version,
        "output_dir": str(output_dir),
    }


def build_probe_set(model: Any, device: str) -> torch.Tensor:
    """Encode held-out patients and append a scalar ``t=48`` KAN input."""
    from torch.utils.data import DataLoader

    from scripts.run_experiment_1 import collate_for_experiment
    from src.config import BATCH_SIZE, NUM_WORKERS
    from src.dataset import load_split

    loader = DataLoader(
        load_split("test"),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_for_experiment,
    )
    representations: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            hidden = model.encode_initial_state(
                batch["values"].to(device),
                batch["mask"].to(device),
                batch["valid_time_mask"].to(device),
            )
            time_column = hidden.new_full((hidden.shape[0], 1), 48.0)
            representations.append(torch.cat([hidden, time_column], dim=-1))
    if not representations:
        raise RuntimeError("Held-out test split produced no probe representations.")
    return torch.cat(representations, dim=0)


def collect_feature_attribution(
    network: Any,
    probe_inputs: torch.Tensor,
) -> tuple[list[float], str]:
    """Run PyKAN activation collection and return its real feature scores."""
    version = require_pykan_capabilities(network)
    network.get_act(probe_inputs)
    network.attribute(plot=False)
    scores = network.feature_score
    if not isinstance(scores, torch.Tensor):
        raise RuntimeError(
            f"PyKAN {version} feature_score did not return a tensor."
        )
    flat_scores = scores.detach().reshape(-1).cpu()
    if flat_scores.numel() != probe_inputs.shape[1]:
        raise RuntimeError(
            f"PyKAN {version} returned {flat_scores.numel()} feature scores "
            f"for {probe_inputs.shape[1]} vector-field inputs."
        )
    if not torch.isfinite(flat_scores).all():
        raise RuntimeError(f"PyKAN {version} returned non-finite feature scores.")
    return [float(value) for value in flat_scores.tolist()], version


def reflection_template(selected_labels: list[str]) -> str:
    """Return a facts-and-placeholders-only Experiment 3 reflection template."""
    return "\n".join(
        [
            "# Experiment 3 reflection template",
            "",
            "- Required length: 150–300 words.",
            f"- Selected dimension 1: `{selected_labels[0]}`.",
            f"- Selected dimension 2: `{selected_labels[1]}`.",
            "- [CITE ACTUAL EDGE-PLOT EVIDENCE FOR DIMENSION 1]",
            "- [CITE ACTUAL EDGE-PLOT EVIDENCE FOR DIMENSION 2]",
            "- [EXPLAIN THE TWO SELECTED LATENT/TIME FUNCTIONS]",
            "- [COMPARE THE OBSERVED KAN BEHAVIOR WITH MLP-ODE]",
            "- [COMPARE THE OBSERVED KAN BEHAVIOR WITH LTC]",
            (
                "- Do not claim direct clinical attribution to a raw ICU variable "
                "unless the architecture actually supplies that mapping."
            ),
            "",
        ]
    )


def run_experiment() -> dict[str, Any]:
    """Create Experiment 3 artifacts from the trained KAN checkpoint."""
    import matplotlib

    matplotlib.use("Agg")

    from src.config import (
        EXPERIMENT_1_DIR,
        EXPERIMENT_3_DIR,
        PART_A_CACHE_PATH,
        SEED,
        resolve_device,
    )
    from src.models.kan_neural_ode import KANNeuralODE, SYMBOLIC_LIBRARY
    from src.utils import package_versions, write_csv, write_json, write_text

    checkpoint_path = (
        EXPERIMENT_1_DIR / "checkpoints" / "kan_neural_ode_best.pt"
    )
    validate_dry_run(
        PART_A_CACHE_PATH,
        checkpoint_path,
        EXPERIMENT_3_DIR,
    )
    device = resolve_device()
    model = KANNeuralODE().to(device)
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    probe_inputs = build_probe_set(model, device)
    network = model.ode_func.network
    attribution_values, pykan_version = collect_feature_attribution(
        network,
        probe_inputs,
    )
    labels = kan_input_labels(model.hidden_dim)
    ranked_indices = sorted(
        range(len(attribution_values)),
        key=lambda index: (-attribution_values[index], index),
    )
    selected_indices = ranked_indices[:2]
    selected_labels = [labels[index] for index in selected_indices]

    output_dir = EXPERIMENT_3_DIR
    edge_directory = output_dir / "kan_edge_functions"
    edge_directory.mkdir(parents=True, exist_ok=True)
    model.plot_learned_edge_functions(
        beta=100,
        folder=str(edge_directory),
        metric="backward",
        in_vars=labels,
        out_vars=[
            f"latent_derivative_{index}" for index in range(model.hidden_dim)
        ],
    )

    isolated_network = copy.deepcopy(network).to("cpu")
    isolated_probe = probe_inputs.detach().cpu()
    isolated_network.get_act(isolated_probe)
    isolated_network.attribute(plot=False)
    pruned_network = isolated_network.prune()
    if pruned_network is None:
        pruned_network = isolated_network
    pruned_network.get_act(isolated_probe)
    pruned_network.auto_symbolic(lib=SYMBOLIC_LIBRARY, verbose=0)
    formulas = pruned_network.symbolic_formula(var=labels)

    top_rows = [
        {
            "rank": rank,
            "input_index": input_index,
            "input_label": labels[input_index],
            "attribution_value": attribution_values[input_index],
        }
        for rank, input_index in enumerate(selected_indices, start=1)
    ]
    write_csv(
        output_dir / "top_two_input_dimensions.csv",
        top_rows,
        ["rank", "input_index", "input_label", "attribution_value"],
    )
    write_text(output_dir / "symbolic_formulas.txt", f"{formulas}\n")
    write_text(
        output_dir / "reflection_e3_template.md",
        reflection_template(selected_labels),
    )
    metadata = {
        "experiment": 3,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "checkpoint": str(checkpoint_path),
        "probe_source": "held-out test encoder representations at t=48 hours",
        "probe_count": int(probe_inputs.shape[0]),
        "ranking_method": (
            "PyKAN feature_score after get_act(probe_inputs) and "
            "attribute(plot=False)"
        ),
        "pykan_version": pykan_version,
        "input_labels": labels,
        "raw_attribution_values": attribution_values,
        "ranked_input_indices": ranked_indices,
        "selected_input_indices": selected_indices,
        "selected_input_labels": selected_labels,
        "edge_figure_directory": str(edge_directory),
        "symbolic_library": SYMBOLIC_LIBRARY,
        "symbolic_and_pruning_used_isolated_copy": True,
        "configuration": configuration_snapshot(),
        "package_versions": package_versions(
            ["torch", "numpy", "pykan", "sympy", "matplotlib"]
        ),
        "device": device,
    }
    write_json(output_dir / "interpretability_metadata.json", metadata)
    return metadata


def main(argv: list[str] | None = None) -> int:
    """Validate or explicitly execute Experiment 3."""
    args = build_parser().parse_args(argv)
    from src.config import EXPERIMENT_1_DIR, EXPERIMENT_3_DIR, PART_A_CACHE_PATH

    checkpoint_path = (
        EXPERIMENT_1_DIR / "checkpoints" / "kan_neural_ode_best.pt"
    )
    if args.dry_run:
        plan = validate_dry_run(
            PART_A_CACHE_PATH,
            checkpoint_path,
            EXPERIMENT_3_DIR,
        )
        print("Experiment 3 dry run")
        print(f"Checkpoint: {plan['checkpoint']}")
        print(f"PyKAN version: {plan['pykan_version']}")
        print(f"Input labels: {', '.join(plan['input_labels'])}")
        print(f"Planned output directory: {EXPERIMENT_3_DIR}")
        print("No directories or files were created.")
        return 0

    run_experiment()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
