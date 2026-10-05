"""Temporary-only smoke test for Part F Experiments 3 and 4."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.dont_write_bytecode = True

from scripts.run_experiment_3 import (  # noqa: E402
    kan_input_labels,
    validate_dry_run as validate_experiment_3,
)
from scripts.run_experiment_4 import (  # noqa: E402
    select_kan_candidate,
    validate_dry_run as validate_experiment_4,
)
from src.config import N_VARS, SEED  # noqa: E402


class TinyCandidate(nn.Module):
    """Tiny parameter-count-only stand-in; it is never trained or evaluated."""

    def __init__(self, parameter_count: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(parameter_count))


def _candidate_factory(
    hidden_dim: int,
    width_multiplier: int,
    grid: int,
) -> TinyCandidate:
    """Create deterministic candidate counts from tiny in-memory objects."""
    del hidden_dim
    return TinyCandidate(parameter_count=10 * width_multiplier + grid)


def _write_minimal_cache(path: Path) -> None:
    """Write only the Part A schema needed by dry-run validation."""
    records = {
        "train": [{"record_id": "1"}],
        "validation": [{"record_id": "2"}],
        "test": [{"record_id": "3"}],
    }
    payload = {
        "records": records,
        "split_record_ids": {
            split: [record["record_id"] for record in split_records]
            for split, split_records in records.items()
        },
        "variables": [f"variable_{index}" for index in range(N_VARS)],
        "seed": SEED,
    }
    path.parent.mkdir(parents=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def main() -> int:
    """Check deterministic selection, truthful labels, and non-writing plans."""
    first, first_candidates = select_kan_candidate(
        target_parameter_count=22,
        hidden_dim=4,
        width_multipliers=(1, 2),
        grid_sizes=(3, 5),
        model_factory=_candidate_factory,
    )
    second, second_candidates = select_kan_candidate(
        target_parameter_count=22,
        hidden_dim=4,
        width_multipliers=(1, 2),
        grid_sizes=(3, 5),
        model_factory=_candidate_factory,
    )
    assert first == second
    assert first_candidates == second_candidates
    assert first["parameter_count"] == 23
    assert first["absolute_parameter_difference"] == 1
    assert first["width_multiplier"] == 2
    assert first["grid"] == 3

    labels = kan_input_labels(hidden_dim=4)
    assert labels == [
        "latent_state_0",
        "latent_state_1",
        "latent_state_2",
        "latent_state_3",
        "time_hours",
    ]
    clinical_names = {"HR", "GCS", "Glucose", "Lactate", "Creatinine"}
    assert clinical_names.isdisjoint(labels)

    temporary_root: Path
    with TemporaryDirectory(prefix="echw2_part_f34_smoke_") as temporary_name:
        temporary_root = Path(temporary_name)
        cache_path = temporary_root / "inputs" / "part_a_cache.pkl"
        checkpoint_path = temporary_root / "inputs" / "kan_best.pt"
        experiment_3_output = temporary_root / "outputs" / "experiment_3"
        experiment_4_output = temporary_root / "outputs" / "experiment_4"
        _write_minimal_cache(cache_path)
        checkpoint_path.write_bytes(b"temporary")

        validate_experiment_3(
            cache_path,
            checkpoint_path,
            experiment_3_output,
            validate_checkpoint_state=False,
            validate_capabilities=False,
        )
        assert not experiment_3_output.exists()

        validate_experiment_4(cache_path, experiment_4_output)
        assert not experiment_4_output.exists()

    assert not temporary_root.exists()
    print("PASS: Part F Experiments 3 and 4 smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
