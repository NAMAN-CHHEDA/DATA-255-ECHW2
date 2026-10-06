"""Temporary-only smoke test for Part F Experiments 1 and 2."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.dont_write_bytecode = True

from scripts.run_experiment_1 import validate_dry_run as validate_experiment_1  # noqa: E402
from scripts.run_experiment_2 import (  # noqa: E402
    drop_record_timestamps,
    validate_dry_run as validate_experiment_2,
)
from src.config import N_VARS, SEED  # noqa: E402


def _synthetic_record() -> dict[str, object]:
    """Create one irregular normalized record with four timestamps."""
    times = np.asarray([0.0, 1.0, 3.0, 6.0], dtype=np.float64)
    values = np.zeros((4, N_VARS), dtype=np.float32)
    mask = np.zeros_like(values)
    values[0, 0], mask[0, 0] = 0.25, 1.0
    values[1, 1], mask[1, 1] = -0.50, 1.0
    values[2, 0], mask[2, 0] = 0.75, 1.0
    values[3, 1], mask[3, 1] = 1.00, 1.0
    return {
        "record_id": "123456",
        "times": times,
        "values": values,
        "mask": mask,
        "delta_t": np.zeros_like(values),
        "label": 1,
    }


def _expected_delta_t(times: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Calculate a direct expected gap matrix for smoke-test assertions."""
    expected = np.zeros_like(mask)
    last_observed: list[float | None] = [None] * mask.shape[1]
    for row, current_time in enumerate(times):
        for column in range(mask.shape[1]):
            previous = last_observed[column]
            expected[row, column] = (
                current_time if previous is None else current_time - previous
            )
            if mask[row, column] == 1:
                last_observed[column] = float(current_time)
    return expected


def _write_minimal_cache(path: Path) -> None:
    """Write only the schema needed by dry-run validation."""
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
    """Check timestamp dropping and non-writing dry-run validation."""
    original = _synthetic_record()
    first = drop_record_timestamps(original, retained_pct=50, seed=SEED)
    second = drop_record_timestamps(original, retained_pct=50, seed=SEED)
    assert np.array_equal(first["times"], second["times"])
    assert np.array_equal(first["values"], second["values"])
    assert len(first["times"]) >= 1
    assert np.all(np.diff(first["times"]) > 0)
    assert np.allclose(
        first["delta_t"],
        _expected_delta_t(first["times"], first["mask"]),
    )

    minimum = drop_record_timestamps(original, retained_pct=25, seed=SEED)
    assert len(minimum["times"]) >= 1
    assert np.all(np.diff(minimum["times"]) > 0)

    temporary_root: Path
    with TemporaryDirectory(prefix="echw2_part_f12_smoke_") as temporary_name:
        temporary_root = Path(temporary_name)
        cache_path = temporary_root / "input" / "part_a_cache.pkl"
        experiment_1_dir = temporary_root / "experiment_1_inputs"
        experiment_1_output = temporary_root / "outputs" / "experiment_1"
        experiment_2_output = temporary_root / "outputs" / "experiment_2"
        _write_minimal_cache(cache_path)

        validate_experiment_1(
            cache_path,
            experiment_1_output,
            include_cfc=False,
        )
        assert not experiment_1_output.exists()

        checkpoint_dir = experiment_1_dir / "checkpoints"
        for model_key in (
            "gru_baseline",
            "mlp_neural_ode",
            "kan_neural_ode",
            "ltc",
        ):
            model_checkpoint_dir = checkpoint_dir / model_key
            model_checkpoint_dir.mkdir(parents=True)
            (model_checkpoint_dir / "best.pt").write_bytes(b"temporary")

        validate_experiment_2(
            cache_path,
            experiment_1_dir,
            experiment_2_output,
            include_cfc=False,
            validate_state_dicts=False,
        )
        assert not experiment_2_output.exists()

    assert not temporary_root.exists()
    print("PASS: Part F Experiments 1 and 2 smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
