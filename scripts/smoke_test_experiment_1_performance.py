"""Synthetic regression check for Experiment 1 materialized input tensors."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.dont_write_bytecode = True

from scripts.run_experiment_1 import collate_for_experiment  # noqa: E402
from src.config import N_VARS, SEED  # noqa: E402
from src.dataset import (  # noqa: E402
    ICUMortalityDataset,
    collate_materialized_stays,
    materialize_records,
)
from src.evaluate import compute_mortality_metrics  # noqa: E402
from src.models.gru_baseline import (  # noqa: E402
    GRUBaseline,
    prepare_hourly_gru_inputs,
)
from src.train import forward_gru  # noqa: E402
from src.utils import set_seed  # noqa: E402


def _synthetic_records() -> list[dict[str, object]]:
    """Create varied irregular stays, including same-hour observations."""
    records: list[dict[str, object]] = []
    labels = [0, 1, 0, 1]
    patient_times = [
        [0.0, 0.5, 1.25, 8.0],
        [0.1, 2.0, 2.75, 12.5, 47.9],
        [0.0, 5.5, 24.0],
        [1.0, 1.2, 1.8, 6.0, 18.0, 36.0],
    ]
    for patient_index, raw_times in enumerate(patient_times):
        times = np.asarray(raw_times, dtype=np.float64)
        values = np.zeros((len(times), N_VARS), dtype=np.float32)
        mask = np.zeros_like(values)
        for row in range(len(times)):
            first_column = (patient_index + row) % N_VARS
            second_column = (first_column + 3) % N_VARS
            values[row, first_column] = patient_index + row / 10.0
            values[row, second_column] = -patient_index - row / 20.0
            mask[row, first_column] = 1.0
            mask[row, second_column] = 1.0
        records.append(
            {
                "record_id": str(900000 + patient_index),
                "times": times,
                "values": values,
                "mask": mask,
                "delta_t": np.zeros_like(values),
                "label": labels[patient_index],
            }
        )
    return records


def _assert_metric_equivalence(
    original: dict[str, float | int | str | None],
    optimized: dict[str, float | int | str | None],
) -> None:
    """Compare scalar metric outputs at a strict numerical tolerance."""
    for key in ("auroc", "auprc", "brier_score"):
        assert math.isclose(
            float(original[key]),
            float(optimized[key]),
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
    assert original["sample_count"] == optimized["sample_count"]
    assert original["note"] == optimized["note"]


def main() -> int:
    """Verify old and materialized preparation paths are semantically equal."""
    records = _synthetic_records()
    original_dataset = ICUMortalityDataset(records, split="train")
    original_items = [
        original_dataset[index] for index in range(len(original_dataset))
    ]
    original_batch = collate_for_experiment(original_items)

    materialized_dataset = materialize_records(records, split="train")
    materialized_items = [
        materialized_dataset[index]
        for index in range(len(materialized_dataset))
    ]
    optimized_batch = collate_materialized_stays(materialized_items)

    for key in (
        "times",
        "values",
        "mask",
        "delta_t",
        "valid_time_mask",
        "lengths",
        "label",
    ):
        assert torch.equal(original_batch[key], optimized_batch[key]), key
    assert original_batch["record_id"] == optimized_batch["record_id"]

    original_hourly_values, original_hourly_mask = prepare_hourly_gru_inputs(
        original_batch["times"],
        original_batch["values"],
        original_batch["mask"],
        original_batch["valid_time_mask"],
    )
    assert torch.equal(
        original_hourly_values,
        optimized_batch["hourly_values"],
    )
    assert torch.equal(original_hourly_mask, optimized_batch["hourly_mask"])

    optimized_gru_batch = collate_materialized_stays(
        materialized_items,
        model_key="gru_baseline",
    )
    assert set(optimized_gru_batch) == {
        "hourly_values",
        "hourly_mask",
        "label",
    }
    set_seed(SEED)
    model = GRUBaseline().eval()
    with torch.no_grad():
        original_logits = model(original_hourly_values, original_hourly_mask)
        optimized_logits = forward_gru(model, optimized_gru_batch)
    assert torch.allclose(
        original_logits,
        optimized_logits,
        rtol=1e-7,
        atol=1e-7,
    )

    original_metrics = compute_mortality_metrics(
        original_batch["label"],
        original_logits,
    )
    optimized_metrics = compute_mortality_metrics(
        optimized_gru_batch["label"],
        optimized_logits,
    )
    _assert_metric_equivalence(original_metrics, optimized_metrics)

    repetitions = 20
    started = time.perf_counter()
    for _ in range(repetitions):
        legacy = collate_for_experiment(original_items)
        prepare_hourly_gru_inputs(
            legacy["times"],
            legacy["values"],
            legacy["mask"],
            legacy["valid_time_mask"],
        )
    original_seconds = time.perf_counter() - started

    started = time.perf_counter()
    for _ in range(repetitions):
        collate_materialized_stays(
            materialized_items,
            model_key="gru_baseline",
        )
    optimized_seconds = time.perf_counter() - started

    print(
        "PASS: Experiment 1 materialized tensors, lengths, labels, logits, "
        "and metrics match the original path"
    )
    print(
        "Synthetic repeated-preparation timing: "
        f"original={original_seconds:.4f}s, optimized={optimized_seconds:.4f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
