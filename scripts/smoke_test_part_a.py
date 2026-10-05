"""Smoke-test Part A preprocessing using only temporary synthetic files.

This script exercises the real preprocessing pipeline. It does not download
PhysioNet data and does not write into the project's data or output folders.
"""

from __future__ import annotations

import csv
import io
import json
import pickle
import sys
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# Avoid leaving import caches in the project when this standalone check runs.
sys.dont_write_bytecode = True

import src.data_preprocessing as preprocessing  # noqa: E402


PATIENT_COUNT = 20
FIRST_RECORD_ID = 100000


def _raw_value(patient_index: int, variable_index: int) -> float:
    """Return a deterministic observed value for a toy admission."""
    return 60.0 + 10.0 * variable_index + float(patient_index**2)


def _write_outcomes(path: Path) -> None:
    """Write balanced binary labels in Outcomes A format."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["RecordID", "In-hospital_death"])
        for patient_index in range(PATIENT_COUNT):
            writer.writerow(
                [FIRST_RECORD_ID + patient_index, patient_index % 2]
            )


def _write_patient(path: Path, patient_index: int) -> None:
    """Write one small Set A-style patient file."""
    record_id = FIRST_RECORD_ID + patient_index
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Time", "Parameter", "Value"])
        writer.writerow(["00:00", "RecordID", record_id])

        # Every selected variable is observed at admission, ensuring that each
        # train-only normalization statistic is defined.
        for variable_index, variable in enumerate(
            preprocessing.SELECTED_VARIABLES
        ):
            writer.writerow(
                [
                    "00:00",
                    variable,
                    _raw_value(patient_index, variable_index),
                ]
            )

        # At 01:00 only HR is observed, creating missing-variable cells.
        writer.writerow(["01:00", "HR", 80.0 + patient_index])
        if patient_index == 0:
            # The expected grouped value is mean(80, 100) = 90.
            writer.writerow(["01:00", "HR", 100.0])

        # At 03:00 only Temp is observed, leaving all other variables missing.
        writer.writerow(["03:00", "Temp", 36.0 + 0.1 * patient_index])


def _observed_raw_values(record_id: str, variable: str) -> list[float]:
    """Return the independently known toy observations for one variable."""
    patient_index = int(record_id) - FIRST_RECORD_ID
    variable_index = preprocessing.SELECTED_VARIABLES.index(variable)
    values = [_raw_value(patient_index, variable_index)]
    if variable == "HR":
        values.append(90.0 if patient_index == 0 else 80.0 + patient_index)
    elif variable == "Temp":
        values.append(36.0 + 0.1 * patient_index)
    return values


def _assert_normalization_uses_train_only(
    statistics: dict[str, Any],
    train_ids: list[str],
) -> None:
    """Compare saved statistics with independent train-only expectations."""
    expected_means: list[float] = []
    expected_stds: list[float] = []
    expected_counts: list[int] = []

    for variable in preprocessing.SELECTED_VARIABLES:
        observed = np.asarray(
            [
                value
                for record_id in train_ids
                for value in _observed_raw_values(record_id, variable)
            ],
            dtype=np.float64,
        )
        expected_means.append(float(observed.mean()))
        standard_deviation = float(observed.std())
        expected_stds.append(1.0 if standard_deviation == 0.0 else standard_deviation)
        expected_counts.append(int(observed.size))

    assert statistics["computed_from_split"] == "train"
    assert statistics["variables"] == list(preprocessing.SELECTED_VARIABLES)
    assert np.allclose(statistics["mean"], expected_means)
    assert np.allclose(statistics["std"], expected_stds)
    assert statistics["observed_train_count"] == expected_counts


def _assert_patient_tensors(
    patient: dict[str, Any],
    statistics: dict[str, Any],
) -> None:
    """Check tensor invariants, duplicate averaging, gaps, and missing cells."""
    times = patient["times"]
    values = patient["values"]
    mask = patient["mask"]
    delta_t = patient["delta_t"]
    variable_count = len(preprocessing.SELECTED_VARIABLES)

    assert np.all(np.diff(times) > 0.0)
    assert np.all((times >= 0.0) & (times <= 48.0))
    assert values.shape == mask.shape == delta_t.shape
    assert values.shape == (len(times), variable_count)
    assert np.all((mask == 0.0) | (mask == 1.0))
    assert np.all(np.isfinite(delta_t))

    # Every variable was seen at 00:00. Therefore all gaps are 1 hour at the
    # 01:00 row. At 03:00 the HR gap is 2 hours and every other gap is 3.
    assert np.allclose(delta_t[0], np.zeros(variable_count))
    assert np.allclose(delta_t[1], np.ones(variable_count))
    expected_last_row = np.full(variable_count, 3.0)
    expected_last_row[preprocessing.SELECTED_VARIABLES.index("HR")] = 2.0
    assert np.allclose(delta_t[2], expected_last_row)

    # Missing entries must stay at the filler value zero after normalization.
    assert np.all(values[mask == 0.0] == 0.0)

    if patient["record_id"] == str(FIRST_RECORD_ID):
        hr_column = preprocessing.SELECTED_VARIABLES.index("HR")
        hr_mean = statistics["mean"][hr_column]
        hr_std = statistics["std"][hr_column]
        recovered_duplicate_mean = values[1, hr_column] * hr_std + hr_mean
        assert np.isclose(recovered_duplicate_mean, 90.0)


def main() -> int:
    """Run the temporary end-to-end Part A smoke test."""
    temporary_root: Path
    with TemporaryDirectory(prefix="echw2_part_a_smoke_") as temporary_name:
        temporary_root = Path(temporary_name)
        set_a_dir = temporary_root / "raw" / "set-a"
        outcomes_path = temporary_root / "raw" / "Outcomes-a.txt"
        processed_dir = temporary_root / "processed"
        set_a_dir.mkdir(parents=True)

        _write_outcomes(outcomes_path)
        for patient_index in range(PATIENT_COUNT):
            record_id = FIRST_RECORD_ID + patient_index
            _write_patient(set_a_dir / f"{record_id}.txt", patient_index)

        temporary_paths = {
            "SET_A_DIR": set_a_dir,
            "OUTCOMES_PATH": outcomes_path,
            "PROCESSED_DATA_DIR": processed_dir,
            "PART_A_CACHE_PATH": processed_dir / "part_a_cache.pkl",
            "SPLITS_PATH": processed_dir / "splits.json",
            "NORMALIZATION_STATS_PATH": processed_dir
            / "normalization_stats.json",
            "PREPROCESSING_SUMMARY_PATH": processed_dir
            / "preprocessing_summary.json",
        }
        original_paths = {
            name: getattr(preprocessing, name) for name in temporary_paths
        }

        try:
            for name, path in temporary_paths.items():
                setattr(preprocessing, name, path)

            # Suppress the pipeline's normal data summary so this smoke test
            # prints only its short PASS line on success.
            with redirect_stdout(io.StringIO()):
                preprocessing.preprocess_dataset()

            assert all(path.is_file() for path in temporary_paths.values() if path.suffix)
            with temporary_paths["PART_A_CACHE_PATH"].open("rb") as handle:
                cache = pickle.load(handle)
            with temporary_paths["SPLITS_PATH"].open(
                "r", encoding="utf-8"
            ) as handle:
                split_file = json.load(handle)
            with temporary_paths["NORMALIZATION_STATS_PATH"].open(
                "r", encoding="utf-8"
            ) as handle:
                statistics = json.load(handle)

            split_ids = split_file["record_ids"]
            split_sets = {
                split: set(record_ids)
                for split, record_ids in split_ids.items()
            }
            assert split_sets["train"].isdisjoint(split_sets["validation"])
            assert split_sets["train"].isdisjoint(split_sets["test"])
            assert split_sets["validation"].isdisjoint(split_sets["test"])
            assert set().union(*split_sets.values()) == {
                str(FIRST_RECORD_ID + index)
                for index in range(PATIENT_COUNT)
            }
            preprocessing.validate_splits(
                split_ids,
                set().union(*split_sets.values()),
            )

            _assert_normalization_uses_train_only(
                statistics,
                split_ids["train"],
            )

            all_patients = [
                patient
                for split in ("train", "validation", "test")
                for patient in cache["records"][split]
            ]
            assert len(all_patients) == PATIENT_COUNT
            for patient in all_patients:
                _assert_patient_tensors(patient, statistics)
        finally:
            for name, path in original_paths.items():
                setattr(preprocessing, name, path)

    assert not temporary_root.exists()
    print(
        "PASS: Part A preprocessing smoke test "
        f"({PATIENT_COUNT} temporary patients)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
