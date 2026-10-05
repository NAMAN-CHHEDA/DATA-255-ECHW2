"""Preprocess PhysioNet Challenge 2012 Set A for mortality prediction.

The cache contains one irregular sequence per patient. Only the 12 variables
in ``SELECTED_VARIABLES`` are retained. Duplicate measurements at the same
timestamp are averaged before tensors are built. Normalization statistics are
estimated from observed training values only.
"""

from __future__ import annotations

import csv
import json
import math
import pickle
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from src.config import (  # noqa: E402
    NORMALIZATION_STATS_PATH,
    MAX_HOURS,
    OUTCOMES_PATH,
    PART_A_CACHE_PATH,
    PREPROCESSING_SUMMARY_PATH,
    PROCESSED_DATA_DIR,
    SEED,
    SET_A_DIR,
    SPLITS_PATH,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    validate_config,
)

SELECTED_VARIABLES = (
    "HR",
    "NIDiasABP",
    "NIMAP",
    "NISysABP",
    "Temp",
    "RespRate",
    "GCS",
    "Glucose",
    "Lactate",
    "Urine",
    "Creatinine",
    "Platelets",
)
TIME_VARYING_PARAMETERS = SELECTED_VARIABLES


def canonical_record_id(value: object) -> str:
    """Return a stable string representation of a PhysioNet RecordID."""
    text = str(value).strip()
    if not text:
        raise ValueError("Encountered an empty RecordID.")
    try:
        numeric = float(text)
    except ValueError:
        return text
    if not numeric.is_integer():
        raise ValueError(f"RecordID must be integer-like, got {text!r}.")
    return str(int(numeric))


def parse_time_hours(value: str) -> float:
    """Convert a PhysioNet ``HH:MM`` timestamp to hours."""
    text = value.strip()
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected time in HH:MM form, got {value!r}.")
    try:
        hours, minutes = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"Expected numeric time, got {value!r}.") from exc
    if hours < 0 or minutes < 0 or minutes >= 60:
        raise ValueError(f"Invalid timestamp {value!r}.")
    return hours + minutes / 60.0


def build_values_mask_delta(
    times_hours: list[float],
    observations: dict[str, list[tuple[float, float]]],
    variable_names: tuple[str, ...] = SELECTED_VARIABLES,
) -> dict[str, Any]:
    """Build one patient's times, values, observation mask, and time gaps.

    Repeated ``(timestamp, variable)`` measurements are replaced by their
    arithmetic mean. For every cell, ``delta_t`` is the time since that
    variable's previous observation. Before its first observation the gap is
    the elapsed time since admission.
    """
    import numpy as np

    grouped: dict[tuple[float, str], list[float]] = defaultdict(list)
    for variable, pairs in observations.items():
        if variable not in variable_names:
            continue
        for time_hours, value in pairs:
            if not 0.0 <= time_hours <= MAX_HOURS:
                continue
            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite value for {variable} at {time_hours:g} hours."
                )
            grouped[(float(time_hours), variable)].append(float(value))

    observed_times = {time for time, _ in grouped}
    supplied_times = {
        float(time) for time in times_hours if 0.0 <= float(time) <= MAX_HOURS
    }
    times = np.asarray(sorted(observed_times | supplied_times), dtype=np.float64)
    if times.size == 0:
        raise ValueError("Patient has no selected observations in [0, 48] hours.")

    time_index = {float(time): index for index, time in enumerate(times)}
    variable_index = {name: index for index, name in enumerate(variable_names)}
    values = np.zeros((len(times), len(variable_names)), dtype=np.float32)
    mask = np.zeros_like(values, dtype=np.float32)

    for (time_hours, variable), duplicates in grouped.items():
        row = time_index[time_hours]
        column = variable_index[variable]
        values[row, column] = float(np.mean(duplicates, dtype=np.float64))
        mask[row, column] = 1.0

    delta_t = np.zeros_like(values, dtype=np.float32)
    last_observed: list[float | None] = [None] * len(variable_names)
    for row, time_hours in enumerate(times):
        current_time = float(time_hours)
        for column in range(len(variable_names)):
            previous = last_observed[column]
            delta_t[row, column] = (
                current_time if previous is None else current_time - previous
            )
            if mask[row, column] == 1.0:
                last_observed[column] = current_time

    return {
        "times": times,
        "values": values,
        "mask": mask,
        "delta_t": delta_t,
    }


def load_outcomes(outcomes_path: Path) -> dict[str, int]:
    """Load ``RecordID -> In-hospital_death`` from Outcomes A."""
    if not outcomes_path.is_file():
        raise FileNotFoundError(
            f"Outcomes file not found: {outcomes_path}. "
            "Run `python scripts/download_data.py` first."
        )

    outcomes: dict[str, int] = {}
    with outcomes_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"RecordID", "In-hospital_death"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{outcomes_path} must contain columns {sorted(required)}."
            )
        for row_number, row in enumerate(reader, start=2):
            record_id = canonical_record_id(row["RecordID"])
            if record_id in outcomes:
                raise ValueError(f"Duplicate RecordID {record_id} in outcomes.")
            try:
                label = int(row["In-hospital_death"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid mortality label on outcomes row {row_number}."
                ) from exc
            if label not in {0, 1}:
                raise ValueError(
                    f"Mortality label for RecordID {record_id} is {label}, not 0 or 1."
                )
            outcomes[record_id] = label
    return outcomes


def read_patient_file(
    patient_path: Path,
    outcomes: dict[str, int],
) -> dict[str, Any] | None:
    """Parse one patient file, returning ``None`` if no selected values exist."""
    observations: dict[str, list[tuple[float, float]]] = defaultdict(list)
    internal_record_id: str | None = None

    with patient_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"Time", "Parameter", "Value"}
        if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
            raise ValueError(f"{patient_path} must contain columns {sorted(expected)}.")

        for row_number, row in enumerate(reader, start=2):
            parameter = row["Parameter"].strip()
            if parameter == "RecordID":
                candidate = canonical_record_id(row["Value"])
                if internal_record_id is not None and candidate != internal_record_id:
                    raise ValueError(f"Conflicting RecordIDs in {patient_path}.")
                internal_record_id = candidate
                continue
            if parameter not in SELECTED_VARIABLES:
                continue

            try:
                time_hours = parse_time_hours(row["Time"])
                value = float(row["Value"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid selected measurement in {patient_path} "
                    f"on row {row_number}: {exc}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite {parameter} value in {patient_path} "
                    f"on row {row_number}."
                )
            if 0.0 <= time_hours <= MAX_HOURS:
                observations[parameter].append((time_hours, value))

    if internal_record_id is None:
        raise ValueError(f"No RecordID row found in {patient_path}.")
    filename_record_id = canonical_record_id(patient_path.stem)
    if internal_record_id != filename_record_id:
        raise ValueError(
            f"RecordID {internal_record_id} in {patient_path} does not match "
            f"filename RecordID {filename_record_id}."
        )
    if internal_record_id not in outcomes:
        raise ValueError(
            f"Patient {internal_record_id} in {patient_path} has no mortality label."
        )
    if not observations:
        return None

    tensors = build_values_mask_delta(
        times_hours=[],
        observations=observations,
        variable_names=SELECTED_VARIABLES,
    )
    patient = {
        "record_id": internal_record_id,
        **tensors,
        "label": outcomes[internal_record_id],
    }
    validate_patient(patient)
    return patient


def validate_patient(patient: dict[str, Any]) -> None:
    """Fail clearly when one patient's tensors violate the cache contract."""
    import numpy as np

    record_id = patient.get("record_id", "<unknown>")
    if "label" not in patient or patient["label"] not in {0, 1}:
        raise ValueError(f"Patient {record_id} has no valid binary label.")

    times = np.asarray(patient["times"])
    values = np.asarray(patient["values"])
    mask = np.asarray(patient["mask"])
    delta_t = np.asarray(patient["delta_t"])
    expected_shape = (len(times), len(SELECTED_VARIABLES))

    if times.ndim != 1 or len(times) == 0:
        raise ValueError(f"Patient {record_id} must have a non-empty 1D times array.")
    if np.any(~np.isfinite(times)) or np.any((times < 0.0) | (times > MAX_HOURS)):
        raise ValueError(f"Patient {record_id} has a time outside [0, 48].")
    if len(times) > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError(f"Patient {record_id} times are not strictly ascending.")
    for name, array in (
        ("values", values),
        ("mask", mask),
        ("delta_t", delta_t),
    ):
        if array.shape != expected_shape:
            raise ValueError(
                f"Patient {record_id} {name} shape is {array.shape}; "
                f"expected {expected_shape}."
            )
    if not np.all((mask == 0.0) | (mask == 1.0)):
        raise ValueError(f"Patient {record_id} mask contains a value other than 0 or 1.")
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(delta_t)):
        raise ValueError(f"Patient {record_id} contains a non-finite tensor value.")
    if np.any(delta_t < 0.0):
        raise ValueError(f"Patient {record_id} contains a negative delta_t.")
    if np.any(values[mask == 0.0] != 0.0):
        raise ValueError(f"Patient {record_id} has nonzero values where mask is 0.")


def split_record_ids(
    patients: list[dict[str, Any]],
) -> tuple[dict[str, list[str]], bool]:
    """Split RecordIDs 70/15/15, stratifying by mortality when possible."""
    from sklearn.model_selection import train_test_split

    ordered = sorted(patients, key=lambda patient: patient["record_id"])
    record_ids = [patient["record_id"] for patient in ordered]
    labels = [patient["label"] for patient in ordered]
    if len(record_ids) < 3:
        raise ValueError("At least three usable patients are required for splitting.")

    stratified = True
    try:
        train_ids, remainder_ids, _, remainder_labels = train_test_split(
            record_ids,
            labels,
            train_size=TRAIN_RATIO,
            random_state=SEED,
            shuffle=True,
            stratify=labels,
        )
        relative_test_size = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
        validation_ids, test_ids = train_test_split(
            remainder_ids,
            test_size=relative_test_size,
            random_state=SEED,
            shuffle=True,
            stratify=remainder_labels,
        )
    except ValueError as exc:
        stratified = False
        warnings.warn(
            f"Stratified splitting was not possible ({exc}); "
            "using a deterministic shuffled split.",
            RuntimeWarning,
            stacklevel=2,
        )
        train_ids, remainder_ids = train_test_split(
            record_ids,
            train_size=TRAIN_RATIO,
            random_state=SEED,
            shuffle=True,
        )
        relative_test_size = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
        validation_ids, test_ids = train_test_split(
            remainder_ids,
            test_size=relative_test_size,
            random_state=SEED,
            shuffle=True,
        )

    splits = {
        "train": sorted(train_ids),
        "validation": sorted(validation_ids),
        "test": sorted(test_ids),
    }
    validate_splits(splits, set(record_ids))
    return splits, stratified


def validate_splits(
    splits: dict[str, list[str]],
    expected_record_ids: set[str],
) -> None:
    """Ensure split RecordIDs are disjoint and cover every usable patient."""
    split_sets = {name: set(ids) for name, ids in splits.items()}
    for name, ids in split_sets.items():
        if len(ids) != len(splits[name]):
            raise ValueError(f"Split {name} contains duplicate RecordIDs.")
    if split_sets["train"] & split_sets["validation"]:
        raise ValueError("Train and validation RecordIDs overlap.")
    if split_sets["train"] & split_sets["test"]:
        raise ValueError("Train and test RecordIDs overlap.")
    if split_sets["validation"] & split_sets["test"]:
        raise ValueError("Validation and test RecordIDs overlap.")
    covered = set().union(*split_sets.values())
    if covered != expected_record_ids:
        missing = sorted(expected_record_ids - covered)
        extra = sorted(covered - expected_record_ids)
        raise ValueError(f"Split coverage mismatch; missing={missing}, extra={extra}.")


def calculate_normalization_stats(
    patients_by_id: dict[str, dict[str, Any]],
    train_ids: list[str],
) -> dict[str, Any]:
    """Calculate per-variable z-score statistics from observed train cells."""
    import numpy as np

    means: list[float] = []
    standard_deviations: list[float] = []
    observation_counts: list[int] = []
    zero_std_variables: list[str] = []

    for column, variable in enumerate(SELECTED_VARIABLES):
        observed = [
            patient["values"][:, column][patient["mask"][:, column] == 1.0]
            for record_id in train_ids
            for patient in [patients_by_id[record_id]]
        ]
        combined = np.concatenate(observed) if observed else np.asarray([])
        if combined.size == 0:
            raise ValueError(
                f"Training split has no observed values for selected variable {variable}."
            )
        mean = float(np.mean(combined, dtype=np.float64))
        std = float(np.std(combined, dtype=np.float64))
        if not math.isfinite(mean) or not math.isfinite(std):
            raise ValueError(f"Non-finite normalization statistic for {variable}.")
        if std == 0.0:
            std = 1.0
            zero_std_variables.append(variable)
        means.append(mean)
        standard_deviations.append(std)
        observation_counts.append(int(combined.size))

    return {
        "variables": list(SELECTED_VARIABLES),
        "mean": means,
        "std": standard_deviations,
        "observed_train_count": observation_counts,
        "zero_std_replaced_with_one": zero_std_variables,
        "computed_from_split": "train",
    }


def normalize_patients(
    patients: list[dict[str, Any]],
    statistics: dict[str, Any],
) -> None:
    """Apply train statistics in place and restore missing cells to zero."""
    import numpy as np

    means = np.asarray(statistics["mean"], dtype=np.float32)
    standard_deviations = np.asarray(statistics["std"], dtype=np.float32)
    for patient in patients:
        observed = patient["mask"] == 1.0
        normalized = (patient["values"] - means) / standard_deviations
        patient["values"] = np.where(observed, normalized, 0.0).astype(np.float32)
        validate_patient(patient)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write formatted JSON atomically."""
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _write_pickle(path: Path, payload: dict[str, Any]) -> None:
    """Write a pickle atomically."""
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def build_summary(
    patients_by_id: dict[str, dict[str, Any]],
    splits: dict[str, list[str]],
    stratified: bool,
) -> dict[str, Any]:
    """Build the required concise preprocessing summary."""
    import numpy as np

    def mortality_rate(record_ids: list[str]) -> float:
        if not record_ids:
            raise ValueError("A split is empty; mortality rate is undefined.")
        return float(
            np.mean([patients_by_id[record_id]["label"] for record_id in record_ids])
        )

    return {
        "total_usable_patients": len(patients_by_id),
        "split_patient_counts": {
            split: len(record_ids) for split, record_ids in splits.items()
        },
        "mortality_rate_by_split": {
            split: mortality_rate(record_ids)
            for split, record_ids in splits.items()
        },
        "selected_variables": list(SELECTED_VARIABLES),
        "median_irregular_timestamps_per_patient": float(
            np.median(
                [len(patient["times"]) for patient in patients_by_id.values()]
            )
        ),
        "stratified_split": stratified,
        "seed": SEED,
        "cache_path": str(PART_A_CACHE_PATH),
    }


def print_summary(summary: dict[str, Any]) -> None:
    """Print the required preprocessing summary."""
    counts = summary["split_patient_counts"]
    rates = summary["mortality_rate_by_split"]
    print("Part A preprocessing complete")
    print(f"Total usable patients: {summary['total_usable_patients']}")
    print(
        "Patient counts (train/validation/test): "
        f"{counts['train']}/{counts['validation']}/{counts['test']}"
    )
    print(
        "Mortality rates (train/validation/test): "
        f"{rates['train']:.4f}/{rates['validation']:.4f}/{rates['test']:.4f}"
    )
    print(f"Selected variables: {', '.join(summary['selected_variables'])}")
    print(
        "Median irregular timestamps per patient: "
        f"{summary['median_irregular_timestamps_per_patient']:g}"
    )
    print(f"Cache path: {summary['cache_path']}")


def preprocess_dataset() -> dict[str, Any]:
    """Run deterministic Set A preprocessing and save all Part A artifacts."""
    validate_config()
    if not SET_A_DIR.is_dir():
        raise FileNotFoundError(
            f"Set A directory not found: {SET_A_DIR}. "
            "Run `python scripts/download_data.py` first."
        )
    patient_paths = sorted(SET_A_DIR.glob("*.txt"))
    if not patient_paths:
        raise FileNotFoundError(f"No patient text files found in {SET_A_DIR}.")

    outcomes = load_outcomes(OUTCOMES_PATH)
    patients: list[dict[str, Any]] = []
    seen_record_ids: set[str] = set()
    for patient_path in patient_paths:
        patient = read_patient_file(patient_path, outcomes)
        if patient is None:
            continue
        record_id = patient["record_id"]
        if record_id in seen_record_ids:
            raise ValueError(f"Duplicate patient RecordID {record_id}.")
        seen_record_ids.add(record_id)
        patients.append(patient)

    if not patients:
        raise ValueError("No usable Set A patients had selected observations.")

    splits, stratified = split_record_ids(patients)
    patients_by_id = {patient["record_id"]: patient for patient in patients}
    statistics = calculate_normalization_stats(patients_by_id, splits["train"])
    normalize_patients(patients, statistics)

    records_by_split = {
        split: [patients_by_id[record_id] for record_id in record_ids]
        for split, record_ids in splits.items()
    }
    cache = {
        "format_version": 1,
        "seed": SEED,
        "variables": list(SELECTED_VARIABLES),
        "records": records_by_split,
        "split_record_ids": splits,
        "normalization_stats": statistics,
    }
    summary = build_summary(patients_by_id, splits, stratified)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    _write_pickle(PART_A_CACHE_PATH, cache)
    _write_json(
        SPLITS_PATH,
        {
            "seed": SEED,
            "ratios": {
                "train": TRAIN_RATIO,
                "validation": VAL_RATIO,
                "test": TEST_RATIO,
            },
            "stratified": stratified,
            "record_ids": splits,
        },
    )
    _write_json(NORMALIZATION_STATS_PATH, statistics)
    _write_json(PREPROCESSING_SUMMARY_PATH, summary)
    print_summary(summary)
    return summary


def main() -> int:
    """Run Part A preprocessing from the command line."""
    preprocess_dataset()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
