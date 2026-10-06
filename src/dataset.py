"""Dataset and padded batching for the Part A pickle cache."""

from __future__ import annotations

import pickle
import sys
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from src.config import N_VARS, processed_path  # noqa: E402
from src.data_preprocessing import SELECTED_VARIABLES  # noqa: E402


class ICUMortalityDataset(Dataset[dict[str, Any]]):
    """In-memory tensor view of one Part A cache split.

    Part B consumes only normalized ``values`` and their observation ``mask``.
    It does not expose static demographic features.
    """

    def __init__(self, records: list[dict[str, Any]], split: str) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be 'train', 'validation', or 'test'.")
        if not records:
            raise ValueError(f"No records were provided for the {split} split.")
        self.records = records
        self.split = split

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one irregular stay using the existing Part A fields."""
        record = self.records[index]
        times = torch.as_tensor(record["times"], dtype=torch.float32)
        values = torch.as_tensor(record["values"], dtype=torch.float32)
        mask = torch.as_tensor(record["mask"], dtype=torch.float32)

        if times.ndim != 1 or times.numel() == 0:
            raise ValueError(f"Record {record.get('record_id')} has invalid times.")
        expected_shape = (times.numel(), N_VARS)
        if values.shape != expected_shape or mask.shape != expected_shape:
            raise ValueError(
                f"Record {record.get('record_id')} must have values and mask "
                f"shape {expected_shape}; got {tuple(values.shape)} and "
                f"{tuple(mask.shape)}."
            )
        if not torch.all(times[1:] > times[:-1]):
            raise ValueError(
                f"Record {record.get('record_id')} times are not strictly ascending."
            )
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError(f"Record {record.get('record_id')} has a non-binary mask.")

        return {
            "record_id": str(record["record_id"]),
            "times": times,
            "values": values,
            "mask": mask,
            "label": torch.tensor(float(record["label"]), dtype=torch.float32),
        }


class MaterializedICUDataset(Dataset[dict[str, Any]]):
    """Reusable model-ready CPU tensors prepared once from one cache split."""

    def __init__(self, records: list[dict[str, Any]], split: str) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be 'train', 'validation', or 'test'.")
        if not records:
            raise ValueError(f"No records were materialized for the {split} split.")
        self.records = records
        self.split = split

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def _record_delta_t(times: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Reproduce padded-collator delta semantics once for one unpadded stay."""
    result = torch.zeros_like(mask)
    last_observed = torch.zeros(mask.shape[1], dtype=times.dtype)
    seen = torch.zeros(mask.shape[1], dtype=torch.bool)
    for row, current_time in enumerate(times):
        result[row] = torch.where(
            seen,
            current_time - last_observed,
            current_time,
        )
        observed = mask[row] > 0
        last_observed = torch.where(observed, current_time, last_observed)
        seen |= observed
    return result.to(dtype=mask.dtype)


def materialize_records(
    records: list[dict[str, Any]],
    split: str,
) -> MaterializedICUDataset:
    """Convert NumPy-backed records into reusable irregular and hourly tensors."""
    from src.models.gru_baseline import irregular_patient_to_hourly

    source = ICUMortalityDataset(records, split)
    materialized: list[dict[str, Any]] = []
    for index in range(len(source)):
        item = source[index]
        delta_t = _record_delta_t(item["times"], item["mask"])
        hourly_values, hourly_mask = irregular_patient_to_hourly(
            times=item["times"],
            values=item["values"],
            mask=item["mask"],
        )
        materialized.append(
            {
                **item,
                "delta_t": delta_t,
                "length": torch.tensor(item["times"].numel(), dtype=torch.long),
                "hourly_values": hourly_values,
                "hourly_mask": hourly_mask,
            }
        )
    return MaterializedICUDataset(materialized, split)


def collate_materialized_stays(
    batch: list[dict[str, Any]],
    model_key: str | None = None,
) -> dict[str, Any]:
    """Batch prebuilt tensors, copying only fields consumed by one model."""
    if not batch:
        raise ValueError("Cannot collate an empty materialized batch.")
    labels = torch.stack([item["label"] for item in batch])

    if model_key == "gru_baseline":
        return {
            "hourly_values": torch.stack(
                [item["hourly_values"] for item in batch]
            ),
            "hourly_mask": torch.stack([item["hourly_mask"] for item in batch]),
            "label": labels,
        }

    lengths = torch.stack([item["length"] for item in batch])
    values = pad_sequence(
        [item["values"] for item in batch],
        batch_first=True,
    )
    mask = pad_sequence(
        [item["mask"] for item in batch],
        batch_first=True,
    )
    max_length = values.shape[1]
    valid_time_mask = (
        torch.arange(max_length, dtype=torch.long).unsqueeze(0)
        < lengths.unsqueeze(1)
    )
    times = pad_sequence(
        [item["times"] for item in batch],
        batch_first=True,
    )
    common = {
        "values": values,
        "mask": mask,
        "valid_time_mask": valid_time_mask,
        "times": times,
        "label": labels,
    }
    delta_t = pad_sequence(
        [item["delta_t"] for item in batch],
        batch_first=True,
    )
    if model_key in {"mlp_neural_ode", "kan_neural_ode", "ltc", "cfc"}:
        return {**common, "delta_t": delta_t}
    if model_key is not None:
        raise ValueError(f"Unknown model key for materialized collation: {model_key}")

    return {
        "record_id": [item["record_id"] for item in batch],
        **common,
        "delta_t": delta_t,
        "lengths": lengths,
        "hourly_values": torch.stack(
            [item["hourly_values"] for item in batch]
        ),
        "hourly_mask": torch.stack([item["hourly_mask"] for item in batch]),
    }


def materialized_collator(model_key: str) -> Any:
    """Return a Windows-pickle-safe model-specific collator."""
    return partial(collate_materialized_stays, model_key=model_key)


def collate_stays(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad irregular stays and mark every real, non-padding timestep."""
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    batch_size = len(batch)
    lengths = torch.tensor(
        [item["times"].numel() for item in batch],
        dtype=torch.long,
    )
    max_length = int(lengths.max().item())
    times = torch.zeros((batch_size, max_length), dtype=torch.float32)
    values = torch.zeros(
        (batch_size, max_length, N_VARS),
        dtype=torch.float32,
    )
    mask = torch.zeros_like(values)
    valid_time_mask = torch.zeros(
        (batch_size, max_length),
        dtype=torch.bool,
    )

    for row, item in enumerate(batch):
        length = int(lengths[row].item())
        times[row, :length] = item["times"]
        values[row, :length] = item["values"]
        mask[row, :length] = item["mask"]
        valid_time_mask[row, :length] = True

    return {
        "record_id": [item["record_id"] for item in batch],
        "times": times,
        "values": values,
        "mask": mask,
        "valid_time_mask": valid_time_mask,
        "lengths": lengths,
        "label": torch.stack([item["label"] for item in batch]),
    }


def load_split(split: str) -> ICUMortalityDataset:
    """Load one split without re-reading any raw PhysioNet files."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be 'train', 'validation', or 'test'.")
    cache = processed_path()
    if not cache.is_file():
        raise FileNotFoundError(
            f"Part A cache not found: {cache}. "
            "Run `python -m src.data_preprocessing` first."
        )
    with cache.open("rb") as handle:
        payload = pickle.load(handle)

    required = {"variables", "records", "split_record_ids"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"Unsupported Part A cache schema in {cache}.")
    variables = payload["variables"]
    if list(variables) != list(SELECTED_VARIABLES) or len(variables) != N_VARS:
        raise ValueError(
            "Part A cache variable order does not match the 12 Part B variables."
        )
    records = payload["records"].get(split)
    if not isinstance(records, list):
        raise ValueError(f"Part A cache does not contain a {split!r} record list.")
    return ICUMortalityDataset(records, split)
