"""Hourly-binned GRU baseline for Part B."""

from __future__ import annotations

import torch
from torch import nn

from src.config import GRU_HIDDEN_SIZE, N_VARS


def irregular_patient_to_hourly(
    times: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    valid_time_mask: torch.Tensor | None = None,
    n_hours: int = 48,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one irregular patient to forward-filled hourly values and mask.

    Observations are assigned to ``floor(time)``, clamped to ``[0, 47]``.
    Duplicate observations for a variable within an hour are averaged.
    Returned masks mark actual hourly observations, not forward-filled cells.
    """
    if times.ndim != 1:
        raise ValueError("times must have shape (T,) for one patient.")
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("values and mask must have matching shape (T, n_vars).")
    if values.shape[0] != times.numel():
        raise ValueError("times, values, and mask must use the same T dimension.")
    if values.shape[1] != N_VARS:
        raise ValueError(f"Part B hourly inputs must contain {N_VARS} variables.")
    if n_hours != 48:
        raise ValueError("The Part B GRU baseline requires exactly 48 hourly bins.")
    times_on_device = times.to(device=values.device)
    if valid_time_mask is None:
        valid = torch.ones_like(times_on_device, dtype=torch.bool)
    else:
        if valid_time_mask.shape != times.shape:
            raise ValueError("valid_time_mask must have shape (T,).")
        valid = valid_time_mask.to(device=values.device, dtype=torch.bool)
    if torch.any(~torch.isfinite(times_on_device[valid])):
        raise ValueError("Valid timestamps must be finite.")

    n_vars = values.shape[1]
    observed_values = torch.zeros(
        (n_hours, n_vars),
        dtype=values.dtype,
        device=values.device,
    )
    hourly_mask = torch.zeros_like(observed_values)
    hour_indices = (
        torch.floor(times_on_device).to(dtype=torch.long).clamp(0, n_hours - 1)
    )
    observed = mask.to(device=values.device) > 0

    for hour in range(n_hours):
        rows = valid & (hour_indices == hour)
        if not torch.any(rows):
            continue
        hour_observed = observed[rows]
        counts = hour_observed.sum(dim=0)
        sums = (values[rows] * hour_observed.to(values.dtype)).sum(dim=0)
        has_observation = counts > 0
        observed_values[hour] = torch.where(
            has_observation,
            sums / counts.clamp_min(1).to(values.dtype),
            torch.zeros_like(sums),
        )
        hourly_mask[hour] = has_observation.to(values.dtype)

    forward_filled = torch.zeros_like(observed_values)
    carried = torch.zeros(n_vars, dtype=values.dtype, device=values.device)
    seen = torch.zeros(n_vars, dtype=torch.bool, device=values.device)
    for hour in range(n_hours):
        actually_observed = hourly_mask[hour] == 1
        carried = torch.where(actually_observed, observed_values[hour], carried)
        seen = seen | actually_observed
        forward_filled[hour] = torch.where(
            seen,
            carried,
            torch.zeros_like(carried),
        )

    return forward_filled, hourly_mask


def prepare_hourly_gru_inputs(
    times: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    valid_time_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a padded irregular batch into tensors shaped ``(B, 48, D)``."""
    if times.ndim != 2 or values.ndim != 3 or mask.shape != values.shape:
        raise ValueError(
            "Expected times (B, T), values (B, T, D), and matching mask."
        )
    if values.shape[:2] != times.shape:
        raise ValueError("times, values, and mask must share B and T dimensions.")
    if valid_time_mask is not None and valid_time_mask.shape != times.shape:
        raise ValueError("valid_time_mask must have shape (B, T).")

    hourly = [
        irregular_patient_to_hourly(
            times=times[index],
            values=values[index],
            mask=mask[index],
            valid_time_mask=(
                None if valid_time_mask is None else valid_time_mask[index]
            ),
        )
        for index in range(times.shape[0])
    ]
    hourly_values = torch.stack([item[0] for item in hourly])
    hourly_masks = torch.stack([item[1] for item in hourly])
    return hourly_values, hourly_masks


class GRUBaseline(nn.Module):
    """GRU classifier over 48 hourly value and observation-mask vectors."""

    def __init__(
        self,
        n_vars: int = N_VARS,
        hidden_size: int = GRU_HIDDEN_SIZE,
    ) -> None:
        super().__init__()
        if n_vars < 1 or hidden_size < 1:
            raise ValueError("n_vars and hidden_size must be positive.")
        self.n_vars = n_vars
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=2 * n_vars,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.mortality_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        hourly_values: torch.Tensor,
        hourly_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return unsquashed mortality logits with shape ``(B,)``."""
        expected_tail = (48, self.n_vars)
        if hourly_values.ndim != 3 or tuple(hourly_values.shape[1:]) != expected_tail:
            raise ValueError(
                f"hourly_values must have shape (B, 48, {self.n_vars})."
            )
        if hourly_mask.shape != hourly_values.shape:
            raise ValueError("hourly_mask must match hourly_values.")

        inputs = torch.cat(
            [hourly_values, hourly_mask.to(hourly_values.dtype)],
            dim=-1,
        )
        _, final_hidden = self.gru(inputs)
        return self.mortality_head(final_hidden[-1]).squeeze(-1)
