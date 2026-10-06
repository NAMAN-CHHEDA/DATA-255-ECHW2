"""LTC and CfC mortality-risk models using the official ncps PyTorch API."""

from __future__ import annotations

import torch
from ncps.torch import LTC, CfC
from ncps.wirings import AutoNCP
from torch import nn

from src.config import (
    LIQUID_BATCH_MODE,
    LIQUID_ODE_UNFOLDS,
    LIQUID_TIMESPAN_MAX,
    LIQUID_TIMESPAN_MIN,
    LIQUID_UNITS,
    N_VARS,
)


class LiquidRiskModel(nn.Module):
    """Configurable LTC or CfC model over irregular ICU measurements.

    ``per_sample_compat`` is the reproducible default because current ncps
    releases can fail when a single batched call receives timespans shaped
    ``(B, T)`` for ``B > 1``. It loops over patients, submits only valid rows,
    and repads returned logits. ``native_batched`` is explicit and accepts
    only batches without padding; benchmark it later only after verifying the
    installed ncps version.
    """

    def __init__(
        self,
        n_vars: int = N_VARS,
        units: int = LIQUID_UNITS,
        cell_type: str = "ltc",
        ode_unfolds: int = LIQUID_ODE_UNFOLDS,
        timespan_min: float = LIQUID_TIMESPAN_MIN,
        timespan_max: float = LIQUID_TIMESPAN_MAX,
        batch_mode: str = LIQUID_BATCH_MODE,
    ) -> None:
        super().__init__()
        normalized_cell_type = cell_type.lower()
        if normalized_cell_type not in {"ltc", "cfc"}:
            raise ValueError(
                f"Unsupported cell_type {cell_type!r}; expected 'ltc' or 'cfc'."
            )
        if n_vars < 1 or units < 1 or ode_unfolds < 1:
            raise ValueError("n_vars, units, and ode_unfolds must be positive.")
        if not 0 < timespan_min <= timespan_max:
            raise ValueError(
                "timespan bounds must satisfy 0 < timespan_min <= timespan_max."
            )
        if batch_mode not in {"per_sample_compat", "native_batched"}:
            raise ValueError(
                "batch_mode must be 'per_sample_compat' or 'native_batched'."
            )

        self.n_vars = n_vars
        self.units = units
        self.cell_type = normalized_cell_type
        self.ode_unfolds = ode_unfolds
        self.timespan_min = timespan_min
        self.timespan_max = timespan_max
        self.batch_mode = batch_mode

        wiring = AutoNCP(units, 1)
        if self.cell_type == "ltc":
            self.recurrent = LTC(
                2 * n_vars,
                wiring,
                batch_first=True,
                return_sequences=True,
                ode_unfolds=ode_unfolds,
            )
        else:
            self.recurrent = CfC(
                2 * n_vars,
                wiring,
                batch_first=True,
                return_sequences=True,
            )

    def _validate_inputs(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta_t: torch.Tensor,
        valid_time_mask: torch.Tensor,
    ) -> None:
        """Validate padded tensor shapes before selecting valid sequences."""
        if values.ndim != 3 or values.shape[-1] != self.n_vars:
            raise ValueError(
                f"values must have shape (B, T, {self.n_vars})."
            )
        if mask.shape != values.shape or delta_t.shape != values.shape:
            raise ValueError("mask and delta_t must have the same shape as values.")
        if valid_time_mask.shape != values.shape[:2]:
            raise ValueError("valid_time_mask must have shape (B, T).")
        if torch.any(valid_time_mask.sum(dim=1) == 0):
            raise ValueError("Every patient must contain at least one valid timestep.")

    def _prepare_inputs(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta_t: torch.Tensor,
        valid_time_mask: torch.Tensor,
        times: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create value-mask features and one clamped timespan per timestep.

        With ``times`` the timespan is the real inter-step gap
        ``times[t] - times[t-1]`` (the first step uses the clamp minimum as
        its epsilon). Without it (smoke tests only) the legacy mean per-variable
        ``delta_t`` is used.
        """
        self._validate_inputs(values, mask, delta_t, valid_time_mask)
        parameter = next(self.recurrent.parameters())
        values = values.to(device=parameter.device, dtype=parameter.dtype)
        mask = mask.to(device=parameter.device, dtype=parameter.dtype)
        delta_t = delta_t.to(device=parameter.device, dtype=parameter.dtype)
        valid = valid_time_mask.to(device=parameter.device, dtype=torch.bool)
        features = torch.cat([values, mask], dim=-1)
        if times is None:
            raw_timespans = delta_t.mean(dim=-1)
        else:
            times = times.to(device=parameter.device, dtype=parameter.dtype)
            raw_timespans = torch.zeros_like(times)
            raw_timespans[:, 1:] = times[:, 1:] - times[:, :-1]
            raw_timespans[:, 0] = self.timespan_min
        if torch.any(~torch.isfinite(raw_timespans[valid])):
            raise ValueError("Valid mean delta_t timespans must be finite.")
        timespans = raw_timespans.clamp(
            min=self.timespan_min,
            max=self.timespan_max,
        )
        if torch.any(~torch.isfinite(features[valid])):
            raise ValueError("Valid values and masks must be finite.")
        return features, timespans, valid

    @staticmethod
    def _extract_logits(output: torch.Tensor) -> torch.Tensor:
        """Convert an ncps output sequence from ``(B, T, 1)`` to ``(B, T)``."""
        if output.ndim != 3 or output.shape[-1] != 1:
            raise RuntimeError(
                "AutoNCP(units, 1) must return ncps outputs shaped (B, T, 1); "
                f"got {tuple(output.shape)}."
            )
        return output.squeeze(-1)

    def _forward_per_sample(
        self,
        features: torch.Tensor,
        timespans: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run ncps once per patient and repad outputs for version safety."""
        batch_size, max_length, _ = features.shape
        trajectory = features.new_zeros((batch_size, max_length))
        final_logits: list[torch.Tensor] = []

        for patient_index in range(batch_size):
            patient_valid = valid[patient_index]
            patient_features = features[patient_index, patient_valid].unsqueeze(0)
            patient_timespans = timespans[
                patient_index, patient_valid
            ].unsqueeze(0)
            output, _ = self.recurrent(
                patient_features,
                timespans=patient_timespans,
            )
            patient_logits = self._extract_logits(output)[0]
            trajectory[patient_index, patient_valid] = patient_logits
            final_logits.append(patient_logits[-1])

        return trajectory, torch.stack(final_logits)

    def _forward_native_batched(
        self,
        features: torch.Tensor,
        timespans: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use native batched timing only for an explicitly unpadded batch."""
        if not torch.all(valid):
            raise ValueError(
                "native_batched mode requires every timestep to be valid; "
                "use per_sample_compat for padded or unequal-length batches."
            )
        output, _ = self.recurrent(features, timespans=timespans)
        trajectory = self._extract_logits(output)
        return trajectory, trajectory[:, -1]

    def _forward_outputs(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta_t: torch.Tensor,
        valid_time_mask: torch.Tensor,
        times: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return padded trajectories and final valid logits."""
        features, timespans, valid = self._prepare_inputs(
            values,
            mask,
            delta_t,
            valid_time_mask,
            times,
        )
        if self.batch_mode == "per_sample_compat":
            return self._forward_per_sample(features, timespans, valid)
        return self._forward_native_batched(features, timespans, valid)

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta_t: torch.Tensor,
        valid_time_mask: torch.Tensor,
        times: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return padded unsquashed risk logits shaped ``(B, T_max)``."""
        trajectory, _ = self._forward_outputs(
            values,
            mask,
            delta_t,
            valid_time_mask,
            times,
        )
        return trajectory

    def forward_final(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta_t: torch.Tensor,
        valid_time_mask: torch.Tensor,
        times: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return each patient's final valid unsquashed logit shaped ``(B,)``."""
        _, final_logits = self._forward_outputs(
            values,
            mask,
            delta_t,
            valid_time_mask,
            times,
        )
        return final_logits


# Backward-compatible project name; both LTC and CfC use LiquidRiskModel.
LiquidModel = LiquidRiskModel
