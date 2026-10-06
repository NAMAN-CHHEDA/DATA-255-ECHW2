"""Shared encoder, solver, and output path for Neural ODE comparisons."""

from __future__ import annotations

import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint

from src.config import (
    MAX_HOURS,
    N_VARS,
    ODE_ATOL,
    ODE_HIDDEN_DIM,
    ODE_MAX_NUM_STEPS,
    ODE_RTOL,
    ODE_SOLVER,
)


class NeuralODEBase(nn.Module):
    """Common Neural ODE implementation used by the MLP and KAN variants.

    Subclasses supply only the latent vector field. The padded masked-mean
    encoder, classification head, query validation, solver invocation, and
    final-risk API are defined once here to guarantee a fair comparison.
    """

    def __init__(
        self,
        ode_func: nn.Module,
        n_vars: int = N_VARS,
        hidden_dim: int = ODE_HIDDEN_DIM,
        method: str = ODE_SOLVER,
        rtol: float = ODE_RTOL,
        atol: float = ODE_ATOL,
        max_num_steps: int = ODE_MAX_NUM_STEPS,
    ) -> None:
        super().__init__()
        if n_vars < 1 or hidden_dim < 1:
            raise ValueError("n_vars and hidden_dim must be positive.")
        if rtol <= 0 or atol <= 0:
            raise ValueError("rtol and atol must be positive.")
        self.n_vars = n_vars
        self.hidden_dim = hidden_dim
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.max_num_steps = max_num_steps
        # Diagnostics only (no effect on outputs): forward NFE and max |h| of
        # the most recent forward pass; the max is kept as a tensor so reading
        # it (one sync) is the caller's choice.
        self.last_forward_nfe = 0
        self.last_max_abs_state: torch.Tensor | None = None
        self.encoder = nn.Sequential(
            # [values, mask, delta_t / 48 per variable, t / 48]
            nn.Linear(3 * n_vars + 1, hidden_dim),
            nn.Tanh(),
        )
        self.ode_func = ode_func
        self.mortality_head = nn.Linear(hidden_dim, 1)

    def encode_initial_state(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        valid_time_mask: torch.Tensor,
        times: torch.Tensor | None = None,
        delta_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Masked-mean timestep embeddings, excluding every padded row.

        Observation time and per-variable ``delta_t`` (both scaled by 48 h)
        are encoder inputs so timing is not discarded. If omitted (smoke
        tests only) they are zero-filled; all training and evaluation
        adapters pass them.
        """
        if values.ndim != 3:
            raise ValueError("values must have shape (B, T, n_vars).")
        if values.shape[-1] != self.n_vars:
            raise ValueError(
                f"values has {values.shape[-1]} variables; expected {self.n_vars}."
            )
        if mask.shape != values.shape:
            raise ValueError("mask must have the same shape as values.")
        if valid_time_mask.shape != values.shape[:2]:
            raise ValueError("valid_time_mask must have shape (B, T).")

        parameter = next(self.encoder.parameters())
        values = values.to(device=parameter.device, dtype=parameter.dtype)
        mask = mask.to(device=parameter.device, dtype=parameter.dtype)
        valid = valid_time_mask.to(device=parameter.device, dtype=torch.bool)
        valid_counts = valid.sum(dim=1)
        if torch.any(valid_counts == 0):
            raise ValueError("Every patient must contain at least one valid timestep.")

        if times is None:
            times = torch.zeros_like(values[..., 0])
        if delta_t is None:
            delta_t = torch.zeros_like(values)
        if times.shape != values.shape[:2] or delta_t.shape != values.shape:
            raise ValueError("times must be (B, T) and delta_t must match values.")
        times = times.to(device=parameter.device, dtype=parameter.dtype)
        delta_t = delta_t.to(device=parameter.device, dtype=parameter.dtype)
        timestep_inputs = torch.cat(
            [
                values,
                mask,
                delta_t.clamp(0.0, MAX_HOURS) / MAX_HOURS,
                times.unsqueeze(-1) / MAX_HOURS,
            ],
            dim=-1,
        )
        timestep_inputs = torch.where(
            valid.unsqueeze(-1),
            timestep_inputs,
            torch.zeros_like(timestep_inputs),
        )
        embeddings = self.encoder(timestep_inputs)
        weights = valid.unsqueeze(-1).to(embeddings.dtype)
        return (embeddings * weights).sum(dim=1) / valid_counts.unsqueeze(-1).to(
            embeddings.dtype
        )

    @staticmethod
    def _validate_query_times(query_times: torch.Tensor) -> None:
        """Require a non-empty, finite, strictly increasing 1-D tensor."""
        if query_times.ndim != 1:
            raise ValueError("query_times must be a sorted 1-D tensor.")
        if query_times.numel() == 0:
            raise ValueError("query_times must not be empty.")
        if not torch.all(torch.isfinite(query_times)):
            raise ValueError("query_times must contain only finite values.")
        if torch.any(query_times < 0):
            raise ValueError("query_times must be non-negative.")
        if query_times.numel() > 1 and not torch.all(
            query_times[1:] > query_times[:-1]
        ):
            raise ValueError(
                "query_times must be sorted in strictly increasing order."
            )

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        valid_time_mask: torch.Tensor,
        query_times: torch.Tensor,
        times: torch.Tensor | None = None,
        delta_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return unsquashed logits with shape ``(n_query_times, B)``."""
        self._validate_query_times(query_times)
        initial_state = self.encode_initial_state(
            values, mask, valid_time_mask, times, delta_t
        )
        integration_times = query_times.to(
            device=initial_state.device,
            dtype=initial_state.dtype,
        )
        prepended_zero = bool((integration_times[0] > 0).item())
        if prepended_zero:
            integration_times = torch.cat(
                [integration_times.new_zeros(1), integration_times]
            )
        trajectory = odeint(
            self.ode_func,
            initial_state,
            integration_times,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            options={"max_num_steps": self.max_num_steps},
        )
        self.last_forward_nfe = int(getattr(self.ode_func, "nfe", 0))
        self.last_max_abs_state = trajectory.detach().abs().max()
        if prepended_zero:
            trajectory = trajectory[1:]
        return self.mortality_head(trajectory).squeeze(-1)

    def latent_trajectory(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        valid_time_mask: torch.Tensor,
        query_times: torch.Tensor,
        times: torch.Tensor | None = None,
        delta_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return latent states ``h(t)`` shaped ``(n_query_times, B, hidden)``.

        Used by Experiment 3 to probe the vector field on states the solver
        actually visits. Same encoder and solver settings as ``forward``.
        """
        self._validate_query_times(query_times)
        initial_state = self.encode_initial_state(
            values, mask, valid_time_mask, times, delta_t
        )
        integration_times = query_times.to(
            device=initial_state.device, dtype=initial_state.dtype
        )
        prepended_zero = bool((integration_times[0] > 0).item())
        if prepended_zero:
            integration_times = torch.cat(
                [integration_times.new_zeros(1), integration_times]
            )
        trajectory = odeint(
            self.ode_func,
            initial_state,
            integration_times,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            options={"max_num_steps": self.max_num_steps},
        )
        return trajectory[1:] if prepended_zero else trajectory

    def final_risk_logits(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        valid_time_mask: torch.Tensor,
        times: torch.Tensor | None = None,
        delta_t: torch.Tensor | None = None,
        final_time: float = 48.0,
    ) -> torch.Tensor:
        """Return one unsquashed mortality logit per patient at ``final_time``."""
        if final_time < 0:
            raise ValueError("final_time must be non-negative.")
        query_times = torch.tensor(
            [0.0] if final_time == 0 else [0.0, final_time],
            dtype=values.dtype if values.is_floating_point() else torch.float32,
            device=values.device,
        )
        return self.forward(
            values=values,
            mask=mask,
            valid_time_mask=valid_time_mask,
            query_times=query_times,
            times=times,
            delta_t=delta_t,
        )[-1]
