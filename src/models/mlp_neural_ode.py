"""MLP-parametrized Neural ODE baseline for Part B."""

from __future__ import annotations

import torch
from torch import nn

from src.config import (
    N_VARS,
    ODE_ATOL,
    ODE_HIDDEN_DIM,
    ODE_RTOL,
    ODE_SOLVER,
    ODE_WIDTH,
)
from src.models.neural_ode_common import NeuralODEBase


class ODEFuncMLP(nn.Module):
    """Time-dependent MLP vector field ``dh/dt = f(t, h)``."""

    def __init__(self, hidden_dim: int, width: int) -> None:
        super().__init__()
        if hidden_dim < 1 or width < 1:
            raise ValueError("hidden_dim and width must be positive.")
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(hidden_dim + 1, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, hidden_dim),
        )

    def forward(self, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Return a derivative with the same shape as ``state``."""
        time_feature = torch.ones(
            (*state.shape[:-1], 1),
            dtype=state.dtype,
            device=state.device,
        ) * time.to(dtype=state.dtype, device=state.device)
        return self.network(torch.cat([state, time_feature], dim=-1))


class MLPNeuralODE(NeuralODEBase):
    """Neural ODE whose latent vector field is a multilayer perceptron."""

    def __init__(
        self,
        n_vars: int = N_VARS,
        hidden_dim: int = ODE_HIDDEN_DIM,
        width: int = ODE_WIDTH,
        method: str = ODE_SOLVER,
        rtol: float = ODE_RTOL,
        atol: float = ODE_ATOL,
    ) -> None:
        if width < 1:
            raise ValueError("width must be positive.")
        super().__init__(
            ode_func=ODEFuncMLP(hidden_dim=hidden_dim, width=width),
            n_vars=n_vars,
            hidden_dim=hidden_dim,
            method=method,
            rtol=rtol,
            atol=atol,
        )
