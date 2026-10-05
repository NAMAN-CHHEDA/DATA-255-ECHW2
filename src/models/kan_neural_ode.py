"""KAN-parametrized Neural ODE and post-training interpretability helpers."""

from __future__ import annotations

import inspect
from typing import Any

import torch
from kan import KAN
from torch import nn

from src.config import (
    KAN_GRID,
    KAN_HIDDEN_WIDTH_MULTIPLIER,
    KAN_SPLINE_ORDER,
    N_VARS,
    ODE_ATOL,
    ODE_HIDDEN_DIM,
    ODE_RTOL,
    ODE_SOLVER,
    SEED,
)
from src.models.neural_ode_common import NeuralODEBase

SYMBOLIC_LIBRARY = ["x", "x^2", "exp", "sin", "tanh", "abs"]


class ODEFuncKAN(nn.Module):
    """KAN vector field over the hidden state concatenated with scalar time."""

    def __init__(
        self,
        hidden_dim: int,
        grid: int = KAN_GRID,
        spline_order: int = KAN_SPLINE_ORDER,
        width_multiplier: int = KAN_HIDDEN_WIDTH_MULTIPLIER,
        seed: int = SEED,
    ) -> None:
        super().__init__()
        if min(hidden_dim, grid, spline_order, width_multiplier) < 1:
            raise ValueError(
                "hidden_dim, grid, spline_order, and width_multiplier "
                "must be positive."
            )
        self.hidden_dim = hidden_dim
        self.width = [
            hidden_dim + 1,
            width_multiplier * hidden_dim + 1,
            hidden_dim,
        ]
        # Disabling auto-save prevents model-history files during ordinary
        # construction and smoke tests. Interpretability remains opt-in.
        kan_kwargs: dict[str, Any] = {
            "width": self.width,
            "grid": grid,
            "k": spline_order,
            "seed": seed,
        }
        if "auto_save" in inspect.signature(KAN).parameters:
            kan_kwargs["auto_save"] = False
        self.network = KAN(**kan_kwargs)

    def forward(self, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Return a derivative matching ``state`` shape and dtype/device."""
        time_feature = torch.ones(
            (*state.shape[:-1], 1),
            dtype=state.dtype,
            device=state.device,
        ) * time.to(dtype=state.dtype, device=state.device)
        inputs = torch.cat([state, time_feature], dim=-1)
        derivative = self.network(inputs)
        if derivative.shape != state.shape:
            raise RuntimeError(
                f"KAN derivative shape {tuple(derivative.shape)} does not match "
                f"hidden-state shape {tuple(state.shape)}."
            )
        return derivative.to(dtype=state.dtype, device=state.device)


class KANNeuralODE(NeuralODEBase):
    """Neural ODE using a KAN drift and the shared fair-comparison path."""

    def __init__(
        self,
        n_vars: int = N_VARS,
        hidden_dim: int = ODE_HIDDEN_DIM,
        grid: int = KAN_GRID,
        spline_order: int = KAN_SPLINE_ORDER,
        width_multiplier: int = KAN_HIDDEN_WIDTH_MULTIPLIER,
        method: str = ODE_SOLVER,
        rtol: float = ODE_RTOL,
        atol: float = ODE_ATOL,
        seed: int = SEED,
    ) -> None:
        super().__init__(
            ode_func=ODEFuncKAN(
                hidden_dim=hidden_dim,
                grid=grid,
                spline_order=spline_order,
                width_multiplier=width_multiplier,
                seed=seed,
            ),
            n_vars=n_vars,
            hidden_dim=hidden_dim,
            method=method,
            rtol=rtol,
            atol=atol,
        )

    def plot_learned_edge_functions(
        self,
        beta: int = 100,
        **plot_kwargs: Any,
    ) -> Any:
        """Plot learned KAN edge functions after Experiment 3 training.

        This may create plot files according to pykan's options. It is never
        called by model construction, forward passes, or smoke tests.
        """
        return self.ode_func.network.plot(beta=beta, **plot_kwargs)

    def prune_weak_edges(self, **prune_kwargs: Any) -> KAN:
        """Prune weak KAN edges after training and return the pruned network.

        Pruning changes the learned model and belongs to Experiment 3
        interpretation, not validation or smoke testing.
        """
        pruned = self.ode_func.network.prune(**prune_kwargs)
        if pruned is not None:
            self.ode_func.network = pruned
        return self.ode_func.network

    def symbolic_formulas(
        self,
        auto_symbolic: bool = False,
        print_formulas: bool = False,
    ) -> Any:
        """Return symbolic formulas, optionally fitting the approved library.

        ``auto_symbolic=True`` runs pykan's symbolic fitting with
        ``['x', 'x^2', 'exp', 'sin', 'tanh', 'abs']``. Both fitting and formula
        inspection are post-training Experiment 3 operations and are never
        invoked by the smoke test.
        """
        if auto_symbolic:
            self.ode_func.network.auto_symbolic(lib=SYMBOLIC_LIBRARY)
        formulas = self.ode_func.network.symbolic_formula()
        if print_formulas:
            print(str(formulas))
        return formulas
