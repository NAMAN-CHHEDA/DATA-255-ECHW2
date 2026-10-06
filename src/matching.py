"""Capacity matching of the KAN-ODE to the MLP-ODE (direct search).

The KAN hidden width (4..64) and grid ({3, 4, 5}, spline order 3) are chosen to
minimize the gap to the MLP-ODE trainable-parameter count at the same latent
dimension. Only trainable parameters count (pykan's symbolic affine is frozen).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.config import KAN_SPLINE_ORDER, N_VARS, ODE_WIDTH

KAN_HIDDEN_WIDTHS = tuple(range(4, 65))
KAN_GRIDS = (3, 4, 5)


def mlp_parameter_count(hidden_dim: int) -> int:
    from src.models.mlp_neural_ode import MLPNeuralODE
    from src.utils import count_trainable_parameters

    return count_trainable_parameters(
        MLPNeuralODE(hidden_dim=hidden_dim, width=ODE_WIDTH)
    )


def analytic_kan_parameter_count(
    hidden_dim: int, hidden_width: int, grid: int, spline_order: int
) -> int:
    """encoder ``(3n+1)h + h`` + head ``h+1`` + KAN edges x ``(G+k+2)``."""
    encoder = (3 * N_VARS + 1) * hidden_dim + hidden_dim
    head = hidden_dim + 1
    edges = (hidden_dim + 1) * hidden_width + hidden_width * hidden_dim
    return encoder + head + edges * (grid + spline_order + 2)


def build_kan(hidden_dim: int, hidden_width: int, grid: int) -> Any:
    from src.models.kan_neural_ode import KANNeuralODE

    return KANNeuralODE(
        hidden_dim=hidden_dim,
        hidden_width=hidden_width,
        grid=grid,
        spline_order=KAN_SPLINE_ORDER,
    )


@lru_cache(maxsize=None)
def _match(hidden_dim: int) -> str:
    import json

    from src.utils import count_trainable_parameters

    target = mlp_parameter_count(hidden_dim)
    candidates = [
        {
            "hidden_width": width,
            "grid": grid,
            "parameters": analytic_kan_parameter_count(
                hidden_dim, width, grid, KAN_SPLINE_ORDER
            ),
        }
        for width in KAN_HIDDEN_WIDTHS
        for grid in KAN_GRIDS
    ]
    for candidate in candidates:
        candidate["gap"] = candidate["parameters"] - target
    candidates.sort(key=lambda c: (abs(c["gap"]), c["hidden_width"], c["grid"]))
    chosen = dict(candidates[0])
    measured = count_trainable_parameters(
        build_kan(hidden_dim, chosen["hidden_width"], chosen["grid"])
    )
    if measured != chosen["parameters"]:
        raise RuntimeError(
            f"Analytic KAN count {chosen['parameters']} != measured {measured} "
            f"for hidden_dim={hidden_dim}, {chosen}."
        )
    chosen["gap_pct"] = 100.0 * chosen["gap"] / target
    return json.dumps(
        {
            "hidden_dim": hidden_dim,
            "mlp_parameters": target,
            "mlp_width": ODE_WIDTH,
            "kan_spline_order": KAN_SPLINE_ORDER,
            "chosen": chosen,
            "runner_ups": candidates[1:4],
        }
    )


def match_kan_to_mlp(hidden_dim: int) -> dict[str, Any]:
    """Return the matching entry for one MLP hidden size (cached)."""
    import json

    return json.loads(_match(hidden_dim))


def format_match(entry: dict[str, Any]) -> str:
    chosen = entry["chosen"]
    return (
        f"KAN capacity match (hidden_dim={entry['hidden_dim']}): "
        f"MLP-ODE={entry['mlp_parameters']:,} params; KAN width="
        f"{chosen['hidden_width']}, grid={chosen['grid']}, "
        f"k={entry['kan_spline_order']} -> {chosen['parameters']:,} params "
        f"({chosen['gap']:+d}, {chosen['gap_pct']:+.2f}%)"
    )


def build_matched_kan(hidden_dim: int) -> Any:
    chosen = match_kan_to_mlp(hidden_dim)["chosen"]
    return build_kan(hidden_dim, chosen["hidden_width"], chosen["grid"])
