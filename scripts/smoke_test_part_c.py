"""In-memory smoke test for the Part C KAN Neural ODE."""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.dont_write_bytecode = True

from src.config import N_VARS, SEED  # noqa: E402
from src.models.kan_neural_ode import KANNeuralODE  # noqa: E402
from src.models.mlp_neural_ode import MLPNeuralODE  # noqa: E402


def _synthetic_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return tiny padded values, observation masks, and valid-time masks."""
    values = torch.zeros((2, 3, N_VARS), dtype=torch.float32)
    mask = torch.zeros_like(values)
    valid_time_mask = torch.tensor(
        [[True, True, False], [True, False, False]],
        dtype=torch.bool,
    )
    values[0, 0, 0], mask[0, 0, 0] = 0.25, 1.0
    values[0, 1, 4], mask[0, 1, 4] = -0.5, 1.0
    values[1, 0, 6], mask[1, 0, 6] = 0.75, 1.0
    return values, mask, valid_time_mask


def main() -> int:
    """Exercise the real KAN ODE without files, plots, or model outputs."""
    torch.manual_seed(SEED)
    values, mask, valid_time_mask = _synthetic_batch()

    # Suppress any informational output from the third-party KAN package so
    # this script has exactly one successful output line.
    with redirect_stdout(io.StringIO()):
        model = KANNeuralODE(hidden_dim=4, grid=3)
        query_times = torch.tensor([0.0, 1.0, 48.0], dtype=torch.float32)
        trajectory = model(values, mask, valid_time_mask, query_times)
        assert trajectory.shape == (3, 2)
        assert torch.isfinite(trajectory).all()

        final_logits = model.final_risk_logits(values, mask, valid_time_mask)
        assert final_logits.shape == (2,)
        assert torch.isfinite(final_logits).all()

        # Both variants inherit these exact implementations from one base.
        assert MLPNeuralODE.forward is KANNeuralODE.forward
        assert MLPNeuralODE.final_risk_logits is KANNeuralODE.final_risk_logits

        try:
            model(
                values,
                mask,
                valid_time_mask,
                torch.tensor([0.0, 48.0, 1.0]),
            )
        except ValueError as exc:
            assert "sorted" in str(exc)
        else:
            raise AssertionError("Unsorted ODE query times did not raise ValueError.")

    print("PASS: Part C KAN Neural ODE smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
