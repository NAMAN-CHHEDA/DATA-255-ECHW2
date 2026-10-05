"""In-memory smoke test for the Part B GRU and MLP Neural ODE."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.dont_write_bytecode = True

from src.config import N_VARS, SEED  # noqa: E402
from src.models.gru_baseline import (  # noqa: E402
    GRUBaseline,
    prepare_hourly_gru_inputs,
)
from src.models.mlp_neural_ode import MLPNeuralODE  # noqa: E402


def _synthetic_padded_batch() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Create two padded irregular sequences entirely in memory."""
    times = torch.tensor(
        [
            [0.2, 1.1, 1.8, 47.9, 0.0],
            [0.5, 2.2, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    values = torch.zeros((2, 5, N_VARS), dtype=torch.float32)
    mask = torch.zeros_like(values)
    valid_time_mask = torch.tensor(
        [
            [True, True, True, True, False],
            [True, True, False, False, False],
        ],
        dtype=torch.bool,
    )

    values[0, 0, 0], mask[0, 0, 0] = 1.0, 1.0
    values[0, 1, 0], mask[0, 1, 0] = 3.0, 1.0
    values[0, 2, 0], mask[0, 2, 0] = 5.0, 1.0
    values[0, 3, 4], mask[0, 3, 4] = 0.25, 1.0
    values[1, 0, 6], mask[1, 0, 6] = -0.5, 1.0
    values[1, 1, 7], mask[1, 1, 7] = 0.75, 1.0
    return times, values, mask, valid_time_mask


def main() -> int:
    """Exercise both Part B models without data files or persistent output."""
    torch.manual_seed(SEED)
    times, values, mask, valid_time_mask = _synthetic_padded_batch()

    hourly_values, hourly_mask = prepare_hourly_gru_inputs(
        times,
        values,
        mask,
        valid_time_mask,
    )
    assert hourly_values.shape == (2, 48, 12)
    assert hourly_mask.shape == (2, 48, 12)
    # The two HR measurements in hour 1 are averaged before forward-fill.
    assert torch.isclose(hourly_values[0, 1, 0], torch.tensor(4.0))
    assert hourly_mask[0, 1, 0] == 1
    assert hourly_mask[0, 2, 0] == 0
    assert torch.isclose(hourly_values[0, 2, 0], torch.tensor(4.0))

    gru = GRUBaseline()
    gru_logits = gru(hourly_values, hourly_mask)
    assert gru_logits.shape == (2,)
    assert torch.isfinite(gru_logits).all()

    neural_ode = MLPNeuralODE()
    query_times = torch.tensor([0.0, 12.0, 48.0], dtype=torch.float32)
    trajectory = neural_ode(values, mask, valid_time_mask, query_times)
    assert trajectory.shape == (len(query_times), 2)
    assert torch.isfinite(trajectory).all()

    final_logits = neural_ode.final_risk_logits(
        values,
        mask,
        valid_time_mask,
    )
    assert final_logits.shape == (2,)
    assert torch.isfinite(final_logits).all()

    initial_state = neural_ode.encode_initial_state(
        values,
        mask,
        valid_time_mask,
    )
    altered_values = values.clone()
    altered_mask = mask.clone()
    padded = ~valid_time_mask
    altered_values[padded] = 1_000_000.0
    altered_mask[padded] = 1.0
    altered_initial_state = neural_ode.encode_initial_state(
        altered_values,
        altered_mask,
        valid_time_mask,
    )
    assert torch.allclose(initial_state, altered_initial_state)

    try:
        neural_ode(
            values,
            mask,
            valid_time_mask,
            torch.tensor([0.0, 48.0, 12.0]),
        )
    except ValueError as exc:
        assert "sorted" in str(exc)
    else:
        raise AssertionError("Unsorted ODE query times did not raise ValueError.")

    print("PASS: Part B GRU and MLP Neural ODE smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
