"""In-memory smoke test for Part D LTC and CfC risk models."""

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
from src.models.liquid_model import LiquidRiskModel  # noqa: E402


def _synthetic_batch() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Create two unequal padded irregular sequences entirely in memory."""
    values = torch.zeros((2, 4, N_VARS), dtype=torch.float32)
    mask = torch.zeros_like(values)
    delta_t = torch.ones_like(values)
    valid_time_mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
        ],
        dtype=torch.bool,
    )

    values[0, 0, 0], mask[0, 0, 0] = 0.25, 1.0
    values[0, 1, 4], mask[0, 1, 4] = -0.5, 1.0
    values[0, 2, 7], mask[0, 2, 7] = 0.75, 1.0
    values[0, 3, 11], mask[0, 3, 11] = 1.0, 1.0
    values[1, 0, 2], mask[1, 0, 2] = -0.25, 1.0
    values[1, 1, 8], mask[1, 1, 8] = 0.5, 1.0

    delta_t[0, 1] = 0.5
    delta_t[0, 2] = 2.0
    delta_t[0, 3] = 4.0
    delta_t[1, 1] = 3.0
    return values, mask, delta_t, valid_time_mask


def main() -> int:
    """Exercise both ncps variants without files, training, or metrics."""
    torch.manual_seed(SEED)
    values, mask, delta_t, valid_time_mask = _synthetic_batch()

    # Keep successful output stable if ncps emits informational messages.
    with redirect_stdout(io.StringIO()):
        for cell_type in ("ltc", "cfc"):
            model = LiquidRiskModel(
                units=8,
                cell_type=cell_type,
                ode_unfolds=2,
            )
            trajectory = model(values, mask, delta_t, valid_time_mask)
            final_logits = model.forward_final(
                values,
                mask,
                delta_t,
                valid_time_mask,
            )
            assert trajectory.shape == (2, 4)
            assert final_logits.shape == (2,)
            assert torch.isfinite(trajectory).all()
            assert torch.isfinite(final_logits).all()

            altered_values = values.clone()
            altered_values[~valid_time_mask] = 1_000_000.0
            altered_final_logits = model.forward_final(
                altered_values,
                mask,
                delta_t,
                valid_time_mask,
            )
            assert torch.allclose(
                final_logits,
                altered_final_logits,
                rtol=1e-5,
                atol=1e-6,
            )

        try:
            LiquidRiskModel(cell_type="unsupported")
        except ValueError as exc:
            assert "Unsupported cell_type" in str(exc)
        else:
            raise AssertionError("Invalid cell_type did not raise ValueError.")

    print("PASS: Part D LTC and CfC smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
