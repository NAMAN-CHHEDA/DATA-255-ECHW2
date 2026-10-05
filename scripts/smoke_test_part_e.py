"""In-memory smoke test for the unified Part E harness."""

from __future__ import annotations

import io
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.dont_write_bytecode = True

from src.config import N_VARS, SEED  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.models.gru_baseline import GRUBaseline  # noqa: E402
from src.models.kan_neural_ode import KANNeuralODE  # noqa: E402
from src.models.liquid_model import LiquidRiskModel  # noqa: E402
from src.models.mlp_neural_ode import MLPNeuralODE  # noqa: E402
from src.train import (  # noqa: E402
    FORWARD_ADAPTERS,
    get_forward_adapter,
    train_one_epoch,
)
from src.utils import (  # noqa: E402
    EarlyStopping,
    PeakMemoryTracker,
    count_trainable_parameters,
    set_seed,
)


def _toy_batch() -> dict[str, torch.Tensor]:
    """Create one padded two-patient batch with both mortality classes."""
    times = torch.tensor(
        [[0.0, 1.5, 4.0], [0.0, 2.0, 0.0]],
        dtype=torch.float32,
    )
    values = torch.zeros((2, 3, N_VARS), dtype=torch.float32)
    mask = torch.zeros_like(values)
    delta_t = torch.ones_like(values)
    valid_time_mask = torch.tensor(
        [[True, True, True], [True, True, False]],
        dtype=torch.bool,
    )
    values[0, 0, 0], mask[0, 0, 0] = 0.25, 1.0
    values[0, 1, 4], mask[0, 1, 4] = -0.5, 1.0
    values[0, 2, 7], mask[0, 2, 7] = 0.75, 1.0
    values[1, 0, 6], mask[1, 0, 6] = -0.25, 1.0
    values[1, 1, 8], mask[1, 1, 8] = 0.5, 1.0
    delta_t[0, 1] = 1.5
    delta_t[0, 2] = 2.5
    delta_t[1, 1] = 2.0
    return {
        "times": times,
        "values": values,
        "mask": mask,
        "delta_t": delta_t,
        "valid_time_mask": valid_time_mask,
        "label": torch.tensor([0.0, 1.0]),
    }


def main() -> int:
    """Exercise adapters, one optimizer step, evaluation, and utilities."""
    set_seed(SEED)
    batch = _toy_batch()

    # Third-party KAN/ncps modules may emit informational text. A successful
    # smoke test deliberately exposes only the required PASS line.
    with redirect_stdout(io.StringIO()):
        models = {
            "gru_baseline": GRUBaseline(hidden_size=4),
            "mlp_neural_ode": MLPNeuralODE(hidden_dim=4, width=8),
            "kan_neural_ode": KANNeuralODE(hidden_dim=4, grid=3),
            "ltc": LiquidRiskModel(
                units=8,
                cell_type="ltc",
                ode_unfolds=2,
            ),
            "cfc": LiquidRiskModel(
                units=8,
                cell_type="cfc",
                ode_unfolds=2,
            ),
        }
        assert set(models).issubset(FORWARD_ADAPTERS)
        for model_key, model in models.items():
            adapter = get_forward_adapter(model_key)
            model.eval()
            with torch.no_grad():
                logits = adapter(model, batch)
            assert logits.shape == (2,)
            assert torch.isfinite(logits).all()

        # Exactly one loader batch means exactly one optimizer step.
        training_model = GRUBaseline(hidden_size=4)
        optimizer = torch.optim.Adam(training_model.parameters(), lr=1e-3)
        mean_loss, elapsed_seconds = train_one_epoch(
            training_model,
            [batch],
            optimizer,
            get_forward_adapter("gru_baseline"),
            "cpu",
        )
        assert math.isfinite(mean_loss)
        assert elapsed_seconds >= 0.0

        metrics = evaluate(
            training_model,
            [batch],
            get_forward_adapter("gru_baseline"),
            "cpu",
        )
        assert {"auroc", "auprc", "brier_score", "sample_count"}.issubset(metrics)
        assert metrics["sample_count"] == 2
        assert math.isfinite(float(metrics["auroc"]))
        assert math.isfinite(float(metrics["auprc"]))
        assert math.isfinite(float(metrics["brier_score"]))
        assert "accuracy" not in metrics

        early_stopping = EarlyStopping(patience=2)
        assert early_stopping.step(0.20)
        assert early_stopping.step(0.30)
        assert not early_stopping.step(0.25)
        assert not early_stopping.should_stop
        assert not early_stopping.step(0.24)
        assert early_stopping.should_stop
        assert early_stopping.best_score == 0.30

        assert count_trainable_parameters(training_model) > 0
        with PeakMemoryTracker("cpu") as memory:
            temporary = [index**2 for index in range(100)]
            assert len(temporary) == 100
        assert memory.peak_bytes >= 0
        assert memory.label == "Python-tracked peak memory (not total RAM)"

    print("PASS: Part E unified training and evaluation harness smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
