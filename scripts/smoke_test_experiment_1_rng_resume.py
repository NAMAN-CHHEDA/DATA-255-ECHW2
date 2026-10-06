"""Temporary CPU/CUDA-map regression test for resume RNG restoration."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.dont_write_bytecode = True

from scripts.run_experiment_1 import load_checkpoint_payload  # noqa: E402
from src.utils import as_cpu_rng_state  # noqa: E402


def _assert_generator_resume(checkpoint_path: Path, map_location: str) -> None:
    """Load one saved state at ``map_location`` and resume its exact stream."""
    payload = load_checkpoint_payload(
        checkpoint_path,
        map_location=map_location,
    )
    loaded_state = payload["train_generator_state"]
    if map_location == "cuda":
        assert loaded_state.device.type == "cuda"
        assert payload["model_state"]["weight"].device.type == "cuda"
        assert (
            payload["optimizer_state"]["state"][0]["exp_avg"].device.type
            == "cuda"
        )
    else:
        assert payload["model_state"]["weight"].device.type == "cpu"
        assert (
            payload["optimizer_state"]["state"][0]["exp_avg"].device.type
            == "cpu"
        )

    restored_state = as_cpu_rng_state(loaded_state)
    assert restored_state.device.type == "cpu"
    assert restored_state.dtype == torch.uint8
    assert restored_state.is_contiguous()
    assert restored_state.grad_fn is None

    generator = torch.Generator(device="cpu")
    generator.set_state(restored_state)
    actual = torch.rand(16, generator=generator)
    assert torch.equal(actual, payload["expected_next_values"].cpu())


def _assert_global_cpu_rng_restore(state: torch.Tensor) -> None:
    """Exercise the same conversion before ``torch.set_rng_state``."""
    original_state = torch.get_rng_state()
    try:
        torch.set_rng_state(as_cpu_rng_state(state))
        first = torch.rand(8)
        torch.set_rng_state(as_cpu_rng_state(state))
        second = torch.rand(8)
        assert torch.equal(first, second)
    finally:
        torch.set_rng_state(as_cpu_rng_state(original_state))


def main() -> int:
    """Verify CPU loading and conditional CUDA-map loading are resumable."""
    source = torch.Generator(device="cpu")
    source.manual_seed(8893)
    torch.rand(7, generator=source)
    saved_state = source.get_state()
    expected_next_values = torch.rand(16, generator=source)

    with TemporaryDirectory(prefix="echw2_exp1_rng_resume_") as temporary:
        checkpoint_path = Path(temporary) / "rng_checkpoint.pt"
        torch.save(
            {
                "model_state": {"weight": torch.ones(2)},
                "optimizer_state": {
                    "state": {0: {"exp_avg": torch.ones(2)}}
                },
                "train_generator_state": saved_state,
                "expected_next_values": expected_next_values,
            },
            checkpoint_path,
        )
        _assert_generator_resume(checkpoint_path, map_location="cpu")
        _assert_global_cpu_rng_restore(saved_state)

        if torch.cuda.is_available():
            _assert_generator_resume(checkpoint_path, map_location="cuda")
            cuda_status = "passed"
        else:
            cuda_status = "skipped (CUDA unavailable)"

    print("PASS: CPU RNG-state checkpoint loading and resume")
    print(f"CUDA map_location RNG-state resume: {cuda_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
