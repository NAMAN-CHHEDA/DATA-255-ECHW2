"""Check that evaluation batch size does not change model outputs.

Evaluates each finished Experiment 1 model's test logits at batch size 32 and
at the configured evaluation batch size and reports the largest logit and
metric differences. Not part of any run; execute it when you want proof.
Read-only: it writes nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiment_1 import PRIMARY_MODEL_NAMES, build_model, load_model_state  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--models", nargs="+", default=list(PRIMARY_MODEL_NAMES))
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader

    from src.config import BATCH_SIZE, EVAL_BATCH_SIZE, EXPERIMENT_1_DIR, NUM_WORKERS, PART_A_CACHE_PATH, resolve_device
    from src.dataset import materialize_records, materialized_collator
    from src.evaluate import compute_mortality_metrics, predict_logits
    from src.train import get_forward_adapter

    from scripts.run_experiment_1 import validate_part_a_cache

    device = resolve_device()
    dataset = materialize_records(validate_part_a_cache(PART_A_CACHE_PATH)["records"]["test"], "test")
    status = 0
    for key in args.models:
        path = EXPERIMENT_1_DIR / "checkpoints" / key / "best.pt"
        if not path.is_file():
            print(f"{key}: no checkpoint, skipped")
            continue
        model = build_model(key).to(device)
        model.load_state_dict(load_model_state(path, map_location=device))
        forward_fn = get_forward_adapter(key)
        results = {}
        for size in (BATCH_SIZE, EVAL_BATCH_SIZE):
            loader = DataLoader(dataset, batch_size=size, shuffle=False, num_workers=NUM_WORKERS, collate_fn=materialized_collator(key))
            labels, logits = predict_logits(model, loader, forward_fn, device)
            results[size] = (logits, compute_mortality_metrics(labels, logits))
        delta = (results[BATCH_SIZE][0] - results[EVAL_BATCH_SIZE][0]).abs().max().item()
        d_auroc = abs(results[BATCH_SIZE][1]["auroc"] - results[EVAL_BATCH_SIZE][1]["auroc"])
        flag = "OK" if delta <= args.tolerance else "DIFFERS"
        status |= flag != "OK"
        print(f"{key}: max|dlogit|={delta:.3e} dAUROC={d_auroc:.3e} -> {flag}")
    return int(status)


if __name__ == "__main__":
    raise SystemExit(main())
