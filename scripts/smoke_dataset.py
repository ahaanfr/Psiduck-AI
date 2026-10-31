"""Exploratory smoke test: run both configs on the first K problems of a dataset.

    python scripts/smoke_dataset.py amc23 --k 3 --max-new-tokens 4096

Writes results/explore/<dataset>_smoke.jsonl. Exploratory only; not part of the reported results.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from psiduck import data  # noqa: E402
from psiduck.answers import is_correct  # noqa: E402
from psiduck.config import Settings  # noqa: E402
from psiduck.hf_env import REPO_ROOT  # noqa: E402
from psiduck.pipeline import build_solver  # noqa: E402
from psiduck.runner import append_jsonl, problem_seed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    args = ap.parse_args()

    settings = replace(Settings.from_env(), max_new_tokens=args.max_new_tokens)
    from psiduck.backbone import SharedModel

    model = SharedModel(settings.model_name, settings.quantization, settings.seed, thinking=settings.thinking,
                        temperature=settings.temperature, top_p=settings.top_p)
    out = REPO_ROOT / "results" / "explore" / f"{args.dataset}_smoke.jsonl"
    for idx, p in enumerate(data.load(args.dataset)[: args.k]):
        for cfg in ("latent", "baseline"):
            r = build_solver(cfg, model, settings).solve(p.question, seed=problem_seed(settings.seed, args.dataset, idx))
            rec = {"config": cfg, "idx": idx, "problem_id": p.problem_id, "gold": p.answer,
                   "prediction": r.prediction, "correct": is_correct(r.prediction, p.answer),
                   "wall_s": round(r.wall_s, 1), **r.compute, **r.detail}
            append_jsonl(out, rec)
            print(f"[{cfg:8}] #{idx} gold={p.answer} pred={r.prediction} correct={rec['correct']} "
                  f"tokens={r.compute['generated_tokens']} cap={r.detail.get('hit_token_cap')} {r.wall_s:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
