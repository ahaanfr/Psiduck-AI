"""Resumable run of both configs on one extra dataset (Experiments 2 and 3).

    python scripts/run_extra.py amc23 --max-new-tokens 4096
    python scripts/run_extra.py gsm8k --max-new-tokens 4096 --shuffled --n 200 --interleave --out gsm8k_4096

Writes results/<out>/{latent,baseline}.jsonl in the same format as the main
runner, so analysis.dataset_section can score it. Re-running skips finished problems.
--shuffled uses the fixed seed-42 order (the same problems, in the same
order and with the same per-problem seeds, as the main GSM8K run). --interleave
runs latent then baseline on each problem before moving on, so a run cut short
still leaves complete pairs.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from psiduck import analysis, data, system  # noqa: E402
from psiduck.answers import is_correct  # noqa: E402
from psiduck.config import Settings  # noqa: E402
from psiduck.hf_env import REPO_ROOT  # noqa: E402
from psiduck.pipeline import build_solver  # noqa: E402
from psiduck.runner import append_jsonl, problem_seed  # noqa: E402

CONFIGS = ("latent", "baseline")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--max-new-tokens", type=int, required=True)
    ap.add_argument("--n", type=int, default=None, help="only the first N problems")
    ap.add_argument("--shuffled", action="store_true", help="use the fixed seed-42 order")
    ap.add_argument("--interleave", action="store_true", help="both configs per problem, then the next problem")
    ap.add_argument("--out", default=None, help="results subfolder (default: the dataset name)")
    args = ap.parse_args()

    settings = replace(Settings.from_env(), max_new_tokens=args.max_new_tokens)
    topic = system.ntfy_topic(REPO_ROOT / "results" / "ntfy_topic.txt")
    problems = data.load(args.dataset)
    if args.shuffled:
        problems = data.shuffled(problems, settings.seed)
    problems = problems[: args.n]
    out = args.out or args.dataset
    from psiduck.backbone import SharedModel

    model = SharedModel(settings.model_name, settings.quantization, settings.seed, thinking=settings.thinking,
                        temperature=settings.temperature, top_p=settings.top_p)
    paths = {cfg: REPO_ROOT / "results" / out / f"{cfg}.jsonl" for cfg in CONFIGS}
    done = {cfg: analysis.read_jsonl(paths[cfg]) for cfg in CONFIGS}
    solvers = {cfg: build_solver(cfg, model, settings) for cfg in CONFIGS}

    def solve(cfg: str, idx: int, p) -> None:
        if idx in done[cfg]:
            return
        base = {"idx": idx, "problem_id": p.problem_id, "gold": p.answer}
        try:
            r = solvers[cfg].solve(p.question, seed=problem_seed(settings.seed, args.dataset, idx))
            rec = {**base, "prediction": r.prediction, "correct": is_correct(r.prediction, p.answer),
                   "wall_s": round(r.wall_s, 3), **r.compute, **r.detail, "text": r.text}
        except Exception as e:
            rec = {**base, "prediction": None, "correct": False, "error": f"{type(e).__name__}: {e}"[:500]}
            print(traceback.format_exc(), flush=True)
            model.free()
        append_jsonl(paths[cfg], rec)
        done[cfg][idx] = rec
        print(f"{out}/{cfg} {len(done[cfg])}/{len(problems)}", flush=True)

    def finished(cfg: str) -> None:
        s = analysis.config_stats(list(done[cfg].values()))
        system.notify(topic, f"{out}/{cfg} finished: {s['correct']}/{s['n']} correct, "
                             f"mean {s['gen_tokens']:.0f} tokens, {s['cap_hits']} hit the cap.")

    if args.interleave:
        for idx, p in enumerate(problems):
            for cfg in CONFIGS:
                solve(cfg, idx, p)
        for cfg in CONFIGS:
            finished(cfg)
    else:
        for cfg in CONFIGS:
            for idx, p in enumerate(problems):
                solve(cfg, idx, p)
            finished(cfg)
    system.notify(topic, f"{out} run DONE. Stop the pod after the results are downloaded.", priority="high")
    return 0


if __name__ == "__main__":
    sys.exit(main())
