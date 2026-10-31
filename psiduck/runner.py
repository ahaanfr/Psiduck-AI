"""Unattended, resumable eval driver.

Order: smoke test (3 GSM8K problems per config, timing only) -> lock N ->
GSM8K latent -> GSM8K baseline -> AIME latent -> AIME baseline -> analysis.

Every finished problem is appended to results/<dataset>/<config>.jsonl and
fsync'd, so a crash or reboot resumes where it stopped. A problem that kills
the process twice is recorded as an error (counted incorrect) and skipped.
"""

from __future__ import annotations

import json
import os
import platform
import threading
import time
import traceback
from collections import Counter
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from psiduck import analysis, data, system
from psiduck.answers import is_correct
from psiduck.config import Settings
from psiduck.hf_env import REPO_ROOT
from psiduck.pipeline import BaselineSolver, build_solver

SMOKE_N = 3
CONFIGS = ("latent", "baseline")
JOBS = (("gsm8k", "latent"), ("gsm8k", "baseline"), ("aime", "latent"), ("aime", "baseline"))
MAX_CONSECUTIVE_ERRORS = 5
MAX_ATTEMPTS = 2
FALLBACK_MODEL = "Qwen/Qwen3-4B"
MIN_N_PRIMARY = 200  # below this, Qwen3-8B counts as "unusably slow"


def choose_n(sec_per_problem: dict[str, float], budget_s: float = 13 * 3600,
             cap: int = 1319, step: int = 50) -> int:
    """Largest multiple of `step` such that every config runs N problems in budget_s."""
    total = sum(sec_per_problem.values())
    n = int(budget_s / total) // step * step if total > 0 else cap
    return max(step, min(cap, n))


def problem_seed(seed: int, dataset: str, idx: int) -> int:
    """Per-problem sampling seed: same for every config (paired), stable across resumes."""
    return seed * 1_000_003 + (0 if dataset == "gsm8k" else 500_000) + idx


def append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


class ConfigFailed(Exception):
    pass


class Runner:
    def __init__(self, settings, make_backbone: Callable, results_dir: Path = REPO_ROOT / "results",
                 readme: Path = REPO_ROOT / "README.md",
                 allow_battery: bool = False, notify: Callable = system.notify, status_interval: float = 30.0,
                 load_dataset: Callable = data.load):
        self.settings = settings
        self.make_backbone = make_backbone
        self.dir = results_dir
        self.readme = readme
        self.allow_battery = allow_battery
        self._notify = notify
        self.status_interval = status_interval
        self.load_dataset = load_dataset
        self.topic = system.ntfy_topic(self.dir / "ntfy_topic.txt")
        self.state_path = self.dir / "state.json"
        self.plan_path = self.dir / "plan_lock.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.backbone = None
        self._stop = threading.Event()
        self._battery_warned = False

    # -- small helpers --------------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            with (self.dir / "run.log").open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def notify(self, msg: str, priority: str = "default") -> None:
        if not self._notify(self.topic, msg, priority=priority):
            self.log(f"(ntfy failed, continuing) {msg}")

    def set_state(self, job: str, value: str) -> None:
        self.state[job] = value
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2))
        tmp.replace(self.state_path)

    def plan(self) -> Optional[dict]:
        return json.loads(self.plan_path.read_text()) if self.plan_path.exists() else None

    def check_battery(self) -> None:
        if system.on_battery():
            if not self._battery_warned:
                self.log("WARNING: laptop is on battery. Plug it in for the overnight run.")
                self.notify("WARNING: laptop is on battery power. Plug it in.", priority="high")
                self._battery_warned = True
        else:
            self._battery_warned = False

    def get_backbone(self):
        if self.backbone is None:
            s = self.settings
            self.log(f"Loading {s.model_name} ({s.quantization}) ...")
            t0 = time.time()
            self.backbone = self.make_backbone(s)
            self.log(f"Model loaded in {time.time() - t0:.0f}s")
        return self.backbone

    # -- problems -------------------------------------------------------------
    def problems(self, dataset: str) -> list:
        if dataset == "gsm8k":
            return data.shuffled(self.load_dataset("gsm8k"), self.settings.seed)
        return self.load_dataset("aime")  # all 60, fixed order: 2024 then 2025

    def job_n(self, dataset: str) -> int:
        return self.plan()["n"] if dataset == "gsm8k" else 60

    # -- smoke test -----------------------------------------------------------
    def smoke(self) -> dict:
        """Time SMOKE_N problems per config. Records timing only, never correctness."""
        if not self.allow_battery and system.on_battery():
            self.log("On battery: waiting for AC power before timing the smoke test (battery throttles the GPU).")
            self.notify("Smoke test is waiting for AC power. Plug the laptop in.", priority="high")
            while system.on_battery():
                time.sleep(30)
        b = self.get_backbone()
        BaselineSolver(b, 8).solve(data.SAMPLE_PROBLEMS[0].question)  # warm-up, untimed
        probs = self.problems("gsm8k")[:SMOKE_N]
        sec = {}
        for cfg in CONFIGS:
            path = self.dir / "smoke" / f"{cfg}.jsonl"
            recs = analysis.read_jsonl(path)
            solver = build_solver(cfg, b, self.settings)
            for i, p in enumerate(probs):
                if i in recs:
                    continue
                self.set_state(f"smoke/{cfg}", f"running {i + 1}/{SMOKE_N}")
                r = solver.solve(p.question, seed=problem_seed(self.settings.seed, "gsm8k", i))
                rec = {"idx": i, "wall_s": r.wall_s, **r.compute, "hit_token_cap": r.detail.get("hit_token_cap")}
                append_jsonl(path, rec)
                recs[i] = rec
            sec[cfg] = sum(r["wall_s"] for r in recs.values()) / len(recs)
            self.set_state(f"smoke/{cfg}", "done")
        return sec

    def _switch_to_fallback(self, reason: str) -> None:
        import gc
        import shutil

        self.log(f"Falling back to {FALLBACK_MODEL}: {reason}")
        self.notify(f"Falling back to {FALLBACK_MODEL}: {reason}"[:300], priority="high")
        self.backbone = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        smoke = self.dir / "smoke"
        if smoke.exists():
            shutil.move(str(smoke), str(self.dir / f"smoke_{self.settings.model_name.replace('/', '_')}"))
        self.settings = replace(self.settings, model_name=FALLBACK_MODEL)

    def lock_plan(self) -> dict:
        plan = self.plan()
        if plan:
            self.settings = Settings(**plan["settings"])  # a resumed run always uses the locked settings
            return plan
        fallback_reason = None
        try:
            sec = self.smoke()
        except Exception as e:
            if self.settings.model_name == FALLBACK_MODEL or "out of memory" not in str(e).lower():
                raise
            fallback_reason = f"out of memory with {self.settings.model_name}: {e}"[:300]
            self._switch_to_fallback(fallback_reason)
            sec = self.smoke()
        budget_s = self.settings.gsm8k_budget_hours * 3600
        n = choose_n(sec, budget_s=budget_s)
        if n < MIN_N_PRIMARY and self.settings.model_name != FALLBACK_MODEL:
            fallback_reason = (f"{self.settings.model_name} too slow: N would be {n} < {MIN_N_PRIMARY} "
                               f"(smoke s/problem {sec})")
            self._switch_to_fallback(fallback_reason)
            sec = self.smoke()
            n = choose_n(sec, budget_s=budget_s)
        import torch
        import transformers

        try:
            import bitsandbytes
            bnb = bitsandbytes.__version__
        except Exception:
            bnb = None
        plan = {
            "n": n,
            "smoke_sec_per_problem": sec,
            "gsm8k_budget_hours": self.settings.gsm8k_budget_hours,
            "settings": self.settings.as_dict(),
            "fallback_reason": fallback_reason,
            "locked_at": datetime.now().isoformat(timespec="seconds"),
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "transformers": transformers.__version__,
                "bitsandbytes": bnb,
            },
        }
        self.dir.mkdir(parents=True, exist_ok=True)
        self.plan_path.write_text(json.dumps(plan, indent=2))
        msg = (f"Smoke done. sec/problem: " + ", ".join(f"{k} {v:.1f}" for k, v in sec.items())
               + f". Locked N = {n} GSM8K problems per config.")
        self.log(msg)
        self.notify(msg)
        return plan

    # -- full runs --------------------------------------------------------------
    def run_job(self, dataset: str, cfg: str) -> None:
        job = f"{dataset}/{cfg}"
        n = self.job_n(dataset)
        probs = self.problems(dataset)[:n]
        path = self.dir / dataset / f"{cfg}.jsonl"
        attempts_path = path.with_suffix(".attempts")
        done = analysis.read_jsonl(path)
        attempts = Counter(
            int(x) for x in (attempts_path.read_text().split() if attempts_path.exists() else [])
        )
        if len(done) >= n:
            self.set_state(job, "done")
            return
        solver = build_solver(cfg, self.get_backbone(), self.settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.set_state(job, "running")
        self.log(f"{job}: {len(done)}/{n} already done, resuming")
        consecutive = 0
        for idx, p in enumerate(probs):
            if idx in done:
                continue
            base = {"idx": idx, "problem_id": p.problem_id, "gold": p.answer}
            if attempts[idx] >= MAX_ATTEMPTS:
                rec = {**base, "prediction": None, "correct": False,
                       "error": f"process died on this problem {attempts[idx]} times"}
                append_jsonl(path, rec)
                done[idx] = rec
                self.log(f"{job}: idx {idx} crashed the process {attempts[idx]}x; recorded as error")
                continue
            with attempts_path.open("a") as f:
                f.write(f"{idx}\n")
            try:
                r = solver.solve(p.question, seed=problem_seed(self.settings.seed, dataset, idx))
                rec = {**base, "prediction": r.prediction, "correct": is_correct(r.prediction, p.answer),
                       "wall_s": round(r.wall_s, 3), **r.compute, **r.detail, "text": r.text}
                consecutive = 0
            except Exception as e:
                consecutive += 1
                rec = {**base, "prediction": None, "correct": False,
                       "error": f"{type(e).__name__}: {e}"[:500]}
                self.log(f"{job}: idx {idx} error: {rec['error']}\n{traceback.format_exc()}")
                try:
                    self.get_backbone().free()
                except Exception:
                    pass
            append_jsonl(path, rec)
            done[idx] = rec
            if consecutive >= MAX_CONSECUTIVE_ERRORS:
                raise ConfigFailed(f"{consecutive} consecutive errors; last: {rec['error']}")
        self.set_state(job, "done")
        s = analysis.config_stats(list(done.values()))
        msg = (f"{job} finished: {s['correct']}/{s['n']} correct ({100 * s['correct'] / s['n']:.1f}%), "
               f"mean {s['gen_tokens']:.0f} generated tokens, {s['wall_s']:.1f}s/problem, {s['errors']} errors.")
        self.log(msg)
        self.notify(msg)

    # -- status -----------------------------------------------------------------
    def status_text(self) -> str:
        plan = self.plan()
        lines = [f"Psiduck eval status  (updated {datetime.now():%Y-%m-%d %H:%M:%S})",
                 f"ntfy topic: {self.topic}   (subscribe at https://ntfy.sh/{self.topic})",
                 f"power: {'BATTERY - plug in!' if system.on_battery() else 'AC'}", ""]
        if plan is None:
            lines.append("Smoke test: " + ", ".join(f"{c}={self.state.get('smoke/' + c, 'pending')}" for c in CONFIGS))
            return "\n".join(lines) + "\n"
        lines.append(f"N (GSM8K, locked) = {plan['n']}")
        lines.append(f"{'job':<18}{'state':<12}{'progress':<14}{'s/problem':<11}ETA")
        smoke = plan["smoke_sec_per_problem"]
        for dataset, cfg in JOBS:
            job = f"{dataset}/{cfg}"
            n = self.job_n(dataset)
            recs = analysis.read_jsonl(self.dir / dataset / f"{cfg}.jsonl")
            k = len(recs)
            walls = [r["wall_s"] for r in recs.values() if "wall_s" in r]
            spp = sum(walls) / len(walls) if walls else smoke.get(cfg, 0)
            eta = (n - k) * spp
            eta_s = "-" if k >= n else f"{eta / 3600:.1f} h"
            lines.append(f"{job:<18}{self.state.get(job, 'pending'):<12}{f'{k}/{n}':<14}{spp:<11.1f}{eta_s}")
        finished = [j for j in JOBS if self.state.get(f"{j[0]}/{j[1]}") == "done"]
        if finished:
            lines += ["", "Finished configs:"]
            for dataset, cfg in finished:
                recs = analysis.read_jsonl(self.dir / dataset / f"{cfg}.jsonl")
                s = analysis.config_stats(list(recs.values()))
                lines.append(f"  {dataset}/{cfg}: {s['correct']}/{s['n']} ({100 * s['correct'] / s['n']:.1f}%), "
                             f"mean gen tokens {s['gen_tokens']:.0f}, {s['wall_s']:.1f}s/problem, errors {s['errors']}")
        summary = self.dir / "SUMMARY.md"
        if summary.exists() and all(self.state.get(f"{d}/{c}") in ("done", "failed") for d, c in JOBS):
            lines += ["", summary.read_text(encoding="utf-8")]
        return "\n".join(lines) + "\n"

    def write_status(self) -> None:
        try:
            tmp = self.dir / "STATUS.tmp"
            tmp.write_text(self.status_text(), encoding="utf-8")
            tmp.replace(self.dir / "STATUS.txt")
        except Exception as e:
            self.log(f"(status write failed: {e})")

    def _status_loop(self) -> None:
        while not self._stop.wait(self.status_interval):
            self.write_status()
            self.check_battery()

    # -- analysis ---------------------------------------------------------------
    def analyze(self) -> str:
        summary = analysis.build_summary(self.dir, self.plan())
        (self.dir.parent / "SUMMARY.md").write_text(summary, encoding="utf-8")
        (self.dir / "SUMMARY.md").write_text(summary, encoding="utf-8")
        analysis.inject_readme(self.readme, analysis.build_summary(self.dir, self.plan(), show_p=False))
        return summary

    # -- main -------------------------------------------------------------------
    def run(self) -> int:
        self.dir.mkdir(parents=True, exist_ok=True)
        awake = system.keep_awake()
        self.log(f"ntfy topic: {self.topic}  (keep-awake {'on' if awake else 'UNAVAILABLE'})")
        self.check_battery()
        self.write_status()
        t = threading.Thread(target=self._status_loop, daemon=True)
        t.start()
        failures = 0
        try:
            try:
                for name in ("gsm8k", "aime"):
                    self.load_dataset(name)  # download + cache before anything slow
                self.lock_plan()
            except Exception as e:
                self.log(f"Setup/smoke failed: {e}\n{traceback.format_exc()}")
                self.notify(f"FAILED during setup/smoke test: {type(e).__name__}: {e}"[:300], priority="high")
                return 2
            for dataset, cfg in JOBS:
                job = f"{dataset}/{cfg}"
                if self.state.get(job) in ("done", "failed"):
                    continue
                try:
                    self.run_job(dataset, cfg)
                except Exception as e:
                    failures += 1
                    self.set_state(job, "failed")
                    self.log(f"{job} FAILED: {e}\n{traceback.format_exc()}")
                    self.notify(f"{job} FAILED: {type(e).__name__}: {e}"[:300], priority="high")
                self.write_status()
            summary = self.analyze()
            self.write_status()
            self.notify("All done. " + ("%d config(s) failed. " % failures if failures else "")
                        + "SUMMARY.md written.", priority="high")
            self.log("All done.\n" + summary)
            return 0
        finally:
            self._stop.set()
            self.write_status()
            system.allow_sleep()
