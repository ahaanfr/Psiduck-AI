"""Benchmark loaders: GSM8K (test, 1,319) and AIME 2024 + 2025 (60).

Each dataset is downloaded from the Hugging Face Hub once, with retries and
exponential backoff, then written to data/<name>.jsonl. Every later load reads
only that local file, so an overnight run never touches the network for data.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

from psiduck import hf_env  # noqa: F401  (sets HF_HOME before datasets is imported)
from psiduck.answers import gsm8k_gold, normalize_number

DATA_DIR = hf_env.REPO_ROOT / "data"


@dataclass(frozen=True)
class Problem:
    problem_id: str
    question: str
    answer: str  # canonical numeric string, see answers.normalize_number


# Hub sources. Pinned by repo id; the cached JSONL is what every run actually uses.
SOURCES = {
    "gsm8k": dict(path="openai/gsm8k", name="main", split="test", expected=1319),
    "aime24": dict(path="HuggingFaceH4/aime_2024", name=None, split="train", expected=30),
    "aime25": dict(path="MathArena/aime_2025", name=None, split="train", expected=30),
    "amc23": dict(path="math-ai/amc23", name=None, split="test", expected=40),
}

SAMPLE_PROBLEMS = [
    Problem("sample-0", "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "72"),
    Problem("sample-1", "Find the remainder when 7^100 is divided by 1000.", "1"),
    Problem("sample-2", "How many positive divisors does 360 have?", "24"),
]


def with_retries(fn: Callable, *, attempts: int = 6, base_delay: float = 2.0, log=print):
    """Call fn(), retrying on any exception with exponential backoff."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # network errors come in many types
            if i == attempts - 1:
                raise
            delay = base_delay * (2**i)
            log(f"  download failed ({type(e).__name__}: {e}); retry {i + 1}/{attempts - 1} in {delay:.0f}s")
            time.sleep(delay)


def _rows_from_hub(name: str) -> list[dict]:
    from datasets import load_dataset

    src = SOURCES[name]
    ds = with_retries(lambda: load_dataset(src["path"], src["name"], split=src["split"]))
    return [dict(r) for r in ds]


def _to_problems(name: str, rows: list[dict]) -> list[Problem]:
    out = []
    for i, r in enumerate(rows):
        if name == "gsm8k":
            out.append(Problem(f"gsm8k-{i}", r["question"], gsm8k_gold(r["answer"])))
        else:
            gold = normalize_number(str(r["answer"]))
            if gold is None:
                raise ValueError(f"{name} row {i}: unparseable answer {r['answer']!r}")
            pid = r.get("id", r.get("problem_idx", i))
            out.append(Problem(f"{name}-{pid}", r.get("problem") or r["question"], gold))
    return out


def load(name: str, *, data_dir: Path = DATA_DIR, fetch: Optional[Callable] = None) -> list[Problem]:
    """Load a benchmark, downloading and caching it on first use."""
    if name == "sample":
        return list(SAMPLE_PROBLEMS)
    if name == "aime":
        return load("aime24", data_dir=data_dir, fetch=fetch) + load("aime25", data_dir=data_dir, fetch=fetch)
    if name not in SOURCES:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(SOURCES)} or 'aime'")
    cache = data_dir / f"{name}.jsonl"
    if not cache.exists():
        rows = (fetch or _rows_from_hub)(name)
        problems = _to_problems(name, rows)
        expected = SOURCES[name]["expected"]
        if len(problems) != expected:
            raise ValueError(f"{name}: expected {expected} problems, got {len(problems)}")
        data_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for p in problems:
                f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
        tmp.replace(cache)
    with cache.open(encoding="utf-8") as f:
        return [Problem(**json.loads(line)) for line in f if line.strip()]


def shuffled(problems: list[Problem], seed: int = 42) -> list[Problem]:
    """The fixed evaluation order: random.Random(seed).shuffle of the list."""
    out = list(problems)
    random.Random(seed).shuffle(out)
    return out
