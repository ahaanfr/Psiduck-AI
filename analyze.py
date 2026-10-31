"""Build SUMMARY.md (and the README results block) from results/.

    python analyze.py
"""

from __future__ import annotations

import json
import sys

from psiduck.analysis import build_limitations, build_summary, build_tldr, inject_readme, write_figure
from psiduck.hf_env import REPO_ROOT


def main() -> int:
    results = REPO_ROOT / "results"
    plan_path = results / "plan_lock.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else None
    summary = build_summary(results, plan)
    (REPO_ROOT / "SUMMARY.md").write_text(summary, encoding="utf-8")
    write_figure(results, plan, REPO_ROOT / "docs" / "accuracy_vs_compute.svg")
    # The README shows significance in words; exact p-values stay in SUMMARY.md.
    inject_readme(REPO_ROOT / "README.md", build_summary(results, plan, show_p=False), tldr=build_tldr(results, plan),
                  limitations=build_limitations(results, plan))
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
