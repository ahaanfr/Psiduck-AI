import json

import pytest

from psiduck import analysis, data, runner, system
from psiduck.config import Settings
from tests.fakes import FakeBackbone


def test_choose_n():
    # 13 h = 46,800 s; 20 + 15 = 35 s/problem -> 1337 -> 1300 -> capped at 1300.
    assert runner.choose_n({"latent": 20, "baseline": 15}) == 1300
    assert runner.choose_n({"latent": 40, "baseline": 30}) == 650  # 668 -> 650
    assert runner.choose_n({"latent": 1, "baseline": 1}) == 1319  # cap
    assert runner.choose_n({"latent": 5000, "baseline": 5000}) == 50  # floor at one block


def test_mcnemar_exact():
    assert analysis.mcnemar_exact(0, 0) == 1.0
    assert analysis.mcnemar_exact(5, 0) == pytest.approx(0.0625)
    assert analysis.mcnemar_exact(3, 3) == 1.0
    # b=2, c=10: 2 * P(X<=2 | n=12, p=.5) = 2 * 79/4096
    assert analysis.mcnemar_exact(2, 10) == pytest.approx(2 * 79 / 4096)


def test_torn_jsonl_line_is_ignored(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"idx": 0, "correct": true}\n{"idx": 1, "corr')
    assert list(analysis.read_jsonl(p)) == [0]


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    monkeypatch.setattr(system, "on_battery", lambda: False)
    monkeypatch.setattr(system, "keep_awake", lambda: True)
    gsm = [data.Problem(f"gsm8k-{i}", f"q{i}", "42" if i % 2 == 0 else "7") for i in range(1319)]
    aime = [data.Problem(f"aime-{i}", f"a{i}", "42") for i in range(60)]

    def load(name):
        return gsm if name == "gsm8k" else aime

    readme = tmp_path / "README.md"
    readme.write_text(f"# R\n{analysis.README_START}\npending\n{analysis.README_END}\nend\n")
    sent = []
    r = runner.Runner(
        Settings(max_new_tokens=64), lambda s: FakeBackbone(), results_dir=tmp_path / "results",
        readme=readme, notify=lambda t, m, **k: sent.append(m) or True,
        status_interval=3600, load_dataset=load,
    )
    monkeypatch.setattr(runner, "choose_n", lambda sec, **k: 50)
    monkeypatch.setattr(runner, "MIN_N_PRIMARY", 0)
    return r, sent, tmp_path


def test_full_run_end_to_end(fake_env):
    r, sent, tmp = fake_env
    assert r.run() == 0
    res = tmp / "results"
    assert json.loads((res / "plan_lock.json").read_text())["n"] == 50
    for dataset, cfg in runner.JOBS:
        recs = analysis.read_jsonl(res / dataset / f"{cfg}.jsonl")
        assert len(recs) == (50 if dataset == "gsm8k" else 60)
    # Smoke records never contain correctness.
    smoke = (res / "smoke" / "latent.jsonl").read_text()
    assert "correct" not in smoke and "prediction" not in smoke
    assert any(m.startswith("Smoke done") for m in sent)
    assert any("gsm8k/latent finished" in m for m in sent)
    assert any(m.startswith("All done") for m in sent)
    summary = (tmp / "SUMMARY.md").read_text()
    assert "McNemar" in summary
    assert "pending" not in (tmp / "README.md").read_text()
    assert "Finished configs" in (res / "STATUS.txt").read_text()


def test_too_slow_primary_falls_back_before_locking(fake_env, monkeypatch):
    r, sent, tmp = fake_env
    monkeypatch.setattr(runner, "MIN_N_PRIMARY", 100)
    calls = iter([50, 300])
    monkeypatch.setattr(runner, "choose_n", lambda sec, **k: next(calls))
    assert r.run() == 0
    plan = json.loads((tmp / "results" / "plan_lock.json").read_text())
    assert plan["settings"]["model_name"] == runner.FALLBACK_MODEL
    assert plan["n"] == 300 and "too slow" in plan["fallback_reason"]
    assert (tmp / "results" / "smoke_Qwen_Qwen3-8B").exists()


def test_resume_skips_done_and_poison_problems(fake_env):
    r, sent, tmp = fake_env
    res = tmp / "results"
    res.mkdir(exist_ok=True)
    (res / "plan_lock.json").write_text(json.dumps({
        "n": 50, "smoke_sec_per_problem": {"latent": 1, "baseline": 1},
        "gsm8k_budget_hours": 13, "settings": Settings().as_dict(), "locked_at": "t",
    }))
    (res / "gsm8k").mkdir(exist_ok=True)
    for i in range(10):
        runner.append_jsonl(res / "gsm8k" / "latent.jsonl", {"idx": i, "correct": True, "sentinel": True})
    (res / "gsm8k" / "latent.attempts").write_text("10\n10\n")  # idx 10 killed the process twice
    assert r.run() == 0
    recs = analysis.read_jsonl(res / "gsm8k" / "latent.jsonl")
    assert len(recs) == 50
    assert all(recs[i].get("sentinel") for i in range(10))  # not re-run
    assert "process died" in recs[10]["error"] and recs[10]["correct"] is False


def test_failing_config_is_logged_and_next_runs(fake_env, monkeypatch):
    r, sent, tmp = fake_env
    real = runner.build_solver

    def build(cfg, b, s):
        solver = real(cfg, b, s)
        if cfg == "latent":
            solver.solve = lambda q: (_ for _ in ()).throw(RuntimeError("CUDA out of memory"))
        return solver

    monkeypatch.setattr(runner, "build_solver", build)
    # Smoke uses build_solver too; lock the plan up front so only full runs fail.
    (tmp / "results").mkdir(exist_ok=True)
    (tmp / "results" / "plan_lock.json").write_text(json.dumps({
        "n": 50, "smoke_sec_per_problem": {"latent": 1, "baseline": 1},
        "gsm8k_budget_hours": 13, "settings": Settings().as_dict(), "locked_at": "t",
    }))
    assert r.run() == 0
    assert r.state["gsm8k/latent"] == "failed"
    assert r.state["gsm8k/baseline"] == "done"
    assert any("gsm8k/latent FAILED" in m for m in sent)


def test_notify_never_raises(monkeypatch):
    def boom(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr(system.urllib.request, "urlopen", boom)
    assert system.notify("psiduck-test-topic", "x") is False
