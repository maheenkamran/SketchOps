"""Unit tests for the D1 code. Run:  pytest -q   (mock mode is switched on in conftest.py)"""
import io

import pytest
from PIL import Image


def _png(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, "PNG")
    return buf.getvalue()


# ---- agent: governance ----
def test_governance():
    from agent.governance import Governance
    g = Governance(4, 0.02, 100)
    assert g.should_stop(1, [0.4], 1) is None
    assert g.should_stop(2, [0.4, 0.4], 1) == "no_improvement"
    assert g.should_stop(4, [0.1, 0.3, 0.5, 0.7], 1) == "max_rounds"
    assert g.should_stop(1, [0.4], 101) == "timeout"


# ---- agent: generator helpers ----
def test_extract_html():
    from agent.generator import extract_html
    assert extract_html("Here:\n```html\n<html>A</html>\n```\nbye") == "<html>A</html>"
    assert extract_html("```html\n<html>cut off") == "<html>cut off"  # closing ``` missing
    assert extract_html("<html>no fence</html>") == "<html>no fence</html>"
    assert extract_html(None) == ""


def test_shrink_large_images_only():
    from shared.images import mime, shrink
    small = _png(800, 600)
    assert shrink(small) is small  # untouched
    big = shrink(_png(4000, 3000))
    assert max(Image.open(io.BytesIO(big)).size) == 1600
    assert mime(big) == "image/jpeg" and mime(small) == "image/png"


# ---- prompts ----
def test_prompts_and_unknown_version():
    from ops.prompts import get_prompt
    for v in ("v1", "v2"):
        p = get_prompt(v)
        assert "{score}" in p["fix"] and "4.3.5" in p["generate"]
    with pytest.raises(KeyError):
        get_prompt("v3")


# ---- agent: full loop ----
def test_full_build_in_mock_mode():
    from agent.build import build
    r = build(b"fake-sketch", "v1")
    for k in ("html", "score", "success", "rounds", "latency_s", "api_calls", "tokens", "model_used"):
        assert k in r
    assert r["rounds"] >= 1


def test_loop_records_every_round():
    from agent.graph import run_agent
    from agent.placeholder_score import score
    from ops.prompts import get_prompt
    st = run_agent(b"x", get_prompt("v1"), score, "trace", "v1")
    assert len(st["history"]) == st["round"] == 2  # placeholder 0.5 twice -> no_improvement
    assert st["stop_reason"] == "no_improvement"
    assert all(h["html"] and h["screenshot"] for h in st["history"])


# ---- eval runner ----
def test_fake_build_is_repeatable():
    from eval.fake_build import build
    assert build(b"same", "v1") == build(b"same", "v1")


def test_fake_eval_logs_to_mlflow(isolated_mlflow):
    from eval.run_eval import run
    from shared.tracking import setup_mlflow
    out = run("fake", "v1", limit=10)
    assert out["metrics"]["n"] == 10
    runs = setup_mlflow("sketchops-prompt-experiments").search_runs()
    assert len(runs) == 1 and runs["params.prompt_version"][0] == "v1"


def test_bad_prompt_scores_lower(isolated_mlflow):
    from eval.run_eval import run
    good = run("fake", "v1", limit=20)["metrics"]["success_rate"]
    bad = run("fake", "v2-bad", limit=20)["metrics"]["success_rate"]
    assert bad < good
