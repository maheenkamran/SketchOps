"""Prompt experiments: run the agent on a fixed set of sketches and log the success metrics to MLflow.

    python -m eval.run_eval --mode real --prompt-version v1     # real agent (needs API keys)
    python -m eval.run_eval --mode fake --prompt-version v1     # fake agent, no keys (tests / CI)

Sketches are read from data/prompt_eval/ (every .png / .jpg in it, in name order).
"""
import argparse
import json
import math
import time
from pathlib import Path

from shared import config
from shared.tracking import git_commit, setup_mlflow

SKETCH_DIR = config.ROOT / "data" / "prompt_eval"
RESULTS = Path(__file__).parent / "results"
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def p95(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, math.ceil(0.95 * len(xs)) - 1)] if xs else 0.0


def load_sketches(mode: str, limit: int = None):
    """Return a list of (id, image bytes). Fake mode invents sketches if the folder is empty."""
    files = sorted(p for p in SKETCH_DIR.glob("*") if p.suffix.lower() in IMAGE_EXTS)
    if files:
        return [(p.stem, p.read_bytes()) for p in files[:limit]]
    if mode == "fake":
        return [(f"fake_{i:02d}", f"fake_{i:02d}".encode()) for i in range(limit or 10)]
    raise FileNotFoundError(f"no sketches in {SKETCH_DIR} (run `dvc pull` or add some)")


def run(mode: str, prompt_version: str, limit: int = None) -> dict:
    if mode == "fake":
        from eval.fake_build import build
    else:
        from agent.build import build

    rows = []
    for sketch_id, sketch in load_sketches(mode, limit):
        t = time.time()
        r = build(sketch, prompt_version)
        r["id"], r["wall_s"] = sketch_id, time.time() - t
        r.pop("html", None)
        rows.append(r)
        print(f"{sketch_id}: success={r['success']} rounds={r['rounds']} score={r['score']:.2f}")

    n = len(rows)
    metrics = dict(
        n=n,
        success_rate=sum(r["success"] for r in rows) / n,
        avg_rounds=sum(r["rounds"] for r in rows) / n,
        p95_latency_s=p95([r["latency_s"] for r in rows]),
        avg_api_calls=sum(r["api_calls"] for r in rows) / n,
        avg_tokens=sum(r.get("tokens", 0) for r in rows) / n,
        fallback_rate=sum(bool(r.get("fallback_used")) for r in rows) / n,
    )
    out = dict(prompt_version=prompt_version, mode=mode, metrics=metrics, rows=rows,
               settings=dict(threshold=config.SCORE_THRESHOLD, max_rounds=config.MAX_ROUNDS,
                             min_improvement=config.MIN_IMPROVEMENT, gemini=config.GEMINI_MODEL,
                             real_scorer=config.USE_REAL_SCORER,
                             viewport=f"{config.VIEWPORT_W}x{config.VIEWPORT_H}"))
    RESULTS.mkdir(exist_ok=True)
    json.dump(out, open(RESULTS / "latest.json", "w"), indent=2)
    _log_mlflow(out)
    return out


def _log_mlflow(out):
    try:
        mlflow = setup_mlflow("sketchops-prompt-experiments")
        with mlflow.start_run(run_name=f"{out['prompt_version']}-{out['mode']}"):
            mlflow.log_params({"prompt_version": out["prompt_version"], "mode": out["mode"], **out["settings"]})
            mlflow.set_tag("git_commit", git_commit())
            mlflow.log_metrics(out["metrics"])
            mlflow.log_artifact(str(RESULTS / "latest.json"))
    except Exception as e:
        print("(mlflow logging skipped:", e, ")")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fake", "real"], default="fake")
    ap.add_argument("--prompt-version", default=config.STABLE_PROMPT_VERSION)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    res = run(a.mode, a.prompt_version, a.limit)
    print(json.dumps(res["metrics"], indent=2))
