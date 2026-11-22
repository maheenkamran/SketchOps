"""Run the real agent on every validated sketch and save every round's page + screenshot.
    python -m pipeline.generate_renders [--limit N]
This is NOT a dvc.yaml stage: it calls paid/limited LLM APIs, takes hours and is not deterministic.
Its output is treated like raw data instead: run it once, then `dvc add data/renders` so everyone gets the same renders.

Resumable: a sketch whose folder has summary.json is skipped, so you can stop (Ctrl+C) and restart any time.
Stops by itself after --max-failures failures in a row (usually the daily API quota is used up; rerun tomorrow).
Output: data/renders/<sketch_id>/round_<n>.{png,html} + summary.json, and data/renders/renders.csv (one row per round)."""
import argparse
import json
import random
import shutil
import time

import pandas as pd

from agent.run_one import run_and_save
from shared import config

MANIFEST = config.ROOT / "data" / "validated" / "manifest.csv"
RAW = config.ROOT / "data" / "raw"
SEED = 42


def write_index(out):
    """renders.csv, rebuilt from every finished summary.json (so it is always complete after a resume)."""
    m = pd.read_csv(MANIFEST, dtype=str).set_index("sketch_id")
    rows = []
    for f in sorted(out.glob("*/summary.json")):
        sid, s = f.parent.name, json.loads(f.read_text())
        for r in s["rounds"]:
            if r["png"]:  # a round whose page failed to render has no screenshot -> nothing to label
                rows.append(dict(sketch_id=sid, webpage_id=m.loc[sid, "webpage_id"], source=m.loc[sid, "source"],
                                 round=r["round"], render_png=f"{sid}/{r['png']}", render_html=f"{sid}/{r['html']}",
                                 model_used=r["model_used"], agent_score=r["agent_score"],
                                 latency_s=r["latency_s"], prompt_version=s["prompt_version"]))
    pd.DataFrame(rows).to_csv(out / "renders.csv", index=False)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="run at most this many new sketches")
    ap.add_argument("--out", default=str(config.ROOT / "data" / "renders"))
    ap.add_argument("--version", default=config.STABLE_PROMPT_VERSION, help="prompt version")
    ap.add_argument("--max-failures", type=int, default=3)
    a = ap.parse_args()
    out = config.ROOT / a.out  # a relative path is taken from the repo root; an absolute one is kept as is

    if config.MOCK_MODE:
        raise SystemExit("MOCK_MODE=1 would save fake renders; set MOCK_MODE=0 in .env")
    m = pd.read_csv(MANIFEST, dtype=str)
    order = sorted(m["sketch_id"])
    random.Random(SEED).shuffle(order)  # random order, so a partial run still covers many different webpages
    paths = dict(zip(m["sketch_id"], m["sketch_path"]))
    todo = [s for s in order if not (out / s / "summary.json").exists()]
    print(f"{len(order) - len(todo)} of {len(order)} sketches already done", end="")
    todo = todo[:a.limit] if a.limit else todo
    print(f", running {len(todo)} now -> {out}")

    done = fails = 0
    t0 = time.time()
    for i, sid in enumerate(todo, 1):
        folder = out / sid
        shutil.rmtree(folder, ignore_errors=True)  # clear rounds left by an interrupted earlier try
        try:
            s = run_and_save(RAW / paths[sid], folder, a.version)
        except KeyboardInterrupt:
            shutil.rmtree(folder, ignore_errors=True)
            print("\nstopped; rerun the same command to continue")
            break
        except Exception as e:
            fails += 1
            shutil.rmtree(folder, ignore_errors=True)
            print(f"[{i}/{len(todo)}] {sid} FAILED ({str(e)[:150]})")
            if fails >= a.max_failures:
                print(f"{fails} failures in a row (quota used up?). Stopping; rerun later to continue.")
                break
            continue
        fails, done = 0, done + 1
        eta = (time.time() - t0) / i * (len(todo) - i) / 60
        models = ",".join(sorted({r["model_used"] for r in s["rounds"]}))
        print(f"[{i}/{len(todo)}] {sid}: {len(s['rounds'])} rounds, {models}, {s['total_s']}s  (~{eta:.0f} min left)")

    n = write_index(out)
    print(f"finished {done} sketches this run; renders.csv has {n} screenshots")


if __name__ == "__main__":
    main()
