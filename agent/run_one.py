"""Run the real agent on one sketch and save every round, to see what it does.
    python -m agent.run_one path/to/sketch.jpg [prompt_version]
Output: data/apps/try/<sketch name>/round_<n>.html and .png  (data/apps is git-ignored)
run_and_save() is also used by pipeline/generate_renders.py to build the dataset."""
import json
import sys
import time
from pathlib import Path

from agent.build import _score_fn
from agent.graph import run_agent
from ops.prompts import get_prompt
from shared import config


def run_and_save(sketch_path: Path, out: Path, version: str = None) -> dict:
    """Run the agent, write round_<n>.html/.png for every round plus summary.json, return the summary.
    summary.json is written last, so its presence means the sketch finished."""
    version = version or config.STABLE_PROMPT_VERSION
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    st = run_agent(Path(sketch_path).read_bytes(), get_prompt(version), _score_fn(), "render", version)
    rounds = []
    for h in st["history"]:
        n = h["round"]
        (out / f"round_{n}.html").write_text(h["html"])
        png = None
        if h["screenshot"]:
            png = f"round_{n}.png"
            (out / png).write_bytes(h["screenshot"])
        rounds.append(dict(round=n, png=png, html=f"round_{n}.html", model_used=h["model_used"],
                           agent_score=h["score"], latency_s=round(h["latency_s"], 1)))
    summary = dict(prompt_version=version, real_scorer=config.USE_REAL_SCORER, rounds=rounds,
                   stop_reason=st.get("stop_reason"), api_calls=st.get("api_calls"), tokens=st.get("tokens"),
                   fallback_used=st.get("fallback_used"), total_s=round(time.time() - t0, 1))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    path = Path(sys.argv[1])
    out = config.ROOT / "data" / "apps" / "try" / path.stem
    s = run_and_save(path, out, *sys.argv[2:3])
    for r in s["rounds"]:
        print(f"round {r['round']}: score={r['agent_score']:.3f}  model={r['model_used']}  {r['latency_s']}s")
    print({k: v for k, v in s.items() if k != "rounds"}, "\nsaved to", out)
