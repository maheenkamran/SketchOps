"""Run the real agent on one sketch and save every round, to see what it does.
    python -m agent.run_one path/to/sketch.jpg [prompt_version]
Output: data/apps/try/<sketch name>/round_<n>.html and .png  (data/apps is git-ignored)"""
import json
import sys
import time
from pathlib import Path

from agent.build import _score_fn
from agent.graph import run_agent
from ops.prompts import get_prompt
from shared import config


def main(path: str, version: str = None):
    version = version or config.STABLE_PROMPT_VERSION
    out = config.ROOT / "data" / "apps" / "try" / Path(path).stem
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    st = run_agent(Path(path).read_bytes(), get_prompt(version), _score_fn(), "try", version)
    for h in st["history"]:
        n = h["round"]
        (out / f"round_{n}.html").write_text(h["html"])
        if h["screenshot"]:
            (out / f"round_{n}.png").write_bytes(h["screenshot"])
        print(f"round {n}: score={h['score']:.3f}  model={h['model_used']}  {h['latency_s']:.1f}s")
    summary = dict(prompt_version=version, rounds=st["round"], stop_reason=st.get("stop_reason"),
                   api_calls=st.get("api_calls"), tokens=st.get("tokens"),
                   fallback_used=st.get("fallback_used"), total_s=round(time.time() - t0, 1))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary, "\nsaved to", out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
