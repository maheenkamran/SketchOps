"""build(sketch, prompt_version) -> dict   (the contract in shared/contracts.py; called by the eval runner and the API)"""
import time

from agent.graph import run_agent
from ops.prompts import get_prompt
from agent.tracing import end_trace, start_trace
from shared import config


def _score_fn():
    if config.USE_REAL_SCORER:
        from scoring.score import score
    else:
        from agent.placeholder_score import score
    return score


def build(sketch: bytes, prompt_version: str = None) -> dict:
    prompt_version = prompt_version or config.STABLE_PROMPT_VERSION
    t0 = time.time()
    prompts = get_prompt(prompt_version)
    trace_id = start_trace(prompt_version)
    st = run_agent(sketch, prompts, _score_fn(), trace_id, prompt_version)
    best = max(st.get("best_score", 0.0), 0.0)
    result = dict(
        html=st.get("best_html", st.get("html", "")),
        score=best,
        success=best >= config.SCORE_THRESHOLD,
        rounds=st.get("round", 0),
        latency_s=round(time.time() - t0, 3),
        api_calls=st.get("api_calls", 0),
        tokens=st.get("tokens", 0),
        model_used=st.get("model_used", ""),
        fallback_used=st.get("fallback_used", False),
        stop_reason=st.get("stop_reason", ""),
        trace_id=trace_id,
    )
    end_trace(trace_id, {k: v for k, v in result.items() if k != "html"})
    return result
