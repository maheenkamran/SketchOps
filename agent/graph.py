"""LangGraph loop:  generate -> render -> score -> (accept | stop | generate again)"""
import logging
import time
from typing import Callable, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agent.generator import generate
from agent.governance import Governance
from agent.tracing import trace_round
from sandbox.render import render
from shared import config

log = logging.getLogger("graph")


class State(TypedDict, total=False):
    html: str
    screenshot: Optional[bytes]
    score: float
    scores: List[float]
    round: int
    best_html: str
    best_score: float
    model_used: str
    api_calls: int
    tokens: int
    fallback_used: bool
    stop_reason: str
    history: List[dict]  # one entry per round (html, screenshot, score...) - used to build the D1 dataset


def run_agent(sketch: bytes, prompts: dict, score_fn: Callable[[bytes, bytes], float],
              trace_id: str, prompt_version: str) -> State:
    t0 = time.time()
    gov = Governance(config.MAX_ROUNDS, config.MIN_IMPROVEMENT, config.BUILD_TIMEOUT_S)
    timing = {"round_start": t0}

    def n_generate(s: State):
        timing["round_start"] = time.time()
        if s.get("round", 0) == 0:
            prompt, prev, shot = prompts["generate"], None, None
        else:
            prompt = prompts["fix"].replace("{score}", f"{s['score']:.2f}")
            prev, shot = s["html"], s.get("screenshot")
        r = generate(prompt, sketch, prev, shot)
        return {"html": r["html"], "model_used": r["model_used"],
                "api_calls": s.get("api_calls", 0) + r["api_calls"],
                "tokens": s.get("tokens", 0) + r.get("tokens", 0),
                "fallback_used": s.get("fallback_used", False) or r["fallback_used"]}

    def n_render(s: State):
        try:
            return {"screenshot": render(s["html"])}
        except Exception as e:
            log.warning("render failed: %s", e)
            return {"screenshot": None}

    def n_score(s: State):
        shot = s.get("screenshot")
        sc = 0.0
        if shot:
            try:
                sc = float(score_fn(sketch, shot))
            except Exception as e:
                log.warning("scorer failed: %s", e)
        scores = s.get("scores", []) + [sc]
        rounds = s.get("round", 0) + 1
        round_latency = time.time() - timing["round_start"]
        history = s.get("history", []) + [dict(
            round=rounds, html=s["html"], screenshot=shot, score=sc,
            model_used=s.get("model_used", ""), latency_s=round(round_latency, 3))]
        out = {"score": sc, "scores": scores, "round": rounds, "history": history}
        if sc > s.get("best_score", -1.0):
            out.update(best_score=sc, best_html=s["html"])
        if sc >= config.SCORE_THRESHOLD:
            out["stop_reason"] = "accepted"
        else:
            out["stop_reason"] = gov.should_stop(rounds, scores, time.time() - t0) or ""
        trace_round(trace_id=trace_id, round_no=rounds, prompt_version=prompt_version,
                    model=s.get("model_used", ""), html=s["html"], score=sc,
                    latency_s=round_latency,
                    api_calls=s.get("api_calls", 0), screenshot_bytes=len(shot or b""))
        return out

    def route(s: State):
        return END if s.get("stop_reason") else "generate"

    g = StateGraph(State)
    g.add_node("generate", n_generate)
    g.add_node("render", n_render)
    g.add_node("score", n_score)
    g.set_entry_point("generate")
    g.add_edge("generate", "render")
    g.add_edge("render", "score")
    g.add_conditional_edges("score", route, {END: END, "generate": "generate"})
    app = g.compile()
    return app.invoke({"round": 0, "scores": [], "api_calls": 0, "tokens": 0, "history": [],
                       "best_score": -1.0},
                      config={"recursion_limit": 4 * config.MAX_ROUNDS + 10})
