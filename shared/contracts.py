"""The agreed function shapes between the parts of the system.
D1 (Maaham) builds them; D2 (Maheen) serves and monitors them. Change only after both agree.

score():  scoring/score.py   - judges a (sketch, screenshot) pair; called by the agent loop
build():  agent/build.py     - turns a sketch into an app; called by the eval runner and the API
get_prompt(): ops/prompts.py - returns the prompt text for a version (v1, v2, ...)
"""
from typing import Protocol, TypedDict


class BuildResult(TypedDict, total=False):
    html: str
    score: float
    success: bool
    rounds: int
    latency_s: float
    api_calls: int
    model_used: str
    fallback_used: bool
    stop_reason: str
    trace_id: str


class ScoreFn(Protocol):
    def __call__(self, sketch: bytes, screenshot: bytes) -> float:
        """Return 0.0-1.0; higher = closer to the sketch."""


class BuildFn(Protocol):
    def __call__(self, sketch: bytes, prompt_version: str) -> BuildResult: ...
