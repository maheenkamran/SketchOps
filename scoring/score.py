"""Hand-tuned score() from shared/contracts.py: fixed mix of SigLIP similarity and the LLM judge.
Used by the agent (USE_REAL_SCORER=1) until the trained scorer is registered in MLflow."""
from scoring.clip_score import clip_score
from scoring.judge import judge
from shared import config


def score(sketch: bytes, screenshot: bytes) -> float:
    """Return 0.0-1.0; higher = closer to the sketch."""
    c = clip_score(sketch, screenshot)
    j = judge(sketch, screenshot)
    if j is None:  # judge unavailable -> CLIP only
        return round(c, 4)
    return round(config.CLIP_WEIGHT * c + (1 - config.CLIP_WEIGHT) * j, 4)
