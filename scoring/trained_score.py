"""The trained scorer (SCORER=trained): loads models:/sketchops-scorer@champion from MLflow and scores a
(sketch, screenshot) pair with the SAME feature code the model was trained on (pipeline/featurize.py and the
featurize settings in params.yaml), so serving can't drift from training.

score() returns 0.0-1.0 like every ScoreFn. The model's own pass/fail threshold (a tag on the registered version,
e.g. 0.66) is mapped onto the agent's SCORE_THRESHOLD (0.75), so "score >= 0.75" in the agent means exactly
"the trained model says pass"."""
import xgboost  # noqa: F401, I001  (load before MLflow pulls in PyTorch; see pipeline/train.py)

import io

import numpy as np
import yaml

from pipeline.featurize import FEATURES, layout_features
from scoring.clip_score import embed
from shared import config

MODEL = "sketchops-scorer"
ALIAS = "champion"
_cache = {}


def load():
    """Load the champion once: model, its threshold, whether it needs embeddings, and its version."""
    if not _cache:
        import torch

        # XGBoost and PyTorch each ship their own OpenMP; on a Mac, PyTorch then FREEZES on its first multi-thread
        # computation (SigLIP). One thread avoids the clash and is fast enough for 2 images per score.
        torch.set_num_threads(1)
        from shared.tracking import setup_mlflow

        mlflow = setup_mlflow("sketchops-scorer")
        mv = mlflow.MlflowClient().get_model_version_by_alias(MODEL, ALIAS)
        flavor = mlflow.xgboost if mv.tags.get("model") == "xgboost" else mlflow.sklearn
        _cache.update(model=flavor.load_model(f"models:/{MODEL}@{ALIAS}"), threshold=float(mv.tags["threshold"]),
                      uses_embeddings=mv.tags.get("uses_embeddings") == "True", version=mv.version,
                      params=yaml.safe_load((config.ROOT / "params.yaml").read_text())["featurize"])
    return _cache


def features(sketch: bytes, screenshot: bytes, uses_embeddings: bool, params: dict) -> np.ndarray:
    """One model input row, built exactly like pipeline/featurize.py + pipeline/train.py build_X."""
    f = layout_features(io.BytesIO(sketch), io.BytesIO(screenshot), params)
    e = embed([sketch, screenshot]).numpy().astype(np.float32)
    f["siglip_sim"] = float(e[0] @ e[1])
    x = np.array([f[k] for k in FEATURES], dtype=np.float64)
    if uses_embeddings:
        x = np.concatenate([x, e[0] * e[1], np.abs(e[0] - e[1])])
    return x.reshape(1, -1)


def to_agent_scale(p: float, threshold: float, target: float) -> float:
    """Map the model's probability so its threshold lands on the agent's threshold (piecewise linear, 0..1)."""
    if p < threshold:
        return target * p / threshold
    return target + (1 - target) * (p - threshold) / (1 - threshold)


def score(sketch: bytes, screenshot: bytes) -> float:
    m = load()
    p = float(m["model"].predict_proba(features(sketch, screenshot, m["uses_embeddings"], m["params"]))[0, 1])
    return round(to_agent_scale(p, m["threshold"], config.SCORE_THRESHOLD), 4)
