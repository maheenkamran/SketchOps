"""Metrics and threshold tuning shared by baseline.py, train.py and evaluate.py, so every model is judged the same way.
"Positive" = pass (label 1). A score >= threshold means the scorer says pass."""
import hashlib

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def classify(y, score, threshold) -> dict:
    y, score = np.asarray(y), np.asarray(score)
    pred = (score >= threshold).astype(int)
    out = dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
               precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred, zero_division=0),
               f1=f1_score(y, pred, zero_division=0), n=len(y))
    # ROC-AUC: how well the score ranks pass above fail, whatever the threshold (0.5 = random)
    out["roc_auc"] = roc_auc_score(y, score) if len(set(y)) == 2 else float("nan")
    return {k: round(float(v), 4) for k, v in out.items()}


def tune_threshold(y, score) -> float:
    """The threshold with the best balanced accuracy on this data (use the VALIDATION set, never test).
    Not F1: with half the pairs passing, "say pass to everything" already gets F1 = 0.67 but balanced accuracy 0.5."""
    candidates = np.unique(score)
    bal = [balanced_accuracy_score(y, (score >= t).astype(int)) for t in candidates]
    return float(candidates[int(np.argmax(bal))])


def report(df, score, threshold) -> dict:
    """Metrics per split, plus per pair kind on test -> {"val": {...}, "test": {...}, "test/agent": {...}}."""
    df = df.assign(_s=score)
    out = {s: classify(g["label"], g["_s"], threshold) for s, g in df.groupby("split") if s != "train"}
    for k, g in df[df["split"] == "test"].groupby("kind"):
        out[f"test/{k}"] = classify(g["label"], g["_s"], threshold)
    return out


def file_md5(path) -> str:
    """Fingerprint of the feature file, logged with every run so you can tell which data a model saw."""
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def flat(metrics: dict) -> dict:
    """{"test": {"f1": .8}} -> {"test_f1": .8}, the form MLflow wants (NaNs dropped)."""
    return {f"{s.replace('/', '_')}_{k}": v for s, m in metrics.items() for k, v in m.items()
            if k != "n" and v == v}
