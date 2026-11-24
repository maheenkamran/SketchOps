"""Stage 6: train the pass/fail scorers. Settings in params.yaml -> train.
    python -m pipeline.train
  logreg   logistic regression on the 6 features            (simple, easy to explain)
  xgboost  gradient-boosted trees on the 6 features         (catches non-linear combinations)
  mlp      2-layer neural network on the 6 features + SigLIP embeddings of both images
Model choice uses 5-fold cross-validation on TRAIN+VAL, grouped by webpage (no page in two folds) and stratified
(every fold gets some fails): each pair is scored by a model that never saw its webpage ("out-of-fold"), the
pass/fail threshold is tuned on those scores, and the "cv" metrics are what evaluate.py compares. This is steadier
than the small VAL set alone. The final model is then refitted on all of TRAIN+VAL. Test is only reported.
Each model is one MLflow run in "sketchops-scorer" (params, metrics, the fitted model itself).
Writes reports/train.json (run id, threshold and metrics per model), which evaluate.py uses to pick the champion."""
import json

# Must be the first heavy import. On Macs, saving a model makes MLflow load PyTorch, which brings its own copy
# of OpenMP; if XGBoost starts its copy after that, the process crashes (segfault, exit -11). Loading XGBoost
# first avoids the clash. Harmless on Linux/Docker.
import xgboost  # noqa: F401, I001

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.featurize import FEATURES
from pipeline.metrics import classify, file_md5, flat, report, tune_threshold
from shared import config
from shared.tracking import git_commit, setup_mlflow

DATA = config.ROOT / "data" / "features"
REPORT = config.ROOT / "reports" / "train.json"
EXPERIMENT = "sketchops-scorer"


def load():
    df = pd.read_csv(DATA / "features.csv", dtype={"webpage_id": str})
    e = np.load(DATA / "embeddings.npz", allow_pickle=True)
    assert (e["pair_id"] == df["pair_id"].to_numpy()).all(), "embeddings are not in the same order as features"
    return df, e["sketch"], e["render"]


def build_X(df, sketch_emb, render_emb, with_embeddings):
    """Model input. The embedding part is (sketch * render) and |sketch - render|: how the two images agree,
    dimension by dimension, independent of which page it is."""
    X = df[FEATURES].to_numpy(dtype=np.float64)
    if with_embeddings:
        X = np.hstack([X, sketch_emb * render_emb, np.abs(sketch_emb - render_emb)])
    return X


def make_model(name, p, seed):
    """-> (unfitted model, uses embeddings?)"""
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=p["C"], class_weight="balanced", max_iter=2000, random_state=seed)), False
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                             learning_rate=p["learning_rate"], subsample=p["subsample"],
                             eval_metric="logloss", random_state=seed, n_jobs=1), False
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(p["hidden"],), alpha=p["alpha"], early_stopping=True,
            max_iter=1000, random_state=seed)), True
    raise ValueError(f"unknown model {name!r}")


def log_model(mlflow, name, model, example):
    """Save the fitted model inside the MLflow run; returns its address (models:/m-...) for registering later."""
    if name == "xgboost":
        return mlflow.xgboost.log_model(model, name="model", input_example=example).model_uri  # XGBoost's own format
    # MLflow saves scikit-learn models with skops, which only loads classes it trusts; the MLP's
    # optimizer is a normal scikit-learn class, so we list it as trusted.
    return mlflow.sklearn.log_model(model, name="model", input_example=example, skops_trusted_types=[
        "sklearn.neural_network._stochastic_optimizers.AdamOptimizer"]).model_uri


def out_of_fold(model, X, y, groups, folds, seed):
    """Score every row with a copy of the model trained on the other folds (never on its own webpage)."""
    oof = np.zeros(len(y))
    for fit_idx, score_idx in StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(X, y, groups):
        m = clone(model).fit(X[fit_idx], y[fit_idx])
        oof[score_idx] = m.predict_proba(X[score_idx])[:, 1]
    return oof


def main():
    params = yaml.safe_load((config.ROOT / "params.yaml").read_text())["train"]
    df, s_emb, r_emb = load()
    dev = df["split"].isin(["train", "val"]).to_numpy()  # everything except test
    y = df["label"].to_numpy()
    groups = df["webpage_id"].to_numpy()
    folds = params["cv_folds"]
    mlflow = setup_mlflow(EXPERIMENT)
    results = {}

    for name in params["models"]:
        p = params[name]
        model, with_emb = make_model(name, p, params["seed"])
        X = build_X(df, s_emb, r_emb, with_emb)
        oof = out_of_fold(model, X[dev], y[dev], groups[dev], folds, params["seed"])
        thr = tune_threshold(y[dev], oof)
        model.fit(X[dev], y[dev])  # final model on all of train+val
        score = model.predict_proba(X)[:, 1]
        m = {k: v for k, v in report(df, score, thr).items() if k != "val"}  # val is now training data
        m["cv"] = classify(y[dev], oof, thr)
        m["train"] = classify(y[dev], score[dev], thr)  # far above cv = overfitting

        with mlflow.start_run(run_name=f"train_{name}") as run:
            mlflow.set_tags(dict(model_type="trained", git_commit=git_commit()))
            mlflow.log_params(dict(model=name, threshold=round(thr, 4), uses_embeddings=with_emb,
                                   n_inputs=X.shape[1], n_train=int(dev.sum()), cv_folds=folds, seed=params["seed"],
                                   features_md5=file_md5(DATA / "features.csv"), **p))
            mlflow.log_metrics(flat(m))
            uri = log_model(mlflow, name, model, X[dev][:2])
        results[name] = dict(run_id=run.info.run_id, model_uri=uri, threshold=round(thr, 4),
                             uses_embeddings=with_emb, **m)
        t, v = m["test"], m["cv"]
        print(f"{name:8s} cv bal_acc={v['balanced_accuracy']:.3f} auc={v['roc_auc']:.3f} | "
              f"test bal_acc={t['balanced_accuracy']:.3f} auc={t['roc_auc']:.3f} | "
              f"train bal_acc={m['train']['balanced_accuracy']:.3f}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
