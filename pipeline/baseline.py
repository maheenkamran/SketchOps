"""Stage 5: simple baselines the trained models must beat. No learning, just one number and a threshold.
    python -m pipeline.baseline
  siglip_fixed  the agent's old hand-tuned critic: SigLIP cosine rescaled with CLIP_LOW/HIGH, pass if >= 0.75
  siglip_tuned  raw SigLIP cosine, threshold chosen on val (a fairer version of the same idea)
  ssim_tuned    SSIM, threshold chosen on val
Thresholds are tuned on val and reported on test. Each baseline is one MLflow run in "sketchops-scorer".
Writes reports/baseline.json."""
import json

import numpy as np
import pandas as pd

from pipeline.metrics import file_md5, flat, report, tune_threshold
from shared import config
from shared.tracking import git_commit, setup_mlflow

FEATURES = config.ROOT / "data" / "features" / "features.csv"
REPORT = config.ROOT / "reports" / "baseline.json"
EXPERIMENT = "sketchops-scorer"


def baselines(df):
    """-> {name: (score per row, threshold, how the threshold was chosen)}"""
    val = df["split"] == "val"
    rescaled = ((df["siglip_sim"] - config.CLIP_LOW) / (config.CLIP_HIGH - config.CLIP_LOW)).clip(0, 1)
    out = {"siglip_fixed": (rescaled, config.SCORE_THRESHOLD, "fixed")}
    for name, col in (("siglip_tuned", "siglip_sim"), ("ssim_tuned", "ssim")):
        thr = tune_threshold(df.loc[val, "label"], df.loc[val, col].to_numpy())
        out[name] = (df[col], thr, "best balanced accuracy on val")
    return out


def main():
    df = pd.read_csv(FEATURES)
    mlflow = setup_mlflow(EXPERIMENT)
    results = {}
    for name, (score, thr, how) in baselines(df).items():
        m = report(df, np.asarray(score), thr)
        results[name] = dict(threshold=round(thr, 4), **m)
        with mlflow.start_run(run_name=f"baseline_{name}"):
            mlflow.set_tags(dict(model_type="baseline", git_commit=git_commit()))
            mlflow.log_params(dict(model=name, threshold=round(thr, 4), threshold_from=how,
                                   features_md5=file_md5(FEATURES), n_pairs=len(df)))
            mlflow.log_metrics(flat(m))
        t = m["test"]
        print(f"{name:13s} threshold={thr:.3f}  test: balanced_acc={t['balanced_accuracy']:.3f}  "
              f"roc_auc={t['roc_auc']:.3f}  precision={t['precision']:.3f}  recall={t['recall']:.3f}")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
