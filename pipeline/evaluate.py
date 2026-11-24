"""Stage 7: compare all models, pick the champion and register it in the MLflow Model Registry.
    python -m pipeline.evaluate
1. Champion = the trained model with the best cross-validated ("cv") balanced accuracy on train+val (ties: cv
   ROC-AUC); see train.py. Test is never used to choose, only reported.
2. Gate: the champion's cv balanced accuracy must beat the best baseline's validation balanced accuracy
   (baselines only tune one threshold, so val is enough for them), otherwise nothing is registered.
3. Register it as model "sketchops-scorer", point the alias "champion" at the new version, and store its
   threshold and metrics as version tags (the serving code in D2 loads models:/sketchops-scorer@champion).
4. Sanity check: load the model back from the registry and confirm it gives the same test result.
Writes reports/evaluate.json and logs an "evaluate" MLflow run with the comparison table."""
import json

import xgboost  # noqa: F401, I001  (load before MLflow pulls in PyTorch; see train.py)

import numpy as np
import pandas as pd

from pipeline import train
from pipeline.metrics import classify, file_md5
from shared import config
from shared.tracking import git_commit, setup_mlflow

REPORTS = config.ROOT / "reports"
REGISTERED_NAME = "sketchops-scorer"
ALIAS = "champion"
MIN_GAIN = 0.0  # champion must beat the best baseline's val balanced accuracy by more than this


def comparison(trained, baselines):
    """One row per model: val/test balanced accuracy and ROC-AUC, plus test accuracy per pair kind
    (each kind holds only one label, e.g. every mismatch is a fail, so plain accuracy is the meaningful number)."""
    rows = []
    for kind, res in (("baseline", baselines), ("trained", trained)):
        for name, r in res.items():
            row = dict(model=name, type=kind, threshold=r["threshold"])
            row["select_bal_acc"] = sel(r)["balanced_accuracy"]  # cv for trained models, val for baselines
            row["select_auc"] = sel(r)["roc_auc"]
            row["test_bal_acc"] = r["test"]["balanced_accuracy"]
            row["test_auc"] = r["test"]["roc_auc"]
            for k in sorted(x for x in r if x.startswith("test/")):
                row[f"{k.replace('/', '_')}_acc"] = r[k]["accuracy"]
            rows.append(row)
    return pd.DataFrame(rows)


def sel(r):
    """The metrics used for choosing: cross-validation for trained models, the val split for baselines."""
    return r["cv"] if "cv" in r else r["val"]


def pick_champion(trained):
    return max(trained, key=lambda n: (sel(trained[n])["balanced_accuracy"], sel(trained[n])["roc_auc"]))


def main():
    trained = json.loads((REPORTS / "train.json").read_text())
    baselines = json.loads((REPORTS / "baseline.json").read_text())
    table = comparison(trained, baselines)
    print(table.to_string(index=False))

    name = pick_champion(trained)
    champ = trained[name]
    best_base = max(baselines, key=lambda n: sel(baselines[n])["balanced_accuracy"])
    gain = sel(champ)["balanced_accuracy"] - sel(baselines[best_base])["balanced_accuracy"]
    passed = gain > MIN_GAIN
    print(f"\nchampion: {name} (cv bal_acc {sel(champ)['balanced_accuracy']:.3f}), "
          f"best baseline: {best_base} -> gain {gain:+.3f} -> {'REGISTER' if passed else 'NOT registered'}")

    mlflow = setup_mlflow(train.EXPERIMENT)
    out = dict(champion=name, gate_passed=bool(passed), gain_over_baseline=round(gain, 4),
               best_baseline=best_base, champion_cv=sel(champ), champion_test=champ["test"])
    with mlflow.start_run(run_name="evaluate"):
        mlflow.set_tags(dict(model_type="evaluation", git_commit=git_commit(), champion=name))
        mlflow.log_params(dict(champion=name, best_baseline=best_base, gate_passed=passed))
        mlflow.log_metrics(dict(gain_over_baseline=gain, champion_cv_bal_acc=sel(champ)["balanced_accuracy"],
                                champion_test_bal_acc=champ["test"]["balanced_accuracy"]))
        mlflow.log_table(table, "comparison.json")

        if passed:
            version = register(mlflow, name, champ)
            out["registered_version"] = int(version)
            out["reloaded_test"] = check_reload(mlflow, champ)
            assert out["reloaded_test"]["balanced_accuracy"] == champ["test"]["balanced_accuracy"], \
                "registered model gives different results from the trained one"
            print(f"registered {REGISTERED_NAME} v{version} as @{ALIAS}; reloaded test result matches")

    (REPORTS / "evaluate.json").write_text(json.dumps(out, indent=2))


def register(mlflow, name, champ):
    client = mlflow.MlflowClient()
    mv = mlflow.register_model(champ["model_uri"], REGISTERED_NAME)
    client.set_registered_model_alias(REGISTERED_NAME, ALIAS, mv.version)
    tags = dict(model=name, threshold=champ["threshold"], uses_embeddings=champ["uses_embeddings"],
                train_run_id=champ["run_id"], git_commit=git_commit(),
                features_md5=file_md5(train.DATA / "features.csv"),
                cv_bal_acc=sel(champ)["balanced_accuracy"], test_bal_acc=champ["test"]["balanced_accuracy"])
    for k, v in tags.items():
        client.set_model_version_tag(REGISTERED_NAME, mv.version, k, str(v))
    client.update_model_version(REGISTERED_NAME, mv.version, description=(
        f"{name} pass/fail scorer for (sketch, screenshot) pairs. Predict pass when "
        f"predict_proba >= threshold tag ({champ['threshold']})."))
    return mv.version


def load_champion(mlflow):
    """Load models:/sketchops-scorer@champion with the right library (XGBoost models were saved in their own format)."""
    mv = mlflow.MlflowClient().get_model_version_by_alias(REGISTERED_NAME, ALIAS)
    flavor = mlflow.xgboost if mv.tags.get("model") == "xgboost" else mlflow.sklearn
    return flavor.load_model(f"models:/{REGISTERED_NAME}@{ALIAS}")


def check_reload(mlflow, champ):
    """Load the registered champion back and score the test pairs again."""
    df, s_emb, r_emb = train.load()
    X = train.build_X(df, s_emb, r_emb, champ["uses_embeddings"])
    model = load_champion(mlflow)
    test = (df["split"] == "test").to_numpy()
    return classify(df["label"].to_numpy()[test], model.predict_proba(X[test])[:, 1], champ["threshold"])


if __name__ == "__main__":
    main()
