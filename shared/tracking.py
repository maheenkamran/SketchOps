"""MLflow helpers shared by every script that logs runs (prompt experiments now, the scorer pipeline later)."""
import subprocess

from shared import config


def setup_mlflow(experiment: str):
    """Point MLflow at our SQLite database and select (or create) an experiment."""
    import mlflow

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment) is None:
        # keep artifacts in <repo>/mlruns no matter which folder the script is started from
        mlflow.create_experiment(experiment, artifact_location=(config.ROOT / "mlruns").as_uri())
    mlflow.set_experiment(experiment)
    return mlflow


def git_commit() -> str:
    """Current commit hash, with '-dirty' if there are uncommitted changes (so every run is traceable)."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        cwd=config.ROOT, text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"
