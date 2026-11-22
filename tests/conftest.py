"""Shared test setup (pytest loads this file automatically before the tests)."""
import os

# Mock mode = fake LLM and fake renderer: tests need no API keys, browser or internet.
# Set before any project import, because shared/config.py reads it once at import time.
os.environ["MOCK_MODE"] = "1"

import pytest  # noqa: E402


@pytest.fixture
def isolated_mlflow(tmp_path, monkeypatch):
    """Send MLflow runs and eval results to a temporary folder, so tests never touch the real mlflow.db."""
    from eval import run_eval
    from shared import config

    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(config, "ROOT", tmp_path)  # artifacts go to tmp_path/mlruns
    monkeypatch.setattr(run_eval, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(run_eval, "SKETCH_DIR", tmp_path / "no_sketches")  # empty -> fake sketches
    return tmp_path
