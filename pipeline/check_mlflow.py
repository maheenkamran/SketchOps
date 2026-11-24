"""Check that the configured MLflow server (MLFLOW_TRACKING_URI) supports everything the pipeline needs.
    python -m pipeline.check_mlflow
Logs a run, saves a tiny model, registers it, sets an alias and loads it back, in a throwaway experiment and
model ("sketchops-connection-check"). Prints OK per step, or the step that failed."""
import xgboost  # noqa: F401, I001  (load before MLflow pulls in PyTorch; see train.py)

import numpy as np
from sklearn.linear_model import LogisticRegression

from shared import config
from shared.tracking import setup_mlflow

NAME = "sketchops-connection-check"


def main():
    print("server:", config.MLFLOW_TRACKING_URI)
    mlflow = setup_mlflow(NAME)
    print("OK  experiment")
    X, y = np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0, 0, 1, 1])
    with mlflow.start_run(run_name="check"):
        mlflow.log_param("p", 1)
        mlflow.log_metric("m", 0.5)
        print("OK  run with params and metrics")
        uri = mlflow.sklearn.log_model(LogisticRegression().fit(X, y), name="model", input_example=X[:1]).model_uri
        print("OK  model saved")
    client = mlflow.MlflowClient()
    v = mlflow.register_model(uri, NAME).version
    client.set_registered_model_alias(NAME, "champion", v)
    client.set_model_version_tag(NAME, v, "threshold", "0.5")
    print(f"OK  registered version {v} with alias @champion")
    m = mlflow.sklearn.load_model(f"models:/{NAME}@champion")
    assert m.predict(X).tolist() == [0, 0, 1, 1]
    print("OK  loaded back from the registry\nAll good: the pipeline can use this server.")


if __name__ == "__main__":
    main()
