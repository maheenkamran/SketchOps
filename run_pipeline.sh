#!/usr/bin/env bash
# The one command that reproduces D1:  dvc pull && dvc repro  (the Docker image runs this by default).
#   bash run_pipeline.sh
# 1. Log in to the DagsHub data remote if DAGSHUB_USER / DAGSHUB_TOKEN are set (kept in .dvc/config.local,
#    which is never committed or copied into the image).
# 2. Pull ONLY the raw inputs (every *.dvc file). Everything after them is recomputed, not downloaded.
# 3. dvc repro: validate -> split -> pairs -> featurize -> baseline -> train -> evaluate (registers the model).
# 4. dvc metrics diff: compare the new results with the ones committed in git. No difference = reproduced.
# 5. If /results is mounted (docker run -v ...:/results), copy the reports and the MLflow store there.
set -euo pipefail

# Docker's --env-file keeps "# comments" and spaces in values, so cut them off
clean() { local v="${1%%#*}"; v="${v#"${v%%[![:space:]]*}"}"; echo "${v%"${v##*[![:space:]]}"}"; }
DAGSHUB_USER="$(clean "${DAGSHUB_USER:-}")"
DAGSHUB_TOKEN="$(clean "${DAGSHUB_TOKEN:-}")"

if [[ -n "$DAGSHUB_USER" && -n "$DAGSHUB_TOKEN" ]]; then
  dvc remote modify origin --local auth basic
  dvc remote modify origin --local user "$DAGSHUB_USER"
  dvc remote modify origin --local password "$DAGSHUB_TOKEN"
fi

echo "== 1/3 pulling raw data =="
dvc pull $(git ls-files '*.dvc')

echo "== 2/3 running the pipeline =="
dvc repro

echo "== 3/3 results compared with the committed ones (empty = identical) =="
dvc metrics diff --precision 4
dvc metrics show --precision 4

if [[ -d /results ]]; then
  cp -r reports /results/
  cp mlflow.db /results/ 2>/dev/null || true
  cp -r mlruns /results/ 2>/dev/null || true
  echo "copied reports/, mlflow.db and mlruns/ to the mounted results folder"
fi
