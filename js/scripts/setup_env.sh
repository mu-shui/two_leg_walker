#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
VENV_PATH=${VENV_PATH:-.venv}

if [ ! -d "$VENV_PATH" ]; then
  echo "Creating venv at $VENV_PATH ..."
  "$PYTHON" -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Done. Activate with: source $VENV_PATH/bin/activate"
