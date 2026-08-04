#!/bin/bash
# 実験パイプライン（事前学習 → 学習 → 評価）を実行する
set -eu

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

python run_experiment.py "$@"
