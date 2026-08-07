#!/bin/bash
# 権力重みパターンを変えた実験を直列に実行する
# 使い方: scripts/run_weight_experiments.sh [設定ファイル...]
#   引数省略時は configs/weights_*.json をすべて実行する
set -eu

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

if [ "$#" -gt 0 ]; then
  CONFIGS=("$@")
else
  shopt -s nullglob
  CONFIGS=(configs/weights_*.json)
fi

if [ ${#CONFIGS[@]} -eq 0 ]; then
  echo "実行する設定ファイルがありません。"
  exit 1
fi

mkdir -p log
RUNNER_LOG="log/weight_experiments_run.log"

for CONFIG_PATH in "${CONFIGS[@]}"; do
  echo "=== 実験開始: ${CONFIG_PATH} ($(date '+%Y-%m-%d %H:%M:%S')) ===" | tee -a "$RUNNER_LOG"
  python run_experiment.py "$CONFIG_PATH" >> "$RUNNER_LOG" 2>&1
  echo "=== 実験完了: ${CONFIG_PATH} ($(date '+%Y-%m-%d %H:%M:%S')) ===" | tee -a "$RUNNER_LOG"
done

echo "すべての実験が完了しました。"
