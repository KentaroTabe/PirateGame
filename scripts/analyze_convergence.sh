#!/bin/bash
# 学習中メトリクスCSVから収束エポックを推定する
# 使い方: scripts/analyze_convergence.sh [CSVパス...]
#   引数省略時は log/log_metrics_*.csv をすべて分析する
set -eu

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

if [ "$#" -gt 0 ]; then
  FILES=("$@")
else
  shopt -s nullglob
  FILES=(log/log_metrics_*.csv)
fi

if [ ${#FILES[@]} -eq 0 ]; then
  echo "log/log_metrics_*.csv が見つかりません。"
  exit 0
fi

python -m tools.convergence "${FILES[@]}"
