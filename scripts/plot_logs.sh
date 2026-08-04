#!/bin/bash
# log/ 配下の学習ログをすべてグラフ化する
set -eu

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

shopt -s nullglob
FILES=(log/log_metrics_*.csv)

if [ ${#FILES[@]} -eq 0 ]; then
  echo "log/log_metrics_*.csv が見つかりません。"
  exit 0
fi

for FILE_PATH in "${FILES[@]}"; do
  echo "====================================="
  echo "${FILE_PATH} を処理しています..."
  python plot_log.py "$FILE_PATH"
done

echo "====================================="
echo "すべてのグラフ作成が完了しました！"
