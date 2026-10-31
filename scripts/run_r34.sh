#!/bin/bash
# 第34ラウンドを実行順の一覧どおりに走らせる。
# 完了済みの設定は飛ばすので、再起動などで止まっても同じコマンドで再開できる。
set -eu

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

while IFS= read -r config; do
  if grep -q -l "設定ファイル: ${config}\$" result/result_*.txt; then
    echo "完了済み: ${config}"
    continue
  fi
  scripts/run_weight_experiments.sh "$config"
done < configs/r34_queue.txt
