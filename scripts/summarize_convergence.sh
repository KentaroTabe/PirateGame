#!/bin/bash
# 収束分析の出力から「秩序収束」の行だけを試行番号つきで並べる。
# 使い方: scripts/summarize_convergence.sh 128 129 130 ...
set -eu
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
printf '%5s  %s\n' "試行" "秩序収束"
for n in "$@"; do
  line=$(scripts/analyze_convergence.sh "log/log_metrics_${n}.csv" 2>/dev/null | grep "秩序収束" || true)
  printf '%5s  %s\n' "$n" "${line:-（記録なし）}"
done
