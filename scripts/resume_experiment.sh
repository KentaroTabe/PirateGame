#!/bin/bash
# scripts/pause_experiment.sh で凍結した学習プロセスを再開する(SIGCONT)。
set -eu

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PARENT=$(pgrep -f "[r]un_experiment.py" || true)
if [ -z "$PARENT" ]; then
  echo "学習プロセスが見つかりません。"
  exit 1
fi

CHILDREN=$(pgrep -P "$PARENT" || true)

# 子を先に動かしてから親を動かす
for c in $CHILDREN; do
  kill -CONT "$c" 2>/dev/null || true
done
kill -CONT "$PARENT"

echo "再開しました:"
ps -o pid,stat,etime,time,command -p "$PARENT" | cut -c1-110
