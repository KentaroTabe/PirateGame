#!/bin/bash
# 実行中の学習プロセスを凍結する(SIGSTOP)。scripts/resume_experiment.sh で再開する。
#
# 注意: これはプロセスをメモリ上に保持したまま止める方式である。
#   - マシンの電源を切る・再起動する・ログアウトすると失われる(スリープは可)
#   - 学習の途中状態を保存するわけではない(チェックポイント機構は未実装)
set -eu

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PARENT=$(pgrep -f "[r]un_experiment.py" || true)
if [ -z "$PARENT" ]; then
  echo "学習プロセスが見つかりません。すでに停止しているか完了しています。"
  exit 1
fi

CHILDREN=$(pgrep -P "$PARENT" || true)

# 親を先に止めてから子を止める(親が新たな指示を出さないようにする)
kill -STOP "$PARENT"
for c in $CHILDREN; do
  kill -STOP "$c" 2>/dev/null || true
done

echo "凍結しました:"
ps -o pid,stat,etime,time,command -p "$PARENT" | cut -c1-110
echo "子プロセス $(echo "$CHILDREN" | wc -w | tr -d ' ') 個も凍結しました。"
echo ""
echo "再開: scripts/resume_experiment.sh"
