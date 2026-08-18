#!/bin/bash
# 保存済みモデルを再評価する(学習はやり直さない)
# 使い方: scripts/reevaluate.sh <設定ファイル> <モデル> [出力ログ] [エピソード数] [詳細エピソード数]
set -eu

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

python -m tools.reevaluate "$@"
