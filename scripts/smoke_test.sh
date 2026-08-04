#!/bin/bash
# パイプライン全体のスモークテストを実行する
# 使い方: scripts/smoke_test.sh [出力ディレクトリ]
set -eu

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

OUT_DIR="${1:-smoke_output}"
python -m tools.smoke_test "$OUT_DIR"
