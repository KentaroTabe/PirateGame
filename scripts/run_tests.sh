#!/bin/bash
# tests/ 配下のユニットテストを実行する
set -eu

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

python -m unittest discover -s tests -t . -p 'test_*.py' -v 2>&1
