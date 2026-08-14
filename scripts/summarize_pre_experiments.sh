#!/bin/bash
# 事前実験（files/result/*.txt）の主要項目を一覧化する
set -eu

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

DIR="${1:-files/result}"

printf "%-5s %-5s %-6s %-5s %-8s %-10s %s\n" \
  "番号" "人数" "宝石" "L" "エポック" "Best報酬" "最終報酬"

for n in $(seq 1 40); do
  f="${DIR}/result_${n}.txt"
  [ -f "$f" ] || continue
  agents=$(grep -o '海賊の人数: [0-9]*' "$f" | grep -o '[0-9]*' || echo "-")
  gems=$(grep -o '宝石の総数: [0-9]*' "$f" | grep -o '[0-9]*' || echo "-")
  L=$(grep -o 'ペナルティ L): [0-9.]*' "$f" | grep -o '[0-9.]*$' || echo "-")
  ep=$(grep -o '実行エポック数: [0-9]*' "$f" | grep -o '[0-9]*' || echo "-")
  best=$(grep -o 'Best): [-0-9.]*' "$f" | grep -o '[-0-9.]*$' || echo "-")
  rewards=$(grep -o 'agent_[A-F]: [-0-9.]*' "$f" | sed 's/agent_//' | tr '\n' ' ')
  printf "%-5s %-5s %-6s %-5s %-8s %-10s %s\n" "$n" "$agents" "$gems" "$L" "$ep" "$best" "$rewards"
done
