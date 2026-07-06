#!/bin/bash

# ログファイルが格納されているディレクトリ名を指定
LOG_DIR="log"

# 1から33までループ処理
for i in {1..33}; do
  FILE_PATH="${LOG_DIR}/log_learning_${i}.txt"
  
  # ファイルが存在するか確認してから実行
  if [ -f "$FILE_PATH" ]; then
    echo "====================================="
    echo "${FILE_PATH} を処理しています..."
    python plot_log.py "$FILE_PATH"
  else
    echo "スキップ: ${FILE_PATH} が見つかりません。"
  fi
done

echo "====================================="
echo "すべてのグラフ作成が完了しました！"