import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# コマンドライン引数からファイルパスを取得
if len(sys.argv) < 2:
    print("使い方: python plot_log.py <ログファイルのパス>")
    sys.exit(1)

log_file_path = sys.argv[1]

# ログファイルが存在しない場合は終了
if not os.path.exists(log_file_path):
    print(f"エラー: {log_file_path} が見つかりません。")
    sys.exit(1)

# ファイル名から拡張子を除いた部分を取得（例: log_learning_1）
base_name = os.path.splitext(os.path.basename(log_file_path))[0]

dropped_counts = []
current_dropped = 0

# 1. ログから「落とされた人数」を抽出（「否決」「可決」のテキストをパース）
with open(log_file_path, "r", encoding="utf-8") as f:
    for line in f:
        if "否決" in line or "💀" in line:
            current_dropped += 1
        elif "可決" in line or "👉" in line:
            dropped_counts.append(current_dropped)
            current_dropped = 0

if not dropped_counts:
    print(f"警告: {log_file_path} から「否決/可決」のデータが抽出できませんでした。")
    sys.exit(0)

# 2. DataFrameの作成と100エピソード移動平均の計算
df = pd.DataFrame({"Dropped": dropped_counts})
df["Moving_Average_100"] = df["Dropped"].rolling(window=10000, min_periods=1).mean()

# 3. グラフの描画
plt.figure(figsize=(10, 6))

# 生データのプロットを削除し、100エピソードの移動平均のみをプロット
plt.plot(df.index, df["Moving_Average_100"], label="100-Episode Moving Average", color="red", linewidth=2)

# グラフの装飾
plt.title(f"Average Dropped Pirates per Episode: {base_name}", fontsize=14)
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Average Dropped (10000-ep window)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# 画像名を区別しやすいように変更して保存
output_filename = f"chart_dropped_avg_{base_name}.png"
plt.savefig(output_filename)

# 次のループのためにメモリを解放
plt.close() 

print(f"{output_filename} を作成しました。")