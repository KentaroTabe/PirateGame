import sys
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# コマンドライン引数からファイルパスを取得
if len(sys.argv) < 2:
    print("使い方: python plot_log.py <ログファイルのパス>")
    sys.exit(1)

log_file_path = sys.argv[1]

if not os.path.exists(log_file_path):
    print(f"エラー: {log_file_path} が見つかりません。")
    sys.exit(1)

base_name = os.path.splitext(os.path.basename(log_file_path))[0]

dropped_counts = []
first_survival = []
approval_rates = []

current_dropped = 0
is_first_proposal = True
current_votes = 0
current_survivors = 0

pattern_stats = re.compile(r"賛成:\s*(\d+)\s*/\s*生存者:\s*(\d+)")

# 1. ログのパース
with open(log_file_path, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern_stats.search(line)
        if match:
            current_votes = int(match.group(1))
            current_survivors = int(match.group(2))
        
        if "否決" in line or "💀" in line:
            is_first_proposal = False
            current_dropped += 1
            
        elif "可決" in line or "👉" in line:
            dropped_counts.append(current_dropped)
            first_survival.append(1 if is_first_proposal else 0)
            
            if current_survivors > 0:
                approval_rate = current_votes / current_survivors
            else:
                approval_rate = 0
            approval_rates.append(approval_rate)

            current_dropped = 0
            is_first_proposal = True
            current_votes = 0
            current_survivors = 0

if not dropped_counts:
    print(f"警告: {log_file_path} から「否決/可決」のデータが抽出できませんでした。")
    sys.exit(0)

# 2. DataFrameの作成
df = pd.DataFrame({
    "Dropped": dropped_counts,
    "First_Survival": first_survival,
    "Approval_Rate": approval_rates
})

# 全体で移動平均を計算
WINDOW_SIZE = 10000
df["MA_Dropped"] = df["Dropped"].rolling(window=WINDOW_SIZE).mean()
df["MA_Survival"] = df["First_Survival"].rolling(window=WINDOW_SIZE).mean()
df["MA_Approval"] = df["Approval_Rate"].rolling(window=WINDOW_SIZE).mean()

# 外れ値を含む初期エピソードを切り捨てる
SKIP_EPISODES = 0  
if len(df) > SKIP_EPISODES:
    df = df.iloc[SKIP_EPISODES:].copy()

# NaN を除外
df = df.dropna()

# 3. グラフの描画
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# (1) 落とされた人数の移動平均
axes[0].plot(df.index, df["MA_Dropped"], color="red", linewidth=2)
axes[0].set_title(f"Average Dropped Pirates ({WINDOW_SIZE}-ep window)", fontsize=12)
axes[0].set_ylabel("Avg Dropped", fontsize=10)
axes[0].grid(True, linestyle="--", alpha=0.6)

# (2) 最初の提案者の生存率の移動平均
axes[1].plot(df.index, df["MA_Survival"], color="blue", linewidth=2)
axes[1].set_title(f"First Proposer Survival Rate ({WINDOW_SIZE}-ep window)", fontsize=12)
axes[1].set_ylabel("Survival Rate", fontsize=10)
axes[1].set_ylim(auto = True)  # 手動でレンジを固定
axes[1].grid(True, linestyle="--", alpha=0.6)

# (3) 可決時の賛成割合の移動平均
axes[2].plot(df.index, df["MA_Approval"], color="green", linewidth=2, label="Approval Rate")
axes[2].axhline(y=0.5, color="black", linestyle="--", linewidth=1.5, label="Threshold (0.5)")
axes[2].set_title(f"Approval Rate at Acceptance ({WINDOW_SIZE}-ep window)", fontsize=12)
axes[2].set_ylabel("Approval Rate", fontsize=10)
axes[2].set_xlabel("Episode", fontsize=12)
axes[2].set_ylim(auto = True)  # 手動でレンジを固定
axes[2].legend(loc="upper right")
axes[2].grid(True, linestyle="--", alpha=0.6)

fig.suptitle(f"Training Progress: {base_name}", fontsize=16, y=0.98)
plt.tight_layout()

output_filename = f"log_deadnumber/log_chart_metrics_full_{base_name}.png"
plt.savefig(output_filename)
plt.close() 

print(f"{output_filename} を作成しました。")