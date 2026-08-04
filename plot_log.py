"""学習中に記録した政治的指標CSV（log/log_metrics_n.csv）をグラフ化する。

使い方: python plot_log.py <メトリクスCSVのパス> [出力ディレクトリ]

プロットする評価指標:
    1. エージェント別 平均報酬（分配の偏り = 強者の傲慢/弱者の妥協の推移）
    2. エージェント別 死亡率（否決されて海に落とされた割合）
    3. 一発可決率（最初の提案がそのまま通った割合 = 交渉の安定度）
    4. 平均エピソード長（長いほど否決ラウンドが多い）
"""

import os
import sys

# CVD検証済みのカテゴリカルパレット（エージェント順に固定で割り当てる）
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                 "#4a3aa7", "#e34948"]
TEXT_PRIMARY = "#1f1f1e"
TEXT_SECONDARY = "#5c5b54"
GRID_COLOR = "#d8d7d0"


def _style_axis(ax, title, ylabel):
    ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, loc="left")
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SECONDARY)
    ax.grid(True, linestyle=":", linewidth=0.8, color=GRID_COLOR, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def _plot_series(ax, x, y, color, label=None):
    """生データを薄く、移動平均を主線として描く。"""
    window = max(1, len(y) // 15)
    smooth = y.rolling(window=window, min_periods=1).mean()
    ax.plot(x, y, color=color, linewidth=1, alpha=0.25)
    ax.plot(x, smooth, color=color, linewidth=2, label=label)


def plot_metrics(csv_path, output_dir="files/plots"):
    # 学習プロセス(マルチプロセスワーカー)に描画ライブラリを読み込ませないよう遅延インポート
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    agent_names = [c[len("rew_"):] for c in df.columns if c.startswith("rew_")]
    x = df["epoch"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.patch.set_facecolor("white")

    # (1) エージェント別 平均報酬
    ax = axes[0][0]
    for i, name in enumerate(agent_names):
        _plot_series(ax, x, df[f"rew_{name}"], SERIES_COLORS[i % len(SERIES_COLORS)], label=name)
    ax.axhline(y=0, color=GRID_COLOR, linewidth=1)
    _style_axis(ax, "Average reward per agent", "Avg reward")

    # (2) エージェント別 死亡率
    ax = axes[0][1]
    for i, name in enumerate(agent_names):
        _plot_series(ax, x, df[f"death_{name}"], SERIES_COLORS[i % len(SERIES_COLORS)], label=name)
    ax.set_ylim(-0.05, 1.05)
    _style_axis(ax, "Death rate per agent (proposal rejected)", "Death rate")

    # (3) 一発可決率
    ax = axes[1][0]
    _plot_series(ax, x, df["first_pass_rate"], SERIES_COLORS[0])
    ax.set_ylim(-0.05, 1.05)
    _style_axis(ax, "First-proposal pass rate", "Rate")
    ax.set_xlabel("Epoch", fontsize=9, color=TEXT_SECONDARY)

    # (4) 平均エピソード長
    ax = axes[1][1]
    _plot_series(ax, x, df["len_mean"], SERIES_COLORS[0])
    _style_axis(ax, "Average episode length (longer = more rejections)", "Steps")
    ax.set_xlabel("Epoch", fontsize=9, color=TEXT_SECONDARY)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(agent_names),
        frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.965),
        labelcolor=TEXT_PRIMARY,
    )

    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    fig.suptitle(f"Training political metrics: {base_name}", fontsize=13,
                 color=TEXT_PRIMARY, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base_name}.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main():
    if len(sys.argv) < 2:
        print("使い方: python plot_log.py <メトリクスCSVのパス> [出力ディレクトリ]")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"エラー: {csv_path} が見つかりません。")
        sys.exit(1)

    output_dir = sys.argv[2] if len(sys.argv) > 2 else "files/plots"
    output_path = plot_metrics(csv_path, output_dir)
    print(f"{output_path} を作成しました。")


if __name__ == "__main__":
    main()
