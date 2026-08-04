"""学習中に記録した政治的指標CSV（log/log_metrics_n.csv）をグラフ化する。

使い方: python plot_log.py <メトリクスCSVのパス> [出力ディレクトリ]

1枚のPNGに以下をまとめる:
    1. エージェント別 平均報酬の推移（全体レンジ: 序盤の死亡ペナルティ込み）
    2. 同・宝石レンジ拡大（収束後の分配の駆け引きが見える）
    3. エージェント別 死亡率の推移
    4. 一発可決率の推移（交渉の安定度）
    5. 平均エピソード長の推移（長いほど否決ラウンドが多い）
    6. 最終盤の平均取り分（横棒: 誰が勝者かの要約）
"""

import os
import sys

# CVD検証済みのカテゴリカルパレット（エージェント順に固定で割り当てる）
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                 "#4a3aa7", "#e34948"]
TEXT_PRIMARY = "#1f1f1e"
TEXT_SECONDARY = "#5c5b54"
GRID_COLOR = "#d8d7d0"

# 日本語フォントが見つかった場合のみ日本語表記を使う
JP_FONTS = ["Hiragino Sans", "Hiragino Kaku Gothic ProN", "Yu Gothic", "Meiryo",
            "IPAexGothic", "Noto Sans CJK JP"]

LABELS_JP = {
    "reward_full": "平均報酬の推移（全体: 死亡ペナルティ込み）",
    "reward_zoom": "平均報酬の推移（宝石レンジ拡大: 分配の駆け引き）",
    "death": "死亡率の推移（提案が否決され海に落とされた割合）",
    "first_pass": "一発可決率の推移（最初の提案がそのまま通った割合）",
    "len": "平均エピソード長の推移（長いほど否決ラウンドが多い）",
    "final": "最終盤の平均取り分（終盤10%区間の平均報酬）",
    "epoch": "エポック",
    "reward": "平均報酬",
    "rate": "割合",
    "steps": "ステップ数",
    "caption": "各点は学習中に実施した貪欲方策（ε=0）の評価対局の平均。薄い線は生データ、濃い線は移動平均。",
}
LABELS_EN = {
    "reward_full": "Average reward (full range, incl. death penalty)",
    "reward_zoom": "Average reward (zoomed to gem range)",
    "death": "Death rate (proposal rejected)",
    "first_pass": "First-proposal pass rate",
    "len": "Average episode length (longer = more rejections)",
    "final": "Final average share (last 10% of training)",
    "epoch": "Epoch",
    "reward": "Avg reward",
    "rate": "Rate",
    "steps": "Steps",
    "caption": "Each point averages greedy (ε=0) evaluation games run during training. Thin = raw, bold = rolling mean.",
}


def _setup_fonts():
    """日本語フォントがあれば設定し、使用するラベル辞書を返す。"""
    import matplotlib
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    available = [f for f in JP_FONTS if f in installed]
    if available:
        matplotlib.rcParams["font.family"] = available + ["sans-serif"]
        return LABELS_JP
    return LABELS_EN


def _style_axis(ax, title, ylabel):
    ax.set_title(title, fontsize=10.5, color=TEXT_PRIMARY, loc="left")
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SECONDARY)
    ax.grid(True, linestyle=":", linewidth=0.8, color=GRID_COLOR, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def _smooth(series, n_rows):
    window = max(1, n_rows // 15)
    return series.rolling(window=window, min_periods=1).mean()


def _plot_series(ax, x, y, color, label=None):
    """生データを薄く、移動平均を主線として描く。"""
    ax.plot(x, y, color=color, linewidth=1, alpha=0.25)
    ax.plot(x, _smooth(y, len(y)), color=color, linewidth=2, label=label)


def _spread_positions(values, min_gap, lo, hi):
    """直接ラベルが重ならないよう、値の順序を保ったまま縦位置をずらす。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    positions = [min(max(v, lo), hi) for v in values]
    for prev, cur in zip(order, order[1:]):
        if positions[cur] - positions[prev] < min_gap:
            positions[cur] = positions[prev] + min_gap
    # 上にはみ出した分は全体を押し下げる（下端は lo でクランプ）
    overflow = max(positions) - hi
    if overflow > 0:
        positions = [max(p - overflow, lo) for p in positions]
    return positions


def _zoom_limits(df, agent_names):
    """収束後（後半区間）の報酬レンジからズーム用のy軸範囲を決める。

    死亡ペナルティによる一時的な急落へ引きずられないよう、
    移動平均系列のパーセンタイルでレンジを決める（範囲外は全体パネルで確認できる）。
    """
    import numpy as np

    half = len(df) // 2
    vals = np.concatenate([
        _smooth(df[f"rew_{a}"], len(df)).iloc[half:].to_numpy() for a in agent_names
    ])
    lo = min(-1.0, float(np.percentile(vals, 2)) - 0.3)
    hi = float(vals.max()) + 0.5
    return lo, hi


def plot_metrics(csv_path, output_dir="files/plots"):
    # 学習プロセス(マルチプロセスワーカー)に描画ライブラリを読み込ませないよう遅延インポート
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    labels = _setup_fonts()

    df = pd.read_csv(csv_path)
    agent_names = [c[len("rew_"):] for c in df.columns if c.startswith("rew_")]
    colors = {a: SERIES_COLORS[i % len(SERIES_COLORS)] for i, a in enumerate(agent_names)}
    x = df["epoch"]

    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    fig.patch.set_facecolor("white")

    # (1) 平均報酬: 全体レンジ
    ax = axes[0][0]
    for a in agent_names:
        _plot_series(ax, x, df[f"rew_{a}"], colors[a], label=a)
    ax.axhline(y=0, color=GRID_COLOR, linewidth=1)
    _style_axis(ax, labels["reward_full"], labels["reward"])

    # (2) 平均報酬: 宝石レンジ拡大 + 右端に直接ラベル
    ax = axes[0][1]
    lo, hi = _zoom_limits(df, agent_names)
    finals = []
    for a in agent_names:
        smooth = _smooth(df[f"rew_{a}"], len(df))
        ax.plot(x, df[f"rew_{a}"], color=colors[a], linewidth=1, alpha=0.25)
        ax.plot(x, smooth, color=colors[a], linewidth=2)
        finals.append(float(smooth.iloc[-1]))
    ax.axhline(y=0, color=GRID_COLOR, linewidth=1)
    ax.set_ylim(lo, hi)
    x_max = float(x.iloc[-1])
    ax.set_xlim(float(x.iloc[0]), x_max + (x_max - float(x.iloc[0])) * 0.06)
    label_ys = _spread_positions(finals, (hi - lo) * 0.05, lo, hi)
    for a, y_pos in zip(agent_names, label_ys):
        ax.text(x_max + (x_max - float(x.iloc[0])) * 0.015, y_pos, a,
                color=colors[a], fontsize=9, fontweight="bold", va="center")
    _style_axis(ax, labels["reward_zoom"], labels["reward"])

    # (3) 死亡率
    ax = axes[1][0]
    for a in agent_names:
        _plot_series(ax, x, df[f"death_{a}"], colors[a])
    ax.set_ylim(-0.05, 1.05)
    _style_axis(ax, labels["death"], labels["rate"])

    # (4) 一発可決率
    ax = axes[1][1]
    _plot_series(ax, x, df["first_pass_rate"], SERIES_COLORS[0])
    ax.set_ylim(-0.05, 1.05)
    _style_axis(ax, labels["first_pass"], labels["rate"])

    # (5) 平均エピソード長
    ax = axes[2][0]
    _plot_series(ax, x, df["len_mean"], SERIES_COLORS[0])
    _style_axis(ax, labels["len"], labels["steps"])
    ax.set_xlabel(labels["epoch"], fontsize=9, color=TEXT_SECONDARY)

    # (6) 最終盤の平均取り分（横棒）
    ax = axes[2][1]
    tail = df.iloc[-max(1, len(df) // 10):]
    final_means = [float(tail[f"rew_{a}"].mean()) for a in agent_names]
    y_pos = list(range(len(agent_names)))[::-1]  # A を一番上に
    ax.barh(y_pos, final_means, height=0.55,
            color=[colors[a] for a in agent_names], edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(agent_names)
    span = max(abs(v) for v in final_means) or 1.0
    for yp, v in zip(y_pos, final_means):
        offset = span * 0.03
        ax.text(v + (offset if v >= 0 else -offset), yp, f"{v:+.2f}",
                color=TEXT_PRIMARY, fontsize=9,
                va="center", ha="left" if v >= 0 else "right")
    ax.axvline(x=0, color=GRID_COLOR, linewidth=1)
    ax.set_xlim(min(0, min(final_means)) - span * 0.15, max(0, max(final_means)) + span * 0.25)
    _style_axis(ax, labels["final"], "")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.8, color=GRID_COLOR, alpha=0.8)
    ax.grid(False, axis="y")

    for col in range(2):
        axes[1][col].set_xlabel("")
    axes[2][0].set_xlabel(labels["epoch"], fontsize=9, color=TEXT_SECONDARY)

    handles, leg_labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", ncol=len(agent_names),
               frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.972),
               labelcolor=TEXT_PRIMARY)

    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    fig.suptitle(f"学習中の政治的指標: {base_name}" if labels is LABELS_JP
                 else f"Training political metrics: {base_name}",
                 fontsize=14, color=TEXT_PRIMARY, y=0.995)
    fig.text(0.5, 0.005, labels["caption"], ha="center", fontsize=8.5,
             color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.015, 1, 0.95))

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
