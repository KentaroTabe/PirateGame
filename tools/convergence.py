"""学習中メトリクスCSVから「いつ収束したか」を推定する。

使い方: python -m tools.convergence <メトリクスCSVのパス...> [--config <設定パス>]

3つの収束を別々に測る:

    - レジーム収束: 「誰が首位か」が最終的な首位と一致し続けるようになった
      エポック。実験の定性的な結論が確定した時点にあたる。
    - 分配収束: 「誰がいくつ取るか」という分配政策が固まったエポック。
    - 秩序収束: 一発可決率が閾値以上・死亡率が閾値以下の状態が定着した
      エポック（体制の安定化）。

判定上の注意が2つある。

1. 報酬の生系列は稀な死亡(-L)のスパイクに支配され平均が安定しないため、
   分配・レジームの判定は死亡が発生しなかった記録点だけを使う。

2. 「以後ずっと条件を満たす」だけで判定すると、最後まで振動し続ける系列でも
   必ず終盤のどこかが収束点として返ってしまう（有限の系列は必ず最後の
   変化を持つため）。そこで収束と認めるのは、安定した区間が全学習長の
   min_stable_tail_ratio 以上を占める場合に限る。これを満たさないものは
   「振動継続」として未収束に分類する。

許容差は評価エピソード数に由来する測定ノイズから決める。ノイズは系列自体の
ばらつきではなく隣接点の差分から推定する（低周波のドリフトを吸収せず、
振動する系列で許容差が過大になるのを防ぐため）。
"""

import json
import math
import os
import sys


def load_settings(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _smooth(series, window_ratio):
    window = max(1, len(series) // window_ratio)
    return series.rolling(window=window, min_periods=1).mean()


def _stable_point(flags, epochs, min_tail_ratio):
    """条件を満たし続ける最初のエポックと、その安定区間が占める割合を返す。

    安定区間が短すぎる（min_tail_ratio 未満）場合は振動継続とみなし
    エポックを None にする。
    """
    violations = [i for i, ok in enumerate(flags) if not ok]
    idx = (violations[-1] + 1) if violations else 0
    if idx >= len(epochs):
        return None, 0.0

    total = epochs[-1]
    tail_ratio = (total - epochs[idx]) / total if total else 0.0
    if tail_ratio < min_tail_ratio:
        return None, tail_ratio
    return epochs[idx], tail_ratio


def _point_noise(series):
    """隣接点の差分から測定ノイズ（1点あたりの標準偏差）を推定する。"""
    diffs = series.diff().dropna()
    if len(diffs) < 2:
        return 0.0
    return float(diffs.std()) / math.sqrt(2.0)


def analyze(csv_path, settings):
    import pandas as pd

    df = pd.read_csv(csv_path)
    agent_names = [c[len("rew_"):] for c in df.columns if c.startswith("rew_")]
    window_ratio = settings["smooth_window_ratio"]
    min_tail = settings["min_stable_tail_ratio"]
    death_cols = [f"death_{a}" for a in agent_names]
    epochs = df["epoch"].tolist()

    # --- 秩序収束 ---
    # 死亡・否決は「起きたかどうか」が意味を持つ事象なので平滑化せず生値で見る
    orderly = [
        float(df["first_pass_rate"].iloc[i]) >= settings["first_pass_threshold"]
        and float(df[death_cols].max(axis=1).iloc[i]) <= settings["death_threshold"]
        for i in range(len(df))
    ]
    order_epoch, order_tail = _stable_point(orderly, epochs, min_tail)

    # 死亡込みの全期間平均。死亡なし記録点の平均は、頻繁に死ぬエージェントほど
    # 「罰を受けた区間」が除外されて有利に見えるため、両方を併記して比較する。
    overall_means = {a: float(df[f"rew_{a}"].mean()) for a in agent_names}

    result = {
        "csv": csv_path,
        "total_epochs": int(epochs[-1]),
        "overall_means": overall_means,
        "order_epoch": order_epoch,
        "order_tail": order_tail,
        "peaceful_points": 0,
        "total_points": len(df),
        "regime_epoch": None,
        "regime_tail": 0.0,
        "regime_changes": None,
        "leader": None,
        "reward_epoch": None,
        "reward_tail": 0.0,
        "tolerance": None,
        "final_rewards": {},
        "policy_epoch": None,
        "policy_tail": 0.0,
        "policy_changes": None,
    }

    # --- 方策収束: 記録された提案そのものが変わらなくなったエポック ---
    # 順位にも報酬の大きさにも依存しないため、首位の分離度による交絡を受けない
    # （首位交代回数が持つ交絡については docs/reports/round6.md を参照）。
    prop_cols = [(a, f"prop_{a}") for a in agent_names if f"prop_{a}" in df.columns]
    if prop_cols:
        def canonical(row_idx):
            """提案を「自分の取り分 + 他者の取り分の多重集合」に正規化する。

            誰が買収されるかは再現しないことが分かっている（同一設定の
            独立実行で買収相手が総入れ替えになる。docs/reports/round3.md）。
            受け取り手の識別子を落とし、取り分の構成だけを方策とみなす。
            """
            form = []
            for idx, (agent, col) in enumerate(prop_cols):
                parts = str(df[col].iloc[row_idx]).split("-")
                if len(parts) != len(prop_cols):
                    form.append(str(df[col].iloc[row_idx]))
                    continue
                own = parts[idx]
                others = sorted(parts[:idx] + parts[idx + 1:])
                form.append(own + "|" + ",".join(others))
            return tuple(form)

        forms = [canonical(i) for i in range(len(df))]
        unchanged = [True] + [forms[i] == forms[i - 1] for i in range(1, len(df))]
        policy_epoch, policy_tail = _stable_point(unchanged, epochs, min_tail)
        result["policy_epoch"] = policy_epoch
        result["policy_tail"] = policy_tail
        result["policy_changes"] = sum(1 for ok in unchanged[1:] if not ok)

        # エージェント別の変更回数。めったに提案しないエージェントの方策は
        # ほとんど学習されず揺れ続けるため、全体をまとめて数えると
        # 主要な提案者の安定性が埋もれる。
        per_agent = {}
        for idx, (agent, _) in enumerate(prop_cols):
            per_agent[agent] = sum(
                1 for i in range(1, len(df)) if forms[i][idx] != forms[i - 1][idx]
            )
        result["policy_changes_per_agent"] = per_agent

    # --- 分配・レジーム収束: 死亡が出なかった記録点のみを使う ---
    peaceful = df[df[death_cols].max(axis=1) == 0].reset_index(drop=True)
    result["peaceful_points"] = len(peaceful)
    if len(peaceful) < window_ratio:
        return result

    p_epochs = peaceful["epoch"].tolist()
    smoothed = {a: _smooth(peaceful[f"rew_{a}"], window_ratio) for a in agent_names}
    tail_start = len(peaceful) - max(1, int(len(peaceful) * settings["reference_tail_ratio"]))
    references = {a: float(smoothed[a].iloc[tail_start:].mean()) for a in agent_names}

    noise = max(_point_noise(peaceful[f"rew_{a}"]) for a in agent_names)
    tolerance = max(settings["tolerance_floor"], noise * settings["tolerance_noise_factor"])

    within = [
        all(abs(float(smoothed[a].iloc[i]) - references[a]) <= tolerance for a in agent_names)
        for i in range(len(peaceful))
    ]
    reward_epoch, reward_tail = _stable_point(within, p_epochs, min_tail)

    leaders = [
        max(agent_names, key=lambda a: float(smoothed[a].iloc[i]))
        for i in range(len(peaceful))
    ]
    final_leader = max(agent_names, key=lambda a: references[a])
    same_leader = [ld == final_leader for ld in leaders]
    regime_epoch, regime_tail = _stable_point(same_leader, p_epochs, min_tail)
    changes = sum(1 for prev, cur in zip(leaders, leaders[1:]) if prev != cur)

    result.update({
        "reward_epoch": reward_epoch,
        "reward_tail": reward_tail,
        "regime_epoch": regime_epoch,
        "regime_tail": regime_tail,
        "regime_changes": changes,
        "leader": final_leader,
        "tolerance": tolerance,
        "final_rewards": references,
    })
    return result


def _format(epoch, tail):
    if epoch is None:
        return f"未収束（振動継続: 安定区間は末尾{tail:.0%}のみ）" if tail else "未収束"
    return f"{epoch}（以後{tail:.0%}が安定）"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    config_path = "configs/convergence.json"
    if "--config" in sys.argv:
        idx = sys.argv.index("--config") + 1
        config_path = sys.argv[idx]
        args = [a for a in args if a != config_path]

    if not args:
        print("使い方: python -m tools.convergence <メトリクスCSVのパス...>")
        sys.exit(1)

    settings = load_settings(config_path)
    print(f"判定条件: 許容差 = max({settings['tolerance_floor']}, "
          f"測定ノイズ x {settings['tolerance_noise_factor']}) / "
          f"安定区間 >= 全体の{settings['min_stable_tail_ratio']:.0%} / "
          f"一発可決率 >= {settings['first_pass_threshold']} / "
          f"死亡率 <= {settings['death_threshold']}")
    print("=" * 78)

    for csv_path in args:
        if not os.path.exists(csv_path):
            print(f"{csv_path}: 見つかりません")
            continue
        r = analyze(csv_path, settings)
        print(f"{os.path.basename(r['csv'])} (全{r['total_epochs']}エポック)")
        print(f"  レジーム収束: {_format(r['regime_epoch'], r['regime_tail'])}"
              + (f" / 首位 {r['leader']}" if r["leader"] else ""))
        if r["regime_changes"] is not None:
            print(f"  首位の交代回数: {r['regime_changes']}回")
        print(f"  分配収束: {_format(r['reward_epoch'], r['reward_tail'])}"
              + (f" / 許容差 {r['tolerance']:.2f}" if r["tolerance"] else ""))
        if r.get("policy_changes") is not None:
            print(f"  方策収束（全員）: {_format(r['policy_epoch'], r['policy_tail'])}"
                  f" / 取り分構成の変更 {r['policy_changes']}回")
            per = " / ".join(
                f"{a}:{n}回" for a, n in r["policy_changes_per_agent"].items()
            )
            print(f"  エージェント別の提案の変更: {per}")
        print(f"  秩序収束: {_format(r['order_epoch'], r['order_tail'])}")
        print(f"  死亡なしの記録点: {r['peaceful_points']}/{r['total_points']}")
        if r["final_rewards"]:
            rewards = " / ".join(f"{a}:{v:+.2f}" for a, v in r["final_rewards"].items())
            print(f"  最終区間の平均取り分（死亡なし記録点）: {rewards}")
        overall = " / ".join(f"{a}:{v:+.2f}" for a, v in r["overall_means"].items())
        print(f"  全期間平均（死亡込み）: {overall}")
        print("-" * 78)


if __name__ == "__main__":
    main()
