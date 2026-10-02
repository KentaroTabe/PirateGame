"""A の全期間平均報酬を「死亡の頻度 × L」と「生存時の取り分」に分解する。

A の1エピソードの報酬は、死ねば -L、生き残れば 0〜宝石数 のいずれかである
（env._resolve_vote()）。したがって各記録点で

    平均報酬 = (1 - 死亡率) × 生存時の平均取り分 - 死亡率 × L

が厳密に成り立つ。全記録点で平均すると、

    生存時の平均取り分（生存で重みづけ） = (平均報酬 + L × 平均死亡率) / (1 - 平均死亡率)

となる。L が変わったとき A の報酬が落ちるのが
「死ぬ頻度が上がるから」なのか「死ぬ頻度は同じで値段（L）が上がるから」なのか
「生き残っても取り分が減るから」なのかを切り分ける。docs/discussion.md で使う。

使い方:
    python scripts/decompose_agent_reward.py 16 17 20 27 ...
"""
import argparse
import csv
import os
import re
import sys

L_RE = re.compile(r"命の重さ\(ペナルティ L\): (\S+)")


def decompose(n, agent="A", root="."):
    """(L, 平均報酬, 平均死亡率, 生存時の平均取り分, 死亡による損失) を返す。"""
    result = os.path.join(root, f"result/result_{n}.txt")
    metrics = os.path.join(root, f"log/log_metrics_{n}.csv")
    if not (os.path.exists(result) and os.path.exists(metrics)):
        return None
    with open(result, encoding="utf-8") as f:
        m = L_RE.search(f.read())
    if not m:
        return None
    L = float(m.group(1))

    with open(metrics, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # エージェント数が少ない試行（n=3 など）には存在しない列がある
    if not rows or f"rew_{agent}" not in rows[0]:
        return None
    rews = [float(r[f"rew_{agent}"]) for r in rows]
    deaths = [float(r[f"death_{agent}"]) for r in rows]
    mean_r = sum(rews) / len(rews)
    mean_d = sum(deaths) / len(deaths)
    survive_gems = (mean_r + L * mean_d) / (1 - mean_d) if mean_d < 1 else float("nan")
    return L, mean_r, mean_d, survive_gems, L * mean_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trials", nargs="+", type=int)
    ap.add_argument("--agent", default="A")
    args = ap.parse_args()

    print(f"{'試行':>5} {'L':>6} {'平均報酬':>9} {'死亡率':>8} {'生存時の取り分':>12} {'死亡による損失':>12}")
    for n in args.trials:
        res = decompose(n, args.agent)
        if res is None:
            print(f"{n:>5} （記録なし）")
            continue
        L, r, d, g, loss = res
        print(f"{n:>5} {L:>6g} {r:>+9.3f} {d:>8.2%} {g:>12.3f} {loss:>12.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
