"""log/log_metrics_n.csv からエージェント別の全期間平均報酬（死亡込み）を出す。

第2ラウンドの訂正で主指標に採用した「全期間平均」を、手写しせずに機械的に求める。
死亡なし記録点の平均は、頻繁に死ぬエージェントほど罰を受けた区間が除外されて
有利に見えるため使わない（docs/findings.md「指標に関する注意」）。

使い方:
    python scripts/summarize_agent_rewards.py 128 129 130
"""
import argparse
import csv
import os
import sys

AGENTS = ["A", "B", "C", "D", "E", "F"]


def summarize(n):
    path = f"log/log_metrics_{n}.csv"
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    out = {}
    for a in AGENTS:
        col = f"rew_{a}"
        if col not in rows[0]:
            continue
        vals = [float(r[col]) for r in rows if r[col] != ""]
        out[a] = sum(vals) / len(vals) if vals else float("nan")
    return out, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trials", nargs="+", type=int)
    args = ap.parse_args()

    print(f"{'試行':>5} " + " ".join(f"{a:>8}" for a in AGENTS)
          + f" {'記録点':>6} {'首位':>4}")
    for n in args.trials:
        res = summarize(n)
        if res is None:
            print(f"{n:>5} （記録なし）")
            continue
        means, points = res
        top = max(means, key=lambda a: means[a])
        print(f"{n:>5} " + " ".join(f"{means[a]:>+8.3f}" for a in AGENTS)
              + f" {points:>6} {top:>4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
