"""均衡距離の一覧（CSV）を条件ごとにまとめ、群間を比較する。

`scripts/summarize_equilibrium_distance.py --all --csv ...` が作った CSV を読み、
指定した列で群分けして平均・範囲を出す。群が2つなら Mann-Whitney の厳密な両側 p も出す。

114本の試行は L・重み構成・罰・票数観測・事前学習がばらばらなので、
**比較する前に `--filter` で条件を揃える**こと。

使い方:
    # 一強・事前学習なし・罰なし・票数観測なしに絞って L ごとに見る
    python scripts/group_equilibrium_distance.py log/equilibrium_distance.csv \\
        --filter weights=50.0-1.0-1.0-1.0-1.0-1.0 pretrain=0 penalty=0.0 vote_tally=0 \\
        --group-by L
"""
import argparse
import csv
import os
import sys
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_cells import mann_whitney_u  # noqa: E402

METRICS = ["proposal_l1", "self_gap", "coalition_gap", "vote_match_either",
           "vote_match_strict", "vote_match_lenient"]
# 厳密な両側 p 値は2群の全分割を数えるので、合計標本数がこれを超えると打ち切る
EXACT_P_MAX_SAMPLES = 20


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def matches(row, filters):
    for key, value in filters.items():
        if key not in row:
            raise SystemExit(f"列 {key} は CSV にありません")
        if row[key].strip() != value:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--filter", nargs="*", default=[], help="列=値 の形で条件を絞る")
    ap.add_argument("--group-by", default="L")
    ap.add_argument("--metrics", nargs="*", default=["proposal_l1", "self_gap",
                                                    "coalition_gap", "vote_match_either"])
    args = ap.parse_args()

    filters = {}
    for item in args.filter:
        if "=" not in item:
            raise SystemExit(f"--filter は 列=値 の形で指定してください: {item}")
        key, value = item.split("=", 1)
        filters[key] = value

    rows = [r for r in load(args.csv) if matches(r, filters)]
    if not rows:
        print("条件に合う試行がありません")
        return 1

    for metric in args.metrics:
        if metric not in METRICS:
            raise SystemExit(f"指標 {metric} は未対応です（{', '.join(METRICS)}）")

    groups = {}
    for row in rows:
        groups.setdefault(row[args.group_by], []).append(row)

    def sort_key(label):
        try:
            return (0, float(label))
        except ValueError:
            return (1, label)

    labels = sorted(groups, key=sort_key)
    print(f"条件: {filters if filters else 'なし'} / 対象 {len(rows)} 本 / "
          f"群分け: {args.group_by}")
    for metric in args.metrics:
        print(f"\n【{metric}】")
        print(f"{args.group_by:>12} {'n':>3} {'平均':>9} {'範囲':>19} 試行")
        values = {}
        for label in labels:
            vals = [float(r[metric]) for r in groups[label]]
            values[label] = vals
            trials = ",".join(r["trial"] for r in groups[label])
            print(f"{label:>12} {len(vals):>3} {sum(vals) / len(vals):>+9.3f} "
                  f"{min(vals):>+9.3f}〜{max(vals):>+9.3f} {trials}")
        for a, b in combinations(labels, 2):
            if len(values[a]) < 2 or len(values[b]) < 2:
                continue
            if len(values[a]) + len(values[b]) > EXACT_P_MAX_SAMPLES:
                # 厳密 p 値は全順列を数えるので、合計がこれを超えると現実的に計算できない
                print(f"    {a} 対 {b}: 標本 {len(values[a])}+{len(values[b])} 本は"
                      f"厳密 p 値の計算対象外（{EXACT_P_MAX_SAMPLES} 本まで）")
                continue
            u, p = mann_whitney_u(values[a], values[b])
            overlap = not (max(values[a]) < min(values[b]) or max(values[b]) < min(values[a]))
            print(f"    {a} 対 {b}: U={u:g} 両側p={p:.4f} "
                  f"重なり={'あり' if overlap else 'なし'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
