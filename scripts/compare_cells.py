"""2つの条件セルの感応度を比べ、事前に決めた基準で判定する。

第20〜23ラウンドの教訓から、判定は**実験前に決めた基準**で機械的に行う。
本スクリプトは次を出力する。

1. 各セルの n・平均・範囲
2. **範囲が重なるかどうか**（本プロジェクトの主判定基準）
3. Mann-Whitney U と厳密な両側 p 値（全順列を数えて求める。参考情報）

**主判定は「範囲が重ならないこと」である。** p 値は補助にとどめ、
重なったのに p 値だけを根拠に主張してはならない。

使い方:
    python scripts/compare_cells.py --a 67 68 95 96 --b 91 92 93 94
"""
import argparse
import json
import os
import re
import sys
from itertools import combinations

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.vote_sensitivity import measure  # noqa: E402

CONFIG_RE = re.compile(r"設定ファイル: (\S+)")


def sensitivity_of(n):
    path = f"result/result_{n}.txt"
    if not os.path.exists(path):
        return None
    m = CONFIG_RE.search(open(path, encoding="utf-8").read())
    if not m or not os.path.exists(m.group(1)):
        return None
    model = f"models/policy_{n}.pth"
    if not os.path.exists(model) or not torch.load(model, map_location="cpu"):
        return None
    return measure(json.load(open(m.group(1))), model)["sensitivity"]


def mann_whitney_u(a, b):
    """U 統計量と、全順列を数えて求めた厳密な両側 p 値。"""
    n1, n2 = len(a), len(b)
    u1 = sum(1 for x in a for y in b if x > y) + 0.5 * sum(
        1 for x in a for y in b if x == y)
    u = min(u1, n1 * n2 - u1)

    pooled = sorted(a + b)
    counts = 0
    hits = 0
    for pick in combinations(range(len(pooled)), n1):
        ga = [pooled[i] for i in pick]
        gb = [pooled[i] for i in range(len(pooled)) if i not in pick]
        v1 = sum(1 for x in ga for y in gb if x > y) + 0.5 * sum(
            1 for x in ga for y in gb if x == y)
        counts += 1
        if min(v1, n1 * n2 - v1) <= u:
            hits += 1
    return u, hits / counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", type=int, required=True, help="条件Aの試行番号")
    ap.add_argument("--b", nargs="+", type=int, required=True, help="条件Bの試行番号")
    ap.add_argument("--label-a", default="条件A")
    ap.add_argument("--label-b", default="条件B")
    args = ap.parse_args()

    va = [v for v in (sensitivity_of(n) for n in args.a) if v is not None]
    vb = [v for v in (sensitivity_of(n) for n in args.b) if v is not None]
    if not va or not vb:
        print("測定できる試行がありません")
        return 1

    for label, vals, ns in ((args.label_a, va, args.a), (args.label_b, vb, args.b)):
        print(f"{label}: n={len(vals)} 平均 {sum(vals)/len(vals):+.4f} "
              f"範囲 {min(vals):+.4f}〜{max(vals):+.4f}")
        print("   " + " ".join(f"{n}:{v:+.4f}" for n, v in zip(ns, vals)))

    overlap = not (max(va) < min(vb) or max(vb) < min(va))
    u, p = mann_whitney_u(va, vb)

    print()
    print(f"範囲の重なり: {'あり' if overlap else 'なし'}")
    print(f"Mann-Whitney U = {u:g}、厳密両側 p = {p:.4f}（参考）")
    print()
    if overlap:
        print("【判定】範囲が重なる → **差は確立しない**。")
        print("        p 値が小さくても、主判定基準は範囲である。")
    else:
        print("【判定】範囲が重ならない → **差は確立する**。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
