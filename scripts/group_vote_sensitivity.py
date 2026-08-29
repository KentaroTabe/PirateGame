"""感応度の測定値を条件ごとにまとめ、平均と範囲を出す。

条件は (L, 宝石, 罰, 票数観測, 乱数次元, 提案者が最後に投票) の組で定義する。
手で表に転記すると条件を取り違えるため、必ずこのスクリプトの出力を使う。

使い方: python scripts/group_vote_sensitivity.py [試行番号...]
"""
import json
import os
import re
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.vote_sensitivity import measure  # noqa: E402

CONFIG_RE = re.compile(r"設定ファイル: (\S+)")


def config_for(n):
    path = f"result/result_{n}.txt"
    if not os.path.exists(path):
        return None
    m = CONFIG_RE.search(open(path, encoding="utf-8").read())
    return m.group(1) if m else None


def key_of(c):
    return (
        c.get("L"),
        c.get("total_gems"),
        c.get("excess_vote_penalty", 0.0),
        bool(c.get("observe_vote_tally")),
        int(c.get("observe_noise_dims", 0)),
        bool(c.get("proposer_votes_last")),
    )


def main(ns):
    groups = defaultdict(list)
    for n in ns:
        cfg_path = config_for(n)
        model = f"models/policy_{n}.pth"
        if cfg_path is None or not os.path.exists(cfg_path) or not os.path.exists(model):
            continue
        if not torch.load(model, map_location="cpu"):
            continue
        config = json.load(open(cfg_path))
        r = measure(config, model)
        groups[key_of(config)].append((n, config.get("seed"), r["sensitivity"]))

    print(f"{'L':>5} {'宝石':>4} {'罰':>4} {'票数':>4} {'乱数':>4} {'最後':>4} "
          f"{'n':>2} {'平均':>8} {'最小':>8} {'最大':>8}  試行")
    for k in sorted(groups, key=lambda x: (x[1], x[0], x[3], x[4], x[2], x[5])):
        vals = [v for _, _, v in groups[k]]
        trials = ",".join(str(n) for n, _, _ in sorted(groups[k]))
        L, gems, pen, tally, noise, last = k
        print(
            f"{L:>5} {gems:>4} {pen:>4} "
            f"{'あり' if tally else 'なし':>4} {noise:>4} "
            f"{'あり' if last else 'なし':>4} "
            f"{len(vals):>2} {sum(vals)/len(vals):>+8.4f} "
            f"{min(vals):>+8.4f} {max(vals):>+8.4f}  {trials}"
        )


if __name__ == "__main__":
    args = sys.argv[1:]
    main([int(a) for a in args] if args else list(range(46, 91)))
