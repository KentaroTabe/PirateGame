"""保存済みモデルを一括で走査し、投票の取り分感応度をまとめる（学習は行わない）。

`tools/vote_sensitivity.py` の細かい指標を、指定した試行番号すべてに適用する。
設定ファイルは `result/result_n.txt` の記録から自動で引く。

第10ラウンド以前のモデルは空で保存されているので自動的に飛ばす
（docs/reports/round10.md 参照）。

使い方: python scripts/summarize_vote_sensitivity.py 59 60 61 ...
"""
import json
import os
import re
import sys

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


def main(ns):
    print(f"{'試行':>4} {'L':>5} {'宝石':>4} {'罰':>4} {'票数':>4} {'乱数':>4} "
          f"{'種':>4} {'感応度':>9} {'局面数':>7}")
    for n in ns:
        cfg_path = config_for(n)
        if cfg_path is None or not os.path.exists(cfg_path):
            print(f"{n:>4}  (設定が引けません)")
            continue
        model = f"models/policy_{n}.pth"
        if not os.path.exists(model):
            print(f"{n:>4}  (モデルがありません)")
            continue
        if not torch.load(model, map_location="cpu"):
            print(f"{n:>4}  (モデルが空: 保存処理の修正前)")
            continue

        config = json.load(open(cfg_path))
        r = measure(config, model)
        print(
            f"{n:>4} {config.get('L', '-'):>5} {config.get('total_gems', '-'):>4} "
            f"{config.get('excess_vote_penalty', 0.0):>4} "
            f"{'あり' if config.get('observe_vote_tally') else 'なし':>4} "
            f"{config.get('observe_noise_dims', 0):>4} "
            f"{config.get('seed', '-'):>4} "
            f"{r['sensitivity']:>+9.4f} {r['n_comparisons']:>7}"
        )


if __name__ == "__main__":
    args = sys.argv[1:]
    main([int(a) for a in args] if args else list(range(46, 91)))
