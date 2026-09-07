"""同一モデルを互いに素な2つの局面集合で測り、測定のばらつきを見積もる。

感応度は同一条件のシード間で 0.045〜0.096 のように大きく散らばる
（docs/results.md 第5節）。この散らばりが

- **測定のばらつき**（走査した局面がたまたま偏った）なのか
- **方策のばらつき**（シードごとに本当に別の方策が学習された）なのか

を切り分ける。走査は決定的なので、提案者を偶数/奇数の2組に分ければ
**同一モデルに対する2つの独立な推定**が得られる。
2つがほぼ一致するなら測定は安定しており、散らばりは方策側に由来する。

使い方:
    python -m tools.sensitivity_split 69 70 79 80
"""
import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.vote_sensitivity import measure  # noqa: E402

CONFIG_RE = re.compile(r"設定ファイル: (\S+)")


def split_measure(n, rich_spread=False):
    """試行番号から、全体・偶数提案者・奇数提案者の3つの感応度を返す。"""
    path = f"result/result_{n}.txt"
    if not os.path.exists(path):
        return None
    m = CONFIG_RE.search(open(path, encoding="utf-8").read())
    if not m or not os.path.exists(m.group(1)):
        return None
    model = f"models/policy_{n}.pth"
    if not os.path.exists(model) or not torch.load(model, map_location="cpu"):
        return None
    config = json.load(open(m.group(1)))
    kw = {"rich_spread": rich_spread}
    whole = measure(config, model, **kw)
    even = measure(config, model, proposer_filter=lambda i: i % 2 == 0, **kw)
    odd = measure(config, model, proposer_filter=lambda i: i % 2 == 1, **kw)
    return whole, even, odd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trials", nargs="+", type=int)
    ap.add_argument("--rich-spread", action="store_true",
                    help="配り方のパターンを増やして走査局面数を上げる")
    args = ap.parse_args()

    print(f"{'試行':>5} {'全体':>9} {'偶数側':>9} {'奇数側':>9} {'半分の差':>9} {'局面数':>7}")
    diffs = []
    for n in args.trials:
        res = split_measure(n, rich_spread=args.rich_spread)
        if res is None:
            print(f"{n:>5} （測定できません）")
            continue
        whole, even, odd = res
        d = abs(even["sensitivity"] - odd["sensitivity"])
        diffs.append(d)
        print(f"{n:>5} {whole['sensitivity']:>+9.4f} {even['sensitivity']:>+9.4f} "
              f"{odd['sensitivity']:>+9.4f} {d:>9.4f} {whole['n_comparisons']:>7}")
    if diffs:
        print()
        print(f"半分同士の差: 平均 {sum(diffs)/len(diffs):.4f} / 最大 {max(diffs):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
