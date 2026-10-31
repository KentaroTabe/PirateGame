"""要因計画の検出力を、既存の2セルの実測値から見積もる。

L=5 と L=100 の差を D とする。別の条件（脱落なし・報酬の縮尺）で D が縮む場合を、
L=100 の実測値を L=5 の平均へ `--shrink` の割合だけ寄せて模擬する。
両条件を各 n シードで復元抽出し、差の差が検出される割合を数える。
正規近似の z 値で判定するので、計画を比べるための目安である。

使い方:
    python scripts/estimate_design_power.py log/onpath_equilibrium.csv \\
        --metric onpath_match_either --n 8 12 16 24 --shrink 0.5 1.0
"""
import argparse
import csv
import os
import sys

import numpy as np

COMPARISONS = 3  # 主要な比較の数（脱落・縮尺・交互作用）
Z_UNCORRECTED = 1.96
Z_BONFERRONI = 2.394


def difference_of_differences_power(low, high, n, shrink, draws, z_threshold, seed=0):
    """D が shrink の割合だけ縮むときに、差の差を検出する割合を返す。"""
    rng = np.random.default_rng(seed)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    shifted = high + shrink * (low.mean() - high.mean())

    def cell(values):
        sample = rng.choice(values, size=(draws, n), replace=True)
        return sample.mean(axis=1), sample.var(axis=1, ddof=1) / n

    means_a, var_a = cell(low)
    means_b, var_b = cell(high)
    means_c, var_c = cell(low)
    means_d, var_d = cell(shifted)
    estimate = (means_c - means_d) - (means_a - means_b)
    se = np.sqrt(var_a + var_b + var_c + var_d)
    return float(np.mean(np.abs(estimate) > z_threshold * np.maximum(se, 1e-12)))


def load_cell(path, metric, filters):
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if all(r[k].strip() == v for k, v in filters.items())]
    return [float(r[metric]) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--metric", default="onpath_match_either")
    ap.add_argument("--low-L", default="5")
    ap.add_argument("--high-L", default="100")
    ap.add_argument("--n", nargs="+", type=int, default=[8, 12, 16, 24])
    ap.add_argument("--shrink", nargs="+", type=float, default=[0.5, 1.0])
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--weights", default="50.0-1.0-1.0-1.0-1.0-1.0")
    args = ap.parse_args()

    base = {"weights": args.weights, "pretrain": "0", "penalty": "0.0", "vote_tally": "0",
            "noise_dims": "0", "total_gems": "5"}
    low = load_cell(args.csv, args.metric, dict(base, L=args.low_L))
    high = load_cell(args.csv, args.metric, dict(base, L=args.high_L))
    if not low or not high:
        print("条件に合う試行がありません")
        return 1

    print(f"指標 {args.metric}")
    print(f"L={args.low_L}: n={len(low)} 平均 {np.mean(low):.3f} 標準偏差 {np.std(low, ddof=1):.3f}")
    print(f"L={args.high_L}: n={len(high)} 平均 {np.mean(high):.3f} "
          f"標準偏差 {np.std(high, ddof=1):.3f}")
    print(f"D = {np.mean(low) - np.mean(high):.3f}")
    print(f"\n{'n':>4} {'縮む割合':>8} {'検出力(補正なし)':>16} "
          f"{'検出力({}比較の補正)'.format(COMPARISONS):>18}")
    for n in args.n:
        for shrink in args.shrink:
            plain = difference_of_differences_power(low, high, n, shrink, args.draws,
                                                    Z_UNCORRECTED)
            corrected = difference_of_differences_power(low, high, n, shrink, args.draws,
                                                        Z_BONFERRONI)
            print(f"{n:>4} {shrink:>8.0%} {plain:>16.1%} {corrected:>18.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
