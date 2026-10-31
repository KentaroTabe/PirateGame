"""第34ラウンドの事前登録した判定を行う（docs/reports/round34.md）。

D = 一致率(L=5) - 一致率(L=100) を各条件で求め、次の3つをブートストラップ区間で判定する。

    C1 脱落の効果: D(脱落あり, 縮尺なし) - D(脱落なし, 縮尺なし)
    C2 縮尺の効果: D(脱落あり, 縮尺なし) - D(脱落あり, 縮尺あり)
    C3 交互作用  : C1 - [D(脱落あり, 縮尺あり) - D(脱落なし, 縮尺あり)]

3比較の補正として、区間の信頼水準は 1 - 0.05/3 とする。

使い方:
    python scripts/analyze_r34.py log/equilibrium_distance_r34.csv
"""
import argparse
import csv
import sys

import numpy as np

ALPHA = 0.05
COMPARISONS = 3
RESAMPLES = 10000
RANDOM_SEED = 34
LOW_L, HIGH_L = 5, 100


def load_cells(path, metric):
    """{(脱落, L, 縮尺あり): [値...]} を返す。"""
    cells = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (int(row["eliminate"]), int(float(row["L"])),
                   float(row["reward_scale"]) != 1.0)
            cells.setdefault(key, []).append(float(row[metric]))
    return cells


def contrast_weights(name):
    """対比ごとに、D の組み合わせの係数 {(脱落, 縮尺あり): 係数} を返す。"""
    if name == "C1":
        return {(1, False): 1, (0, False): -1}
    if name == "C2":
        return {(1, False): 1, (1, True): -1}
    if name == "C3":
        return {(1, False): 1, (0, False): -1, (1, True): -1, (0, True): 1}
    raise ValueError(f"未知の対比: {name}")


def contrast(cells, weights):
    return sum(c * (np.mean(cells[(e, LOW_L, s)]) - np.mean(cells[(e, HIGH_L, s)]))
               for (e, s), c in weights.items())


def bootstrap_interval(cells, weights, resamples=RESAMPLES, seed=RANDOM_SEED,
                       alpha=ALPHA / COMPARISONS):
    """セルごとに復元抽出した対比のパーセンタイル区間を返す。"""
    rng = np.random.default_rng(seed)
    total = np.zeros(resamples)
    for (e, s), c in weights.items():
        for L, sign in ((LOW_L, 1), (HIGH_L, -1)):
            values = np.asarray(cells[(e, L, s)], dtype=float)
            draws = rng.choice(values, size=(resamples, len(values)), replace=True)
            total += c * sign * draws.mean(axis=1)
    return (float(np.quantile(total, alpha / 2)), float(np.quantile(total, 1 - alpha / 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--metric", default="vote_match_either")
    args = ap.parse_args()

    cells = load_cells(args.csv, args.metric)
    print(f"指標 {args.metric}")
    for e in (1, 0):
        for s in (False, True):
            low, high = cells[(e, LOW_L, s)], cells[(e, HIGH_L, s)]
            print(f"  脱落{'あり' if e else 'なし'}・縮尺{'あり' if s else 'なし'}: "
                  f"L=5 {np.mean(low):.3f}(n={len(low)}) L=100 {np.mean(high):.3f}"
                  f"(n={len(high)}) D={np.mean(low) - np.mean(high):+.3f}")
    for name in ("C1", "C2", "C3"):
        weights = contrast_weights(name)
        low, high = bootstrap_interval(cells, weights)
        verdict = "0 を含まない" if low > 0 or high < 0 else "0 を含む"
        print(f"  {name}: {contrast(cells, weights):+.3f} 区間 [{low:+.3f}, {high:+.3f}] {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
