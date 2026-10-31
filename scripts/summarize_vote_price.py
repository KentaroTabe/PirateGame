"""各試行の投票者の「値段」を一覧にする（学習は行わない）。

`tools.vote_price.measure()` を試行ごとに呼び、設定と並べて表示し、
`--csv` を付ければ CSV に落とす（`scripts/group_equilibrium_distance.py` で群比較できる）。

使い方:
    python scripts/summarize_vote_price.py 128 130 132 134 129 131 133 135
    python scripts/summarize_vote_price.py --all --csv log/vote_price.csv
"""
import argparse
import csv
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.vote_price import measure  # noqa: E402

CONFIG_RE = re.compile(r"設定ファイル: (\S+)")

FIELDS = ["trial", "config", "num_agents", "total_gems", "L", "weights", "penalty",
          "vote_tally", "proposer_last", "noise_dims", "pretrain", "seed",
          "mean_price", "gap_strict", "gap_cont", "share_never", "share_always",
          "tie_accept_share", "n_contexts", "n_tie_contexts"]


def all_trials():
    numbers = []
    for name in os.listdir("result"):
        m = re.search(r"result_(\d+)\.txt$", name)
        if m:
            numbers.append(int(m.group(1)))
    return sorted(numbers)


def row_for(trial):
    path = f"result/result_{trial}.txt"
    if not os.path.exists(path):
        return None, "result がない"
    with open(path, encoding="utf-8") as f:
        m = CONFIG_RE.search(f.read())
    if not m or not os.path.exists(m.group(1)):
        return None, "設定ファイルが追えない"
    model = f"models/policy_{trial}.pth"
    if not os.path.exists(model):
        return None, "モデルがない"
    if not torch.load(model, map_location="cpu"):
        return None, "モデルが空"

    with open(m.group(1), encoding="utf-8") as f:
        config = json.load(f)
    r = measure(config, model)
    return {
        "trial": trial,
        "config": os.path.basename(m.group(1)),
        "num_agents": config["num_agents"],
        "total_gems": config["total_gems"],
        "L": config["L"],
        "weights": "-".join(str(w) for w in config["agent_weights"]),
        "penalty": config.get("excess_vote_penalty", 0.0),
        "vote_tally": int(bool(config.get("observe_vote_tally", False))),
        "proposer_last": int(bool(config.get("proposer_votes_last", False))),
        "noise_dims": int(config.get("observe_noise_dims", 0)),
        "pretrain": int(bool(config.get("pretrain", True))),
        "seed": config.get("seed", 42),
        "mean_price": round(r["mean_price"], 4),
        "gap_strict": round(r["gap_strict"], 4),
        "gap_cont": round(r["gap_cont"], 4),
        "share_never": round(r["share_never"], 4),
        "share_always": round(r["share_always"], 4),
        "tie_accept_share": round(r["tie_accept_share"], 4),
        "n_contexts": r["n_contexts"],
        "n_tie_contexts": r["n_tie_contexts"],
    }, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trials", nargs="*", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--csv")
    args = ap.parse_args()

    trials = all_trials() if args.all else args.trials
    if not trials:
        print("試行番号を指定するか --all を付けてください")
        return 1

    print(f"{'試行':>5} {'L':>5} {'事前':>4} {'平均値段':>8} {'厳側との差':>10} "
          f"{'継続価値との差':>12} {'0個で賛成':>9} {'同額で賛成':>9}")
    rows = []
    for trial in trials:
        row, reason = row_for(trial)
        if row is None:
            print(f"{trial:>5} （{reason}）")
            continue
        rows.append(row)
        print(f"{row['trial']:>5} {row['L']:>5g} {'あり' if row['pretrain'] else 'なし':>4} "
              f"{row['mean_price']:>8.3f} {row['gap_strict']:>+10.3f} "
              f"{row['gap_cont']:>+12.3f} {row['share_always']:>8.1%} "
              f"{row['tie_accept_share']:>8.1%}")

    if args.csv and rows:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n{len(rows)} 件を {args.csv} に書き出しました")
    return 0


if __name__ == "__main__":
    sys.exit(main())
