"""各試行の到達局面での均衡との差を一覧にして CSV に落とす（学習は行わない）。

使い方:
    python scripts/summarize_onpath_equilibrium.py --all --csv log/onpath_equilibrium.csv
    python scripts/group_equilibrium_distance.py log/onpath_equilibrium.csv \\
        --filter weights=50.0-1.0-1.0-1.0-1.0-1.0 pretrain=0 penalty=0.0 vote_tally=0 \\
        noise_dims=0 total_gems=5 --group-by L --metrics onpath_match_either
"""
import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from summarize_equilibrium_distance import all_trials, config_of  # noqa: E402
from tools.onpath_equilibrium import measure  # noqa: E402

CONDITION_FIELDS = ["trial", "config", "num_agents", "total_gems", "L", "weights", "penalty",
                    "vote_tally", "proposer_last", "noise_dims", "pretrain", "seed"]
METRIC_FIELDS = ["onpath_l1", "onpath_gap_strict", "onpath_gap_lenient",
                 "onpath_match_strict", "onpath_match_lenient", "onpath_match_either",
                 "first_reject", "expected_rejections",
                 "first_keep", "first_keep_strict", "first_keep_lenient"]


def row_for(trial):
    """1試行の測定結果。測れない場合は (None, 理由) を返す。"""
    config_path, config = config_of(trial)
    if config is None:
        return None, "設定ファイルが追えない"
    model = f"models/policy_{trial}.pth"
    if not os.path.exists(model):
        return None, "モデルがない"
    if not torch.load(model, map_location="cpu"):
        return None, "モデルが空"

    result = measure(config, model)
    row = {
        "trial": trial,
        "config": os.path.basename(config_path),
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
    }
    row.update({key: round(result[key], 4) for key in METRIC_FIELDS})
    return row, None


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

    rows, skipped = [], {}
    for trial in trials:
        row, reason = row_for(trial)
        if row is None:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        rows.append(row)
        print(f"{trial:>4} L={row['L']:>5} 一致 {row['onpath_match_either']:.3f} "
              f"初回否決 {row['first_reject']:.3f} 初回の手元 {row['first_keep']:.2f} "
              f"（均衡 {row['first_keep_strict']:.2f}〜{row['first_keep_lenient']:.2f}）")
    print(f"\n測定 {len(rows)} 本 / 除外 {skipped}")

    if args.csv and rows:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CONDITION_FIELDS + METRIC_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
