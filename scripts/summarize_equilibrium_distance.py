"""各試行の方策が均衡からどれだけ離れているかを一覧にする。

`tools.equilibrium_distance.measure()` を試行ごとに呼び、
設定（L・宝石数・重み構成・罰・票数観測・事前学習・シード）と並べて表示し、
`--csv` を付ければ CSV に落とす。

モデルに中身がない試行（manager.state_dict() が空を返していた時期に保存されたもの）は
`（モデルが空）` として飛ばす。

使い方:
    python scripts/summarize_equilibrium_distance.py 46 47 48
    python scripts/summarize_equilibrium_distance.py --all --csv log/equilibrium_distance.csv
"""
import argparse
import csv
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.equilibrium_distance import measure  # noqa: E402

CONFIG_RE = re.compile(r"設定ファイル: (\S+)")
RESULT_GLOB_RE = re.compile(r"result_(\d+)\.txt$")

FIELDS = [
    "trial", "config", "num_agents", "total_gems", "L", "weights", "penalty",
    "vote_tally", "proposer_last", "noise_dims", "pretrain", "seed",
    "proposal_l1", "self_gap", "coalition_gap",
    "vote_match_strict", "vote_match_lenient", "vote_match_either",
    "n_proposal_states", "n_vote_states",
]


def all_trials():
    numbers = []
    for name in os.listdir("result"):
        m = RESULT_GLOB_RE.search(name)
        if m:
            numbers.append(int(m.group(1)))
    return sorted(numbers)


def config_of(trial):
    path = f"result/result_{trial}.txt"
    if not os.path.exists(path):
        return None, None
    m = CONFIG_RE.search(open(path, encoding="utf-8").read())
    if not m or not os.path.exists(m.group(1)):
        return None, None
    return m.group(1), json.load(open(m.group(1), encoding="utf-8"))


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
    return {
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
        "proposal_l1": round(result["proposal_l1"], 4),
        "self_gap": round(result["self_gap"], 4),
        "coalition_gap": round(result["coalition_gap"], 4),
        "vote_match_strict": round(result["vote_match_strict"], 4),
        "vote_match_lenient": round(result["vote_match_lenient"], 4),
        "vote_match_either": round(result["vote_match_either"], 4),
        "n_proposal_states": result["n_proposal_states"],
        "n_vote_states": result["n_vote_states"],
    }, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trials", nargs="*", type=int)
    ap.add_argument("--all", action="store_true", help="result/ にある全試行を対象にする")
    ap.add_argument("--csv", help="結果を書き出す CSV のパス")
    args = ap.parse_args()

    trials = all_trials() if args.all else args.trials
    if not trials:
        print("試行番号を指定するか --all を付けてください")
        return 1

    print(f"{'試行':>5} {'L':>6} {'宝石':>4} {'罰':>4} {'票数':>4} "
          f"{'提案L1':>7} {'取り分差':>8} {'連合差':>7} {'投票一致(どちらか)':>18}")
    rows = []
    for trial in trials:
        row, reason = row_for(trial)
        if row is None:
            print(f"{trial:>5} （{reason}）")
            continue
        rows.append(row)
        print(f"{row['trial']:>5} {row['L']:>6g} {row['total_gems']:>4} {row['penalty']:>4g} "
              f"{'あり' if row['vote_tally'] else 'なし':>4} "
              f"{row['proposal_l1']:>7.3f} {row['self_gap']:>+8.3f} "
              f"{row['coalition_gap']:>+7.3f} {row['vote_match_either']:>17.1%}")

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
