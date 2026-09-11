"""本プロジェクトのゲーム（ランダム順・否決で脱落）の理論均衡を出す。

docs/discussion.md の「理論均衡」に使う。`solver.RandomOrderSolver` で、
設定ファイルの重み構成と L ごとに次を表示する。

    - 全員生存時の各エージェントの期待価値
    - A が提案者のときに手元に残す宝石
    - A が投票者のとき（他者が提案者のとき）に賛成する＝連合に入る確率
    - 可決できない (生存集合, 提案者) の数（0 なら -L は均衡経路に現れない）

さらに A の重みだけを動かし、選出確率への見返り
（recognition probability へのリターン）が正か負かを見る。

宝石が整数なので、投票者が「無差別なら反対」か「無差別なら賛成」かで予測が変わる。
既定は前者（FixedOrderSolver・事前学習と同じ）。--accept-indifferent で後者になる。
理論予測は両者で挟まれる幅として読む。

使い方:
    python scripts/random_order_benchmark.py configs/weights_flat.json configs/weights_dictator.json
    python scripts/random_order_benchmark.py configs/weights_dictator.json --accept-indifferent
"""
import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import RandomOrderSolver  # noqa: E402

PROPOSER_A = 0


def count_failing(solver, n_agents):
    """可決できない (生存集合, 提案者) の数。"""
    failing = 0
    for size in range(2, n_agents + 1):
        for subset in itertools.combinations(range(n_agents), size):
            alive = frozenset(subset)
            for proposer in alive:
                _, passes, _ = solver.proposal_outcome(alive, proposer)
                if not passes:
                    failing += 1
    return failing


def summarize(solver, n_agents):
    """全員生存時の (期待価値, A の手元, A が投票者のとき連合に入る確率, A の選出確率)。"""
    full = frozenset(range(n_agents))
    value = solver.value(full)
    payoff_a, _, _ = solver.proposal_outcome(full, PROPOSER_A)
    probs = solver.recognition_probabilities(full)
    others = [p for p in full if p != PROPOSER_A]
    weight = sum(probs[p] for p in others)
    inclusion = sum(probs[p] * solver.proposal_outcome(full, p)[2][PROPOSER_A] for p in others) / weight
    return value, payoff_a[PROPOSER_A], inclusion, probs[PROPOSER_A]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", help="重み構成を読む設定ファイル")
    ap.add_argument("--L", nargs="+", type=float, default=[5, 20, 30, 50, 100])
    ap.add_argument("--sweep", nargs="+", type=float, default=[1, 1.5, 2, 3, 5, 10, 20, 50, 100],
                    help="A の重み（他は1）を動かして選出確率へのリターンを見る")
    ap.add_argument("--accept-indifferent", action="store_true",
                    help="投票者は無差別なら賛成する（既定は無差別なら反対）")
    args = ap.parse_args()
    rule = "無差別なら賛成" if args.accept_indifferent else "無差別なら反対"

    def make(n, gems, L, weights):
        return RandomOrderSolver(n, gems, L, weights, accept_when_indifferent=args.accept_indifferent)

    print(f"【重み構成ごとの理論均衡】（投票の規則: {rule}）")
    for path in args.configs:
        cfg = json.load(open(path, encoding="utf-8"))
        n, gems, weights = cfg["num_agents"], cfg["total_gems"], cfg["agent_weights"]
        print(f"\n{os.path.basename(path)}  重み {weights}  宝石 {gems}")
        for L in args.L:
            solver = make(n, gems, L, weights)
            value, keep, incl, prob = summarize(solver, n)
            fails = count_failing(solver, n)
            print(f"  L={L:>5g}: 期待価値 [{', '.join(f'{v:.3f}' for v in value)}]"
                  f" / A の手元 {keep:.3f} / A が連合に入る確率 {incl:.3f}"
                  f" / A の選出確率 {prob:.3f} / 可決不能 {fails}")

    base = json.load(open(args.configs[0], encoding="utf-8"))
    n, gems = base["num_agents"], base["total_gems"]
    L = args.L[-1]
    print(f"\n【選出確率へのリターン】（A の重み w・他は1、{n}人・宝石{gems}個・L={L:g}・{rule}）")
    print(f"{'w':>6} {'A の選出確率':>10} {'A の期待価値':>10} {'A の手元':>8} {'連合に入る確率':>12}")
    for w in args.sweep:
        weights = [w] + [1.0] * (n - 1)
        value, keep, incl, prob = summarize(make(n, gems, L, weights), n)
        print(f"{w:>6g} {prob:>10.3f} {value[PROPOSER_A]:>10.3f} {keep:>8.3f} {incl:>12.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
