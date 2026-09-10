"""感応度の走査局面のうち、実際の対局で到達しうる局面の割合を数える。

`tools/vote_sensitivity.measure()` は票数観測ありの設定で、票の途中経過
[これまでの賛成数, 投票済み人数] を全組み合わせ走査する。
しかし投票順は環境が決める（`env._build_voting_order()`：生存者の固定順、
proposer_votes_last なら提案者だけ末尾）ので、ある投票者がある提案者のもとで
投票するとき「投票済み人数」は1通りに決まる。残りの組み合わせは
**対局では決して現れない観測**であり、そこでのネットワークの出力は外挿である。

本スクリプトは走査を忠実に再現し（学習済みモデルは使わない）、
到達可能な局面の割合を設定ごとに出す。docs/discussion.md で使う。

使い方:
    python scripts/scan_reachability.py configs/r30_tally_L100_seed1.json ...
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import PirateGemEnv  # noqa: E402


def reachability(config):
    """(到達可能な走査局面数, 走査局面数) を返す。票数観測なしなら (None, None)。"""
    env = PirateGemEnv(config)
    if not env.observe_vote_tally:
        return None, None

    n = env.n_agents
    # tools/vote_sensitivity.measure() の tallies と同じ列挙
    tallies = [(yes, voted) for voted in range(n) for yes in range(voted + 1)]
    # measure() は配り方2通り × 投票者の取り分 0..宝石数-1 を走査する
    per_tally = 2 * env.total_gems

    env.reset(seed=0)
    reachable = total = 0
    for v in range(n):
        for p in range(n):
            if p == v:
                continue
            env.proposer = env.possible_agents[p]
            order = env._build_voting_order()
            position = order.index(env.possible_agents[v])
            for _, voted in tallies:
                total += per_tally
                if voted == position:
                    reachable += per_tally
    return reachable, total


def main():
    print(f"{'設定':<44} {'人数':>4} {'提案者最後':>8} {'到達可能':>8} {'走査':>6} {'割合':>7}")
    for path in sys.argv[1:]:
        config = json.load(open(path, encoding="utf-8"))
        reachable, total = reachability(config)
        name = os.path.basename(path)
        last = "あり" if config.get("proposer_votes_last") else "なし"
        if total is None:
            print(f"{name:<44} {config['num_agents']:>4} {last:>8}   （票数観測なし: 走査は全局面が到達可能）")
            continue
        print(f"{name:<44} {config['num_agents']:>4} {last:>8} "
              f"{reachable:>8} {total:>6} {reachable / total:>7.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
