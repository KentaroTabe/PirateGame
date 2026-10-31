"""投票者の「値段」——賛成に転じる最小の宝石数——を測る。

感応度は「取り分に反応するしきい値があるか」を測るが、**その高さは測らない**
（docs/methodology.md 3.5節）。撤回した訂正2 の機構「被支配者が安く売る」は
値段の水準についての主張なので、感応度では確かめられない。本ツールはそこを測る。

各局面（生存集合 × 提案者 × 投票者 × 残りの配り方）で、投票者の取り分を 0 から
宝石数まで上げていき、**初めて賛成に転じる個数**を値段とする。
これを、同じ局面での均衡の値段と比べる。

    均衡の値段（厳しい側）= 否決時の継続価値を真に上回る最小の整数（無差別なら反対）
    均衡の値段（緩い側）  = 継続価値以上の最小の整数（無差別なら賛成）

宝石が整数なので、継続価値がちょうど整数の局面では2つの規則が1個ずれる。
その局面で学習された投票がどちらに転ぶかも集計する（docs/discussion.md 3.1節の検証）。

使い方:
    python -m tools.vote_price <設定ファイル> <モデル>
"""
import itertools
import json
import sys

import numpy as np
import torch
from tianshou.data import Batch

from env import PirateGemEnv
from eval import load_policy_manager
from pretrain import _build_observation
from solver import RandomOrderSolver, _price
from tools.equilibrium_distance import _vote_tally_for

BATCH_ROWS = 4096
# 継続価値が「ちょうど整数」かを判定する許容誤差（期待値の足し算の誤差を吸収する）
INTEGER_TOLERANCE = 1e-9


def _build_proposal(env, alive, proposer, voter, gems, spread):
    """投票者に gems を渡し、残りを提案者が独占するか他の生存者に散らす。"""
    dist = [0] * env.n_agents
    dist[voter] = gems
    rest = env.total_gems - gems
    others = [i for i in sorted(alive) if i not in (voter, proposer)]
    if not spread or not others:
        dist[proposer] += rest
        return tuple(dist)
    for k in range(rest):
        dist[others[k % len(others)]] += 1
    return tuple(dist)


def measure(config, model_path):
    """値段の分布と、均衡の値段との差を返す。"""
    env = PirateGemEnv(config)
    manager = load_policy_manager(config, model_path)
    policies = manager.policies
    agents = env.possible_agents
    solver = RandomOrderSolver(
        env.n_agents, env.total_gems, env.L, list(env.agent_weights),
        excess_vote_penalty=env.excess_vote_penalty,
    )

    vote_mask = np.zeros(env.TOTAL_ACTIONS, dtype=bool)
    vote_mask[env.ACTION_YES] = True
    vote_mask[env.ACTION_NO] = True

    rows = {a: [] for a in agents}
    keys = {a: [] for a in agents}
    for size in range(2, env.n_agents + 1):
        for subset in itertools.combinations(range(env.n_agents), size):
            alive = frozenset(subset)
            for proposer in sorted(alive):
                cont = solver.value(alive - {proposer})
                for voter in sorted(alive):
                    if voter == proposer:
                        continue
                    agent = agents[voter]
                    tally = _vote_tally_for(env, alive, proposer, voter)
                    for spread in (False, True):
                        for gems in range(env.total_gems + 1):
                            proposal = _build_proposal(
                                env, alive, proposer, voter, gems, spread)
                            rows[agent].append(_build_observation(
                                env, alive, proposer, proposal, vote_tally=tally))
                            keys[agent].append(
                                (alive, proposer, voter, spread, gems, float(cont[voter])))

    records = []
    for agent in agents:
        if not rows[agent]:
            continue
        actions = []
        for start in range(0, len(rows[agent]), BATCH_ROWS):
            obs = np.stack(rows[agent][start:start + BATCH_ROWS])
            mask = np.repeat(vote_mask[None, :], len(obs), axis=0)
            batch = Batch(obs=Batch(obs=obs, mask=mask), info={})
            with torch.no_grad():
                actions.extend(int(a) for a in policies[agent](batch).act)

        # 同じ (生存集合, 提案者, 投票者, 配り方) の中で、賛成に転じる最小の取り分を探す
        current = None
        price = None
        for (alive, proposer, voter, spread, gems, cont), action in zip(keys[agent], actions):
            context = (alive, proposer, voter, spread)
            if context != current:
                if current is not None:
                    records.append((current, price, current_cont))
                current, price, current_cont = context, None, cont
            if price is None and action == env.ACTION_YES:
                price = gems
        if current is not None:
            records.append((current, price, current_cont))

    gaps_strict, gaps_cont, prices = [], [], []
    never = always = 0
    tie_total = tie_accept = 0
    for (_, _, _, _), price, cont in records:
        strict_price = _price(cont, False)
        if price is None:
            never += 1
            continue
        prices.append(price)
        if price == 0:
            always += 1
        gaps_strict.append(price - strict_price)
        gaps_cont.append(price - cont)
        # 継続価値がちょうど整数の局面では、2つの規則が1個ずれる
        if abs(cont - round(cont)) < INTEGER_TOLERANCE and 0 <= round(cont) <= env.total_gems:
            tie_total += 1
            if price <= round(cont):
                tie_accept += 1

    n = len(records)
    return {
        "mean_price": float(np.mean(prices)) if prices else float("nan"),
        "gap_strict": float(np.mean(gaps_strict)) if gaps_strict else float("nan"),
        "gap_cont": float(np.mean(gaps_cont)) if gaps_cont else float("nan"),
        "share_never": never / n if n else float("nan"),
        "share_always": always / n if n else float("nan"),
        "tie_accept_share": tie_accept / tie_total if tie_total else float("nan"),
        "n_contexts": n,
        "n_tie_contexts": tie_total,
    }


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    with open(sys.argv[1], encoding="utf-8") as f:
        config = json.load(f)
    r = measure(config, sys.argv[2])
    print(f"平均の値段: {r['mean_price']:.3f} 個")
    print(f"均衡（厳しい側）との差: {r['gap_strict']:+.3f} 個")
    print(f"継続価値との差: {r['gap_cont']:+.3f} 個")
    print(f"何個でも賛成しない局面: {r['share_never']:.1%} / 0個で賛成する局面: {r['share_always']:.1%}")
    print(f"継続価値ちょうどで賛成する割合: {r['tie_accept_share']:.1%}"
          f"（該当 {r['n_tie_contexts']} 局面）")
    print(f"走査した局面: {r['n_contexts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
