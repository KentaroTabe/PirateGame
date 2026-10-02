"""学習後の方策が、本実験のゲームの厳密解（均衡）からどれだけ離れているかを測る。

`solver.RandomOrderSolver` が返す部分ゲーム完全均衡（ランダム順・否決で脱落）を基準に、
学習済みの方策を3つの量で比べる。

1. **提案の距離** `proposal_l1`
   全生存集合 × 自分が提案者の局面で、学習方策が選ぶ分配案と、
   均衡の最適提案（誰を買うかだけが異なる複数の候補）のうち最も近いものとの L1 距離。
   0 なら均衡どおり。最大は宝石数の2倍。
2. **自分の取り分の差** `self_gap`
   学習方策が自分に残す個数 − 均衡で自分に残る個数（符号つき）。
   正なら均衡より強欲、負なら均衡より譲っている。支払い総額は均衡で一意なので、
   この差は買収先の選び方によらず定まる。
3. **投票の一致率** `vote_match`
   全生存集合 × 提案者 × 提示されうる全分配案 × 各投票者について、
   学習方策の投票が均衡の投票と一致する割合。
   宝石が整数なので「無差別なら反対／賛成」の2つの規則で均衡投票が食い違うことがある。
   両方の規則で測り、どちらかに一致すれば一致とみなす `either` も返す。
4. **連合の大きさの乖離** `coalition_gap`
   学習方策の提案で正の宝石を受け取る投票者の人数 − 均衡で買収される人数。
   正なら必要より多く買っている（超過連合）。

票数観測ありの設定では、均衡の側に票の途中経過という概念がない（ピボタル仮定）。
そこで観測に入れる票数は、**その投票者が実際に投票する位置**（`env._build_voting_order()`）と、
**ピボタルに最も近い賛成数**（必要票数 − 1 を超えない最大）を使う。

使い方:
    python -m tools.equilibrium_distance <設定ファイル> <モデル>
"""
import itertools
import json
import sys
from math import ceil

import numpy as np
import torch
from tianshou.data import Batch

from env import PirateGemEnv
from eval import load_policy_manager
from pretrain import _build_observation, _valid_dist_indices
from solver import RandomOrderSolver

BATCH_ROWS = 4096


def _solver_for(config, accept_when_indifferent):
    env = PirateGemEnv(config)
    return RandomOrderSolver(
        env.n_agents, env.total_gems, env.L, list(env.agent_weights),
        excess_vote_penalty=env.excess_vote_penalty,
        accept_when_indifferent=accept_when_indifferent,
    )


def _vote_tally_for(env, alive_set, proposer, voter):
    """票数観測ありの設定で使う [賛成数, 投票済み人数]。観測しない設定では None。"""
    if not env.observe_vote_tally:
        return None
    members = [a for a in range(env.n_agents) if a in alive_set]
    order = [i for i in members]
    if env.proposer_votes_last and proposer in order:
        order.remove(proposer)
        order.append(proposer)
    voted = order.index(voter)
    required = ceil(len(alive_set) / 2)
    yes = min(required - 1, voted)
    return [float(yes), float(voted)]


def _argmax_actions(policy, rows, masks):
    """まとめて貪欲行動を返す。"""
    out = []
    for start in range(0, len(rows), BATCH_ROWS):
        obs = np.stack(rows[start:start + BATCH_ROWS])
        mask = np.stack(masks[start:start + BATCH_ROWS])
        batch = Batch(obs=Batch(obs=obs, mask=mask), info={})
        with torch.no_grad():
            out.extend(int(a) for a in policy(batch).act)
    return out


def measure(config, model_path):
    """提案の距離・自分の取り分の差・投票の一致率・連合の大きさの乖離を返す。"""
    env = PirateGemEnv(config)
    manager = load_policy_manager(config, model_path)
    policies = manager.policies
    agents = env.possible_agents
    n, gems = env.n_agents, env.total_gems
    strict = _solver_for(config, False)
    lenient = _solver_for(config, True)

    subsets = [
        frozenset(s)
        for size in range(2, n + 1)
        for s in itertools.combinations(range(n), size)
    ]

    # ---- 提案の距離 ----
    prop_rows, prop_masks, prop_keys = {a: [] for a in agents}, {a: [] for a in agents}, {a: [] for a in agents}
    zero = (0,) * n
    for alive in subsets:
        valid = _valid_dist_indices(env, alive)
        mask = np.zeros(env.TOTAL_ACTIONS, dtype=bool)
        mask[valid] = True
        for proposer in sorted(alive):
            agent = agents[proposer]
            prop_rows[agent].append(_build_observation(env, alive, proposer, zero))
            prop_masks[agent].append(mask.copy())
            prop_keys[agent].append((alive, proposer))

    l1_all, self_gap_all, coalition_gap_all = [], [], []
    per_agent = {}
    for agent in agents:
        if not prop_rows[agent]:
            continue
        actions = _argmax_actions(policies[agent], prop_rows[agent], prop_masks[agent])
        l1s, gaps, coal = [], [], []
        for (alive, proposer), action in zip(prop_keys[agent], actions):
            if action >= len(env.DISTRIBUTIONS):
                continue  # 有効な分配案が選ばれなかった局面は数えない
            learned = env.DISTRIBUTIONS[action]
            equilibria, affordable = strict.equilibrium_proposals(alive, proposer)
            if not affordable:
                continue
            l1s.append(min(sum(abs(learned[i] - eq[i]) for i in range(n)) for eq in equilibria))
            gaps.append(learned[proposer] - equilibria[0][proposer])
            learned_paid = sum(1 for i in alive if i != proposer and learned[i] > 0)
            equil_paid = sum(1 for i in alive if i != proposer and equilibria[0][i] > 0)
            coal.append(learned_paid - equil_paid)
        per_agent[agent[-1]] = {
            "proposal_l1": float(np.mean(l1s)) if l1s else float("nan"),
            "self_gap": float(np.mean(gaps)) if gaps else float("nan"),
        }
        l1_all.extend(l1s)
        self_gap_all.extend(gaps)
        coalition_gap_all.extend(coal)

    # ---- 投票の一致率 ----
    vote_rows, vote_masks, vote_truth = ({a: [] for a in agents}, {a: [] for a in agents},
                                         {a: [] for a in agents})
    vote_mask = np.zeros(env.TOTAL_ACTIONS, dtype=bool)
    vote_mask[env.ACTION_YES] = True
    vote_mask[env.ACTION_NO] = True
    for alive in subsets:
        for proposer in sorted(alive):
            for di in _valid_dist_indices(env, alive):
                proposal = env.DISTRIBUTIONS[di]
                s_votes = strict.equilibrium_votes(alive, proposer, proposal)
                l_votes = lenient.equilibrium_votes(alive, proposer, proposal)
                for voter in sorted(alive):
                    agent = agents[voter]
                    tally = _vote_tally_for(env, alive, proposer, voter)
                    vote_rows[agent].append(
                        _build_observation(env, alive, proposer, proposal, vote_tally=tally))
                    vote_masks[agent].append(vote_mask.copy())
                    vote_truth[agent].append((s_votes[voter], l_votes[voter]))

    strict_hits = lenient_hits = either_hits = total_votes = 0
    for agent in agents:
        if not vote_rows[agent]:
            continue
        actions = _argmax_actions(policies[agent], vote_rows[agent], vote_masks[agent])
        for (s_vote, l_vote), action in zip(vote_truth[agent], actions):
            learned_yes = action == env.ACTION_YES
            strict_hits += int(learned_yes == s_vote)
            lenient_hits += int(learned_yes == l_vote)
            either_hits += int(learned_yes in (s_vote, l_vote))
            total_votes += 1

    return {
        "proposal_l1": float(np.mean(l1_all)) if l1_all else float("nan"),
        "self_gap": float(np.mean(self_gap_all)) if self_gap_all else float("nan"),
        "coalition_gap": float(np.mean(coalition_gap_all)) if coalition_gap_all else float("nan"),
        "vote_match_strict": strict_hits / total_votes if total_votes else float("nan"),
        "vote_match_lenient": lenient_hits / total_votes if total_votes else float("nan"),
        "vote_match_either": either_hits / total_votes if total_votes else float("nan"),
        "n_proposal_states": len(l1_all),
        "n_vote_states": total_votes,
        "max_proposal_l1": 2 * gems,
        "per_agent": per_agent,
    }


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    config = json.load(open(sys.argv[1], encoding="utf-8"))
    result = measure(config, sys.argv[2])
    print(f"提案の L1 距離: {result['proposal_l1']:.3f}（最大 {result['max_proposal_l1']}）")
    print(f"自分の取り分の差: {result['self_gap']:+.3f}")
    print(f"連合の大きさの乖離: {result['coalition_gap']:+.3f}")
    print(f"投票の一致率: 反対寄り {result['vote_match_strict']:.1%}"
          f" / 賛成寄り {result['vote_match_lenient']:.1%}"
          f" / どちらか {result['vote_match_either']:.1%}")
    print(f"走査した局面: 提案 {result['n_proposal_states']} / 投票 {result['n_vote_states']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
