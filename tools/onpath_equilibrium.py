"""学習後の方策どうしが実際に通る局面だけで、均衡との差を測る。

`tools/equilibrium_distance.py` は全局面を等しい重みで平均する。そのため、
対局ではほとんど通らない小さな部分ゲームが値を左右する。
本ツールは、学習済みの貪欲方策どうしを対局させたときの**到達確率**で重みをつける。

提案者の選出だけが確率的なので、対局を標本抽出せず、
到達しうる局面をすべてたどって厳密な確率を求める（`walk()`）。
`walk()` と `summarize()` はネットワークに依存せず、関数を渡してテストできる。

使い方:
    python -m tools.onpath_equilibrium <設定ファイル> <モデル>
"""
import json
import sys
from math import ceil

import numpy as np
import torch
from tianshou.data import Batch

from env import PirateGemEnv
from eval import load_policy_manager
from pretrain import _build_observation, _valid_dist_indices
from solver import NoEliminationSolver, RandomOrderSolver

RECORD_FIELDS = ["onpath_match_either", "onpath_l1", "first_reject", "expected_rejections",
                 "first_keep"]


def voting_order(alive, proposer, proposer_votes_last):
    """env._build_voting_order() と同じ規則の投票順。"""
    order = sorted(alive)
    if proposer_votes_last:
        order.remove(proposer)
        order.append(proposer)
    return order


def walk(n_agents, weights, propose, vote, proposer_votes_last=False):
    """到達しうる全局面を、到達確率つきで列挙する。

    Args:
        propose: (alive, proposer, stale_tally) -> 分配案のタプル。
            stale_tally は前の提案の票の途中経過（env は次の投票まで前の値を見せる）。
        vote: (alive, proposer, voter, proposal, tally) -> True なら賛成。

    Returns:
        訪問の一覧。各要素は alive, proposer, prob, proposal, votes, passed を持つ。
    """
    reach = {(frozenset(range(n_agents)), (0.0, 0.0)): 1.0}
    visits = []
    for size in range(n_agents, 1, -1):
        keys = [k for k in reach if len(k[0]) == size]
        for alive, stale in keys:
            prob = reach[(alive, stale)]
            total_weight = sum(weights[i] for i in alive)
            for proposer in sorted(alive):
                p_proposer = prob * weights[proposer] / total_weight
                if p_proposer == 0.0:
                    continue
                proposal = propose(alive, proposer, stale)
                yes = voted = 0
                votes = {}
                for voter in voting_order(alive, proposer, proposer_votes_last):
                    votes[voter] = vote(alive, proposer, voter, proposal,
                                        (float(yes), float(voted)))
                    yes += int(votes[voter])
                    voted += 1
                passed = yes >= ceil(size / 2)
                visits.append({"alive": alive, "proposer": proposer, "prob": p_proposer,
                               "proposal": proposal, "votes": votes, "passed": passed,
                               "first": size == n_agents})
                if not passed:
                    key = (alive - {proposer}, (float(yes), float(voted)))
                    reach[key] = reach.get(key, 0.0) + p_proposer
    return visits


def walk_rounds(n_agents, weights, max_rounds, propose, vote):
    """脱落なしの環境で到達しうる全局面を列挙する。

    局面の鍵は回の番号で、`alive` の欄に入れる（solver の引数と同じ扱いにするため）。
    propose は (round_index, proposer)、vote は (round_index, proposer, voter, proposal)。
    """
    visits = []
    total_weight = sum(weights)
    reach = 1.0
    required = ceil(n_agents / 2)
    for round_index in range(max_rounds):
        rejected = 0.0
        for proposer in range(n_agents):
            p_proposer = reach * weights[proposer] / total_weight
            if p_proposer == 0.0:
                continue
            proposal = propose(round_index, proposer)
            votes = {voter: vote(round_index, proposer, voter, proposal)
                     for voter in range(n_agents)}
            passed = sum(votes.values()) >= required
            visits.append({"alive": round_index, "proposer": proposer, "prob": p_proposer,
                           "proposal": proposal, "votes": votes, "passed": passed,
                           "first": round_index == 0})
            if not passed:
                rejected += p_proposer
        reach = rejected
        if reach == 0.0:
            break
    return visits


def summarize(visits, strict, lenient):
    """訪問の一覧から、到達確率で重みづけした指標を返す。"""
    total = sum(v["prob"] for v in visits)
    l1 = gap_strict = gap_lenient = 0.0
    match_strict = match_lenient = match_either = vote_weight = 0.0
    first = rejected = expected_rejections = 0.0
    first_keep = first_keep_strict = first_keep_lenient = 0.0
    n_agents = strict.n_agents

    if any(not strict.equilibrium_proposals(v["alive"], v["proposer"])[1]
           or not lenient.equilibrium_proposals(v["alive"], v["proposer"])[1] for v in visits):
        raise ValueError("可決できる均衡提案がない局面に到達しました（この指標の対象外です）")

    for v in visits:
        alive, proposer, prob, proposal = v["alive"], v["proposer"], v["prob"], v["proposal"]
        strict_set, _ = strict.equilibrium_proposals(alive, proposer)
        lenient_set, _ = lenient.equilibrium_proposals(alive, proposer)
        l1 += prob * min(sum(abs(proposal[i] - e[i]) for i in range(n_agents)) for e in strict_set)
        gap_strict += prob * (proposal[proposer] - strict_set[0][proposer])
        gap_lenient += prob * (proposal[proposer] - lenient_set[0][proposer])

        strict_votes = strict.equilibrium_votes(alive, proposer, proposal)
        lenient_votes = lenient.equilibrium_votes(alive, proposer, proposal)
        for voter, learned in v["votes"].items():
            vote_weight += prob
            match_strict += prob * (learned == strict_votes[voter])
            match_lenient += prob * (learned == lenient_votes[voter])
            match_either += prob * (learned in (strict_votes[voter], lenient_votes[voter]))

        if not v["passed"]:
            expected_rejections += prob
        if v["first"]:
            first += prob
            rejected += prob * (not v["passed"])
            first_keep += prob * proposal[proposer]
            first_keep_strict += prob * strict_set[0][proposer]
            first_keep_lenient += prob * lenient_set[0][proposer]

    return {
        "onpath_l1": l1 / total,
        "onpath_gap_strict": gap_strict / total,
        "onpath_gap_lenient": gap_lenient / total,
        "onpath_match_strict": match_strict / vote_weight,
        "onpath_match_lenient": match_lenient / vote_weight,
        "onpath_match_either": match_either / vote_weight,
        "first_reject": rejected / first,
        "expected_rejections": expected_rejections,
        "first_keep": first_keep / first,
        "first_keep_strict": first_keep_strict / first,
        "first_keep_lenient": first_keep_lenient / first,
    }


def _greedy(policy, obs, mask):
    batch = Batch(obs=Batch(obs=obs[None, :], mask=mask[None, :]), info={})
    with torch.no_grad():
        return int(policy(batch).act[0])


def make_solvers(env):
    """（無差別なら反対, 無差別なら賛成）の2つの均衡を返す。"""
    weights = [float(w) for w in env.agent_weights]
    if not env.eliminate_on_reject:
        return tuple(NoEliminationSolver(env.n_agents, env.total_gems, env.L, weights,
                                         env.max_rounds, accept_when_indifferent=rule)
                     for rule in (False, True))
    return tuple(RandomOrderSolver(env.n_agents, env.total_gems, env.L, weights,
                                   excess_vote_penalty=env.excess_vote_penalty,
                                   accept_when_indifferent=rule)
                 for rule in (False, True))


def measure_policies(env, policies, solvers):
    """方策の辞書から到達局面での指標を求める。"""
    agents = env.possible_agents
    zero = (0,) * env.n_agents
    vote_mask = np.zeros(env.TOTAL_ACTIONS, dtype=bool)
    vote_mask[env.ACTION_YES] = True
    vote_mask[env.ACTION_NO] = True

    def tally_of(tally):
        return list(tally) if env.observe_vote_tally else None

    def propose(alive, proposer, stale):
        mask = np.zeros(env.TOTAL_ACTIONS, dtype=bool)
        mask[_valid_dist_indices(env, alive)] = True
        obs = _build_observation(env, alive, proposer, zero, vote_tally=tally_of(stale))
        return env.DISTRIBUTIONS[_greedy(policies[agents[proposer]], obs, mask)]

    def vote(alive, proposer, voter, proposal, tally):
        obs = _build_observation(env, alive, proposer, proposal, vote_tally=tally_of(tally))
        return _greedy(policies[agents[voter]], obs, vote_mask) == env.ACTION_YES

    weights = [float(w) for w in env.agent_weights]
    if env.eliminate_on_reject:
        visits = walk(env.n_agents, weights, propose, vote, env.proposer_votes_last)
        return summarize(visits, solvers[0], solvers[1])

    if env.observe_vote_tally:
        raise ValueError("脱落なしの環境では票数観測つきの指標に対応していません")
    everyone = frozenset(range(env.n_agents))
    all_mask = np.zeros(env.TOTAL_ACTIONS, dtype=bool)
    all_mask[:len(env.DISTRIBUTIONS)] = True

    def propose_round(round_index, proposer):
        obs = _build_observation(env, everyone, proposer, zero, round_index=round_index)
        return env.DISTRIBUTIONS[_greedy(policies[agents[proposer]], obs, all_mask)]

    def vote_round(round_index, proposer, voter, proposal):
        obs = _build_observation(env, everyone, proposer, proposal, round_index=round_index)
        return _greedy(policies[agents[voter]], obs, vote_mask) == env.ACTION_YES

    visits = walk_rounds(env.n_agents, weights, env.max_rounds, propose_round, vote_round)
    return summarize(visits, solvers[0], solvers[1])


class OnPathRecorder:
    """学習中の到達局面の指標を CSV に書く。方策を読むだけで学習には影響しない。"""

    def __init__(self, env, path):
        self.env = env
        self.path = path
        self.solvers = make_solvers(env)
        with open(path, "w", encoding="utf-8") as f:
            f.write(",".join(["epoch"] + RECORD_FIELDS) + "\n")

    def record(self, epoch, policies):
        result = measure_policies(self.env, policies, self.solvers)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(",".join([str(epoch)] + [f"{result[k]:.4f}" for k in RECORD_FIELDS]) + "\n")


def measure(config, model_path):
    """保存済みモデルの到達局面での指標を返す。"""
    env = PirateGemEnv(config)
    policies = load_policy_manager(config, model_path).policies
    return measure_policies(env, policies, make_solvers(env))


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    with open(sys.argv[1], encoding="utf-8") as f:
        config = json.load(f)
    for key, value in measure(config, sys.argv[2]).items():
        print(f"{key}: {value:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
