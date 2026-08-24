"""保存済みモデルの投票方策を直接調べる（学習は行わない）。

第15ラウンドで、投票が「宝石をもらったかどうか」と対応していないことが分かった。
本ツールは、**自分の取り分だけを 0〜total_gems と変えて**投票を読み出し、
投票が自分の取り分に反応しているかを調べる。

反応しなければ、投票の Q 値が報酬信号で決まっていない（学習が届いていない）
ことになる。

使い方:
    python -m tools.probe_votes <設定ファイル> <モデル> [提案者記号]
"""
import json
import sys

import numpy as np
import torch
from tianshou.data import Batch

from env import PirateGemEnv
from eval import load_policy_manager
from pretrain import _build_observation


def probe(config, model_path, proposer="A"):
    env = PirateGemEnv(config)
    manager = load_policy_manager(config, model_path)
    policies = manager.policies
    agents = env.possible_agents
    p_idx = ord(proposer) - ord("A")

    alive = set(range(env.n_agents))
    n_dist = len(env.DISTRIBUTIONS)
    required = int(np.ceil(env.n_agents / 2))

    vote_mask = np.zeros(n_dist + 2, dtype=bool)
    vote_mask[env.ACTION_YES] = True
    vote_mask[env.ACTION_NO] = True

    print(f"設定: L={env.L}, 重み={list(env.agent_weights)}, "
          f"必要票={required}, 票数観測={env.observe_vote_tally}")
    print(f"モデル: {model_path}")
    print(f"提案者: {agents[p_idx]}\n")
    print("各投票者が『自分の取り分』ごとにどう投票するか")
    print("（局面: 全員生存・あと1票で可決という状況＝自分が決定票）\n")

    header = "投票者 | " + " | ".join(f"{g}個" for g in range(env.total_gems + 1))
    print(header)
    print("-" * len(header))

    insensitive = []
    for v_idx, agent in enumerate(agents):
        if v_idx == p_idx:
            continue
        row = []
        for gems in range(env.total_gems + 1):
            dist = [0] * env.n_agents
            dist[v_idx] = gems
            dist[p_idx] = env.total_gems - gems  # 残りは提案者が取る
            tally = [float(required - 1), float(env.n_agents - 2)]
            obs = _build_observation(env, alive, p_idx, tuple(dist), vote_tally=tally)
            batch = Batch(obs=Batch(obs=obs[None, :], mask=vote_mask[None, :]), info={})
            with torch.no_grad():
                action = int(policies[agent](batch).act[0])
            row.append("賛成" if action == env.ACTION_YES else "反対")
        print(f"{agent[-1]:>5}  | " + " | ".join(f"{v:>3}" for v in row))
        if len(set(row)) == 1:
            insensitive.append((agent[-1], row[0]))

    print()
    if insensitive:
        names = ", ".join(f"{a}({v}で固定)" for a, v in insensitive)
        print(f"取り分に一切反応しない投票者: {len(insensitive)}/{env.n_agents - 1} — {names}")
    else:
        print("すべての投票者が取り分に反応している。")
    return insensitive


def probe_tally(config, model_path, proposer="A"):
    """自分の取り分を0個に固定し、票の途中経過だけを変えて投票を読み出す。"""
    env = PirateGemEnv(config)
    manager = load_policy_manager(config, model_path)
    policies = manager.policies
    agents = env.possible_agents
    p_idx = ord(proposer) - ord("A")

    alive = set(range(env.n_agents))
    n_dist = len(env.DISTRIBUTIONS)
    required = int(np.ceil(env.n_agents / 2))
    vote_mask = np.zeros(n_dist + 2, dtype=bool)
    vote_mask[env.ACTION_YES] = True
    vote_mask[env.ACTION_NO] = True

    # 提案者が全部取る案（投票者の取り分は0で固定）
    dist = [0] * env.n_agents
    dist[p_idx] = env.total_gems

    print("\n各投票者が『これまでの賛成数』ごとにどう投票するか")
    print(f"（自分の取り分は0個で固定、投票済み人数は {env.n_agents - 2} で固定）\n")
    header = "投票者 | " + " | ".join(f"賛成{k}" for k in range(env.n_agents - 1))
    print(header)
    print("-" * len(header))

    sensitive = []
    for v_idx, agent in enumerate(agents):
        if v_idx == p_idx:
            continue
        row = []
        for yes in range(env.n_agents - 1):
            tally = [float(yes), float(env.n_agents - 2)]
            obs = _build_observation(env, alive, p_idx, tuple(dist), vote_tally=tally)
            batch = Batch(obs=Batch(obs=obs[None, :], mask=vote_mask[None, :]), info={})
            with torch.no_grad():
                action = int(policies[agent](batch).act[0])
            row.append("賛成" if action == env.ACTION_YES else "反対")
        print(f"{agent[-1]:>5}  | " + " | ".join(f"{v:>3}" for v in row))
        if len(set(row)) > 1:
            sensitive.append(agent[-1])

    print()
    print(f"票の途中経過に反応する投票者: {len(sensitive)}/{env.n_agents - 1}"
          + (f" — {', '.join(sensitive)}" if sensitive else ""))
    return sensitive


def simulate_sequence(config, model_path, proposer="A"):
    """提案者の貪欲な提案に対する投票列を、実際の順序どおりに再現する。"""
    env = PirateGemEnv(config)
    manager = load_policy_manager(config, model_path)
    policies = manager.policies
    agents = env.possible_agents
    p_idx = ord(proposer) - ord("A")

    alive = set(range(env.n_agents))
    n_dist = len(env.DISTRIBUTIONS)
    required = int(np.ceil(env.n_agents / 2))

    propose_mask = np.zeros(n_dist + 2, dtype=bool)
    propose_mask[:n_dist] = True
    obs = _build_observation(env, alive, p_idx, (0,) * env.n_agents)
    batch = Batch(obs=Batch(obs=obs[None, :], mask=propose_mask[None, :]), info={})
    with torch.no_grad():
        action = int(policies[agents[p_idx]](batch).act[0])
    proposal = env.DISTRIBUTIONS[action]

    order = [a for a in agents if a != agents[p_idx]] if env.proposer_votes_last \
        else list(agents)
    if env.proposer_votes_last:
        order = order + [agents[p_idx]]

    vote_mask = np.zeros(n_dist + 2, dtype=bool)
    vote_mask[env.ACTION_YES] = True
    vote_mask[env.ACTION_NO] = True

    print(f"\n提案 {list(proposal)} に対する投票列（貪欲方策で再現）\n")
    yes = 0
    voted = 0
    others_yes = 0
    for agent in order:
        tally = [float(yes), float(voted)]
        idx = agents.index(agent)
        o = _build_observation(env, alive, p_idx, proposal, vote_tally=tally)
        b = Batch(obs=Batch(obs=o[None, :], mask=vote_mask[None, :]), info={})
        with torch.no_grad():
            a = int(policies[agent](b).act[0])
        is_yes = (a == env.ACTION_YES)
        mark = "👍賛成" if is_yes else "👎反対"
        note = "（提案者）" if idx == p_idx else f"取り分{proposal[idx]}個"
        print(f"  賛成{yes}/投票{voted} → {agent[-1]} {mark}  {note}")
        if is_yes:
            yes += 1
            if idx != p_idx:
                others_yes += 1
        voted += 1

    print(f"\n最終: 賛成 {yes} / 必要 {required} → "
          f"{'可決' if yes >= required else '否決'}、超過票 {max(0, yes - required)}")
    print(f"**提案者以外の賛成: {others_yes} 票**"
          f"（必要票 {required} に{'届いている → 提案者は反対できる' if others_yes >= required else '届かない → 提案者は賛成せざるを得ない'}）")
    return others_yes


def main(config_path, model_path, proposer="A"):
    with open(config_path) as f:
        config = json.load(f)
    probe(config, model_path, proposer)
    probe_tally(config, model_path, proposer)
    simulate_sequence(config, model_path, proposer)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "A")
