"""固定順バックワードインダクションの一般解を各エージェントの Q ネットワークに埋め込む事前学習。

ランダム順ゲームで訪れうる全状態（任意の生存集合 × 任意の提案者 × 任意の分配案）を列挙し、
一般解から導いた Q 値を教師として回帰する:
    - 提案行動: 均衡投票の下で可決するなら自分の取り分、否決なら -L
    - 投票行動: 賛成 = 可決時の自分の取り分、反対 = 否決時の継続価値（ピボタル仮定）

この重みを初期値として、agent_weights に基づくランダム順環境で DQN 学習を行うことで、
「合理的な一般解」を出発点に政治的駆け引きの学習・創発を観察できる。
"""

import itertools

import numpy as np
import torch

from env import PirateGemEnv
from network import Net
from solver import FixedOrderSolver


def _build_observation(env, alive_set, proposer, proposal, vote_tally=None):
    """env.observe() と同一フォーマットの観測ベクトルを構築する。

    env.observe_vote_tally が有効な場合は末尾に投票の途中経過を足す。
    vote_tally 未指定なら「まだ誰も投票していない」= [0, 0] とみなす。
    """
    alive_flag = [1.0 if i in alive_set else 0.0 for i in range(env.n_agents)]
    weights = list(env.agent_weights)
    proposer_onehot = [1.0 if i == proposer else 0.0 for i in range(env.n_agents)]
    features = alive_flag + weights + proposer_onehot + list(proposal)

    if env.observe_vote_tally:
        tally = [0.0] * env.VOTE_TALLY_DIM if vote_tally is None else list(vote_tally)
        if len(tally) != env.VOTE_TALLY_DIM:
            raise ValueError(
                f"vote_tally の長さ {len(tally)} が VOTE_TALLY_DIM {env.VOTE_TALLY_DIM} と一致しません"
            )
        features = features + tally

    return np.array(features, dtype=np.float32)


def _valid_dist_indices(env, alive_set):
    """生存者以外に宝石を配らない分配案のインデックス一覧。"""
    return [
        i for i, dist in enumerate(env.DISTRIBUTIONS)
        if all(amount == 0 or j in alive_set for j, amount in enumerate(dist))
    ]


def build_dataset(env, solver, agent_idx):
    """agent_idx 視点の (観測, 行動マスク, 目標Q) を全状態にわたり列挙する。"""
    obs_list, mask_list, target_list = [], [], []
    zero_proposal = (0,) * env.n_agents

    all_indices = range(env.n_agents)
    for size in range(1, env.n_agents + 1):
        for subset in itertools.combinations(all_indices, size):
            alive = frozenset(subset)
            if agent_idx not in alive:
                continue
            for proposer in alive:
                dist_indices = _valid_dist_indices(env, alive)

                # 提案フェーズ（自分が提案者のときのみ意思決定がある）
                if proposer == agent_idx:
                    mask = np.zeros(env.TOTAL_ACTIONS, dtype=bool)
                    target = np.zeros(env.TOTAL_ACTIONS, dtype=np.float32)
                    for di in dist_indices:
                        mask[di] = True
                        target[di] = solver.proposal_q(alive, proposer, env.DISTRIBUTIONS[di])
                    obs_list.append(_build_observation(env, alive, proposer, zero_proposal))
                    mask_list.append(mask)
                    target_list.append(target)

                # 投票フェーズ（提示されうる全分配案について賛成/反対の Q 値）
                for di in dist_indices:
                    proposal = env.DISTRIBUTIONS[di]
                    q_yes, q_no = solver.vote_q(alive, proposer, proposal, agent_idx)
                    mask = np.zeros(env.TOTAL_ACTIONS, dtype=bool)
                    mask[env.ACTION_YES] = True
                    mask[env.ACTION_NO] = True
                    target = np.zeros(env.TOTAL_ACTIONS, dtype=np.float32)
                    target[env.ACTION_YES] = q_yes
                    target[env.ACTION_NO] = q_no
                    obs_list.append(_build_observation(env, alive, proposer, proposal))
                    mask_list.append(mask)
                    target_list.append(target)

    return (
        np.stack(obs_list),
        np.stack(mask_list),
        np.stack(target_list),
    )


def _greedy_match_rate(net, obs, mask, target, device):
    """貪欲方策が教師 Q 値のもとで最適行動になっている割合（同点最適も正解扱い）。"""
    with torch.inference_mode():
        logits, _ = net({"observation": obs, "action_mask": mask.astype(np.int8)})
        logits = logits.cpu().numpy()
    logits[~mask] = -np.inf
    greedy = logits.argmax(axis=1)

    masked_target = np.where(mask, target, -np.inf)
    best_q = masked_target.max(axis=1)
    chosen_q = target[np.arange(len(target)), greedy]
    return float(np.mean(np.isclose(chosen_q, best_q)))


def pretrain_agent(env, solver, agent_idx, hidden_sizes, device,
                   epochs=200, lr=1e-3, batch_size=512, seed=0):
    """1エージェント分のネットワークを一般解の Q 値に回帰させる。"""
    torch.manual_seed(seed + agent_idx)
    obs, mask, target = build_dataset(env, solver, agent_idx)

    agent = env.possible_agents[agent_idx]
    obs_shape = env.observation_spaces[agent]["observation"].shape
    net = Net(obs_shape, env.TOTAL_ACTIONS, hidden_sizes=hidden_sizes, device=device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    obs_t = torch.as_tensor(obs, device=device)
    mask_t = torch.as_tensor(mask, device=device)
    target_t = torch.as_tensor(target, device=device)
    n = len(obs_t)

    loss_value = float("nan")
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            logits = net.model(obs_t[idx])
            diff = (logits - target_t[idx]) * mask_t[idx]
            loss = diff.pow(2).sum() / mask_t[idx].sum()
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_value = loss.item()

    match_rate = _greedy_match_rate(net, obs, mask, target, device)
    return net, {"loss": loss_value, "match_rate": match_rate, "n_samples": n}


def pretrain_agents(config, device="cpu", hidden_sizes=(128, 128),
                    epochs=200, lr=1e-3, batch_size=512, seed=0, verbose=True):
    """全エージェントを事前学習し、{agent名: state_dict} と統計を返す。"""
    env = PirateGemEnv(config)
    # 固定順一般解は投票の途中経過を状態に持たないため、observe_vote_tally が
    # 有効だと投票フェーズの目標 Q が定義できない（誤った目標を埋め込むより落とす）。
    if env.observe_vote_tally:
        raise ValueError(
            "observe_vote_tally が有効なときは事前学習を使えません"
            "（固定順一般解が投票の途中経過を扱わないため）。"
            "pretrain を false にしてください。"
        )
    solver = FixedOrderSolver(env.n_agents, env.total_gems, env.L, env.excess_vote_penalty)

    state_dicts, stats = {}, {}
    for idx, agent in enumerate(env.possible_agents):
        net, agent_stats = pretrain_agent(
            env, solver, idx, list(hidden_sizes), device,
            epochs=epochs, lr=lr, batch_size=batch_size, seed=seed,
        )
        state_dicts[agent] = net.state_dict()
        stats[agent] = agent_stats
        if verbose:
            print(
                f"[事前学習] {agent}: samples={agent_stats['n_samples']}, "
                f"loss={agent_stats['loss']:.4f}, "
                f"一般解一致率={agent_stats['match_rate']:.1%}"
            )
    return state_dicts, stats
