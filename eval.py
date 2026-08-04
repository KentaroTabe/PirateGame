"""学習済みポリシーの評価ロジック。

run_game() が1ゲームを実行し、evaluate() が複数エピソードを走らせて
平均報酬・死亡率・可決までの提案回数などの統計を集計する。
ランダム順（agent_weights による確率的な提案者選出）の環境では
1ゲームだけでは挙動が分からないため、統計評価を基本とする。
"""

import os
import sys

import numpy as np
import torch
from tianshou.data import Batch
from tianshou.env import PettingZooEnv
from tianshou.policy import DQNPolicy, MultiAgentPolicyManager

from env import PirateGemEnv
from network import Net


class Logger(object):
    """標準出力を log/log_n.txt にリダイレクトする（watch() 用）。"""

    def __init__(self):
        self.terminal = sys.stdout
        os.makedirs("log", exist_ok=True)
        n = 1
        while os.path.exists(f"log/log_{n}.txt"):
            n += 1
        self.log_filepath = f"log/log_{n}.txt"
        self.log_file = open(self.log_filepath, "w", encoding="utf-8")

    def write(self, message):
        self.log_file.write(message)

    def flush(self):
        self.log_file.flush()

    def close(self):
        self.log_file.close()
        self.terminal.write(f"\n[INFO] ログを {self.log_filepath} に保存しました。\n")


def load_policy_manager(config, model_path):
    """保存済みモデルから MultiAgentPolicyManager を再構築する。"""
    raw_env = PirateGemEnv(config)
    env = PettingZooEnv(raw_env)
    policies = {}
    for agent in raw_env.possible_agents:
        obs_shape = raw_env.observation_spaces[agent]["observation"].shape
        act_shape = raw_env.action_spaces[agent].n
        net = Net(obs_shape, act_shape, hidden_sizes=[128, 128], device='cpu')
        policies[agent] = DQNPolicy(model=net, optim=torch.optim.Adam(net.parameters()), is_double=True)

    manager = MultiAgentPolicyManager(
        policies=[policies[agent] for agent in raw_env.possible_agents], env=env,
    )
    manager.load_state_dict(torch.load(model_path, map_location='cpu'))
    return manager


def _print_events(env):
    for event in env.pop_events():
        if event["type"] == "vote_result":
            print(f"\n[判定] 賛成: {event['yes']} / 生存者: {event['n_alive']}")
            if event["passed"]:
                print("👉 提案は【可決】されました！")
            else:
                print(f"💀 提案は【否決】されました... {event['proposer']} は海に落とされます。")
        elif event["type"] == "last_survivor":
            print(f"🏆 {event['agent']} が最後の生存者となり、宝石を独占します！")


def run_game(policy_manager=None, config=None, model_path='policy.pth', seed=None, verbose=True):
    """1ゲームを実行し、{'rewards', 'deaths', 'n_proposals'} を返す。"""
    raw_env = PirateGemEnv(config)
    agents = raw_env.possible_agents

    if policy_manager is None:
        try:
            policy_manager = load_policy_manager(config, model_path)
            if verbose:
                print(f"モデル '{model_path}' を正常に読み込みました。")
        except FileNotFoundError:
            print(f"エラー: '{model_path}' が見つかりません。")
            return None

    policy_manager.eval()
    for policy in policy_manager.policies.values():
        policy.set_eps(0.0)

    if verbose:
        print("\n=========================================")
        print("ゲーム開始！")
        print("=========================================")
        print("【環境設定】")
        print(f" - 海賊の人数: {len(agents)}人")
        print(f" - 宝石の総数: {raw_env.total_gems}個")
        print(f" - 命の重さ(ペナルティ L): {raw_env.L}")
        weight_str = ", ".join(f"{a.split('_')[1]}:{w}" for a, w in zip(agents, raw_env.agent_weights))
        print(f" - 権力ウェイト(発言力): [{weight_str}]")
        order_str = "固定順" if raw_env.fixed_order else "権力ウェイトに基づくランダム順"
        print(f" - 提案者の選出: {order_str}")
        print("=========================================\n")

    raw_env.reset(seed=seed)

    final_rewards = {a: 0.0 for a in agents}
    n_proposals = 0

    with torch.inference_mode():
        for agent in raw_env.agent_iter():
            obs, reward, termination, truncation, info = raw_env.last()

            if termination or truncation:
                final_rewards[agent] = reward
                raw_env.step(None)
                continue

            batch = Batch(obs=Batch([obs]), info=Batch([info]))
            action = policy_manager.policies[agent](batch).act[0]

            if verbose:
                if raw_env.phase == "PROPOSE":
                    print(f"💬 => {agent} は提案行動 [ {list(raw_env.DISTRIBUTIONS[action])} ] を選択しました！")
                else:
                    vote_str = "👍 賛成 (YES)" if action == raw_env.ACTION_YES else "👎 反対 (NO)"
                    print(f"💬 => {agent} は {vote_str} を選択しました！")

            if raw_env.phase == "PROPOSE":
                n_proposals += 1

            raw_env.step(action)

            if verbose:
                _print_events(raw_env)
                raw_env.render()
            else:
                raw_env.pop_events()

    deaths = [a for a in agents if not raw_env.alive[a]]

    if verbose:
        print("\n=========================================")
        print("ゲーム終了！最終的な報酬（宝石の数 / ペナルティ）:")
        for a, r in final_rewards.items():
            print(f" - {a}: {r}")

    return {"rewards": final_rewards, "deaths": deaths, "n_proposals": n_proposals}


def evaluate(policy_manager=None, config=None, model_path='policy.pth',
             n_episodes=100, verbose_episodes=3, base_seed=0):
    """複数エピソードを実行し、エージェントごとの統計を返す。

    Returns:
        dict: {
            'avg_rewards': {agent: 平均報酬},
            'death_rates': {agent: 死亡率},
            'avg_proposals': 可決/決着までの平均提案回数,
            'n_episodes': エピソード数,
        }
    """
    if policy_manager is None:
        policy_manager = load_policy_manager(config, model_path)

    agents = PirateGemEnv(config).possible_agents
    reward_sums = {a: 0.0 for a in agents}
    death_counts = {a: 0 for a in agents}
    total_proposals = 0

    for ep in range(n_episodes):
        verbose = ep < verbose_episodes
        if verbose:
            print(f"\n########## 評価エピソード {ep + 1} ##########")
        result = run_game(
            policy_manager=policy_manager, config=config,
            seed=base_seed + ep, verbose=verbose,
        )
        for a in agents:
            reward_sums[a] += result["rewards"][a]
            if a in result["deaths"]:
                death_counts[a] += 1
        total_proposals += result["n_proposals"]

    stats = {
        "avg_rewards": {a: reward_sums[a] / n_episodes for a in agents},
        "death_rates": {a: death_counts[a] / n_episodes for a in agents},
        "avg_proposals": total_proposals / n_episodes,
        "n_episodes": n_episodes,
    }

    print(f"\n========== 評価統計 ({n_episodes} エピソード) ==========")
    print(f"平均提案回数: {stats['avg_proposals']:.2f}")
    for a in agents:
        print(f" - {a}: 平均報酬 {stats['avg_rewards'][a]:+.2f} / 死亡率 {stats['death_rates'][a]:.1%}")

    return stats


def watch():
    logger = Logger()
    sys.stdout = logger
    try:
        run_game(verbose=True)
    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == '__main__':
    watch()
