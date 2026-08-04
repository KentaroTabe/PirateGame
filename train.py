"""Tianshou を用いた並列 DQN 学習ロジック。

各エージェントは独立した Double DQN ポリシーを持ち、MultiAgentPolicyManager で統括される。
pretrained_state_dicts を渡すと、事前学習済みネットワーク（固定順一般解）を
初期値として学習を開始する。
"""

import argparse

import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import PettingZooEnv, SubprocVectorEnv
from tianshou.policy import DQNPolicy, MultiAgentPolicyManager
from tianshou.trainer import OffpolicyTrainer

from env import PirateGemEnv, make_pirate_env
from network import Net


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='乱数シード')
    parser.add_argument('--eps-test', type=float, default=0.05, help='テスト時のイプシロン')
    parser.add_argument('--eps-train', type=float, default=0.1, help='学習時のイプシロン')
    parser.add_argument('--buffer-size', type=int, default=20000, help='リプレイバッファのサイズ')
    parser.add_argument('--lr', type=float, default=1e-3, help='学習率')
    parser.add_argument('--gamma', type=float, default=0.99, help='割引率')
    parser.add_argument('--n-step', type=int, default=3, help='N-stepリターンのN')
    parser.add_argument('--target-update-freq', type=int, default=320, help='ターゲットネットワークの更新頻度')
    parser.add_argument('--epoch', type=int, default=50, help='学習エポック数')
    parser.add_argument('--step-per-epoch', type=int, default=1000, help='1エポックあたりのステップ数')
    parser.add_argument('--step-per-collect', type=int, default=10, help='1回の収集あたりのステップ数')
    parser.add_argument('--update-per-step', type=float, default=0.1, help='1ステップあたりのネットワーク更新回数')
    parser.add_argument('--batch-size', type=int, default=64, help='バッチサイズ')
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 128], help='隠れ層のサイズ')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='デバイス')
    parser.add_argument('--num-envs', type=int, default=4, help='並列学習する環境の数（プロセス数）')
    # 呼び出し元スクリプト固有の引数と衝突しないよう、未知の引数は無視する
    args, _ = parser.parse_known_args()
    return args


def get_env(config=None):
    return PettingZooEnv(PirateGemEnv(config))


def build_policy_manager(env, args, pretrained_state_dicts=None):
    """エージェントごとの DQN ポリシーを構築し、MultiAgentPolicyManager にまとめる。"""
    agents = env.env.possible_agents
    policies = {}
    for agent in agents:
        obs_shape = env.env.observation_spaces[agent]["observation"].shape
        act_shape = env.env.action_spaces[agent].n
        net = Net(obs_shape, act_shape, hidden_sizes=args.hidden_sizes, device=args.device)
        if pretrained_state_dicts is not None and agent in pretrained_state_dicts:
            net.load_state_dict(pretrained_state_dicts[agent])
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)

        # DQNPolicy はコンストラクタで model をターゲットネットワークに複製するため、
        # 事前学習済み重みのロードはポリシー生成前に行う
        policies[agent] = DQNPolicy(
            model=net, optim=optim, discount_factor=args.gamma,
            estimation_step=args.n_step, target_update_freq=args.target_update_freq,
            is_double=True,
        )

    manager = MultiAgentPolicyManager(policies=[policies[agent] for agent in agents], env=env)
    return manager, policies


def train_agent(args=None, config=None, model_path='policy.pth',
                pretrained_state_dicts=None, show_progress=True, verbose=True):
    if args is None:
        args = get_args()

    if config is not None and "train_epochs" in config:
        args.epoch = config["train_epochs"]

    env = get_env(config)

    train_envs = SubprocVectorEnv([lambda: PettingZooEnv(make_pirate_env(config)) for _ in range(args.num_envs)])
    test_envs = SubprocVectorEnv([lambda: PettingZooEnv(make_pirate_env(config)) for _ in range(args.num_envs)])

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    policy_manager, policies = build_policy_manager(env, args, pretrained_state_dicts)
    agents = env.env.possible_agents

    train_collector = Collector(
        policy_manager, train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy_manager, test_envs, exploration_noise=True)

    def train_fn(epoch, env_step):
        for a in agents:
            policies[a].set_eps(args.eps_train)

    def test_fn(epoch, env_step):
        for a in agents:
            policies[a].set_eps(args.eps_test)

    print(f"Training started on {args.device} with {args.num_envs} parallel envs...")
    trainer = OffpolicyTrainer(
        policy=policy_manager, train_collector=train_collector, test_collector=test_collector,
        max_epoch=args.epoch, step_per_epoch=args.step_per_epoch, step_per_collect=args.step_per_collect,
        episode_per_test=10, batch_size=args.batch_size, train_fn=train_fn, test_fn=test_fn,
        update_per_step=args.update_per_step, show_progress=show_progress, verbose=verbose,
    )
    result = trainer.run()

    train_envs.close()
    test_envs.close()

    print("\nTraining finished!")
    torch.save(policy_manager.state_dict(), model_path)
    print(f"モデルを '{model_path}' に保存しました。")

    return result, policy_manager
