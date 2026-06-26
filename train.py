import argparse
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import DQNPolicy, MultiAgentPolicyManager
from tianshou.trainer import OffpolicyTrainer
from env import PirateGemEnv
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
    return parser.parse_args()

# configを外部から渡せるように修正
def get_env(config=None):
    return PettingZooEnv(PirateGemEnv(config if config is not None else {}))

# 外部引数に対応できるように修正
def train_agent(args=None, config=None, model_path='policy.pth', show_progress=True):
    if args is None:
        args = get_args()
        
    # configにエポック数がある場合は上書き
    if config is not None and "train_epochs" in config:
        args.epoch = config["train_epochs"]

    env = get_env(config)
    train_envs = DummyVectorEnv([lambda: get_env(config) for _ in range(1)])
    test_envs = DummyVectorEnv([lambda: get_env(config) for _ in range(1)])
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    
    policies = {}
    agents = env.env.possible_agents
    
    for agent in agents:
        obs_shape = env.env.observation_spaces[agent]["observation"].shape
        act_shape = env.env.action_spaces[agent].n
        net = Net(obs_shape, act_shape, hidden_sizes=args.hidden_sizes, device=args.device)
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        
        policy = DQNPolicy(
            model=net, optim=optim, discount_factor=args.gamma,
            estimation_step=args.n_step, target_update_freq=args.target_update_freq,
            is_double=True
        )
        policies[agent] = policy
        
    policy_manager = MultiAgentPolicyManager(policies=[policies[agent] for agent in agents], env=env)
    
    train_collector = Collector(policy_manager, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs)), exploration_noise=True)
    test_collector = Collector(policy_manager, test_envs, exploration_noise=True)
    
    def train_fn(epoch, env_step):
        for a in agents: policies[a].set_eps(args.eps_train)

    def test_fn(epoch, env_step):
        for a in agents: policies[a].set_eps(args.eps_test)

    print(f"Training started on {args.device}...")
    trainer = OffpolicyTrainer(
        policy=policy_manager, train_collector=train_collector, test_collector=test_collector,
        max_epoch=args.epoch, step_per_epoch=args.step_per_epoch, step_per_collect=args.step_per_collect,
        episode_per_test=10, batch_size=args.batch_size, train_fn=train_fn, test_fn=test_fn,
        update_per_step=args.update_per_step, show_progress=show_progress
    )
    result = trainer.run()
    
    print("\nTraining finished!")
    torch.save(policy_manager.state_dict(), model_path)
    print(f"モデルを '{model_path}' に保存しました。")
    
    return result, policy_manager