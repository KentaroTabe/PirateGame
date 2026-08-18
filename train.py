"""Tianshou を用いた並列 DQN 学習ロジック。

各エージェントは独立した Double DQN ポリシーを持ち、MultiAgentPolicyManager で統括される。
pretrained_state_dicts を渡すと、事前学習済みネットワーク（固定順一般解）を
初期値として学習を開始する。
"""

import argparse

import numpy as np
import torch
from tianshou.data import Batch, Collector, VectorReplayBuffer
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


def record_proposals(policies, env, agents):
    """各エージェントが「全員生存・自分が提案者」の局面で選ぶ分配案を返す。

    順位や報酬に依らず方策そのものを観測するための指標。学習中に方策が
    何回変わったかを数えれば、首位の分離度に交絡されない安定性が測れる
    （首位交代回数は、首位が他を引き離しているほど起きにくいという
    交絡を持つ。docs/reports/round6.md 参照）。
    """
    from pretrain import _build_observation

    alive = set(range(env.n_agents))
    zero_proposal = (0,) * env.n_agents
    n_dist = len(env.DISTRIBUTIONS)

    proposals = []
    for idx, agent in enumerate(agents):
        obs = _build_observation(env, alive, idx, zero_proposal)
        # 全員生存の局面ではすべての分配案が有効（死者に配る案が存在しない）
        mask = np.zeros(n_dist + 2, dtype=bool)
        mask[:n_dist] = True
        batch = Batch(obs=Batch(obs=obs[None, :], mask=mask[None, :]), info={})
        with torch.no_grad():
            action = int(policies[agent](batch).act[0])
        if action < len(env.DISTRIBUTIONS):
            proposals.append("-".join(str(v) for v in env.DISTRIBUTIONS[action]))
        else:
            proposals.append("invalid")
    return proposals


def collect_political_metrics(policy_manager, policies, test_collector, agents, n_episode):
    """貪欲方策（ε=0）で評価対局を行い、政治的指標を集計する。

    Returns:
        dict: rew_mean / death_rate はエージェント順の配列、
        len_mean は平均エピソード長、first_pass_rate は最初の提案が可決された割合。
    """
    policy_manager.eval()
    for a in agents:
        policies[a].set_eps(0.0)
    test_collector.reset_env()
    test_collector.reset_buffer()
    result = test_collector.collect(n_episode=n_episode)

    rews = np.asarray(result["rews"])  # (エピソード数, エージェント数)
    lens = np.asarray(result["lens"])
    return {
        "rew_mean": rews.mean(axis=0),
        "death_rate": (rews < 0).mean(axis=0),  # 負の報酬 = 否決されて死亡
        "len_mean": float(lens.mean()),
        # 最初の提案で決着 = 1提案 + 全員投票 = (エージェント数 + 1) ステップ
        "first_pass_rate": float((lens == len(agents) + 1).mean()),
    }


def train_agent(args=None, config=None, model_path='policy.pth',
                pretrained_state_dicts=None, metrics_path=None,
                metrics_interval=10, metrics_episodes=30,
                show_progress=True, verbose=True):
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
    if metrics_path is None:
        result = trainer.run()
    else:
        # エポック単位でイテレートし、一定間隔で政治的指標をCSVに記録する
        result = {}
        names = [a.split('_')[1] for a in agents]
        with open(metrics_path, "w", encoding="utf-8") as mf:
            header = (
                ["epoch", "env_step", "len_mean", "first_pass_rate"]
                + [f"rew_{x}" for x in names] + [f"death_{x}" for x in names]
                + [f"prop_{x}" for x in names]
            )
            mf.write(",".join(header) + "\n")
            for epoch, epoch_stat, info in trainer:
                result = info
                if epoch % metrics_interval == 0 or epoch == args.epoch:
                    m = collect_political_metrics(
                        policy_manager, policies, test_collector, agents, metrics_episodes,
                    )
                    row = (
                        [epoch, epoch_stat["env_step"], m["len_mean"], m["first_pass_rate"]]
                        + list(m["rew_mean"]) + list(m["death_rate"])
                        + record_proposals(policies, env.env, agents)
                    )
                    mf.write(",".join(
                        f"{v:.4f}" if isinstance(v, float) else str(v) for v in row
                    ) + "\n")
                    mf.flush()

    train_envs.close()
    test_envs.close()

    print("\nTraining finished!")
    # MultiAgentPolicyManager.policies は素の dict で nn.ModuleDict ではないため、
    # manager.state_dict() は空を返す。エージェントごとに明示的に保存する。
    torch.save({agent: policies[agent].state_dict() for agent in agents}, model_path)
    print(f"モデルを '{model_path}' に保存しました。")

    return result, policy_manager
