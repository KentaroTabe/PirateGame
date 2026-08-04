"""実験パイプライン全体（事前学習 → ランダム順 DQN 学習 → 統計評価）のスモークテスト。

小規模設定・短時間で一連の流れが動くことだけを確認する。
使い方: scripts/smoke_test.sh <出力ディレクトリ>
"""

import os
import sys

import torch

from eval import evaluate
from plot_log import plot_metrics
from pretrain import pretrain_agents
from train import get_args, train_agent


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    config = {
        "num_agents": 4,
        "total_gems": 3,
        "L": 5.0,
        "agent_weights": [8.0, 4.0, 2.0, 1.0],
        "fixed_order": False,
        "excess_vote_penalty": 0.0,
    }

    print("=== フェーズ0: 事前学習 ===")
    state_dicts, stats = pretrain_agents(config, device="cpu", epochs=80, verbose=True)
    pretrained_path = os.path.join(out_dir, "pretrained_smoke.pth")
    torch.save(state_dicts, pretrained_path)

    print("\n=== フェーズ1: ランダム順 DQN 学習（短縮版） ===")
    args = get_args()
    args.device = "cpu"
    args.num_envs = 2
    args.epoch = 4
    args.step_per_epoch = 500
    args.buffer_size = 5000

    model_path = os.path.join(out_dir, "policy_smoke.pth")
    metrics_path = os.path.join(out_dir, "log_metrics_smoke.csv")
    train_result, policy_manager = train_agent(
        args=args, config=config, model_path=model_path,
        pretrained_state_dicts=state_dicts,
        metrics_path=metrics_path, metrics_interval=1, metrics_episodes=10,
        show_progress=False, verbose=False,
    )
    print(f"best_reward: {train_result.get('best_reward', 0):.2f}")

    plot_path = plot_metrics(metrics_path, os.path.join(out_dir, "plots"))
    print(f"メトリクスのグラフを {plot_path} に保存しました。")

    print("\n=== フェーズ2: 統計評価 ===")
    eval_stats = evaluate(
        policy_manager=policy_manager, config=config,
        n_episodes=20, verbose_episodes=1,
    )

    assert eval_stats["n_episodes"] == 20
    print("\n✅ スモークテスト完了")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "smoke_output")
