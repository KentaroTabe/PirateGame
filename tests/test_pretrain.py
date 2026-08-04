"""事前学習（一般解の Q 値回帰）のテスト。"""

import unittest

import numpy as np
import torch

from env import PirateGemEnv
from pretrain import build_dataset, pretrain_agents, _build_observation
from solver import FixedOrderSolver

CONFIG = {
    "num_agents": 3,
    "total_gems": 4,
    "L": 5.0,
    "agent_weights": [3.0, 2.0, 1.0],
    "fixed_order": False,
}


class TestDataset(unittest.TestCase):
    def test_observation_matches_env_observe(self):
        env = PirateGemEnv(CONFIG)
        env.reset(seed=0)
        proposer_idx = env.agent_name_mapping[env.proposer]
        obs = _build_observation(env, frozenset(range(3)), proposer_idx, (0, 0, 0))
        env_obs = env.observe(env.proposer)["observation"]
        np.testing.assert_allclose(obs, env_obs)

    def test_dataset_targets_match_solver(self):
        env = PirateGemEnv(CONFIG)
        solver = FixedOrderSolver(3, 4, 5.0)
        obs, mask, target = build_dataset(env, solver, agent_idx=0)
        self.assertEqual(len(obs), len(mask))
        self.assertEqual(len(obs), len(target))

        # 全生存・自分が提案者の提案状態を探し、最適提案の Q 値が自分の取り分と一致するか確認
        propose_obs = _build_observation(env, frozenset(range(3)), 0, (0, 0, 0))
        rows = np.where((obs == propose_obs).all(axis=1) & mask[:, :env.NUM_DISTRIBUTIONS].any(axis=1))[0]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        best_action = int(np.where(mask[row], target[row], -np.inf).argmax())
        optimal, passes = solver.optimal_proposal(frozenset(range(3)), 0)
        self.assertTrue(passes)
        self.assertEqual(target[row][best_action], optimal[0])


class TestPretraining(unittest.TestCase):
    def test_pretrained_greedy_policy_matches_general_solution(self):
        torch.manual_seed(0)
        state_dicts, stats = pretrain_agents(
            CONFIG, device="cpu", hidden_sizes=(64, 64),
            epochs=150, lr=1e-3, batch_size=256, seed=0, verbose=False,
        )
        for agent, agent_stats in stats.items():
            self.assertGreaterEqual(
                agent_stats["match_rate"], 0.9,
                f"{agent} の一般解一致率が低すぎます: {agent_stats['match_rate']:.1%}",
            )
        # state_dict が Net にロード可能であること
        self.assertEqual(set(state_dicts.keys()), {"agent_A", "agent_B", "agent_C"})


if __name__ == '__main__':
    unittest.main()
