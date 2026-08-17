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

    def test_observation_matches_env_observe_with_vote_tally(self):
        """observe_vote_tally を有効にしても env.observe() と一致する。

        観測の組み立てが env.observe() と _build_observation() の2箇所に
        重複しているため、片方だけ更新すると学習時に次元不一致で落ちる
        （第10ラウンド 試行43 で実際に発生）。両方を突き合わせて守る。
        """
        config = dict(CONFIG, observe_vote_tally=True)
        env = PirateGemEnv(config)
        env.reset(seed=0)
        proposer_idx = env.agent_name_mapping[env.proposer]

        obs = _build_observation(env, frozenset(range(3)), proposer_idx, (0, 0, 0))
        env_obs = env.observe(env.proposer)["observation"]
        self.assertEqual(obs.shape, env_obs.shape)
        np.testing.assert_allclose(obs, env_obs)

    def test_observation_dim_matches_observation_space(self):
        """既定・票数入りの両方で、観測次元が observation_space と一致する。"""
        for observe_vote_tally in (False, True):
            with self.subTest(observe_vote_tally=observe_vote_tally):
                env = PirateGemEnv(dict(CONFIG, observe_vote_tally=observe_vote_tally))
                env.reset(seed=0)
                expected = env.observation_space("agent_A")["observation"].shape
                obs = _build_observation(env, frozenset(range(3)), 0, (0, 0, 0))
                self.assertEqual(obs.shape, expected)

    def test_build_observation_rejects_wrong_tally_length(self):
        env = PirateGemEnv(dict(CONFIG, observe_vote_tally=True))
        with self.assertRaises(ValueError):
            _build_observation(env, frozenset(range(3)), 0, (0, 0, 0), vote_tally=[1.0])

    def test_pretrain_rejects_vote_tally_observation(self):
        """一般解が投票の途中経過を扱えないので、事前学習は明示的に落とす。"""
        with self.assertRaises(ValueError):
            pretrain_agents(dict(CONFIG, observe_vote_tally=True), epochs=1, verbose=False)

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
