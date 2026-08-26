"""PirateGemEnv（環境ロジック）のテスト。"""

import unittest

from env import PirateGemEnv


def make_env(**overrides):
    config = {
        "num_agents": 3,
        "total_gems": 4,
        "L": 1.0,
        "fixed_order": True,
        "agent_weights": [3.0, 2.0, 1.0],
    }
    config.update(overrides)
    return PirateGemEnv(config)


class TestPirateGemEnv(unittest.TestCase):
    def _propose(self, env, dist):
        self.assertEqual(env.phase, "PROPOSE")
        env.step(env.DISTRIBUTIONS.index(tuple(dist)))

    def _vote(self, env, yes):
        self.assertEqual(env.phase, "VOTE")
        env.step(env.ACTION_YES if yes else env.ACTION_NO)

    def test_accepted_proposal_distributes_rewards(self):
        env = make_env()
        env.reset(seed=0)
        self.assertEqual(env.proposer, "agent_A")

        self._propose(env, (2, 1, 1))
        self._vote(env, True)   # A
        self._vote(env, False)  # B
        self._vote(env, True)   # C

        self.assertTrue(all(env.terminations.values()))
        self.assertEqual(env.rewards["agent_A"], 2.0)
        self.assertEqual(env.rewards["agent_B"], 1.0)
        self.assertEqual(env.rewards["agent_C"], 1.0)

        events = env.pop_events()
        self.assertEqual(events[0]["type"], "vote_result")
        self.assertTrue(events[0]["passed"])

    def test_rejected_proposer_dies_and_next_takes_over(self):
        env = make_env()
        env.reset(seed=0)

        self._propose(env, (4, 0, 0))
        self._vote(env, True)   # A
        self._vote(env, False)  # B
        self._vote(env, False)  # C → 1票 < 過半数(2)で否決

        self.assertFalse(env.alive["agent_A"])
        self.assertTrue(env.terminations["agent_A"])
        self.assertEqual(env.rewards["agent_A"], -1.0)
        # 固定順: 次の提案者は B
        self.assertEqual(env.proposer, "agent_B")
        self.assertEqual(env.phase, "PROPOSE")

        # 死亡した A に配る分配案はマスクされる
        mask = env.observe("agent_B")["action_mask"]
        idx_gives_to_dead = env.DISTRIBUTIONS.index((1, 3, 0))
        idx_valid = env.DISTRIBUTIONS.index((0, 4, 0))
        self.assertEqual(mask[idx_gives_to_dead], 0)
        self.assertEqual(mask[idx_valid], 1)

        # B は自分の1票（2人中1票 = 半数）で可決できる
        self._propose(env, (0, 4, 0))
        self._vote(env, True)   # B
        self._vote(env, False)  # C
        self.assertTrue(all(env.terminations.values()))
        self.assertEqual(env.rewards["agent_B"], 4.0)
        self.assertEqual(env.rewards["agent_C"], 0.0)

    def test_last_survivor_takes_all(self):
        env = make_env(num_agents=2, agent_weights=[2.0, 1.0])
        env.reset(seed=0)

        # A が自分の提案に反対して自滅すると、B が独占する
        self._propose(env, (4, 0))
        self._vote(env, False)  # A
        self._vote(env, False)  # B

        self.assertFalse(env.alive["agent_A"])
        self.assertEqual(env.rewards["agent_A"], -1.0)
        self.assertEqual(env.rewards["agent_B"], 4.0)
        self.assertTrue(all(env.terminations.values()))
        events = env.pop_events()
        self.assertEqual(events[-1]["type"], "last_survivor")
        self.assertEqual(events[-1]["agent"], "agent_B")

    def test_excess_vote_penalty(self):
        env = make_env(excess_vote_penalty=1.0)
        env.reset(seed=0)

        self._propose(env, (2, 1, 1))
        self._vote(env, True)  # A
        self._vote(env, True)  # B
        self._vote(env, True)  # C → 必要2票に対し3票（過剰1票）

        self.assertEqual(env.rewards["agent_A"], 2.0 - 1.0)
        self.assertEqual(env.rewards["agent_B"], 1.0)

    def test_random_order_uses_weights(self):
        env = make_env(fixed_order=False, agent_weights=[1000.0, 1.0, 1.0])
        counts = {a: 0 for a in env.possible_agents}
        for seed in range(50):
            env.reset(seed=seed)
            counts[env.proposer] += 1
        # 圧倒的なウェイトを持つ A がほぼ毎回初期提案者になる
        self.assertGreater(counts["agent_A"], 45)

    def test_reset_is_reproducible_with_seed(self):
        env = make_env(fixed_order=False)
        env.reset(seed=123)
        first = env.proposer
        env.reset(seed=123)
        self.assertEqual(env.proposer, first)

    # ------------------------------------------------------------------
    # proposer_votes_last（既定は False = 従来どおり固定順）
    # ------------------------------------------------------------------
    def test_voting_order_defaults_to_fixed_order(self):
        env = make_env()
        env.reset(seed=0)
        self.assertEqual(env.proposer, "agent_A")

        self._propose(env, (2, 1, 1))
        self.assertEqual(env.voting_order, ["agent_A", "agent_B", "agent_C"])
        self.assertEqual(env.agent_selection, "agent_A")

    def test_proposer_votes_last_moves_proposer_to_end(self):
        env = make_env(proposer_votes_last=True)
        env.reset(seed=0)
        self.assertEqual(env.proposer, "agent_A")

        self._propose(env, (2, 1, 1))
        self.assertEqual(env.voting_order, ["agent_B", "agent_C", "agent_A"])
        self.assertEqual(env.agent_selection, "agent_B")

    def test_proposer_votes_last_keeps_others_relative_order(self):
        """提案者が中間の場合、他者の相対順は崩れない。"""
        env = make_env(proposer_votes_last=True, fixed_order=False,
                       agent_weights=[1.0, 1000.0, 1.0])
        env.reset(seed=0)
        self.assertEqual(env.proposer, "agent_B")

        self._propose(env, (1, 2, 1))
        self.assertEqual(env.voting_order, ["agent_A", "agent_C", "agent_B"])

    def test_proposer_votes_last_can_reject_own_proposal_after_seeing_votes(self):
        """提案者が最後に投票し、可決を保ったまま過剰票を1つ減らせる。"""
        env = make_env(proposer_votes_last=True, excess_vote_penalty=1.0)
        env.reset(seed=0)

        self._propose(env, (2, 1, 1))
        self._vote(env, True)   # B
        self._vote(env, True)   # C → この時点で必要2票に到達
        self._vote(env, False)  # A（提案者）は反対しても可決は保たれる

        self.assertTrue(all(env.terminations.values()))
        # 賛成2票 = 必要2票ちょうどなので過剰票は0、罰は発生しない
        self.assertEqual(env.rewards["agent_A"], 2.0)

    def test_proposer_votes_last_survives_after_a_rejection(self):
        """否決で提案者が代わっても、新しい提案者が末尾に回る。"""
        env = make_env(proposer_votes_last=True)
        env.reset(seed=0)

        self._propose(env, (4, 0, 0))
        self._vote(env, False)  # B
        self._vote(env, False)  # C
        self._vote(env, False)  # A（提案者）

        self.assertFalse(env.alive["agent_A"])
        self.assertEqual(env.proposer, "agent_B")

        self._propose(env, (0, 3, 1))
        self.assertEqual(env.voting_order, ["agent_C", "agent_B"])

    # ------------------------------------------------------------------
    # observe_vote_tally（既定は False = 従来どおり 4N 次元）
    # ------------------------------------------------------------------
    def test_observation_excludes_vote_tally_by_default(self):
        env = make_env()
        env.reset(seed=0)
        expected_dim = 4 * env.n_agents
        self.assertEqual(env.observation_space("agent_A")["observation"].shape, (expected_dim,))
        self.assertEqual(env.observe("agent_A")["observation"].shape, (expected_dim,))

    def test_observation_includes_vote_tally_when_enabled(self):
        env = make_env(observe_vote_tally=True)
        env.reset(seed=0)
        expected_dim = 4 * env.n_agents + env.VOTE_TALLY_DIM
        self.assertEqual(env.observation_space("agent_A")["observation"].shape, (expected_dim,))

        # 提案前は誰も投票していない
        self.assertEqual(list(env.observe("agent_A")["observation"][-2:]), [0.0, 0.0])

        self._propose(env, (2, 1, 1))
        self._vote(env, True)   # A
        self.assertEqual(list(env.observe("agent_B")["observation"][-2:]), [1.0, 1.0])

        self._vote(env, False)  # B → 賛成1・投票2
        self.assertEqual(list(env.observe("agent_C")["observation"][-2:]), [1.0, 2.0])

    # ------------------------------------------------------------------
    # observe_noise_dims（既定は 0 = 乱数次元なし）
    # ------------------------------------------------------------------
    def test_observation_has_no_noise_dims_by_default(self):
        env = make_env()
        env.reset(seed=0)
        self.assertEqual(env.observe_noise_dims, 0)
        self.assertEqual(env.observe(env.proposer)["observation"].shape,
                         (4 * env.n_agents,))

    def test_noise_dims_extend_the_observation(self):
        env = make_env(observe_noise_dims=3)
        env.reset(seed=0)
        expected = (4 * env.n_agents + 3,)
        self.assertEqual(env.observation_space("agent_A")["observation"].shape, expected)
        self.assertEqual(env.observe("agent_A")["observation"].shape, expected)

    def test_noise_dims_change_between_observations(self):
        """乱数次元は毎回引き直される（定数ではない）。"""
        env = make_env(observe_noise_dims=4)
        env.reset(seed=0)
        first = env.observe("agent_A")["observation"][-4:]
        second = env.observe("agent_A")["observation"][-4:]
        self.assertFalse((first == second).all(), "乱数次元が変化していない")

    def test_noise_dims_do_not_disturb_proposer_selection(self):
        """乱数次元は専用 RNG から引くので、提案者の並びを変えない。

        self._rng を進めてしまうと、乱数次元なしの実験と比較できなくなる。
        """
        plain = make_env(fixed_order=False)
        noisy = make_env(fixed_order=False, observe_noise_dims=4)
        for seed in range(20):
            plain.reset(seed=seed)
            noisy.reset(seed=seed)
            # 観測を引いて乱数 RNG を進めても提案者は一致する
            noisy.observe("agent_A")
            noisy.observe("agent_B")
            self.assertEqual(plain.proposer, noisy.proposer, f"seed={seed} で提案者がずれた")

    def test_negative_noise_dims_rejected(self):
        with self.assertRaises(ValueError):
            make_env(observe_noise_dims=-1)

    def test_vote_tally_resets_for_the_next_proposal(self):
        env = make_env(observe_vote_tally=True)
        env.reset(seed=0)

        self._propose(env, (4, 0, 0))
        self._vote(env, False)  # A
        self._vote(env, False)  # B
        self._vote(env, False)  # C → 否決、A が死亡

        self.assertEqual(env.proposer, "agent_B")
        self._propose(env, (0, 3, 1))
        self.assertEqual(list(env.observe("agent_B")["observation"][-2:]), [0.0, 0.0])


if __name__ == '__main__':
    unittest.main()
