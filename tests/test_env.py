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


if __name__ == '__main__':
    unittest.main()
