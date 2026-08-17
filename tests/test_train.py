"""学習ループの補助関数のテスト。

方策指標の記録（record_proposals）は学習中に毎エポック呼ばれるため、
ここで観測次元がずれると学習が長時間走ったあとに落ちる。
第10ラウンド 試行43 で実際に発生したので、環境設定の組み合わせを守る。
"""

import unittest

from train import build_policy_manager, get_env, get_args, record_proposals

BASE_CONFIG = {
    "num_agents": 3,
    "total_gems": 4,
    "L": 5.0,
    "agent_weights": [3.0, 2.0, 1.0],
    "fixed_order": False,
}


class TestRecordProposals(unittest.TestCase):
    def _run(self, **overrides):
        config = dict(BASE_CONFIG, **overrides)
        env = get_env(config)
        args = get_args()
        args.device = "cpu"
        _, policies = build_policy_manager(env, args)
        agents = env.env.possible_agents
        return record_proposals(policies, env.env, agents), env.env

    def test_records_a_proposal_for_every_agent(self):
        proposals, env = self._run()
        self.assertEqual(len(proposals), env.n_agents)
        self.assertNotIn("invalid", proposals)

    def test_works_with_vote_tally_observation(self):
        """観測に票数を足しても次元が合い、学習中に落ちない。"""
        proposals, env = self._run(observe_vote_tally=True)
        self.assertEqual(len(proposals), env.n_agents)
        self.assertNotIn("invalid", proposals)

    def test_works_with_proposer_votes_last(self):
        proposals, env = self._run(proposer_votes_last=True)
        self.assertEqual(len(proposals), env.n_agents)
        self.assertNotIn("invalid", proposals)

    def test_works_with_both_options(self):
        proposals, env = self._run(proposer_votes_last=True, observe_vote_tally=True)
        self.assertEqual(len(proposals), env.n_agents)
        self.assertNotIn("invalid", proposals)


if __name__ == '__main__':
    unittest.main()
