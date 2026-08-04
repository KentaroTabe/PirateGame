"""FixedOrderSolver（バックワードインダクション一般解）のテスト。"""

import unittest

from solver import FixedOrderSolver


class TestClassicPirateGame(unittest.TestCase):
    """古典的な海賊ゲーム（5人・100枚）の既知解と一致することを確認する。"""

    def test_classic_solution_98_0_1_0_1(self):
        solver = FixedOrderSolver(n_agents=5, total_gems=100, L=1.0)
        proposal, passes = solver.optimal_proposal(frozenset(range(5)), proposer=0)
        self.assertTrue(passes)
        self.assertEqual(proposal, (98, 0, 1, 0, 1))

    def test_value_matches_proposal(self):
        solver = FixedOrderSolver(n_agents=5, total_gems=100, L=1.0)
        value = solver.value(frozenset(range(5)))
        self.assertEqual(list(value), [98, 0, 1, 0, 1])


class TestProjectConfig(unittest.TestCase):
    """config.json 相当（6人・5個・L=100）の解を確認する。"""

    def setUp(self):
        self.solver = FixedOrderSolver(n_agents=6, total_gems=5, L=100.0)

    def test_full_set_solution(self):
        proposal, passes = self.solver.optimal_proposal(frozenset(range(6)), proposer=0)
        self.assertTrue(passes)
        self.assertEqual(proposal, (3, 0, 1, 0, 1, 0))

    def test_single_survivor_takes_all(self):
        value = self.solver.value(frozenset({5}))
        self.assertEqual(value[5], 5.0)

    def test_arbitrary_proposer_random_order_state(self):
        # ランダム順ゲームで訪れうる「提案者が最若番でない」状態でも解ける
        proposal, passes = self.solver.optimal_proposal(frozenset({0, 1, 2}), proposer=2)
        self.assertTrue(passes)
        # 否決後は {0,1} の固定順ゲーム: 0が全取り(5)、1は0。よって1を宝石1個で買収する
        self.assertEqual(proposal, (0, 1, 4, 0, 0, 0))

    def test_optimal_votes(self):
        alive = frozenset(range(6))
        proposal = (3, 0, 1, 0, 1, 0)
        # 提案者は賛成（反対なら死亡 -L）
        self.assertTrue(self.solver.optimal_vote(alive, 0, proposal, 0))
        # 継続価値より多くもらえる者は賛成
        self.assertTrue(self.solver.optimal_vote(alive, 0, proposal, 2))
        self.assertTrue(self.solver.optimal_vote(alive, 0, proposal, 4))
        # 継続価値以下しかもらえない者は反対（無差別でも反対）
        self.assertFalse(self.solver.optimal_vote(alive, 0, proposal, 1))
        self.assertFalse(self.solver.optimal_vote(alive, 0, proposal, 3))
        self.assertFalse(self.solver.optimal_vote(alive, 0, proposal, 5))


class TestInfeasibleProposal(unittest.TestCase):
    """宝石が足りず買収不能な場合、提案者の死が確定する。"""

    def test_proposer_doomed_when_votes_unaffordable(self):
        solver = FixedOrderSolver(n_agents=6, total_gems=1, L=10.0)
        # 5人（1..5）・宝石1個: 必要2票の買収に2個必要で不可能
        proposal, passes = solver.optimal_proposal(frozenset(range(1, 6)), proposer=1)
        self.assertFalse(passes)
        value = solver.value(frozenset(range(1, 6)))
        self.assertEqual(value[1], -10.0)

    def test_doomed_voter_can_be_bought_for_free(self):
        solver = FixedOrderSolver(n_agents=6, total_gems=1, L=10.0)
        # 全員生存時: 死が確定している1は継続価値 -L のため 0 個でも賛成する
        proposal, passes = solver.optimal_proposal(frozenset(range(6)), proposer=0)
        self.assertTrue(passes)
        self.assertTrue(solver.optimal_vote(frozenset(range(6)), 0, proposal, 1))


if __name__ == '__main__':
    unittest.main()
