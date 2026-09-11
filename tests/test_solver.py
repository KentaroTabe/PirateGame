"""FixedOrderSolver（バックワードインダクション一般解）と RandomOrderSolver のテスト。"""

import unittest

import numpy as np

from solver import FixedOrderSolver, RandomOrderSolver


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


class TestRandomOrderSolver(unittest.TestCase):
    """ランダム順・否決で脱落するゲーム（学習環境そのもの）の理論均衡。

    期待値は手計算で導ける小さなゲームで確かめる。
    """

    def test_two_survivors_split_by_recognition(self):
        # 2人なら提案者自身の1票で可決するので、選ばれた方が独占する
        solver = RandomOrderSolver(2, 5, L=10.0, weights=[3.0, 1.0])
        value = solver.value(frozenset({0, 1}))
        self.assertAlmostEqual(value[0], 5 * 0.75)
        self.assertAlmostEqual(value[1], 5 * 0.25)

    def test_three_agents_one_gem_proposer_keeps_nothing(self):
        # 否決後の2人ゲームで各投票者の期待値は 0.5。真に上回るには1個要るので、
        # 提案者は唯一の宝石を2人のどちらかに（無作為に）渡し、手元は0になる
        solver = RandomOrderSolver(3, 1, L=10.0, weights=[1.0, 1.0, 1.0])
        payoff, passes, yes_prob = solver.proposal_outcome(frozenset(range(3)), 0)
        self.assertTrue(passes)
        np.testing.assert_allclose(payoff, [0.0, 0.5, 0.5])
        np.testing.assert_allclose(yes_prob, [0.0, 0.5, 0.5])
        np.testing.assert_allclose(solver.value(frozenset(range(3))), [1 / 3] * 3)

    def test_unaffordable_subgame_makes_L_matter(self):
        # 4人ゲームの各自の期待値は 1/4（3人ゲームの 1/3 を 3/4 の確率で受け取る）。
        # 5人では必要2票の買収に2個要り、宝石1個では誰も可決させられない。
        # 各自の期待値は「1/5 で提案者になり -L」+「4/5 で残って 1/4」
        L = 10.0
        solver = RandomOrderSolver(5, 1, L=L, weights=[1.0] * 5)
        _, passes, _ = solver.proposal_outcome(frozenset(range(5)), 0)
        self.assertFalse(passes)
        np.testing.assert_allclose(solver.value(frozenset(range(5))), [(1 - L) / 5] * 5)

    def test_values_do_not_depend_on_L_when_every_subgame_passes(self):
        # 6人・宝石5個では可決できない部分ゲームがないので、-L は均衡経路に現れない
        full = frozenset(range(6))
        for weights in ([1.0] * 6, [50.0, 1.0, 1.0, 1.0, 1.0, 1.0]):
            low = RandomOrderSolver(6, 5, L=5.0, weights=weights).value(full)
            high = RandomOrderSolver(6, 5, L=100.0, weights=weights).value(full)
            np.testing.assert_allclose(low, high)
            self.assertAlmostEqual(low.sum(), 5.0)

    def test_symmetric_weights_give_equal_values(self):
        value = RandomOrderSolver(6, 5, L=100.0, weights=[1.0] * 6).value(frozenset(range(6)))
        np.testing.assert_allclose(value, [5 / 6] * 6)

    def test_tie_rule_changes_the_proposer_share(self):
        # 均等重み6人・宝石5個: 否決後の5人ゲームの各自の価値は対称性から 5/5 = 1.0 ちょうど。
        # 無差別なら反対 → 1人2個で2票に4個、手元1。無差別なら賛成 → 1人1個で2票に2個、手元3。
        # どちらの規則でも対称性から全員の期待価値は 5/6 のまま。
        full = frozenset(range(6))
        weights = [1.0] * 6
        strict = RandomOrderSolver(6, 5, L=100.0, weights=weights)
        lenient = RandomOrderSolver(6, 5, L=100.0, weights=weights, accept_when_indifferent=True)
        self.assertAlmostEqual(strict.proposal_outcome(full, 0)[0][0], 1.0)
        self.assertAlmostEqual(lenient.proposal_outcome(full, 0)[0][0], 3.0)
        np.testing.assert_allclose(strict.value(full), [5 / 6] * 6)
        np.testing.assert_allclose(lenient.value(full), [5 / 6] * 6)

    def test_rejects_mismatched_weights(self):
        with self.assertRaises(ValueError):
            RandomOrderSolver(3, 5, L=10.0, weights=[1.0, 1.0])


if __name__ == '__main__':
    unittest.main()
