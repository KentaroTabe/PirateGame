"""学習ループの補助関数のテスト。

方策指標の記録（record_proposals）は学習中に毎エポック呼ばれるため、
ここで観測次元がずれると学習が長時間走ったあとに落ちる。
第10ラウンド 試行43 で実際に発生したので、環境設定の組み合わせを守る。
"""

import unittest

from train import (build_policy_manager, get_env, get_args, record_proposals,
                   record_proposer_votes)

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

    def test_works_with_noise_dims(self):
        """乱数次元を足しても観測次元が合い、学習中に落ちない。

        第10ラウンド 試行43 と同じ種類の不整合（観測の組み立てが2箇所に
        重複している）を、新しい設定でも起こさないことを守る。
        """
        proposals, env = self._run(observe_noise_dims=4)
        self.assertEqual(len(proposals), env.n_agents)
        self.assertNotIn("invalid", proposals)

    def test_works_with_noise_dims_and_tally(self):
        proposals, env = self._run(observe_noise_dims=4, observe_vote_tally=True)
        self.assertEqual(len(proposals), env.n_agents)
        self.assertNotIn("invalid", proposals)


class TestRecordProposerVotes(unittest.TestCase):
    """自己反対ルートか決定票ルートかの読み出し。"""

    def _votes(self, **overrides):
        config = dict(BASE_CONFIG, **overrides)
        env = get_env(config)
        args = get_args()
        args.device = "cpu"
        _, policies = build_policy_manager(env, args)
        agents = env.env.possible_agents
        return record_proposer_votes(policies, env.env, agents), env.env

    def test_returns_na_without_vote_tally_observation(self):
        """票数が観測に入らない設定では読み出せないので n/a を返す。"""
        votes, env = self._votes(observe_vote_tally=False)
        self.assertEqual(votes, ["n/a"] * env.n_agents)

    def test_returns_a_binary_vote_per_agent(self):
        votes, env = self._votes(observe_vote_tally=True)
        self.assertEqual(len(votes), env.n_agents)
        for v in votes:
            self.assertIn(v, ("yes", "no"))

    def test_works_with_proposer_votes_last(self):
        votes, env = self._votes(observe_vote_tally=True, proposer_votes_last=True)
        self.assertEqual(len(votes), env.n_agents)
        for v in votes:
            self.assertIn(v, ("yes", "no"))

    def test_reads_the_policy_not_a_constant(self):
        """異なる重みの方策からは、同じ値が返るとは限らない形で読めている。

        （毎回同じ定数を返す実装になっていないことを、観測が実際に
        ネットワークを通っていることで担保する。）
        """
        import torch

        config = dict(BASE_CONFIG, observe_vote_tally=True)
        env = get_env(config)
        args = get_args()
        args.device = "cpu"
        _, policies = build_policy_manager(env, args)
        agents = env.env.possible_agents

        first = record_proposer_votes(policies, env.env, agents)

        # 最終層のバイアスを操作して NO を強制し、読み出しが追従することを見る
        for a in agents:
            last = [m for m in policies[a].model.model if isinstance(m, torch.nn.Linear)][-1]
            with torch.no_grad():
                last.bias[env.env.ACTION_NO] = 1e6
                last.bias[env.env.ACTION_YES] = -1e6
        forced = record_proposer_votes(policies, env.env, agents)

        self.assertEqual(forced, ["no"] * env.env.n_agents)
        self.assertEqual(len(first), env.env.n_agents)


class TestModelRoundTrip(unittest.TestCase):
    """保存したモデルを読み直すと同じ行動になることを守る。

    MultiAgentPolicyManager.policies は素の dict なので manager.state_dict() は
    空を返す。これに気づかず保存していた時期のモデルは中身が空で、読み込んでも
    ランダム初期化のままだった（第10ラウンドで発覚）。
    """

    def test_saved_model_restores_the_same_behaviour(self):
        import tempfile

        import torch

        from eval import load_policy_manager, run_game

        config = dict(BASE_CONFIG)
        env = get_env(config)
        args = get_args()
        args.device = "cpu"
        _, policies = build_policy_manager(env, args)
        agents = env.env.possible_agents

        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/policy.pth"
            torch.save({a: policies[a].state_dict() for a in agents}, path)

            saved = torch.load(path, map_location="cpu")
            self.assertEqual(sorted(saved), sorted(agents))
            for a in agents:
                self.assertGreater(len(saved[a]), 0, "重みが空のまま保存されている")

            restored = load_policy_manager(config, path)
            for a in agents:
                # 行動を決めるネットワーク（model）が厳密に一致すること。
                # model_old はターゲット用で行動選択に関与しない。
                original = policies[a].model.state_dict()
                loaded = restored.policies[a].model.state_dict()
                self.assertEqual(sorted(original), sorted(loaded))
                self.assertGreater(len(original), 0)
                for k in original:
                    self.assertTrue(torch.equal(original[k], loaded[k]), f"{a}.{k} が一致しない")

            # 同じ種で1ゲーム回すと同じ結果になる
            before = run_game(policy_manager=None, config=config, model_path=path,
                              seed=7, verbose=False)
            after = run_game(policy_manager=restored, config=config, seed=7, verbose=False)
            self.assertEqual(before["rewards"], after["rewards"])

    def test_loading_an_empty_model_raises(self):
        import tempfile

        import torch

        from eval import load_policy_manager

        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/empty.pth"
            torch.save({}, path)
            with self.assertRaises(ValueError):
                load_policy_manager(dict(BASE_CONFIG), path)


if __name__ == '__main__':
    unittest.main()
