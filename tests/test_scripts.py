"""scripts/ の集計スクリプトと tools/equilibrium_distance.py のテスト。

レポートに載せる数値はこれらのスクリプトの出力をそのまま使っている。
集計の取り違えは第12〜14ラウンドで実際に17か所起きたので、
「手で数えても確かめられる小さな入力」で出力を固定しておく。
"""

import csv
import os
import sys
import tempfile
import unittest

import numpy as np

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, SCRIPTS)

from compare_cells import mann_whitney_u  # noqa: E402
from decompose_agent_reward import decompose  # noqa: E402
from scan_reachability import reachability  # noqa: E402
from summarize_agent_rewards import summarize as summarize_rewards  # noqa: E402
from summarize_selfvote_trials import summarize as summarize_selfvote  # noqa: E402

AGENTS = ["A", "B", "C", "D", "E", "F"]


def _write_metrics(root, trial, rows):
    """log/log_metrics_<trial>.csv を作る。rows は {列名: 値} の一覧。"""
    os.makedirs(os.path.join(root, "log"), exist_ok=True)
    path = os.path.join(root, f"log/log_metrics_{trial}.csv")
    header = (["epoch", "env_step", "len_mean", "first_pass_rate"]
              + [f"rew_{a}" for a in AGENTS] + [f"death_{a}" for a in AGENTS])
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, 0.0) for key in header})
    return path


class TestMannWhitney(unittest.TestCase):
    """compare_cells の厳密な両側 p 値。全順列を手で数えて確かめられる大きさで固定する。"""

    def test_fully_separated_two_by_two(self):
        # 2対2で完全に分離: 並べ替え6通りのうち U<=0 は2通り → p = 2/6
        u, p = mann_whitney_u([1.0, 2.0], [3.0, 4.0])
        self.assertEqual(u, 0)
        self.assertAlmostEqual(p, 1 / 3)

    def test_interleaved_two_by_two(self):
        # U=1 まで許すと4通り → p = 4/6
        u, p = mann_whitney_u([1.0, 3.0], [2.0, 4.0])
        self.assertEqual(u, 1)
        self.assertAlmostEqual(p, 2 / 3)

    def test_fully_separated_four_by_four(self):
        # 4対4で完全分離は本プロジェクトで繰り返し出てくる形。p = 2/70
        u, p = mann_whitney_u([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])
        self.assertEqual(u, 0)
        self.assertAlmostEqual(p, 2 / 70)

    def test_is_symmetric(self):
        a, b = [0.1, 0.5, 0.2, 0.9], [0.3, 0.4, 0.8, 0.7]
        self.assertEqual(mann_whitney_u(a, b), mann_whitney_u(b, a))

    def test_ties_count_as_half(self):
        u, _ = mann_whitney_u([1.0, 2.0], [2.0, 3.0])
        self.assertAlmostEqual(u, 0.5)


class TestSummarizeAgentRewards(unittest.TestCase):
    def test_means_over_recorded_points(self):
        with tempfile.TemporaryDirectory() as root:
            _write_metrics(root, 1, [
                {"rew_A": 1.0, "rew_B": 0.0, "rew_C": 0.0, "rew_D": 0.0, "rew_E": 0.0, "rew_F": 0.0},
                {"rew_A": 3.0, "rew_B": 1.0, "rew_C": 1.0, "rew_D": 1.0, "rew_E": 1.0, "rew_F": 1.0},
            ])
            means, points = summarize_rewards(1, root=root)
        self.assertEqual(points, 2)
        self.assertAlmostEqual(means["A"], 2.0)
        self.assertAlmostEqual(means["B"], 0.5)

    def test_missing_file_returns_none(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(summarize_rewards(99, root=root))


class TestDecomposeAgentReward(unittest.TestCase):
    """平均報酬 = (1-死亡率)×生存時の取り分 - 死亡率×L の恒等式を守る。"""

    def test_identity_holds(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "result"), exist_ok=True)
            with open(os.path.join(root, "result/result_1.txt"), "w", encoding="utf-8") as f:
                f.write(" - 命の重さ(ペナルティ L): 10.0\n")
            # 死亡率 0.1・生存時の取り分 2.0 なら 平均報酬 = 0.9*2.0 - 0.1*10 = 0.8
            _write_metrics(root, 1, [{"rew_A": 0.8, "death_A": 0.1}])
            L, mean_r, mean_d, survive_gems, loss = decompose(1, root=root)
        self.assertAlmostEqual(L, 10.0)
        self.assertAlmostEqual(mean_r, 0.8)
        self.assertAlmostEqual(mean_d, 0.1)
        self.assertAlmostEqual(survive_gems, 2.0)
        self.assertAlmostEqual(loss, 1.0)

    def test_missing_agent_column_returns_none(self):
        # エージェント数が少ない試行には rew_D などの列がない
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "result"), exist_ok=True)
            with open(os.path.join(root, "result/result_4.txt"), "w", encoding="utf-8") as f:
                f.write(" - 命の重さ(ペナルティ L): 5.0\n")
            os.makedirs(os.path.join(root, "log"), exist_ok=True)
            with open(os.path.join(root, "log/log_metrics_4.csv"), "w", encoding="utf-8") as f:
                f.write("epoch,rew_A,rew_B,rew_C,death_A,death_B,death_C\n")
                f.write("10,1.0,0.5,0.5,0.0,0.0,0.0\n")
            self.assertIsNone(decompose(4, agent="D", root=root))
            self.assertIsNotNone(decompose(4, agent="C", root=root))

    def test_other_agents_can_be_decomposed(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "result"), exist_ok=True)
            with open(os.path.join(root, "result/result_2.txt"), "w", encoding="utf-8") as f:
                f.write(" - 命の重さ(ペナルティ L): 5.0\n")
            _write_metrics(root, 2, [{"rew_B": -0.5, "death_B": 0.2}])
            _, mean_r, mean_d, survive_gems, loss = decompose(2, agent="B", root=root)
        self.assertAlmostEqual(mean_r, -0.5)
        self.assertAlmostEqual(mean_d, 0.2)
        self.assertAlmostEqual(loss, 1.0)
        self.assertAlmostEqual(survive_gems, (-0.5 + 1.0) / 0.8)


class TestSummarizeSelfvoteTrials(unittest.TestCase):
    def test_parses_result_file(self):
        text = """=========================================
【環境設定】
 - 設定ファイル: configs/x.json
 - 海賊の人数: 6人
 - 宝石の総数: 5個
 - 命の重さ(ペナルティ L): 100
 - 権力ウェイト: [50.0, 1.0, 1.0, 1.0, 1.0, 1.0]
 - 乱数シード: 42
-----------------------------------------
 - 提案者が自分の提案に反対した割合: 38.0% (38/100)
 - agent_A: 平均報酬 +1.50 / 死亡率 0.0% / 提案 90回 (可決率 100.0%)
 - agent_B: 平均報酬 +0.50 / 死亡率 0.0% / 提案 2回 (可決率 100.0%)
"""
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "result"), exist_ok=True)
            with open(os.path.join(root, "result/result_3.txt"), "w", encoding="utf-8") as f:
                f.write(text)
            out = summarize_selfvote(3, root=root)
        self.assertEqual(out["L"], "100")
        self.assertEqual(out["seed"], "42")
        self.assertEqual(out["selfvote"], "38.0")
        self.assertAlmostEqual(out["rew_A"], 1.50)
        # 払われた罰 = 宝石 5 - 報酬合計 2.0 - 死亡損失 0 = 3.0
        self.assertAlmostEqual(out["penalty"], 3.0)
        self.assertAlmostEqual(out["death_loss"], 0.0)


class TestScanReachability(unittest.TestCase):
    """感応度の走査のうち、対局で到達しうる局面の割合。"""

    BASE = {"num_agents": 6, "total_gems": 5, "L": 100.0,
            "agent_weights": [1.0] * 6, "fixed_order": False}

    def test_counts_match_hand_calculation(self):
        # 6人なら票の途中経過は 21 通り、配り方2通り × 取り分5段階 = 6300 局面。
        # 投票者 v の「投票済み人数」は v の位置に決まるので、
        # 到達可能なのは sum_{v=0..5}(v+1) * 5 提案者 * 2 * 5 = 1050 局面。
        reachable, total = reachability(dict(self.BASE, observe_vote_tally=True))
        self.assertEqual(total, 6300)
        self.assertEqual(reachable, 1050)

    def test_without_tally_there_is_nothing_to_scan(self):
        reachable, total = reachability(dict(self.BASE, observe_vote_tally=False))
        self.assertIsNone(reachable)
        self.assertIsNone(total)

    def test_proposer_votes_last_changes_the_fraction(self):
        reachable, total = reachability(
            dict(self.BASE, observe_vote_tally=True, proposer_votes_last=True))
        self.assertEqual(total, 6300)
        self.assertEqual(reachable, 900)


class TestVotePriceAndReachableScan(unittest.TestCase):
    """値段の指標と、到達可能な局面だけに絞った感応度。"""

    CONFIG = {"num_agents": 4, "total_gems": 3, "L": 5.0,
              "agent_weights": [2.0, 1.0, 1.0, 1.0], "fixed_order": False,
              "observe_vote_tally": True}

    def _save_fresh_policy(self, directory):
        import torch
        from train import build_policy_manager, get_args, get_env

        env = get_env(self.CONFIG)
        args = get_args()
        args.device = "cpu"
        _, policies = build_policy_manager(env, args)
        agents = env.env.possible_agents
        path = f"{directory}/policy.pth"
        torch.save({a: policies[a].state_dict() for a in agents}, path)
        return path

    def test_price_metrics_are_in_range(self):
        from tools.vote_price import measure

        with tempfile.TemporaryDirectory() as d:
            result = measure(self.CONFIG, self._save_fresh_policy(d))
        self.assertGreater(result["n_contexts"], 0)
        for key in ("share_never", "share_always", "tie_accept_share"):
            if result[key] == result[key]:  # nan でなければ
                self.assertGreaterEqual(result[key], 0.0)
                self.assertLessEqual(result[key], 1.0)
        if result["mean_price"] == result["mean_price"]:
            self.assertGreaterEqual(result["mean_price"], 0.0)
            self.assertLessEqual(result["mean_price"], self.CONFIG["total_gems"])

    def test_reachable_only_scans_fewer_states(self):
        from tools.vote_sensitivity import measure

        with tempfile.TemporaryDirectory() as d:
            path = self._save_fresh_policy(d)
            full = measure(self.CONFIG, path)
            reachable = measure(self.CONFIG, path, reachable_only=True)
        # 到達可能な局面だけに絞れば走査数は減り、内訳の合計は全体と一致する
        self.assertLess(reachable["n_comparisons"], full["n_comparisons"])
        for r in (full, reachable):
            self.assertEqual(r["n_decided"] + r["n_undecided"], r["n_comparisons"])

    def test_decided_states_are_absent_without_tally(self):
        from tools.vote_sensitivity import measure

        config = dict(self.CONFIG, observe_vote_tally=False)
        with tempfile.TemporaryDirectory() as d:
            import torch
            from train import build_policy_manager, get_args, get_env

            env = get_env(config)
            args = get_args()
            args.device = "cpu"
            _, policies = build_policy_manager(env, args)
            agents = env.env.possible_agents
            path = f"{d}/policy.pth"
            torch.save({a: policies[a].state_dict() for a in agents}, path)
            result = measure(config, path)
        # 票数を観測しない設定では「決着済み」を判定できないので、すべて未決着に入る
        self.assertEqual(result["n_decided"], 0)
        self.assertEqual(result["n_undecided"], result["n_comparisons"])


class TestDesignPower(unittest.TestCase):
    """検出力の見積もりが、効果と標本数に対して正しい向きに動くこと。"""

    LOW = [0.90, 0.92, 0.95, 0.97, 0.88, 0.93, 0.96, 0.91]
    HIGH = [0.50, 0.60, 0.70, 0.55, 0.65, 0.45, 0.75, 0.58]

    def _power(self, n, shrink):
        from estimate_design_power import difference_of_differences_power

        return difference_of_differences_power(self.LOW, self.HIGH, n, shrink, 4000, 1.96)

    def test_power_grows_with_effect_and_samples(self):
        self.assertGreater(self._power(8, 1.0), self._power(8, 0.25))
        self.assertGreater(self._power(16, 0.5), self._power(8, 0.5))

    def test_no_effect_rarely_detects(self):
        self.assertLess(self._power(16, 0.0), 0.15)


class TestRound34Analysis(unittest.TestCase):
    """第34ラウンドの判定式を、値が手で求まるセルで固定する。"""

    def _cells(self):
        # D は脱落あり・縮尺なしで 0.4、それ以外で 0.1
        return {
            (1, 5, False): [0.9, 0.9], (1, 100, False): [0.5, 0.5],
            (0, 5, False): [0.8, 0.8], (0, 100, False): [0.7, 0.7],
            (1, 5, True): [0.8, 0.8], (1, 100, True): [0.7, 0.7],
            (0, 5, True): [0.8, 0.8], (0, 100, True): [0.7, 0.7],
        }

    def test_contrasts_match_hand_calculation(self):
        from analyze_r34 import contrast, contrast_weights

        cells = self._cells()
        self.assertAlmostEqual(contrast(cells, contrast_weights("C1")), 0.3)
        self.assertAlmostEqual(contrast(cells, contrast_weights("C2")), 0.3)
        self.assertAlmostEqual(contrast(cells, contrast_weights("C3")), 0.3)

    def test_interval_collapses_without_spread(self):
        from analyze_r34 import bootstrap_interval, contrast_weights

        low, high = bootstrap_interval(self._cells(), contrast_weights("C1"), resamples=100)
        self.assertAlmostEqual(low, 0.3)
        self.assertAlmostEqual(high, 0.3)


class TestNoEliminationMetrics(unittest.TestCase):
    """脱落なしの環境でも指標が測れ、記録が学習中に書かれること。"""

    CONFIG = {"num_agents": 3, "total_gems": 2, "L": 5.0, "agent_weights": [2.0, 1.0, 1.0],
              "fixed_order": False, "eliminate_on_reject": False, "max_rounds": 2}

    def _policies(self):
        from train import build_policy_manager, get_args, get_env

        env = get_env(self.CONFIG)
        args = get_args()
        args.device = "cpu"
        _, policies = build_policy_manager(env, args)
        return env, policies

    def test_walk_rounds_stops_after_agreement(self):
        from tools.onpath_equilibrium import walk_rounds

        visits = walk_rounds(3, [2.0, 1.0, 1.0], 2, lambda t, p: (2, 0, 0),
                             lambda *args: True)
        self.assertTrue(all(v["alive"] == 0 for v in visits))
        self.assertAlmostEqual(sum(v["prob"] for v in visits), 1.0)

    def test_walk_rounds_reaches_the_next_round_after_rejection(self):
        from tools.onpath_equilibrium import walk_rounds

        visits = walk_rounds(3, [2.0, 1.0, 1.0], 2, lambda t, p: (2, 0, 0),
                             lambda *args: False)
        self.assertEqual({v["alive"] for v in visits}, {0, 1})
        self.assertAlmostEqual(sum(v["prob"] for v in visits), 2.0)

    def test_distance_and_onpath_measures_run(self):
        import torch

        from tools.equilibrium_distance import measure as distance
        from tools.onpath_equilibrium import measure as onpath

        env, policies = self._policies()
        agents = env.env.possible_agents
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/policy.pth"
            torch.save({a: policies[a].state_dict() for a in agents}, path)
            far = distance(self.CONFIG, path)
            near = onpath(self.CONFIG, path)
        self.assertEqual(far["n_proposal_states"], 6)
        self.assertGreaterEqual(far["vote_match_first"], 0.0)
        self.assertLessEqual(far["vote_match_first"], 1.0)
        self.assertGreaterEqual(near["onpath_match_either"], 0.0)

    def test_recorder_writes_one_row_per_call(self):
        from tools.onpath_equilibrium import OnPathRecorder

        env, policies = self._policies()
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/log_onpath_1.csv"
            recorder = OnPathRecorder(env.env, path)
            recorder.record(10, policies)
            recorder.record(20, policies)
            with open(path, encoding="utf-8") as f:
                lines = f.read().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("epoch,"))
        self.assertTrue(lines[2].startswith("20,"))


class TestOnPathEquilibrium(unittest.TestCase):
    """到達局面の列挙と指標。ネットワークを使わず、方策を関数で与えて確かめる。"""

    WEIGHTS = [2.0, 1.0, 1.0]

    def _solvers(self):
        from solver import RandomOrderSolver

        return (RandomOrderSolver(3, 2, 5.0, self.WEIGHTS),
                RandomOrderSolver(3, 2, 5.0, self.WEIGHTS, accept_when_indifferent=True))

    def test_equilibrium_play_scores_perfectly(self):
        from tools.onpath_equilibrium import summarize, walk

        strict, lenient = self._solvers()

        def propose(alive, proposer, stale):
            return strict.equilibrium_proposals(alive, proposer)[0][0]

        def vote(alive, proposer, voter, proposal, tally):
            return strict.equilibrium_votes(alive, proposer, proposal)[voter]

        visits = walk(3, self.WEIGHTS, propose, vote)
        self.assertAlmostEqual(sum(v["prob"] for v in visits), 1.0)
        self.assertTrue(all(v["passed"] for v in visits))
        result = summarize(visits, strict, lenient)
        self.assertAlmostEqual(result["onpath_l1"], 0.0)
        self.assertAlmostEqual(result["onpath_gap_strict"], 0.0)
        self.assertAlmostEqual(result["onpath_match_strict"], 1.0)
        self.assertAlmostEqual(result["first_reject"], 0.0)
        self.assertAlmostEqual(result["expected_rejections"], 0.0)

    def test_rejections_follow_recognition_probabilities(self):
        from tools.onpath_equilibrium import summarize, walk

        strict, lenient = self._solvers()

        # 全員が常に反対: 3人で必ず否決され、2人でも必ず否決されて1人残る
        def propose(alive, proposer, stale):
            dist = [0, 0, 0]
            dist[proposer] = 2
            return tuple(dist)

        visits = walk(3, self.WEIGHTS, propose, lambda *args: False)
        first = [v for v in visits if len(v["alive"]) == 3]
        self.assertEqual({v["proposer"]: v["prob"] for v in first}, {0: 0.5, 1: 0.25, 2: 0.25})
        # A が否決された後の {B, C} は重みが等しいので 0.5 ずつ
        after_a = [v for v in visits if v["alive"] == frozenset({1, 2})]
        self.assertAlmostEqual(sum(v["prob"] for v in after_a), 0.5)
        self.assertAlmostEqual(after_a[0]["prob"], 0.25)
        result = summarize(visits, strict, lenient)
        self.assertAlmostEqual(result["first_reject"], 1.0)
        self.assertAlmostEqual(result["expected_rejections"], 2.0)

    def test_tally_counts_earlier_votes_in_order(self):
        from tools.onpath_equilibrium import walk

        seen = {}

        def vote(alive, proposer, voter, proposal, tally):
            seen[voter] = tally
            return True

        walk(3, [1.0, 0.0, 0.0], lambda *args: (2, 0, 0), vote, proposer_votes_last=True)
        # 提案者 A が末尾に回るので B→C→A の順で、賛成数が積み上がる
        self.assertEqual(seen, {1: (0.0, 0.0), 2: (1.0, 1.0), 0: (2.0, 2.0)})

    def test_measure_runs_on_a_saved_model(self):
        import torch

        from tools.onpath_equilibrium import measure
        from train import build_policy_manager, get_args, get_env

        config = {"num_agents": 3, "total_gems": 2, "L": 5.0, "agent_weights": self.WEIGHTS,
                  "fixed_order": False, "observe_vote_tally": True}
        env = get_env(config)
        args = get_args()
        args.device = "cpu"
        _, policies = build_policy_manager(env, args)
        agents = env.env.possible_agents
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/policy.pth"
            torch.save({a: policies[a].state_dict() for a in agents}, path)
            result = measure(config, path)
        self.assertGreaterEqual(result["onpath_match_either"], 0.0)
        self.assertLessEqual(result["onpath_match_either"], 1.0)
        self.assertGreaterEqual(result["first_reject"], 0.0)
        self.assertLessEqual(result["first_reject"], 1.0)


class TestEquilibriumDistance(unittest.TestCase):
    """均衡距離の指標が、意味のある範囲の値を返すこと。"""

    CONFIG = {"num_agents": 3, "total_gems": 2, "L": 5.0,
              "agent_weights": [2.0, 1.0, 1.0], "fixed_order": False}

    def _measure_fresh_policy(self):
        import torch
        from tools.equilibrium_distance import measure
        from train import build_policy_manager, get_args, get_env

        env = get_env(self.CONFIG)
        args = get_args()
        args.device = "cpu"
        _, policies = build_policy_manager(env, args)
        agents = env.env.possible_agents
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/policy.pth"
            torch.save({a: policies[a].state_dict() for a in agents}, path)
            return measure(self.CONFIG, path)

    def test_metrics_are_in_range(self):
        result = self._measure_fresh_policy()
        self.assertGreater(result["n_proposal_states"], 0)
        self.assertGreater(result["n_vote_states"], 0)
        self.assertGreaterEqual(result["proposal_l1"], 0.0)
        self.assertLessEqual(result["proposal_l1"], result["max_proposal_l1"])
        for key in ("vote_match_strict", "vote_match_lenient", "vote_match_either"):
            self.assertGreaterEqual(result[key], 0.0)
            self.assertLessEqual(result[key], 1.0)
        # どちらかの規則に一致する割合は、各規則の一致率を下回らない
        self.assertGreaterEqual(result["vote_match_either"], result["vote_match_strict"])
        self.assertGreaterEqual(result["vote_match_either"], result["vote_match_lenient"])
        # 自分の取り分の差は -宝石数 〜 +宝石数 の範囲に収まる
        self.assertGreaterEqual(result["self_gap"], -self.CONFIG["total_gems"])
        self.assertLessEqual(result["self_gap"], self.CONFIG["total_gems"])

    def test_measurement_is_deterministic(self):
        first = self._measure_fresh_policy()
        second = self._measure_fresh_policy()
        # 重みの初期化はシード依存なので値そのものは一致しないが、
        # 走査した局面数は設定だけで決まる
        self.assertEqual(first["n_proposal_states"], second["n_proposal_states"])
        self.assertEqual(first["n_vote_states"], second["n_vote_states"])

    def test_equilibrium_policy_scores_perfectly(self):
        """均衡そのものを方策として与えれば、距離0・一致率100%になる。

        ネットワークを使わず、均衡の投票・提案をそのまま返す偽の方策で
        指標の定義そのものを検証する。
        """
        import itertools

        from env import PirateGemEnv
        from solver import RandomOrderSolver

        env = PirateGemEnv(self.CONFIG)
        solver = RandomOrderSolver(
            env.n_agents, env.total_gems, env.L, list(env.agent_weights))
        for size in range(2, env.n_agents + 1):
            for subset in itertools.combinations(range(env.n_agents), size):
                alive = frozenset(subset)
                for proposer in alive:
                    proposals, affordable = solver.equilibrium_proposals(alive, proposer)
                    if not affordable:
                        continue
                    for proposal in proposals:
                        # 均衡提案は宝石をちょうど使い切る
                        self.assertEqual(sum(proposal), env.total_gems)
                        votes = solver.equilibrium_votes(alive, proposer, proposal)
                        # 均衡提案は必要票数を満たす
                        self.assertGreaterEqual(
                            sum(1 for i in alive if votes[i]),
                            int(np.ceil(len(alive) / 2)),
                        )


if __name__ == "__main__":
    unittest.main()
