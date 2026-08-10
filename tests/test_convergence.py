"""収束判定ロジック（tools.convergence）のユニットテスト。"""

import csv
import os
import tempfile
import unittest

from tools.convergence import analyze, load_settings

AGENTS = ["A", "B"]

# 実際に使う設定でテストする（テスト用に別の閾値を持たない）
SETTINGS = load_settings("configs/convergence.json")


def write_metrics(path, rows):
    """rows: [(epoch, first_pass_rate, rew_A, rew_B, death_A, death_B), ...]"""
    header = ["epoch", "env_step", "len_mean", "first_pass_rate",
              "rew_A", "rew_B", "death_A", "death_B"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for epoch, first_pass, rew_a, rew_b, death_a, death_b in rows:
            writer.writerow([epoch, epoch * 100, 7.0, first_pass,
                             rew_a, rew_b, death_a, death_b])


class ConvergenceTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def path(self, name):
        return os.path.join(self.tmpdir.name, name)


class TestSettledRun(ConvergenceTestCase):
    """前半で体制が入れ替わり、後半は安定し続ける系列。"""

    def setUp(self):
        super().setUp()
        rows = []
        for i in range(100):
            epoch = (i + 1) * 100
            if epoch <= 3000:
                # 前半: B が首位、死亡も発生する荒れた時期
                rows.append((epoch, 0.5, 0.5, 3.0, 0.2, 0.0))
            else:
                # 後半: A が首位で完全に安定
                rows.append((epoch, 1.0, 3.0, 0.5, 0.0, 0.0))
        self.csv_path = self.path("settled.csv")
        write_metrics(self.csv_path, rows)
        self.result = analyze(self.csv_path, SETTINGS)

    def test_regime_converges_after_the_switch(self):
        self.assertIsNotNone(self.result["regime_epoch"])
        self.assertEqual(self.result["leader"], "A")
        # 移行期の平滑化ぶんだけ遅れるが、荒れた前半には収まらない
        self.assertGreater(self.result["regime_epoch"], 3000)
        self.assertLess(self.result["regime_epoch"], 5000)

    def test_distribution_converges(self):
        self.assertIsNotNone(self.result["reward_epoch"])
        self.assertGreater(self.result["reward_epoch"], 3000)

    def test_order_converges_after_deaths_stop(self):
        self.assertIsNotNone(self.result["order_epoch"])
        self.assertGreater(self.result["order_epoch"], 3000)

    def test_death_rows_are_excluded_from_distribution(self):
        # 前半30点は死亡ありなので除外される
        self.assertEqual(self.result["peaceful_points"], 70)
        self.assertEqual(self.result["total_points"], 100)


class TestCyclingRun(ConvergenceTestCase):
    """最後まで首位が入れ替わり続ける系列は未収束と判定されること。"""

    def setUp(self):
        super().setUp()
        rows = []
        for i in range(100):
            epoch = (i + 1) * 100
            # 20点ごとに首位が入れ替わる（最終盤でも交代が起きる）
            if (i // 20) % 2 == 0:
                rows.append((epoch, 1.0, 3.0, 0.5, 0.0, 0.0))
            else:
                rows.append((epoch, 1.0, 0.5, 3.0, 0.0, 0.0))
        self.csv_path = self.path("cycling.csv")
        write_metrics(self.csv_path, rows)
        self.result = analyze(self.csv_path, SETTINGS)

    def test_regime_never_converges(self):
        self.assertIsNone(self.result["regime_epoch"])

    def test_distribution_never_converges(self):
        self.assertIsNone(self.result["reward_epoch"])


class TestUnstableOrder(ConvergenceTestCase):
    """死亡が最後まで続く系列は秩序未収束と判定されること。"""

    def setUp(self):
        super().setUp()
        rows = []
        for i in range(100):
            epoch = (i + 1) * 100
            # 10点ごとに死亡が発生し続ける
            death = 0.3 if i % 10 == 0 else 0.0
            rows.append((epoch, 0.6 if death else 1.0, 1.5, 1.5, death, 0.0))
        self.csv_path = self.path("unstable.csv")
        write_metrics(self.csv_path, rows)
        self.result = analyze(self.csv_path, SETTINGS)

    def test_order_never_converges(self):
        self.assertIsNone(self.result["order_epoch"])


if __name__ == "__main__":
    unittest.main()
