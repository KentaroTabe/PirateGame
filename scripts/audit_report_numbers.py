"""レポートに書かれた数値を、生データから計算した値と機械的に突合する。

第12〜14ラウンドで、ある試行の値を別の試行のものとして17か所に書く転記ミスが起きた
（docs/methodology.md 6.1）。同じ誤りが他に残っていないかを人手で探すのは割に合わないので、
「試行番号が書かれた行に出てくる小数が、その試行の実際の値のどれにも一致しない」場合を
**要確認として列挙する**。

突合する値の出どころ:
    - `result/result_N.txt`: L・宝石数・シード・エージェント別の平均報酬・死亡率・自己反対率・平均提案回数
    - `log/log_metrics_N.csv`: 全期間平均・死亡率・A優位度・生存時の取り分・死亡による損失
    - `log/equilibrium_distance.csv`（あれば）: 均衡距離の各指標
    - `log/vote_sensitivity.csv`（あれば）: 投票の取り分感応度

**これは「誤りの一覧」ではなく「確認すべき箇所の一覧」である。**
別の量（p 値・エポック数・割合の合計など）を書いた行も拾うので、1件ずつ目で見る必要がある。

使い方:
    python scripts/audit_report_numbers.py
    python scripts/audit_report_numbers.py --docs docs/reports/round14.md --tolerance 0.01
"""
import argparse
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decompose_agent_reward import decompose  # noqa: E402
from summarize_agent_rewards import summarize as summarize_rewards  # noqa: E402
from summarize_selfvote_trials import summarize as summarize_selfvote  # noqa: E402

AGENTS = ["A", "B", "C", "D", "E", "F"]
# 「試行11・13」「試行7・8・10・11」のように番号を並べる書き方も拾う
TRIAL_RE = re.compile(r"試行\s*\d+(?:\s*[・,、/]\s*\d+)*")
TRIAL_NUMBER_RE = re.compile(r"\d+")
NUMBER_RE = re.compile(r"[+-]?\d+\.\d{2,4}")
# p 値・U 統計量・章番号などは突合の対象にしない
SKIP_PREFIX_RE = re.compile(r"(p\s*=|U\s*=|n\s*=|第|ラウンド|%\))\s*$")

RESULT_PATTERNS = {
    "L": re.compile(r"命の重さ\(ペナルティ L\): (\S+)"),
    "gems": re.compile(r"宝石の総数: (\d+)"),
}
RESULT_AGENT_RE = re.compile(
    r"- agent_(\w): 平均報酬 ([+-][\d.]+) / 死亡率 ([\d.]+)% / 提案 (\d+)回")
SELFVOTE_RE = re.compile(r"自分の提案に反対した割合: ([\d.]+)%")
PROPOSALS_RE = re.compile(r"平均提案回数: ([\d.]+)")


def canonical_values(trial):
    """その試行について、レポートに書かれうる値の集合。"""
    values = set()

    path = f"result/result_{trial}.txt"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for pattern in RESULT_PATTERNS.values():
            m = pattern.search(text)
            if m:
                values.add(float(m.group(1)))
        for _, reward, death, proposals in RESULT_AGENT_RE.findall(text):
            values.add(float(reward))
            values.add(float(death))
            values.add(float(death) / 100.0)
            values.add(float(proposals))
        for pattern in (SELFVOTE_RE, PROPOSALS_RE):
            for hit in pattern.findall(text):
                values.add(float(hit))
                values.add(float(hit) / 100.0)

    summary = summarize_rewards(trial)
    if summary is not None:
        means, _ = summary
        for agent, mean in means.items():
            values.add(round(mean, 4))
        others = [means[a] for a in AGENTS if a != "A" and a in means]
        if others:
            values.add(round(means["A"] - sum(others) / len(others), 4))
    for agent in AGENTS:
        row = decompose(trial, agent=agent)
        if row is None:
            continue
        _, mean_r, mean_d, survive, loss = row
        values.update(round(v, 4) for v in (mean_r, mean_d, mean_d * 100, survive, loss))

    # レポートでは「死亡なし記録点の平均」と「払われた罰」も引用している
    metrics_path = f"log/log_metrics_{trial}.csv"
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for agent in AGENTS:
            key, death_key = f"rew_{agent}", f"death_{agent}"
            if not rows or key not in rows[0]:
                continue
            alive_rows = [r for r in rows if float(r[death_key]) == 0.0]
            if alive_rows:
                values.add(round(sum(float(r[key]) for r in alive_rows) / len(alive_rows), 4))
    selfvote = summarize_selfvote(trial)
    if selfvote is not None and selfvote.get("penalty") is not None:
        values.add(round(selfvote["penalty"], 4))
        values.add(round(selfvote["death_loss"], 4))

    for csv_path, keys in (
        ("log/equilibrium_distance.csv",
         ["proposal_l1", "self_gap", "coalition_gap",
          "vote_match_strict", "vote_match_lenient", "vote_match_either"]),
        ("log/vote_sensitivity.csv", ["sensitivity"]),
    ):
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if int(row["trial"]) != trial:
                    continue
                for key in keys:
                    if row.get(key):
                        values.add(round(float(row[key]), 4))
                        values.add(round(float(row[key]) * 100, 4))
    return values


def matches(number, values, tolerance):
    return any(abs(number - value) <= tolerance for value in values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="*", default=None, help="対象のドキュメント（既定は docs 配下すべて）")
    ap.add_argument("--tolerance", type=float, default=0.0051,
                    help="一致とみなす絶対誤差（既定はレポートの丸め幅）")
    args = ap.parse_args()

    paths = args.docs or sorted(glob.glob("docs/**/*.md", recursive=True))
    cache = {}
    flagged = 0
    checked = 0

    for path in paths:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        for lineno, line in enumerate(lines, start=1):
            trials = [
                int(number)
                for group in TRIAL_RE.findall(line)
                for number in TRIAL_NUMBER_RE.findall(group)
            ]
            if not trials:
                continue
            values = set()
            for trial in trials:
                if trial not in cache:
                    cache[trial] = canonical_values(trial)
                values |= cache[trial]
            if not values:
                continue
            unmatched = []
            for m in NUMBER_RE.finditer(line):
                if SKIP_PREFIX_RE.search(line[max(0, m.start() - 8):m.start()]):
                    continue
                checked += 1
                number = float(m.group())
                if not matches(number, values, args.tolerance):
                    unmatched.append(m.group())
            if unmatched:
                flagged += 1
                print(f"{path}:{lineno} 試行{trials} 一致しない数値: {', '.join(unmatched)}")
                print(f"    {line.strip()[:160]}")

    print(f"\n突合した数値 {checked} 件 / 要確認の行 {flagged} 件")
    print("別の量（p 値・割合の合計・予測値など）を書いた行も拾うので、1件ずつ確認が必要。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
