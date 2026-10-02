"""指定した値が、どの試行のどのエージェントの平均報酬かを探す。

レポートに書かれた数値の出どころを特定するために使う
（`scripts/audit_report_numbers.py` が要確認として挙げた数値の追跡用）。

使い方: python scripts/find_value_in_results.py +2.74 +1.84
"""
import glob
import re
import sys

AGENT_RE = re.compile(r"- (agent_\w): 平均報酬 ([+-][\d.]+) / 死亡率 ([\d.]+)%")
TRIAL_RE = re.compile(r"result_(\d+)\.txt$")


def main():
    targets = [float(a) for a in sys.argv[1:]]
    if not targets:
        print(__doc__)
        return 1

    for target in targets:
        print(f"\n=== {target:+.2f} に一致する平均報酬 ===")
        hits = []
        for path in sorted(glob.glob("result/result_*.txt")):
            trial = int(TRIAL_RE.search(path).group(1))
            with open(path, encoding="utf-8") as f:
                text = f.read()
            for agent, reward, death in AGENT_RE.findall(text):
                if abs(float(reward) - target) < 0.005:
                    hits.append((trial, agent, float(reward), float(death)))
        if not hits:
            print("  該当なし")
        for trial, agent, reward, death in sorted(hits):
            print(f"  試行{trial:>4} {agent}: {reward:+.2f}（死亡率 {death}%）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
