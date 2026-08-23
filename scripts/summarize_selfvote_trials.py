"""result_n.txt から自己反対率・払われた罰・A の報酬を機械的に集計する。

手で表に転記すると取り違えが起きるため（第15ラウンドで実際に発生）、
レポートに載せる数値はこのスクリプトの出力を使う。

払われた罰 = 宝石の総数 − 報酬合計 − 死亡による損失
  （死亡損失 = Σ 死亡率 × L。死亡0なら単純に 総数 − 報酬合計）

使い方: python scripts/summarize_selfvote_trials.py 41 43 46 49 ...
"""
import re
import sys

FIELDS = {
    "config": re.compile(r"設定ファイル: (\S+)"),
    "L": re.compile(r"命の重さ\(ペナルティ L\): (\S+)"),
    "weights": re.compile(r"権力ウェイト: (\[.*\])"),
    "seed": re.compile(r"乱数シード: (\d+)"),
    "gems": re.compile(r"宝石の総数: (\d+)"),
    "selfvote": re.compile(r"自分の提案に反対した割合: ([\d.]+)%"),
}
AGENT = re.compile(r"- (agent_\w+): 平均報酬 ([+-][\d.]+) / 死亡率 ([\d.]+)%")


def summarize(n):
    path = f"result/result_{n}.txt"
    try:
        text = open(path, encoding="utf-8").read()
    except FileNotFoundError:
        return None

    out = {"n": n}
    for key, pat in FIELDS.items():
        m = pat.search(text)
        out[key] = m.group(1) if m else "-"

    rewards, deaths = {}, {}
    for name, rew, death in AGENT.findall(text):
        rewards[name] = float(rew)
        deaths[name] = float(death) / 100.0

    out["rew_A"] = rewards.get("agent_A")
    total = sum(rewards.values())
    gems = float(out["gems"]) if out["gems"] != "-" else 0.0
    L = float(out["L"]) if out["L"] != "-" else 0.0
    death_loss = sum(deaths.values()) * L
    out["penalty"] = gems - total - death_loss
    out["death_loss"] = death_loss
    return out


def main(ns):
    print(f"{'試行':>4} {'L':>5} {'種':>4} {'自己反対':>8} {'罰':>7} {'A報酬':>7}  重み / 設定")
    for n in ns:
        r = summarize(n)
        if r is None:
            print(f"{n:>4}  (result_{n}.txt がありません)")
            continue
        rew = f"{r['rew_A']:+.2f}" if r["rew_A"] is not None else "  -  "
        note = "" if r["death_loss"] == 0 else f"  ※死亡損 {r['death_loss']:.2f}"
        print(
            f"{r['n']:>4} {r['L']:>5} {r['seed']:>4} {r['selfvote']:>7}% "
            f"{r['penalty']:>7.2f} {rew:>7}  {r['weights']}{note}"
        )


if __name__ == "__main__":
    args = sys.argv[1:]
    main([int(a) for a in args] if args else list(range(37, 63)))
