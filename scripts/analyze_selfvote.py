"""評価ログを解析し、提案者の自己投票と可決マージン(超過賛成票)を集計する。

使い方: python analyze_selfvote.py log/log_eval_37.txt
"""
import re
import sys
from collections import Counter

path = sys.argv[1]

prop_re = re.compile(r"=> (agent_\w+) は提案行動 \[ \[([\d, ]+)\] \]")
vote_re = re.compile(r"=> (agent_\w+) は (👍 賛成|👎 反対)")
judge_re = re.compile(r"\[判定\] 賛成: (\d+) / 生存者: (\d+)")

proposer = None
alloc = None
votes = {}

self_no = []           # 提案者が自分の提案に反対した事例
margins = []           # (賛成票, 必要票, 超過分)
proposer_take = Counter()
proposer_n = Counter()
coalition_sizes = []

with open(path) as f:
    for line in f:
        m = prop_re.search(line)
        if m:
            proposer = m.group(1)
            alloc = [int(x) for x in m.group(2).split(",")]
            votes = {}
            continue

        m = vote_re.search(line)
        if m:
            votes[m.group(1)] = (m.group(2) == "👍 賛成")
            continue

        m = judge_re.search(line)
        if m and proposer is not None:
            yes, alive = int(m.group(1)), int(m.group(2))
            need = (alive + 1) // 2
            if votes.get(proposer) is False:
                self_no.append((proposer, alloc, yes, alive))
            if yes >= need:
                margins.append((yes, need, yes - need))
            idx = ord(proposer[-1]) - ord("A")
            if idx < len(alloc):
                proposer_take[proposer] += alloc[idx]
                proposer_n[proposer] += 1
            # 提案者が配った先(0個より多く受け取った他者)の人数
            coalition_sizes.append(sum(1 for i, v in enumerate(alloc)
                                       if v > 0 and i != idx))
            proposer = None

print("ファイル:", path)
print("提案の総数:", sum(proposer_n.values()))
print()
print("--- 提案者の自己反対 ---")
print("件数:", len(self_no))
for p, a, yes, alive in self_no:
    print("  {} 案={} 賛成{}/生存{}".format(p, a, yes, alive))
print()
print("--- 可決時の超過賛成票 ---")
if margins:
    exc = [m[2] for m in margins]
    print("可決数:", len(margins))
    print("超過票の平均:", round(sum(exc) / len(exc), 3))
    print("超過票の分布:", dict(sorted(Counter(exc).items())))
print()
print("--- 提案者が自分に取った宝石(平均) ---")
for p in sorted(proposer_n):
    print("  {}: {:.2f} ({}回)".format(
        p, proposer_take[p] / proposer_n[p], proposer_n[p]))
print()
print("--- 買収した人数の分布 ---")
print(dict(sorted(Counter(coalition_sizes).items())))
