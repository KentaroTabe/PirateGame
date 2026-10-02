"""投票が自分の取り分にどれだけ反応するかを、細かい連続量として測る。

`tools/probe_votes.py` の指標は「5体中いくつが取り分に一切反応しないか」という
**0〜5 の整数**で、第21ラウンドで同一条件のシード間に 2〜5 の幅があると分かった。
n=4 では平均差 1.0 程度を分離できない（docs/reports/round21.md）。

本ツールはもっと細かい指標を返す。**多数の局面を走査し、
「自分の取り分を1個増やしたとき、賛成する側に動いた割合」**を測る。

    感応度 = （取り分を増やすと賛成に転じた回数 - 反対に転じた回数）/ 走査した局面数

- +1.0 に近いほど、取り分が増えれば賛成するという素直な反応
- 0.0 は取り分と投票が無関係
- 走査する局面数は 投票者数 × 分配案の刻み × 票の途中経過 の全組み合わせなので、
  probe_votes の 5 段階に対して数百段階の分解能になる

全エージェントを投票者として、提案者・分配の散らし方・票の途中経過を走査する。

使い方:
    python -m tools.vote_sensitivity <設定ファイル> <モデル>
"""
import json
import sys

import numpy as np
import torch
from tianshou.data import Batch

from env import PirateGemEnv
from eval import load_policy_manager
from pretrain import _build_observation


def measure(config, model_path, proposer="A", proposer_filter=None, rich_spread=False,
            reachable_only=False):
    """感応度と、比較に使った局面数を返す。

    proposer_filter に「提案者インデックス -> bool」を渡すと、
    走査する提案者をその部分集合に絞る。走査は決定的なので、
    互いに素な2つの部分集合で測れば**同一モデルに対する2つの独立な推定**が得られる
    （測定のばらつきとシード間のばらつきを切り分けるために使う。
    docs/methodology.md「感応度のばらつきの出どころ」を参照）。
    既定の None では全提案者を走査し、従来と同じ値を返す。

    rich_spread=True にすると、残りの宝石の配り方を
    「提案者が独占」「他者に均等」の2通りから
    「他者1人が総取り」の各パターンを加えたものに広げ、走査局面数を増やす。
    測定の精度を上げたいときに使う。**既定は False**（過去の全測定と同じ値を返す）。

    reachable_only=True にすると、票数観測ありの設定で、
    **対局で実際に現れる票の途中経過だけ**を走査する（投票順から投票済み人数が1通りに決まる）。
    既定の False では全組み合わせを走査するので、6体では局面の83%が
    学習中に一度も現れない観測になる（docs/discussion.md 6.1節）。
    どちらの場合も、戻り値には「勝敗が決着済みの局面」と「未決着の局面」に分けた
    感応度が入る（決着済みでは合理的な投票者は無差別なので、取り分に反応しなくてよい）。
    """
    env = PirateGemEnv(config)
    manager = load_policy_manager(config, model_path)
    policies = manager.policies
    agents = env.possible_agents
    p_idx = ord(proposer) - ord("A")

    alive = set(range(env.n_agents))
    n_dist = len(env.DISTRIBUTIONS)
    vote_mask = np.zeros(n_dist + 2, dtype=bool)
    vote_mask[env.ACTION_YES] = True
    vote_mask[env.ACTION_NO] = True

    # 走査する票の途中経過。票数観測が無効なら1通りで十分（観測に入らないため）。
    if env.observe_vote_tally:
        all_tallies = [
            [float(yes), float(voted)]
            for voted in range(env.n_agents)
            for yes in range(voted + 1)
        ]
    else:
        all_tallies = [None]

    required = int(np.ceil(env.n_agents / 2))

    def tallies_for(v_idx, prop_idx):
        """この投票者・この提案者のもとで走査する票の途中経過。

        reachable_only=True なら、投票順（env._build_voting_order() と同じ規則）から
        決まる「投票済み人数」だけに絞る。全員生存の局面を走査しているので、
        投票者の順番は提案者が誰かと proposer_votes_last だけで決まる。
        """
        if not env.observe_vote_tally or not reachable_only:
            return all_tallies
        order = list(range(env.n_agents))
        if env.proposer_votes_last:
            order.remove(prop_idx)
            order.append(prop_idx)
        voted = order.index(v_idx)
        return [[float(yes), float(voted)] for yes in range(voted + 1)]

    def is_decided(tally):
        """その時点で勝敗が決まっているか（決まっていれば合理的な投票者は無差別）。"""
        if tally is None:
            return False
        yes, voted = int(tally[0]), int(tally[1])
        if yes >= required:
            return True  # すでに可決が確定
        return (voted - yes) > env.n_agents - required  # 残り全員が賛成しても届かない

    def vote(agent, prop_idx, dist, tally):
        obs = _build_observation(env, alive, prop_idx, tuple(dist), vote_tally=tally)
        batch = Batch(obs=Batch(obs=obs[None, :], mask=vote_mask[None, :]), info={})
        with torch.no_grad():
            return int(policies[agent](batch).act[0]) == env.ACTION_YES

    def build(v_idx, prop_idx, v_gems, spread):
        """投票者に v_gems を渡し、残りを spread の指定どおりに配る。

        spread が False なら提案者が独占、True なら他者に均等。
        整数（0 以上）なら「他者のうち spread 番目が総取り」。
        """
        dist = [0] * env.n_agents
        dist[v_idx] = v_gems
        rest = env.total_gems - v_gems
        others = [i for i in range(env.n_agents) if i not in (v_idx, prop_idx)]
        if spread is False:
            dist[prop_idx] += rest
            return dist
        if spread is True:
            for k in range(rest):
                dist[others[k % len(others)]] += 1
            return dist
        dist[others[spread % len(others)]] += rest
        return dist

    spreads = [False, True]
    if rich_spread:
        spreads += list(range(max(env.n_agents - 2, 1)))

    to_yes = 0
    to_no = 0
    total = 0
    split = {"decided": [0, 0, 0], "undecided": [0, 0, 0]}  # [賛成転換, 反対転換, 局面数]
    per_agent = {}

    for v_idx, agent in enumerate(agents):
        a_yes = a_no = a_total = 0
        # 提案者も走査する（投票者自身が提案者の局面は除く）
        for prop_idx in range(env.n_agents):
            if prop_idx == v_idx:
                continue
            if proposer_filter is not None and not proposer_filter(prop_idx):
                continue
            for spread in spreads:
                for tally in tallies_for(v_idx, prop_idx):
                    bucket = split["decided" if is_decided(tally) else "undecided"]
                    for gems in range(env.total_gems):
                        low = build(v_idx, prop_idx, gems, spread)
                        high = build(v_idx, prop_idx, gems + 1, spread)
                        before = vote(agent, prop_idx, low, tally)
                        after = vote(agent, prop_idx, high, tally)
                        a_total += 1
                        bucket[2] += 1
                        if after and not before:
                            a_yes += 1
                            bucket[0] += 1
                        elif before and not after:
                            a_no += 1
                            bucket[1] += 1

        per_agent[agent[-1]] = (a_yes - a_no) / a_total if a_total else 0.0
        to_yes += a_yes
        to_no += a_no
        total += a_total

    sensitivity = (to_yes - to_no) / total if total else 0.0

    def bucket_sensitivity(name):
        yes, no, count = split[name]
        return ((yes - no) / count if count else float("nan")), count

    decided_sensitivity, n_decided = bucket_sensitivity("decided")
    undecided_sensitivity, n_undecided = bucket_sensitivity("undecided")

    return {
        "sensitivity": sensitivity,
        "n_comparisons": total,
        "sensitivity_decided": decided_sensitivity,
        "n_decided": n_decided,
        "sensitivity_undecided": undecided_sensitivity,
        "n_undecided": n_undecided,
        "to_yes": to_yes,
        "to_no": to_no,
        "per_agent": per_agent,
    }


def main(config_path, model_path, proposer="A"):
    with open(config_path) as f:
        config = json.load(f)
    r = measure(config, model_path, proposer)
    print(f"モデル: {model_path}")
    print(f"走査した局面: {r['n_comparisons']} 通り"
          f"（票数観測 {'あり' if config.get('observe_vote_tally') else 'なし'}）")
    print(f"取り分を増やして賛成に転じた: {r['to_yes']} / 反対に転じた: {r['to_no']}")
    print(f"**感応度: {r['sensitivity']:+.4f}**")
    print("エージェント別: " + ", ".join(
        f"{a}={v:+.3f}" for a, v in sorted(r["per_agent"].items())))
    return r


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "A")
