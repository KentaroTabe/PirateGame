"""第34ラウンドの設定を作り、実行順の一覧を書き出す。

要因は「脱落（あり／なし）× L（5／100）× 報酬の縮尺（なし／1/L）」の8セル、各8シード。
脱落あり・縮尺なしの2セルは既存の16本と同じ設定で、学習中の記録を取るために再実行する。
シードごとに8セルを順に走らせるので、途中で止まっても釣り合った比較が残る。

使い方: python scripts/make_r34_configs.py
"""
import json
import os

BASE = "configs/r4_L20.json"
SEEDS = [42, 7, 13, 99, 1, 2, 3, 4]
L_VALUES = [5, 100]
MAX_ROUNDS = 5  # 脱落ありのゲームの最大提案回数（6人なら5回）に揃える
QUEUE = "configs/r34_queue.txt"


def build_config(base, eliminate, L, scaled, seed):
    config = dict(base, L=L, seed=seed, record_onpath=True)
    if not eliminate:
        config["eliminate_on_reject"] = False
        config["max_rounds"] = MAX_ROUNDS
    if scaled:
        config["reward_scale"] = 1.0 / L
    return config


def main():
    with open(BASE, encoding="utf-8") as f:
        base = json.load(f)
    queue = []
    for seed in SEEDS:
        for eliminate in (True, False):
            for L in L_VALUES:
                for scaled in (False, True):
                    name = (f"r34_{'elim' if eliminate else 'noelim'}_L{L}_"
                            f"{'scaled' if scaled else 'raw'}_seed{seed}.json")
                    path = os.path.join("configs", name)
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(build_config(base, eliminate, L, scaled, seed), f,
                                  indent=4, ensure_ascii=False)
                        f.write("\n")
                    queue.append(path)
    with open(QUEUE, "w", encoding="utf-8") as f:
        f.write("\n".join(queue) + "\n")
    print(f"{len(queue)} 本の設定を作り、{QUEUE} に実行順を書きました")


if __name__ == "__main__":
    main()
