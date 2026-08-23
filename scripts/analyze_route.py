"""log_metrics_n.csv から、罰の回避ルートの推移を2つの側面で要約する。

第15ラウンドで、この2つが食い違いうることが分かった。

1. **投票側の傾向**（`selfvote_*` 列）: 「必要票が他者だけで揃っている」局面で
   投じる票。no = 自己反対する構え / yes = 賛成する構え。
   ただしこれは**反実仮想の局面**であり、提案の作り方によっては一度も訪れない。

2. **実際に実現するルート**（`prop_*` 列の買収人数）: 提案者が
   必要票 −1 人しか買収しなければ自分が決定票になり、投票側の傾向が no でも
   自己反対する機会がない（決定票ルート）。必要票と同数を買収して初めて
   自己反対が実現する（自己反対ルート）。

使い方: python scripts/analyze_route.py log/log_metrics_59.csv [エージェント記号]
"""
import csv
import sys


def buy_count(prop, idx):
    """分配案の文字列から、提案者以外で宝石を受け取った人数を数える。"""
    try:
        dist = [int(v) for v in prop.split("-")]
    except ValueError:
        return None
    return sum(1 for j, v in enumerate(dist) if v > 0 and j != idx)


def summarize_switches(series, label):
    switches = [
        (series[i][0], series[i - 1][1], series[i][1])
        for i in range(1, len(series))
        if series[i][1] != series[i - 1][1]
    ]
    print(f"\n--- {label} ---")
    print(f"切り替わり回数: {len(switches)}")
    print(f"最終値: {series[-1][1]}")
    if switches:
        last = switches[-1][0]
        span = series[-1][0] - last
        print(f"最後の切り替わり: epoch {last}"
              f"（以後 {span} エポック = 全体の {span / series[-1][0]:.0%} が不動）")
        print("切り替わり（先頭10件）:")
        for epoch, before, after in switches[:10]:
            print(f"  epoch {epoch:>5}: {before} → {after}")
    else:
        print("学習を通じて一度も切り替わっていない。")
    return switches


def main(path, agent="A"):
    col = f"selfvote_{agent}"
    prop_col = f"prop_{agent}"
    idx = ord(agent) - ord("A")

    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if col not in rows[0]:
        print(f"{col} 列がありません（{path}）")
        return 1

    # 実際に実現するルート（買収人数から判定）
    n_agents = sum(1 for k in rows[0] if k.startswith("prop_"))
    required = (n_agents + 1) // 2
    route = []
    for r in rows:
        b = buy_count(r[prop_col], idx)
        if b is None:
            continue
        if b >= required:
            route.append((int(r["epoch"]), "自己反対"))
        elif b == required - 1:
            route.append((int(r["epoch"]), "決定票"))
        else:
            route.append((int(r["epoch"]), f"その他({b}人買収)"))

    series = [(int(r["epoch"]), r[col]) for r in rows if r[col] in ("yes", "no")]
    if not series:
        print(f"{col} に yes/no がありません（observe_vote_tally 無効の可能性）")
        return 1

    n = len(series)
    n_no = sum(1 for _, v in series if v == "no")
    print(f"ファイル: {path}  エージェント: {agent}（必要票 {required}）")
    print(f"記録点: {n}（エポック {series[0][0]}〜{series[-1][0]}）")
    print(f"投票側が no（自己反対の構え）だった記録点: {n_no} / {n} = {n_no / n:.1%}")

    summarize_switches(series, f"投票側の傾向（{col}）")
    if route:
        summarize_switches(route, f"実際に実現するルート（{prop_col} の買収人数）")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "A"))
