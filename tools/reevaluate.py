"""保存済みモデルを再評価する（学習をやり直さずに評価だけ行う）。

用途:
    - 詳細ログのエピソード数を増やして、特定の行動を実際に観察する
    - 評価指標を後から追加したとき、過去のモデルに遡って測る

使い方:
    python -m tools.reevaluate <設定ファイル> <モデル> [出力ログ] [エピソード数] [詳細エピソード数]
"""

import json
import os
import sys

from eval import evaluate
from run_experiment import DualLogger


def main(config_path, model_path, log_path=None, n_episodes=100, verbose_episodes=3):
    with open(config_path) as f:
        config = json.load(f)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"モデルが見つかりません: {model_path}")

    print(f"設定: {config_path}")
    print(f"モデル: {model_path}")

    logger = DualLogger(log_path, quiet=True) if log_path else None
    if logger:
        sys.stdout = logger
    try:
        stats = evaluate(
            policy_manager=None, config=config, model_path=model_path,
            n_episodes=n_episodes, verbose_episodes=verbose_episodes,
        )
    finally:
        if logger:
            sys.stdout = logger.terminal
            logger.close()

    return stats


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3] if len(sys.argv) > 3 else None,
        int(sys.argv[4]) if len(sys.argv) > 4 else 100,
        int(sys.argv[5]) if len(sys.argv) > 5 else 3,
    )
