"""実験パイプラインの一元管理スクリプト。

フェーズ構成:
    0. 事前学習: 固定順バックワードインダクションの一般解を各エージェントの
       Q ネットワークに回帰で埋め込む（config の "pretrain" で有効/無効を切り替え）。
    1. 学習: 一般解を初期値として、agent_weights に基づくランダム順の環境で
       DQN 学習を行う（"fixed_order": false）。
    2. 評価: 複数エピソードの統計（平均報酬・死亡率・平均提案回数）と
       サンプルゲームの詳細ログを出力する。
"""

import json
import os
import sys

import torch

from eval import evaluate
from plot_log import plot_metrics
from pretrain import pretrain_agents
from train import get_args, train_agent


class DualLogger(object):
    """標準出力をファイルへ複製する（quiet=True でターミナル出力を抑制）。"""

    def __init__(self, filepath, quiet=False):
        self.terminal = sys.stdout
        self.filepath = filepath
        self.file = open(self.filepath, "w", encoding="utf-8")
        self.quiet = quiet

    def write(self, message):
        if not self.quiet:
            self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        if not self.quiet:
            self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def setup_directories():
    for d in ["log", "result", "models"]:
        os.makedirs(d, exist_ok=True)

    n = 1
    while os.path.exists(f"result/result_{n}.txt"):
        n += 1
    return n


def run_experiment(config_path="config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    n = setup_directories()
    log_pretrain_path = f"log/log_pretrain_{n}.txt"
    log_learning_path = f"log/log_learning_{n}.txt"
    log_metrics_path = f"log/log_metrics_{n}.csv"
    log_eval_path = f"log/log_eval_{n}.txt"
    pretrained_path = f"models/pretrained_{n}.pth"
    model_path = f"models/policy_{n}.pth"
    result_path = f"result/result_{n}.txt"

    # ==========================================
    # フェーズ0: 一般解の事前学習 (Pretrain)
    # ==========================================
    pretrained_state_dicts = None
    pretrain_stats = None
    if config.get("pretrain", True):
        print(f"--- 事前学習開始: 固定順一般解の埋め込み (ログ: {log_pretrain_path}) ---")
        logger = DualLogger(log_pretrain_path, quiet=False)
        sys.stdout = logger

        pretrained_state_dicts, pretrain_stats = pretrain_agents(
            config,
            device="cpu",
            epochs=config.get("pretrain_epochs", 200),
            verbose=True,
        )
        torch.save(pretrained_state_dicts, pretrained_path)
        print(f"事前学習済みモデルを '{pretrained_path}' に保存しました。")

        sys.stdout = logger.terminal
        logger.close()

    # ==========================================
    # フェーズ1: ランダム順環境での DQN 学習 (Training)
    # ==========================================
    logger = DualLogger(log_learning_path, quiet=True)
    sys.stdout = logger
    print(f"--- 学習開始 (ログ: {log_learning_path}) ---")

    args = get_args()
    args.device = 'cpu'  # 必要に応じて 'cuda' などに変更してください
    args.num_envs = 4    # 同時並列処理する環境（プロセス）数。CPUコア数に合わせて調整してください

    epochs = config.get("train_epochs", 50)
    args.epoch = epochs

    calc_buffer = int(20000 * (epochs / 50.0))
    args.buffer_size = max(10000, min(100000, calc_buffer))

    calc_lr = 1e-3 * (50.0 / epochs)
    args.lr = max(1e-4, min(1e-3, calc_lr))

    # AECでは提案から否決確定(-L)までバッファ上で最大「生存者数」ステップ離れるため、
    # n-stepリターンの窓をエピソード最大長(全ラウンドの提案+投票ステップ数)以上にし、
    # 終端報酬が全エージェントの全遷移に届くようにする(実質モンテカルロリターン)
    n = config.get("num_agents", 5)
    args.n_step = n * (n + 1) // 2 + n - 2

    print(f"💡 【自動調整】エポック数: {epochs}")
    print(f"    - バッファサイズ: {args.buffer_size}")
    print(f"    - 学習率 (lr): {args.lr:.5f}")
    print(f"    - 並列プロセス数: {args.num_envs}")
    print(f"    - n-stepリターンの窓: {args.n_step}(エピソード最大長)")
    print(f"    - 提案者の選出: {'固定順' if config.get('fixed_order') else 'agent_weights に基づくランダム順'}")
    print(f"    - 事前学習の適用: {'あり（固定順一般解）' if pretrained_state_dicts else 'なし'}")

    logger.terminal.write(f"--- 学習開始 (ログ: {log_learning_path}) ---\n")

    train_result, policy_manager = train_agent(
        args=args,
        config=config,
        model_path=model_path,
        pretrained_state_dicts=pretrained_state_dicts,
        metrics_path=log_metrics_path,
        # 全体でおよそ100点の学習曲線が得られる間隔で記録する
        metrics_interval=max(1, epochs // 100),
        metrics_episodes=config.get("metrics_episodes", 30),
        show_progress=False,
        verbose=False,
    )

    sys.stdout = logger.terminal
    logger.close()

    # ==========================================
    # フェーズ2: 統計評価 (Evaluation)
    # ==========================================
    logger = DualLogger(log_eval_path, quiet=True)
    sys.stdout = logger
    logger.terminal.write(f"--- 評価開始 (ログ: {log_eval_path}) ---\n")
    print(f"--- 評価開始 (ログ: {log_eval_path}) ---")

    eval_stats = evaluate(
        policy_manager=policy_manager,
        config=config,
        n_episodes=config.get("eval_episodes", 100),
        verbose_episodes=config.get("eval_verbose_episodes", 3),
    )

    sys.stdout = logger.terminal
    logger.close()

    # ==========================================
    # フェーズ3: 学習曲線のグラフ化 (Plot)
    # ==========================================
    plot_path = None
    try:
        plot_path = plot_metrics(log_metrics_path)
        print(f"学習曲線のグラフを {plot_path} に保存しました。")
    except Exception as e:  # グラフ化の失敗で実験結果を失わないようにする
        print(f"警告: 学習曲線のグラフ化に失敗しました: {e}")

    # ==========================================
    # フェーズ4: サマリーの出力 (Result)
    # ==========================================
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("=========================================\n")
        f.write("【環境設定】\n")
        f.write(f" - 設定ファイル: {config_path}\n")
        f.write(f" - 海賊の人数: {config['num_agents']}人\n")
        f.write(f" - 宝石の総数: {config['total_gems']}個\n")
        f.write(f" - 命の重さ(ペナルティ L): {config['L']}\n")
        f.write(f" - 権力ウェイト: {config['agent_weights']}\n")
        f.write(f" - 提案者の選出: {'固定順' if config.get('fixed_order') else 'agent_weights に基づくランダム順'}\n")
        f.write("-----------------------------------------\n")
        f.write("【事前学習（固定順一般解の埋め込み）】\n")
        if pretrain_stats:
            for a, s in pretrain_stats.items():
                f.write(f" - {a}: 一般解一致率 {s['match_rate']:.1%} (loss={s['loss']:.4f})\n")
            f.write(f" - 保存先: {pretrained_path}\n")
        else:
            f.write(" - 実施なし\n")
        f.write("-----------------------------------------\n")
        f.write("【学習したパラメータ（結果指標）】\n")
        f.write(f" - 実行エポック数: {config['train_epochs']}\n")
        f.write(f" - 最終テスト報酬(Best): {train_result.get('best_reward', 0):.2f}\n")
        f.write(f" - モデル保存先: {model_path}\n")
        f.write(f" - 学習中メトリクス: {log_metrics_path}\n")
        if plot_path:
            f.write(f" - 学習曲線グラフ: {plot_path}\n")
        f.write("-----------------------------------------\n")
        f.write(f"【評価統計（{eval_stats['n_episodes']} エピソード）】\n")
        f.write(f" - 平均提案回数: {eval_stats['avg_proposals']:.2f}\n")
        for a in eval_stats['avg_rewards']:
            pass_rate = eval_stats['pass_rates'][a]
            pass_str = f"{pass_rate:.1%}" if pass_rate is not None else "---"
            f.write(
                f" - {a}: 平均報酬 {eval_stats['avg_rewards'][a]:+.2f}"
                f" / 死亡率 {eval_stats['death_rates'][a]:.1%}"
                f" / 提案 {eval_stats['propose_counts'][a]}回 (可決率 {pass_str})\n"
            )
        f.write("=========================================\n")

    print(f"\n✅ すべての処理が完了しました！\n結果は {result_path} に保存されました。")


if __name__ == '__main__':
    run_experiment(sys.argv[1] if len(sys.argv) > 1 else "config.json")
