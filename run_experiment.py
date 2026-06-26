import json
import os
import sys

# 他ファイルから関数やクラスをインポート
from train import train_agent, get_args
from eval import run_game

class DualLogger(object):
    """標準出力をターミナルと指定したファイルの両方に書き出すロガー"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.filepath = filepath
        self.file = open(self.filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()
        
    def close(self):
        self.file.close()

def setup_directories():
    for d in ["log", "result", "models"]:
        if not os.path.exists(d):
            os.makedirs(d)
    
    n = 1
    while os.path.exists(f"result/result_{n}.txt"):
        n += 1
    return n

def run_experiment():
    # 1. 設定の読み込み
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
        
    n = setup_directories()
    log_learning_path = f"log/log_learning_{n}.txt"
    log_eval_path = f"log/log_eval_{n}.txt"
    model_path = f"models/policy_{n}.pth"
    result_path = f"result/result_{n}.txt"

    # ==========================================
    # フェーズ1: 学習 (Training) -> train.py の関数を利用
    # ==========================================
    logger = DualLogger(log_learning_path)
    sys.stdout = logger
    print(f"--- 学習開始 (ログ: {log_learning_path}) ---")
    
    # train.pyの引数オブジェクトをベースに作成
    args = get_args()
    args.device = 'cpu' # 必要に応じて変更してください

    epochs = config.get("train_epochs", 50)
    args.epoch = epochs
    
    # 1. バッファサイズの自動調整 (比例)
    # 50エポックなら20000、200エポックなら80000 (上限100000, 下限10000)
    calc_buffer = int(20000 * (epochs / 50.0))
    args.buffer_size = max(10000, min(100000, calc_buffer))
    
    # 2. 学習率(lr)の自動調整 (反比例)
    # 50エポックなら0.001、200エポックなら0.00025 (上限0.001, 下限0.0001)
    calc_lr = 1e-3 * (50.0 / epochs)
    args.lr = max(1e-4, min(1e-3, calc_lr))
    
    print(f"💡 【自動調整】エポック数: {epochs}")
    print(f"    - バッファサイズ: {args.buffer_size}")
    print(f"    - 学習率 (lr): {args.lr:.5f}")
    
    # train_agentを呼び出して学習を実行
    train_result, policy_manager = train_agent(
        args=args, 
        config=config, 
        model_path=model_path, 
        show_progress=False
    )
    
    sys.stdout = logger.terminal
    logger.close()

    # ==========================================
    # フェーズ2: 評価・観察 (Evaluation) -> eval.py の関数を利用
    # ==========================================
    logger = DualLogger(log_eval_path)
    sys.stdout = logger
    print(f"--- 評価ゲーム開始 (ログ: {log_eval_path}) ---")
    
    # eval.pyから切り出したゲーム実行関数を呼び出す
    final_rewards = run_game(
        policy_manager=policy_manager, 
        config=config, 
        model_path=model_path,
        sleep_time=0.0
    )

    sys.stdout = logger.terminal
    logger.close()

    # ==========================================
    # フェーズ3: サマリーの出力 (Result)
    # ==========================================
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("=========================================\n")
        f.write("【環境設定】\n")
        f.write(f" - 海賊の人数: {config['num_agents']}人\n")
        f.write(f" - 宝石の総数: {config['total_gems']}個\n")
        f.write(f" - 命の重さ(ペナルティ L): {config['L']}\n")
        f.write(f" - 権力ウェイト: {config['agent_weights']}\n")
        f.write("-----------------------------------------\n")
        f.write("【学習したパラメータ（結果指標）】\n")
        f.write(f" - 実行エポック数: {config['train_epochs']}\n")
        f.write(f" - 最終テスト報酬(Best): {train_result.get('best_reward', 0):.2f}\n")
        f.write(f" - モデル保存先: {model_path}\n")
        f.write("-----------------------------------------\n")
        f.write("【ゲーム結果（最終報酬）】\n")
        if final_rewards:
            for a, r in final_rewards.items():
                f.write(f" - {a}: {r}\n")
        else:
            f.write(" - 評価エラーのため結果を取得できませんでした。\n")
        f.write("=========================================\n")
        
    print(f"\n✅ すべての処理が完了しました！\n結果は {result_path} に保存されました。")

if __name__ == '__main__':
    run_experiment()