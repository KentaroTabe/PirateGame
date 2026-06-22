import os
import sys
import time
import torch
from tianshou.data import Batch
from tianshou.policy import MultiAgentPolicyManager, DQNPolicy
from tianshou.env import PettingZooEnv
from env import PirateGemEnv, DISTRIBUTIONS, ACTION_YES
from network import Net

class Logger(object):
    """標準出力をターミナルとログファイルの両方に書き出すためのクラス"""
    def __init__(self):
        self.terminal = sys.stdout
        
        # logディレクトリが存在しない場合は作成
        if not os.path.exists("log"):
            os.makedirs("log")
        
        # log_n.txt の n を決定
        n = 1
        while os.path.exists(f"log/log_{n}.txt"):
            n += 1
            
        self.log_filepath = f"log/log_{n}.txt"
        self.log_file = open(self.log_filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()
        print(f"\n[INFO] ログを {self.log_filepath} に保存しました。")

def watch():
    # 標準出力をフックしてファイルにも書き出す
    logger = Logger()
    sys.stdout = logger

    try:
        # 描画用の環境初期化
        raw_env = PirateGemEnv()
        env = PettingZooEnv(raw_env)
        agents = raw_env.possible_agents
        
        # モデルのロード準備
        policies = {}
        for agent in agents:
            obs_shape = raw_env.observation_spaces[agent]["observation"].shape
            act_shape = raw_env.action_spaces[agent].n
            net = Net(obs_shape, act_shape, hidden_sizes=[128, 128], device='cpu')
            policy = DQNPolicy(model=net, optim=torch.optim.Adam(net.parameters()), is_double=True)
            policies[agent] = policy
            
        policy_manager = MultiAgentPolicyManager(policies=[policies[agent] for agent in agents], env=env)
        
        # 保存されたモデルの読み込み
        try:
            policy_manager.load_state_dict(torch.load('policy.pth'))
            print("モデル 'policy.pth' を正常に読み込みました。")
        except FileNotFoundError:
            print("エラー: 'policy.pth' が見つかりません。先に train.py を実行してください。")
            return

        # 評価モードへ切り替え（ランダムな探索をゼロにする）
        policy_manager.eval()
        for agent in agents:
            policies[agent].set_eps(0.0)

        print("\n=========================================")
        print("ゲーム開始！")
        print("=========================================")
        
        raw_env.reset()
        raw_env.render()
        
        # エージェントごとの最終報酬を記録するための辞書
        final_rewards = {a: 0.0 for a in agents}
        
        # ゲームループ
        for agent in raw_env.agent_iter():
            obs, reward, termination, truncation, info = raw_env.last()
            
            # すでに死亡またはゲーム終了状態の場合は行動をスキップ
            if termination or truncation:
                # 報酬がリセットされる前に、各エージェントの最終的な累積報酬を保存
                final_rewards[agent] = reward
                raw_env.step(None)
                continue
                
            time.sleep(1.0) # ターミナル上で見やすいように少し待機
            
            # 現在のエージェントの観測をTianshouのBatch形式に変換
            batch = Batch(obs=Batch([obs]), info=Batch([info]))
            
            # ネットワークに推論させて行動を取得
            action_result = policies[agent](batch)
            action = action_result.act[0]
            
            # どんな行動を取ったかを表示
            if raw_env.phase == "PROPOSE":
                print(f"💬 => {agent} は提案行動 [ {DISTRIBUTIONS[action]} ] を選択しました！")
            else:
                vote_str = "👍 賛成 (YES)" if action == ACTION_YES else "👎 反対 (NO)"
                print(f"💬 => {agent} は {vote_str} を選択しました！")
                
            # 環境に行動を適用
            raw_env.step(action)
            raw_env.render()
            
        print("\n=========================================")
        print("ゲーム終了！最終的な報酬（宝石の数 / ペナルティ）:")
        for a, r in final_rewards.items():
            print(f" - {a}: {r}")

    finally:
        # 最後にロガーを閉じて標準出力を元に戻す
        sys.stdout = logger.terminal
        logger.close()

if __name__ == '__main__':
    watch()