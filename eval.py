import os
import sys
import time
import torch
from tianshou.data import Batch
from tianshou.policy import MultiAgentPolicyManager, DQNPolicy
from tianshou.env import PettingZooEnv
from env import PirateGemEnv
from network import Net

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        if not os.path.exists("log"):
            os.makedirs("log")
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

# 実験用スクリプト等から設定やポリシーを使い回せるように関数化
def run_game(policy_manager=None, config=None, model_path='policy.pth', sleep_time=0.0):
    raw_env = PirateGemEnv(config if config is not None else {})
    agents = raw_env.possible_agents
    
    # ポリシーマネージャーが渡されなかった場合は、新しく作ってモデルファイルをロードする
    if policy_manager is None:
        env = PettingZooEnv(raw_env)
        policies = {}
        for agent in agents:
            obs_shape = raw_env.observation_spaces[agent]["observation"].shape
            act_shape = raw_env.action_spaces[agent].n
            net = Net(obs_shape, act_shape, hidden_sizes=[128, 128], device='cpu')
            policy = DQNPolicy(model=net, optim=torch.optim.Adam(net.parameters()), is_double=True)
            policies[agent] = policy
            
        policy_manager = MultiAgentPolicyManager(policies=[policies[agent] for agent in agents], env=env)
        
        try:
            policy_manager.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"モデル '{model_path}' を正常に読み込みました。")
        except FileNotFoundError:
            print(f"エラー: '{model_path}' が見つかりません。")
            return None

    # 評価モードへ切り替え
    policy_manager.eval()
    for policy in policy_manager.policies.values():
        policy.set_eps(0.0)

    print("\n=========================================")
    print("ゲーム開始！")
    print("=========================================")
    print(f"【環境設定】")
    print(f" - 海賊の人数: {len(agents)}人")
    print(f" - 宝石の総数: {raw_env.total_gems}個")
    print(f" - 命の重さ(ペナルティ L): {raw_env.L}")
    
    weight_str = ", ".join([f"{a.split('_')[1]}:{w}" for a, w in zip(agents, raw_env.agent_weights)])
    print(f" - 権力ウェイト(発言力): [{weight_str}]")
    print("=========================================\n")
    
    raw_env.reset()
    raw_env.render()
    
    final_rewards = {a: 0.0 for a in agents}
    
    for agent in raw_env.agent_iter():
        obs, reward, termination, truncation, info = raw_env.last()
        
        if termination or truncation:
            final_rewards[agent] = reward
            raw_env.step(None)
            continue
            
        if sleep_time > 0:
            time.sleep(sleep_time)
            
        batch = Batch(obs=Batch([obs]), info=Batch([info]))
        
        # policy_managerから直接行動をサンプリング（インデックスにマッピングされたポリシーを呼ぶ）
        action_result = policy_manager.policies[agent](batch)
        action = action_result.act[0]
        
        if raw_env.phase == "PROPOSE":
            print(f"💬 => {agent} は提案行動 [ {raw_env.DISTRIBUTIONS[action]} ] を選択しました！")
        else:
            vote_str = "👍 賛成 (YES)" if action == raw_env.ACTION_YES else "👎 反対 (NO)"
            print(f"💬 => {agent} は {vote_str} を選択しました！")
            
        raw_env.step(action)
        raw_env.render()
        
    print("\n=========================================")
    print("ゲーム終了！最終的な報酬（宝石の数 / ペナルティ）:")
    for a, r in final_rewards.items():
        print(f" - {a}: {r}")
        
    return final_rewards

def watch():
    logger = Logger()
    sys.stdout = logger
    try:
        run_game(sleep_time=1.0)
    finally:
        sys.stdout = logger.terminal
        logger.close()

if __name__ == '__main__':
    watch()