import json
import os
import sys
import torch
import numpy as np
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import DQNPolicy, MultiAgentPolicyManager
from tianshou.trainer import OffpolicyTrainer

from env import PirateGemEnv
from network import Net

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
    # フェーズ1: 学習 (Training)
    # ==========================================
    logger = DualLogger(log_learning_path)
    sys.stdout = logger
    print(f"--- 学習開始 (ログ: {log_learning_path}) ---")
    
    def get_env():
        return PettingZooEnv(PirateGemEnv(config))

    env = get_env()
    train_envs = DummyVectorEnv([lambda: get_env() for _ in range(1)])
    test_envs = DummyVectorEnv([lambda: get_env() for _ in range(1)])
    
    policies = {}
    agents = env.env.possible_agents
    
    for agent in agents:
        obs_shape = env.env.observation_spaces[agent]["observation"].shape
        act_shape = env.env.action_spaces[agent].n
        net = Net(obs_shape, act_shape, hidden_sizes=[128, 128], device='cpu')
        optim = torch.optim.Adam(net.parameters(), lr=1e-3)
        
        policy = DQNPolicy(
            model=net, optim=optim, discount_factor=0.99, estimation_step=3,
            target_update_freq=320, is_double=True
        )
        policies[agent] = policy
        
    policy_manager = MultiAgentPolicyManager(policies=[policies[agent] for agent in agents], env=env)
    
    train_collector = Collector(policy_manager, train_envs, VectorReplayBuffer(20000, len(train_envs)), exploration_noise=True)
    test_collector = Collector(policy_manager, test_envs, exploration_noise=True)
    
    def train_fn(epoch, env_step):
        for a in agents: policies[a].set_eps(0.1)

    def test_fn(epoch, env_step):
        for a in agents: policies[a].set_eps(0.05)

    # ログファイルがプログレスバーで埋まらないよう show_progress=False を指定
    trainer = OffpolicyTrainer(
        policy=policy_manager,
        train_collector=train_collector, test_collector=test_collector,
        max_epoch=config.get("train_epochs", 50), step_per_epoch=1000, step_per_collect=10,
        episode_per_test=10, batch_size=64, train_fn=train_fn, test_fn=test_fn,
        update_per_step=0.1, show_progress=False
    )
    
    train_result = trainer.run()
    torch.save(policy_manager.state_dict(), model_path)
    print(f"\nモデルを {model_path} に保存しました。")
    
    sys.stdout = logger.terminal
    logger.close()

    # ==========================================
    # フェーズ2: 評価・観察 (Evaluation)
    # ==========================================
    logger = DualLogger(log_eval_path)
    sys.stdout = logger
    print(f"--- 評価ゲーム開始 (ログ: {log_eval_path}) ---")
    
    raw_env = PirateGemEnv(config)
    env = PettingZooEnv(raw_env)
    
    # 評価モードへの切り替え
    policy_manager.eval()
    for agent in agents:
        policies[agent].set_eps(0.0)

    raw_env.reset()
    raw_env.render()
    
    final_rewards = {a: 0.0 for a in agents}
    
    for agent in raw_env.agent_iter():
        obs, reward, termination, truncation, info = raw_env.last()
        
        if termination or truncation:
            final_rewards[agent] = reward
            raw_env.step(None)
            continue
            
        batch = Batch(obs=Batch([obs]), info=Batch([info]))
        action_result = policies[agent](batch)
        action = action_result.act[0]
        
        if raw_env.phase == "PROPOSE":
            print(f"💬 => {agent} は提案行動 [ {raw_env.DISTRIBUTIONS[action]} ] を選択しました！")
        else:
            vote_str = "👍 賛成 (YES)" if action == raw_env.ACTION_YES else "👎 反対 (NO)"
            print(f"💬 => {agent} は {vote_str} を選択しました！")
            
        raw_env.step(action)
        raw_env.render()

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
        for a, r in final_rewards.items():
            f.write(f" - {a}: {r}\n")
        f.write("=========================================\n")
        
    print(f"\n✅ すべての処理が完了しました！\n結果は {result_path} に保存されました。")

if __name__ == '__main__':
    run_experiment()