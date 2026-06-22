import functools
import itertools
import numpy as np
from gymnasium.spaces import Discrete, Box, Dict
from pettingzoo import AECEnv

# 分配パターンの全列挙（合計total_gemsになるnum_agents個の非負整数の組み合わせ）
def generate_distributions(total_gems, num_agents):
    distributions = []
    for bars in itertools.combinations(range(total_gems + num_agents - 1), num_agents - 1):
        dist = []
        prev = -1
        for b in bars:
            dist.append(b - prev - 1)
            prev = b
        dist.append(total_gems + num_agents - 1 - prev - 1)
        distributions.append(dist)
    return distributions

class PirateGemEnv(AECEnv):
    metadata = {'render_modes': ['human'], "name": "pirate_gem_v1"}

    def __init__(self, config):
        super().__init__()
        # configから環境設定を読み込む
        self.L = config.get("L", 1.5)
        self.total_gems = config.get("total_gems", 10)
        self.n_agents = config.get("num_agents", 5)
        
        # エージェント名と権力ウェイトの設定
        self.possible_agents = [f"agent_{chr(65+i)}" for i in range(self.n_agents)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        
        default_weights = [10.0 - i for i in range(self.n_agents)] # デフォルト値
        self.agent_weights = np.array(config.get("agent_weights", default_weights), dtype=np.float32)
        
        # 行動空間の動的計算
        self.DISTRIBUTIONS = generate_distributions(self.total_gems, self.n_agents)
        self.NUM_DISTRIBUTIONS = len(self.DISTRIBUTIONS)
        self.ACTION_YES = self.NUM_DISTRIBUTIONS
        self.ACTION_NO = self.NUM_DISTRIBUTIONS + 1
        self.TOTAL_ACTIONS = self.NUM_DISTRIBUTIONS + 2
        
        self.action_spaces = {agent: Discrete(self.TOTAL_ACTIONS) for agent in self.possible_agents}
        
        # 観測空間の次元計算: [生存フラグ(N), 権力ウェイト(N), 現在の提案者(N), 現在の分配案(N)] = 4 * N 次元
        obs_dim = 4 * self.n_agents
        self.observation_spaces = {
            agent: Dict({
                "observation": Box(low=0.0, high=float(max(100, self.total_gems)), shape=(obs_dim,), dtype=np.float32),
                "action_mask": Box(low=0, high=1, shape=(self.TOTAL_ACTIONS,), dtype=np.int8)
            }) for agent in self.possible_agents
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.agents = self.possible_agents[:]
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}

        self.alive = {agent: True for agent in self.possible_agents}
        self.current_proposal = [0] * len(self.possible_agents)
        self.votes = {}
        
        self.proposer = self._select_next_proposer()
        self.phase = "PROPOSE"
        self.agent_selection = self.proposer
        self.voting_order = []

    def _select_next_proposer(self):
        alive_agents = [a for a in self.possible_agents if self.alive[a]]
        if len(alive_agents) == 0:
            return None
        weights = [self.agent_weights[self.agent_name_mapping[a]] for a in alive_agents]
        prob = np.array(weights) / sum(weights)
        return np.random.choice(alive_agents, p=prob)

    def observe(self, agent):
        alive_flag = [1.0 if self.alive[a] else 0.0 for a in self.possible_agents]
        weights = list(self.agent_weights)
        proposer_onehot = [1.0 if a == self.proposer else 0.0 for a in self.possible_agents]
        proposal = self.current_proposal[:]
        
        obs = np.array(alive_flag + weights + proposer_onehot + proposal, dtype=np.float32)
        
        mask = np.zeros(self.TOTAL_ACTIONS, dtype=np.int8)
        if self.alive[agent]:
            if self.phase == "PROPOSE" and agent == self.proposer:
                for i, dist in enumerate(self.DISTRIBUTIONS):
                    valid = True
                    for j, amount in enumerate(dist):
                        if amount > 0 and not self.alive[self.possible_agents[j]]:
                            valid = False
                            break
                    if valid:
                        mask[i] = 1
            elif self.phase == "VOTE" and agent == self.agent_selection:
                mask[self.ACTION_YES] = 1
                mask[self.ACTION_NO] = 1
                
        return {"observation": obs, "action_mask": mask}

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self._clear_rewards()

        if self.phase == "PROPOSE":
            self.current_proposal = self.DISTRIBUTIONS[action]
            self.phase = "VOTE"
            self.votes = {}
            self.voting_order = [a for a in self.possible_agents if self.alive[a]]
            self._next_voter()
            
        elif self.phase == "VOTE":
            self.votes[agent] = (action == self.ACTION_YES)
            self.voting_order.remove(agent)
            
            if len(self.voting_order) > 0:
                self._next_voter()
            else:
                self._resolve_vote()

        self._accumulate_rewards()

    def _next_voter(self):
        if self.voting_order:
            self.agent_selection = self.voting_order[0]

    def _resolve_vote(self):
        alive_agents = [a for a in self.possible_agents if self.alive[a]]
        yes_count = sum(1 for v in self.votes.values() if v)
        
        print(f"\n[判定] 賛成: {yes_count} / 生存者: {len(alive_agents)}")
        
        if yes_count >= len(alive_agents) / 2:
            print(f"👉 提案は【可決】されました！")
            for i, a in enumerate(self.possible_agents):
                if self.alive[a]:
                    self.rewards[a] = float(self.current_proposal[i])
            for a in self.agents:
                self.terminations[a] = True
        else:
            print(f"💀 提案は【否決】されました... {self.proposer} は海に落とされます。")
            self.rewards[self.proposer] = -self.L
            self.alive[self.proposer] = False
            self.terminations[self.proposer] = True
            
            remaining_alive = [a for a in self.possible_agents if self.alive[a]]
            if len(remaining_alive) == 1:
                winner = remaining_alive[0]
                print(f"🏆 {winner} が最後の生存者となり、宝石を独占します！")
                self.rewards[winner] = float(self.total_gems)
                for a in self.agents:
                    self.terminations[a] = True
            else:
                self.proposer = self._select_next_proposer()
                self.current_proposal = [0] * len(self.possible_agents)
                self.phase = "PROPOSE"
                self.agent_selection = self.proposer

    def render(self):
        print("-" * 40)
        alive_str = ", ".join([a.split('_')[1] for a in self.possible_agents if self.alive[a]])
        print(f"生存者: [{alive_str}] | 現在のフェーズ: {self.phase}")
        if self.phase == "PROPOSE":
            print(f"👑 提案者 {self.proposer} が分配案を考えています...")
        elif self.phase == "VOTE":
            print(f"💡 現在の分配案: {self.current_proposal}")
            if self.agent_selection in self.voting_order:
                print(f"🗳️ {self.agent_selection} の投票待ち...")