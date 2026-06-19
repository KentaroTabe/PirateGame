import functools
import itertools
import numpy as np
from gymnasium.spaces import Discrete, Box, Dict
from pettingzoo import AECEnv

# 分配パターンの全列挙（合計10になる5つの非負整数の組み合わせ）
def generate_distributions(total_gems=10, num_agents=5):
    distributions = []
    # Stars and bars（星と仕切り）のアルゴリズムを応用
    for bars in itertools.combinations(range(total_gems + num_agents - 1), num_agents - 1):
        dist = []
        prev = -1
        for b in bars:
            dist.append(b - prev - 1)
            prev = b
        dist.append(total_gems + num_agents - 1 - prev - 1)
        distributions.append(dist)
    return distributions

DISTRIBUTIONS = generate_distributions()
NUM_DISTRIBUTIONS = len(DISTRIBUTIONS)

# 行動IDの定義
# 0 ~ 1000: 提案アクション（1001通りの分配パターン）
# 1001: 賛成 (YES)
# 1002: 反対 (NO)
ACTION_YES = NUM_DISTRIBUTIONS
ACTION_NO = NUM_DISTRIBUTIONS + 1
TOTAL_ACTIONS = NUM_DISTRIBUTIONS + 2

class PirateGemEnv(AECEnv):
    metadata = {'render_modes': ['human'], "name": "pirate_gem_v0"}

    def __init__(self, L=1.5, total_gems=10):
        super().__init__()
        self.L = L
        self.total_gems = total_gems
        self.possible_agents = ["agent_A", "agent_B", "agent_C", "agent_D", "agent_E"]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        
        # 権力ウェイトの設定（発言力・次期提案者になる確率に影響）
        self.agent_weights = np.array([10.0, 8.0, 6.0, 4.0, 2.0], dtype=np.float32)
        
        # 共通の行動空間と観測空間
        self.action_spaces = {agent: Discrete(TOTAL_ACTIONS) for agent in self.possible_agents}
        
        # 観測: [生存フラグ(5), 権力ウェイト(5), 現在の提案者(5), 現在の分配案(5)] = 20次元
        self.observation_spaces = {
            agent: Dict({
                "observation": Box(low=0.0, high=float(max(10, total_gems)), shape=(20,), dtype=np.float32),
                "action_mask": Box(low=0, high=1, shape=(TOTAL_ACTIONS,), dtype=np.int8)
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

        # 状態の初期化
        self.alive = {agent: True for agent in self.possible_agents}
        self.current_proposal = [0] * len(self.possible_agents)
        self.votes = {}
        
        # 最初の提案者をウェイト比例で選出
        self.proposer = self._select_next_proposer()
        
        self.phase = "PROPOSE"  # "PROPOSE" または "VOTE"
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
        # 20次元の観測ベクトルを作成
        alive_flag = [1.0 if self.alive[a] else 0.0 for a in self.possible_agents]
        weights = list(self.agent_weights)
        proposer_onehot = [1.0 if a == self.proposer else 0.0 for a in self.possible_agents]
        proposal = self.current_proposal[:]
        
        obs = np.array(alive_flag + weights + proposer_onehot + proposal, dtype=np.float32)
        
        # 行動マスクの作成（無効な行動を0、有効な行動を1にする）
        mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)
        if self.alive[agent]:
            if self.phase == "PROPOSE" and agent == self.proposer:
                # 提案フェーズ：提案者は、死亡者に宝石を割り当てない分配パターンの身を提案可能
                for i, dist in enumerate(DISTRIBUTIONS):
                    valid = True
                    for j, amount in enumerate(dist):
                        if amount > 0 and not self.alive[self.possible_agents[j]]:
                            valid = False
                            break
                    if valid:
                        mask[i] = 1
            elif self.phase == "VOTE" and agent == self.agent_selection:
                # 投票フェーズ：順番が回ってきたら「賛成」または「反対」のみ可能
                mask[ACTION_YES] = 1
                mask[ACTION_NO] = 1
                
        return {"observation": obs, "action_mask": mask}

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self._clear_rewards()

        if self.phase == "PROPOSE":
            # 提案を状態に保存し、投票フェーズへ移行
            self.current_proposal = DISTRIBUTIONS[action]
            self.phase = "VOTE"
            self.votes = {}
            # 生存している全員で投票を行うための順序を設定
            self.voting_order = [a for a in self.possible_agents if self.alive[a]]
            self._next_voter()
            
        elif self.phase == "VOTE":
            # 投票の記録
            self.votes[agent] = (action == ACTION_YES)
            self.voting_order.remove(agent)
            
            if len(self.voting_order) > 0:
                self._next_voter()
            else:
                # 全員の投票が完了したら結果を判定
                self._resolve_vote()

        self._accumulate_rewards()

    def _next_voter(self):
        if self.voting_order:
            self.agent_selection = self.voting_order[0]

    def _resolve_vote(self):
        alive_agents = [a for a in self.possible_agents if self.alive[a]]
        yes_count = sum(1 for v in self.votes.values() if v)
        
        # 半数以上の賛成で可決
        if yes_count >= len(alive_agents) / 2:
            # 提案通りの報酬を分配し、ゲーム終了
            for i, a in enumerate(self.possible_agents):
                if self.alive[a]:
                    self.rewards[a] = float(self.current_proposal[i])
            for a in self.agents:
                self.terminations[a] = True
        else:
            # 否決された場合：提案者はペナルティを受けて死亡
            self.rewards[self.proposer] = -self.L
            self.alive[self.proposer] = False
            self.terminations[self.proposer] = True
            
            # 生存者が残り1人になった場合は、その人が全取りしてゲーム終了
            remaining_alive = [a for a in self.possible_agents if self.alive[a]]
            if len(remaining_alive) == 1:
                winner = remaining_alive[0]
                self.rewards[winner] = float(self.total_gems)
                for a in self.agents:
                    self.terminations[a] = True
            else:
                # まだ生存者がいる場合は、次の提案者を選出して提案フェーズへ戻る
                self.proposer = self._select_next_proposer()
                self.current_proposal = [0] * len(self.possible_agents)
                self.phase = "PROPOSE"
                self.agent_selection = self.proposer

    def render(self):
        # 今回は学習メインのため描画処理は省略します
        pass