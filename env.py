"""海賊の宝石分配ゲーム環境（PettingZoo AECEnv）。

ルール概要:
    1. 提案者が宝石の分配案を提示する（PROPOSE フェーズ）。
    2. 生存者全員（提案者含む）が順に賛成/反対を投票する（VOTE フェーズ）。
       投票順は生存者の固定順（A, B, C…）。proposer_votes_last=True なら
       提案者だけを末尾に回すので、提案者は他者の票を見てから決められる。
    3. 賛成が過半数（生存者数の半分以上）なら可決し、分配どおりの報酬でゲーム終了。
    4. 否決なら提案者は海に落とされ（報酬 -L で死亡）、次の提案者が選ばれる。
       次の提案者は fixed_order=True なら生存者中の最若番、
       False なら agent_weights（権力ウェイト）に比例した確率で選出される。
    5. 生存者が1人になったら、その者が宝石を独占してゲーム終了。
"""

import itertools

import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
from pettingzoo import AECEnv

PHASE_PROPOSE = "PROPOSE"
PHASE_VOTE = "VOTE"


def generate_distributions(total_gems, num_agents):
    """合計 total_gems になる num_agents 個の非負整数の組み合わせを全列挙する。"""
    distributions = []
    for bars in itertools.combinations(range(total_gems + num_agents - 1), num_agents - 1):
        dist = []
        prev = -1
        for b in bars:
            dist.append(b - prev - 1)
            prev = b
        dist.append(total_gems + num_agents - 1 - prev - 1)
        distributions.append(tuple(dist))
    return distributions


class PirateGemEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "pirate_gem_v1"}

    def __init__(self, config=None):
        super().__init__()
        config = config if config is not None else {}

        self.L = config.get("L", 1.5)
        self.total_gems = config.get("total_gems", 10)
        self.n_agents = config.get("num_agents", 5)
        self.fixed_order = config.get("fixed_order", False)
        # 過剰得票ペナルティ: 可決時、必要最小票を超えた賛成1票につき提案者の報酬を減らす。
        # ゲーム理論的な一般解と一致させたい場合は 0.0 にする。
        self.excess_vote_penalty = config.get("excess_vote_penalty", 0.0)
        # 投票順: True なら提案者を最後に回す（提案者は他者の票を見てから決められる）。
        self.proposer_votes_last = config.get("proposer_votes_last", False)
        # 観測に投票の途中経過（これまでの賛成数・投票済み人数）を含めるか。
        self.observe_vote_tally = config.get("observe_vote_tally", False)

        self.possible_agents = [f"agent_{chr(65 + i)}" for i in range(self.n_agents)]
        self.agent_name_mapping = dict(zip(self.possible_agents, range(self.n_agents)))

        default_weights = [10.0 - i for i in range(self.n_agents)]
        self.agent_weights = np.array(config.get("agent_weights", default_weights), dtype=np.float32)
        if len(self.agent_weights) != self.n_agents:
            raise ValueError(
                f"agent_weights の長さ {len(self.agent_weights)} が num_agents {self.n_agents} と一致しません"
            )

        # 行動空間: [全分配案..., 賛成, 反対]
        self.DISTRIBUTIONS = generate_distributions(self.total_gems, self.n_agents)
        self.NUM_DISTRIBUTIONS = len(self.DISTRIBUTIONS)
        self.ACTION_YES = self.NUM_DISTRIBUTIONS
        self.ACTION_NO = self.NUM_DISTRIBUTIONS + 1
        self.TOTAL_ACTIONS = self.NUM_DISTRIBUTIONS + 2

        self.action_spaces = {agent: Discrete(self.TOTAL_ACTIONS) for agent in self.possible_agents}

        # 観測: [生存フラグ(N), 権力ウェイト(N), 提案者one-hot(N), 現在の分配案(N)] = 4N 次元
        # observe_vote_tally が True の場合、末尾に [これまでの賛成数, 投票済み人数] を足す。
        self.VOTE_TALLY_DIM = 2
        obs_dim = 4 * self.n_agents
        if self.observe_vote_tally:
            obs_dim += self.VOTE_TALLY_DIM
        obs_high = float(max(100, self.total_gems))
        self.observation_spaces = {
            agent: Dict({
                "observation": Box(low=0.0, high=obs_high, shape=(obs_dim,), dtype=np.float32),
                "action_mask": Box(low=0, high=1, shape=(self.TOTAL_ACTIONS,), dtype=np.int8),
            }) for agent in self.possible_agents
        }

        self._rng = np.random.default_rng()
        self.event_log = []

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.agents = self.possible_agents[:]
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}

        self.alive = {agent: True for agent in self.possible_agents}
        self.current_proposal = (0,) * self.n_agents
        self.votes = {}
        self.voting_order = []
        self.event_log = []

        self.proposer = self._select_next_proposer()
        self.phase = PHASE_PROPOSE
        self.agent_selection = self.proposer

    def observe(self, agent):
        alive_flag = [1.0 if self.alive[a] else 0.0 for a in self.possible_agents]
        weights = list(self.agent_weights)
        proposer_onehot = [1.0 if a == self.proposer else 0.0 for a in self.possible_agents]
        proposal = list(self.current_proposal)

        features = alive_flag + weights + proposer_onehot + proposal
        if self.observe_vote_tally:
            features = features + self._vote_tally()

        obs = np.array(features, dtype=np.float32)

        mask = np.zeros(self.TOTAL_ACTIONS, dtype=np.int8)
        if self.alive[agent]:
            if self.phase == PHASE_PROPOSE and agent == self.proposer:
                for i in self.valid_distribution_indices():
                    mask[i] = 1
            elif self.phase == PHASE_VOTE and agent == self.agent_selection:
                mask[self.ACTION_YES] = 1
                mask[self.ACTION_NO] = 1

        return {"observation": obs, "action_mask": mask}

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self._clear_rewards()

        if self.phase == PHASE_PROPOSE:
            self.current_proposal = self.DISTRIBUTIONS[action]
            self.phase = PHASE_VOTE
            self.votes = {}
            self.voting_order = self._build_voting_order()
            self.agent_selection = self.voting_order[0]

        elif self.phase == PHASE_VOTE:
            self.votes[agent] = (action == self.ACTION_YES)
            self.voting_order.remove(agent)

            if self.voting_order:
                self.agent_selection = self.voting_order[0]
            else:
                self._resolve_vote()

        self._accumulate_rewards()

    def render(self):
        alive_str = ", ".join(a.split("_")[1] for a in self.possible_agents if self.alive[a])
        print("-" * 40)
        print(f"生存者: [{alive_str}] | 現在のフェーズ: {self.phase}")
        if self.phase == PHASE_PROPOSE:
            print(f"👑 提案者 {self.proposer} が分配案を考えています...")
        elif self.phase == PHASE_VOTE:
            print(f"💡 現在の分配案: {list(self.current_proposal)}")
            if self.agent_selection in self.voting_order:
                print(f"🗳️ {self.agent_selection} の投票待ち...")

    # ------------------------------------------------------------------
    # 内部ロジック
    # ------------------------------------------------------------------
    def valid_distribution_indices(self):
        """死亡したエージェントに宝石を配らない分配案のインデックス一覧。"""
        indices = []
        for i, dist in enumerate(self.DISTRIBUTIONS):
            if all(amount == 0 or self.alive[self.possible_agents[j]] for j, amount in enumerate(dist)):
                indices.append(i)
        return indices

    def _build_voting_order(self):
        """投票順を作る。proposer_votes_last なら提案者を末尾に回す。"""
        order = [a for a in self.possible_agents if self.alive[a]]
        if self.proposer_votes_last and self.proposer in order:
            order.remove(self.proposer)
            order.append(self.proposer)
        return order

    def _vote_tally(self):
        """観測用の投票の途中経過 [これまでの賛成数, 投票済み人数]。"""
        return [float(sum(1 for v in self.votes.values() if v)), float(len(self.votes))]

    def _select_next_proposer(self):
        alive_agents = [a for a in self.possible_agents if self.alive[a]]
        if not alive_agents:
            return None

        if self.fixed_order:
            # 生存者中で最もインデックスが若いエージェントが提案者になる
            return alive_agents[0]

        # 権力ウェイトに比例した確率でランダム選出
        weights = np.array([self.agent_weights[self.agent_name_mapping[a]] for a in alive_agents])
        return self._rng.choice(alive_agents, p=weights / weights.sum())

    def _resolve_vote(self):
        alive_agents = [a for a in self.possible_agents if self.alive[a]]
        yes_count = sum(1 for v in self.votes.values() if v)
        required_votes = int(np.ceil(len(alive_agents) / 2))
        passed = yes_count >= required_votes

        self.event_log.append({
            "type": "vote_result",
            "yes": yes_count,
            "n_alive": len(alive_agents),
            "passed": passed,
            "proposer": self.proposer,
            "proposal": self.current_proposal,
        })

        if passed:
            for i, a in enumerate(self.possible_agents):
                if self.alive[a]:
                    reward = float(self.current_proposal[i])
                    if a == self.proposer:
                        reward -= self.excess_vote_penalty * max(0, yes_count - required_votes)
                    self.rewards[a] = reward
            for a in self.agents:
                self.terminations[a] = True
            return

        # 否決: 提案者は海に落とされる
        self.event_log.append({"type": "eliminated", "agent": self.proposer})
        self.rewards[self.proposer] = -self.L
        self.alive[self.proposer] = False
        self.terminations[self.proposer] = True

        remaining = [a for a in self.possible_agents if self.alive[a]]
        if not remaining:
            # 最後の1人が自分の提案に反対して自滅した場合、全滅でゲーム終了
            for a in self.agents:
                self.terminations[a] = True
        elif len(remaining) == 1:
            winner = remaining[0]
            self.event_log.append({"type": "last_survivor", "agent": winner})
            self.rewards[winner] = float(self.total_gems)
            for a in self.agents:
                self.terminations[a] = True
        else:
            self.proposer = self._select_next_proposer()
            self.current_proposal = (0,) * self.n_agents
            self.phase = PHASE_PROPOSE
            self.agent_selection = self.proposer

    def pop_events(self):
        """直近の step で発生したイベントを取り出す（評価スクリプトの実況用）。"""
        events, self.event_log = self.event_log, []
        return events


def make_pirate_env(config=None):
    """マルチプロセス並列化（Pickle）に対応するための環境生成関数。"""
    return PirateGemEnv(config)
