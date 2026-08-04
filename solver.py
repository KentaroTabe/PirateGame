"""固定順の海賊ゲームをバックワードインダクション（後ろ向き帰納法）で解く一般解ソルバー。

固定順ルール: 提案が否決されると提案者は死亡し、生存者の中で最もインデックスが
若いエージェントが次の提案者になる。このとき任意の生存集合 S について、
ゲームの価値 V(S) は生存人数に関する帰納法で厳密に計算できる。

均衡の仮定（標準的な海賊ゲームの解）:
    - 投票者は「可決時の取り分 > 否決時の継続価値」のときだけ賛成する
      （無差別なら反対 = 提案者は継続価値より真に多く払う必要がある）。
    - 提案者は自分の提案に必ず賛成する。
    - 提案者は必要最小限の票を最も安い投票者から買収し、残りを独占する。

ランダム順ゲームの状態（任意の生存集合・任意の提案者）に対しても、
「否決後は固定順ゲーム V(S - {提案者}) が始まる」とみなした一般解として
最適提案・投票の Q 値を返せるため、事前学習の教師として利用できる。
"""

from math import ceil

import numpy as np

from env import generate_distributions


class FixedOrderSolver:
    def __init__(self, n_agents, total_gems, L, excess_vote_penalty=0.0):
        self.n_agents = n_agents
        self.total_gems = total_gems
        self.L = float(L)
        self.excess_vote_penalty = float(excess_vote_penalty)
        self.distributions = generate_distributions(total_gems, n_agents)
        self._value_cache = {}

    # ------------------------------------------------------------------
    # 価値関数
    # ------------------------------------------------------------------
    def value(self, alive):
        """生存集合 alive・提案者 min(alive) でゲームが始まるときの各エージェントの価値。

        Returns:
            np.ndarray (n_agents,): 死亡済みエージェントの成分は 0。
        """
        alive = frozenset(alive)
        if not alive:
            raise ValueError("生存集合が空です")
        if alive in self._value_cache:
            return self._value_cache[alive]

        v = np.zeros(self.n_agents, dtype=np.float64)
        proposer = min(alive)

        if len(alive) == 1:
            v[proposer] = float(self.total_gems)
        else:
            proposal, passes = self.optimal_proposal(alive, proposer)
            if passes:
                for i in alive:
                    v[i] = float(proposal[i])
                yes = self._yes_count(alive, proposer, proposal)
                v[proposer] -= self.excess_vote_penalty * max(0, yes - self._required_votes(alive))
            else:
                # どの提案も可決できず、提案者は必ず死亡する
                cont = self.value(alive - {proposer})
                for i in alive:
                    v[i] = cont[i]
                v[proposer] = -self.L

        self._value_cache[alive] = v
        return v

    def continuation_value(self, alive, proposer, agent):
        """提案が否決されたとき（提案者死亡後）の agent の価値。"""
        if agent == proposer:
            return -self.L  # 提案者は必ず海に落とされる（最後の1人でも同様）
        return self.value(frozenset(alive) - {proposer})[agent]

    # ------------------------------------------------------------------
    # 最適戦略
    # ------------------------------------------------------------------
    def optimal_proposal(self, alive, proposer):
        """提案者 proposer（min(alive) でなくてもよい）の最適分配案。

        Returns:
            (proposal, passes): proposal は長さ n_agents のタプル。
            passes=False は「どの提案も可決不能で提案者の死が確定」を意味する。
        """
        alive = frozenset(alive)
        proposal = [0] * self.n_agents

        if len(alive) == 1:
            proposal[proposer] = self.total_gems
            return tuple(proposal), True

        cont = self.value(alive - {proposer})
        # 各投票者の買収コスト: 継続価値より真に多い最小の整数（負なら 0 で足りる）
        costs = sorted(
            (self._buy_cost(cont[i]), i) for i in alive if i != proposer
        )
        needed = self._required_votes(alive) - 1  # 提案者自身の賛成を除いた必要票数

        total_cost = sum(c for c, _ in costs[:needed])
        if total_cost > self.total_gems:
            proposal[proposer] = self.total_gems
            return tuple(proposal), False

        for c, i in costs[:needed]:
            proposal[i] = c
        proposal[proposer] = self.total_gems - total_cost
        return tuple(proposal), True

    def vote_q(self, alive, proposer, proposal, voter):
        """投票の Q 値 (q_yes, q_no)。

        自分の票が結果を左右する（ピボタル）と仮定し、
        賛成 = 可決時の自分の取り分、反対 = 否決時の継続価値。
        """
        q_yes = float(proposal[voter])
        if voter == proposer:
            yes = self._yes_count(alive, proposer, proposal)
            q_yes -= self.excess_vote_penalty * max(0, yes - self._required_votes(alive))
        q_no = self.continuation_value(alive, proposer, voter)
        return q_yes, q_no

    def proposal_q(self, alive, proposer, proposal):
        """分配案の Q 値: 均衡投票の下で可決なら自分の取り分、否決なら -L。"""
        alive = frozenset(alive)
        required = self._required_votes(alive)
        yes = self._yes_count(alive, proposer, proposal)
        if yes >= required:
            return float(proposal[proposer]) - self.excess_vote_penalty * max(0, yes - required)
        return -self.L

    def optimal_vote(self, alive, proposer, proposal, voter):
        """True=賛成。無差別なら反対（q_yes > q_no のときのみ賛成）。"""
        q_yes, q_no = self.vote_q(alive, proposer, proposal, voter)
        return q_yes > q_no

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------
    @staticmethod
    def _required_votes(alive):
        return ceil(len(alive) / 2)

    @staticmethod
    def _buy_cost(continuation):
        """継続価値 continuation の投票者に賛成させる最小の宝石数。"""
        if continuation < 0:
            return 0
        return int(np.floor(continuation)) + 1

    def _yes_count(self, alive, proposer, proposal):
        """均衡投票（無差別なら反対、提案者は賛成）の下での賛成票数。"""
        alive = frozenset(alive)
        if len(alive) == 1:
            return 1  # 提案者のみが投票する
        cont = self.value(alive - {proposer})
        yes = 1  # 提案者自身
        for i in alive:
            if i != proposer and proposal[i] > cont[i]:
                yes += 1
        return yes
