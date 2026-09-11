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


# 期待値の足し算で生じる浮動小数点誤差（例: 1.0 が 0.9999999 になる）で
# 買収価格が1個ずれないよう、継続価値を整数と比べるときにだけ使う許容誤差。
_INTEGER_TOLERANCE = 1e-9


def _price(continuation, accept_when_indifferent=False):
    """期待継続価値 continuation の投票者に賛成させる最小の宝石数。

    accept_when_indifferent=False（既定。FixedOrderSolver と同じ）なら継続価値を
    真に上回る最小の整数、True なら継続価値以上の最小の整数（無差別なら賛成）。
    分配が連続なら2つの規則の予測は一致するが、宝石が整数なので大きく食い違いうる。
    """
    if accept_when_indifferent:
        if continuation <= _INTEGER_TOLERANCE:
            return 0
        return int(np.ceil(continuation - _INTEGER_TOLERANCE))
    if continuation < -_INTEGER_TOLERANCE:
        return 0
    return int(np.floor(continuation + _INTEGER_TOLERANCE)) + 1


class RandomOrderSolver:
    """ランダム順（権力ウェイト比例）の海賊ゲームを解く部分ゲーム完全均衡ソルバー。

    本プロジェクトの学習環境（`fixed_order=False`）そのものの理論均衡を返す:
        - 提案者は生存者から agent_weights に比例した確率で選ばれる
          （Baron-Ferejohn モデルの recognition probability に当たる）。
        - 否決された提案者は -L を負って脱落し、残りの生存者から次の提案者が選ばれる。
    否決のたびに生存者が1人減るので、生存人数についての帰納法で厳密に解ける
    （否決されても誰も脱落しない Baron-Ferejohn モデルと違い、不動点を解く必要がない）。

    均衡の仮定は FixedOrderSolver と同じ:
        - 投票者は「可決時の取り分 > 否決時の継続価値（期待値）」のときだけ賛成する
          （ピボタル仮定。無差別なら反対）。accept_when_indifferent=True にすると
          「無差別なら賛成」に切り替わる。宝石が整数なので、理論予測はこの2つの規則で
          挟まれる幅として読む。
        - 提案者は自分の提案に賛成し、必要最小限の票を最も安い投票者から買収する。
          同じ値段の投票者が複数いれば、その中から一様に無作為に選ぶ
          （Baron-Ferejohn の定常均衡と同じく、誰を買うかは均衡では決まらない）。

    value(alive) は「その生存集合で、提案者がまだ選ばれていない」時点の期待価値を返す。
    """

    def __init__(self, n_agents, total_gems, L, weights, excess_vote_penalty=0.0,
                 accept_when_indifferent=False):
        if len(weights) != n_agents:
            raise ValueError(f"weights の長さ {len(weights)} が n_agents {n_agents} と一致しません")
        self.n_agents = n_agents
        self.total_gems = total_gems
        self.L = float(L)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.excess_vote_penalty = float(excess_vote_penalty)
        self.accept_when_indifferent = bool(accept_when_indifferent)
        self._value_cache = {}
        self._outcome_cache = {}

    def recognition_probabilities(self, alive):
        """生存集合 alive で各エージェントが提案者に選ばれる確率。"""
        members = sorted(alive)
        w = self.weights[members]
        return dict(zip(members, w / w.sum()))

    def value(self, alive):
        """提案者が選ばれる前の、各エージェントの期待価値（死亡済みの成分は 0）。"""
        alive = frozenset(alive)
        if not alive:
            raise ValueError("生存集合が空です")
        if alive in self._value_cache:
            return self._value_cache[alive]

        v = np.zeros(self.n_agents, dtype=np.float64)
        if len(alive) == 1:
            v[next(iter(alive))] = float(self.total_gems)
        else:
            for proposer, prob in self.recognition_probabilities(alive).items():
                payoff, _, _ = self.proposal_outcome(alive, proposer)
                v += prob * payoff

        self._value_cache[alive] = v
        return v

    def proposal_outcome(self, alive, proposer):
        """提案者 proposer が最適に提案したときの結果。

        Returns:
            (payoff, passes, yes_prob):
                payoff は各エージェントの期待報酬（買収先の無作為選択について期待値をとる）、
                passes は可決するか、yes_prob は各投票者が賛成する確率。
        """
        alive = frozenset(alive)
        key = (alive, proposer)
        if key in self._outcome_cache:
            return self._outcome_cache[key]

        payoff = np.zeros(self.n_agents, dtype=np.float64)
        yes_prob = np.zeros(self.n_agents, dtype=np.float64)
        voters = sorted(alive - {proposer})
        needed = ceil(len(alive) / 2) - 1  # 提案者自身の賛成を除いた必要票数

        if needed == 0:
            payoff[proposer] = float(self.total_gems)
            result = (payoff, True, yes_prob)
            self._outcome_cache[key] = result
            return result

        cont = self.value(alive - {proposer})
        prices = {i: _price(cont[i], self.accept_when_indifferent) for i in voters}
        threshold = sorted(prices.values())[needed - 1]
        below = [i for i in voters if prices[i] < threshold]
        tied = [i for i in voters if prices[i] == threshold]
        slots = needed - len(below)
        total_cost = sum(prices[i] for i in below) + slots * threshold

        if total_cost > self.total_gems:
            # どの提案も可決できず、提案者は必ず脱落する
            for i in alive:
                payoff[i] = cont[i]
            payoff[proposer] = -self.L
            result = (payoff, False, yes_prob)
            self._outcome_cache[key] = result
            return result

        share = slots / len(tied)
        for i in below:
            payoff[i] = float(prices[i])
            yes_prob[i] = 1.0
        for i in tied:
            payoff[i] = threshold * share
            # 値段0（否決後の価値が負）の投票者は、買収されなくても0個で賛成する
            yes_prob[i] = 1.0 if threshold == 0 else share

        yes = 1 + int(round(yes_prob.sum()))
        excess = max(0, yes - (needed + 1))
        payoff[proposer] = self.total_gems - total_cost - self.excess_vote_penalty * excess
        result = (payoff, True, yes_prob)
        self._outcome_cache[key] = result
        return result
