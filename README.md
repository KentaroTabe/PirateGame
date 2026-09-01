# 海賊の宝石分配問題：マルチエージェント強化学習

このプロジェクトは、ゲーム理論における古典的な「海賊の宝石分配問題」に、不確実性（確率的な提案権の移行）と有限のリスク（命の重さ）を導入したマルチエージェント強化学習（MARL）シミュレーション環境です。
AIエージェントが、自律的に「政治的駆け引き」や「強者の傲慢・弱者の妥協」といった戦略を学習・創発する過程を検証することを目的としています。

---

## 🎯 環境ルール

- **エージェント数**: `num_agents` で設定（デフォルト 5人: A, B, C, ...）
- **宝石の総数**: `total_gems` で設定（高速化・検証用として 3〜5個への変更も可能）
- **ゲームの流れ**:
  1. 提案者が宝石の分配案を提示する
  2. 生存者全員（提案者含む）が順に賛成/反対を投票する
  3. 賛成が過半数（生存者数の半分以上）なら可決し、分配どおりの報酬でゲーム終了
  4. 否決なら提案者は海に落とされ死亡（ペナルティ $-L$）、次の提案者が選ばれる
  5. 生存者が1人になったら、その者が宝石を独占する
- **提案者の選出**（`fixed_order`）:
  - `true`: 生存者中で最もインデックスが若いエージェント（古典的な海賊ゲームの固定順）
  - `false`: 「権力ウェイト」（`agent_weights`）に比例した確率でランダム選出

---

## 🧠 実験デザイン: 一般解を初期知識とした創発学習

本プロジェクトの実験は3フェーズで構成されます。

### フェーズ0: 固定順一般解の事前学習（`solver.py` + `pretrain.py`）
固定順ルールの海賊ゲームは、バックワードインダクション（後ろ向き帰納法）で厳密に解けます。
`FixedOrderSolver` は任意の生存集合・任意の提案者について、

- **最適提案**: 否決後の各投票者の継続価値を計算し、必要最小限の票を最も安い投票者から買収する
- **最適投票**: 「可決時の取り分 > 否決時の継続価値」のときのみ賛成（無差別なら反対）

という標準的な均衡解を返します。`pretrain.py` はランダム順ゲームで訪れうる全状態
（生存集合 × 提案者 × 分配案）を列挙し、一般解から導いた Q 値

- 提案行動: 均衡投票の下で可決するなら自分の取り分、否決なら $-L$
- 投票行動: 賛成 = 可決時の取り分、反対 = 否決時の継続価値（ピボタル仮定）

を各エージェントの Q ネットワークに回帰で埋め込みます。

### フェーズ1: ランダム順環境での DQN 学習（`train.py`）
一般解を初期値として、`agent_weights` に基づくランダムな順番で提案権が移る環境
（`"fixed_order": false`）で Double DQN の学習を行います。
固定順の合理的解を出発点に、提案権の不確実性（権力構造）の下での
「政治的駆け引き」「強者の傲慢・弱者の妥協」の創発を観察します。

### フェーズ2: 統計評価（`eval.py`）
ランダム順環境は1ゲームごとに展開が変わるため、複数エピソード（デフォルト100）を実行し、
エージェントごとの平均報酬・死亡率・平均提案回数を集計します。
最初の数エピソードは提案・投票の詳細ログも出力します。

### エージェントのアーキテクチャ
- **行動マスク付き Double DQN**: ルール違反の行動（死者への分配、フェーズ外の行動）の
  Q 値を $-10^9$ に固定し、有効な行動のみを探索
- **入力（状態）**: 「生存フラグ」「権力ウェイト」「現在の提案者（One-hot）」「現在の分配案」の $4N$ 次元
  （`observe_vote_tally` を有効にすると「これまでの賛成数」「投票済み人数」が加わり $4N+2$ 次元）
- **ネットワーク**: 隠れ層 `[128, 128]` の MLP
- **MultiAgentPolicyManager + SubprocVectorEnv**: 独立ポリシーの統括とマルチプロセス並列学習

---

## 📦 モジュール構成

### 主要な外部ライブラリ
- **PettingZoo (`pettingzoo`)**: マルチエージェント環境（AECEnv）の標準インターフェース
- **Tianshou (`tianshou`)**: DQN アルゴリズム、並列環境、ポリシーマネージャーを提供
- **PyTorch (`torch`)**: ニューラルネットワークの構築と最適化

### 内部モジュール
- `env.py`: PettingZoo を継承した海賊ゲーム環境（ルール・Action Mask・イベントログ）
- `solver.py`: 固定順バックワードインダクションの一般解ソルバー
- `pretrain.py`: 一般解の Q 値を各エージェントのネットワークに埋め込む事前学習
- `network.py`: 行動マスク機能付き MLP（DQN のネットワーク）
- `train.py`: Tianshou による並列 DQN 学習ロジック
- `eval.py`: 1ゲーム実行（`run_game`）と複数エピソードの統計評価（`evaluate`）
- `run_experiment.py`: 事前学習 → 学習 → 評価 → サマリー出力を一元管理
- `export_onnx.py`: 学習済みモデルの ONNX エクスポート
- `tests/`: ソルバー・環境・事前学習のユニットテスト
- `tools/smoke_test.py`: パイプライン全体の短時間動作確認
- `scripts/`: 実行用シェルスクリプト（venv の有効化込み）

---

## 🛠️ 環境構築手順

1. Python 3.8 以上の環境を用意します。
2. 仮想環境（`.venv`）を作成し、アクティベートします。

**Windows の場合:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux の場合:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. 必要なパッケージをインストールします。

```bash
pip install -r requirements.txt
```

---

## 🚀 実行方法

### 1. 環境設定 (`config.json`)

```json
{
    "num_agents": 6,
    "total_gems": 5,
    "L": 100,
    "agent_weights": [10.0, 15.0, 2.0, 1.0, 1.0, 1.0],
    "train_epochs": 50000,
    "fixed_order": false,
    "excess_vote_penalty": 0.0,
    "pretrain": true,
    "pretrain_epochs": 200,
    "eval_episodes": 100,
    "eval_verbose_episodes": 3
}
```

- `fixed_order`: 学習・評価環境の提案者選出（`false` = 権力ウェイトに基づくランダム順）
- `excess_vote_penalty`: 可決時、必要最小票を超えた賛成1票につき提案者の報酬から引く量
  （一般解と整合させる場合は `0.0`）
- `proposer_votes_last`: `true` で提案者だけを投票順の末尾に回す（既定 `false`）。
  提案者が他者の票を見てから自分の票を決められるようになる
- `observe_vote_tally`: `true` で観測の末尾に `[これまでの賛成数, 投票済み人数]` を足す
  （既定 `false`、観測が 4N → 4N+2 次元になる）
- `observe_noise_dims`: 観測の末尾に足す**無意味な乱数**の次元数（既定 `0`）。
  `observe_vote_tally` の効果が「予測力のある特徴だから」なのか
  「次元が増えたから」なのかを切り分けるための対照
- `pretrain` / `pretrain_epochs`: 固定順一般解の事前学習の有効化と回帰エポック数
- `eval_episodes` / `eval_verbose_episodes`: 評価エピソード数と詳細ログを出すエピソード数

### 2. 実験の実行

```bash
scripts/run_experiment.sh
```

事前学習 → 学習 → 統計評価 → 結果保存が一括で行われます。

### 3. 出力ファイルの確認

- **`result/result_n.txt`**: 環境設定・一般解一致率・評価統計をまとめたサマリー
- **`models/pretrained_n.pth`**: 一般解を埋め込んだ事前学習済みネットワーク
- **`models/policy_n.pth`**: DQN 学習後のネットワーク（`{エージェント名: state_dict}` 形式）。
  `scripts/reevaluate.sh <設定> <モデル> [ログ] [エピソード数] [詳細数]` で学習をやり直さずに再評価できる
- **`log/log_pretrain_n.txt`**: 事前学習のログ（一般解一致率など）
- **`log/log_learning_n.txt`**: エポックごとの学習ログ
- **`log/log_metrics_n.csv`**: 学習中に定期記録した政治的指標
  （エージェント別平均報酬・死亡率・一発可決率・平均エピソード長・
  各エージェントの提案 `prop_*`・罰の回避ルート `selfvote_*`）
- **`files/plots/log_metrics_n.png`**: 上記メトリクスの学習曲線グラフ
  （実験終了時に自動生成。全体/宝石レンジ拡大の報酬推移・死亡率・
  一発可決率・エピソード長・最終盤の平均取り分）
- **`log/log_eval_n.txt`**: 評価エピソードの駆け引きログと統計

### 4. テスト・動作確認

```bash
scripts/run_tests.sh    # ユニットテスト（ソルバー・環境・事前学習）
scripts/smoke_test.sh   # パイプライン全体の短時間動作確認
```

### 5. (オプション) 学習曲線の再グラフ化 / ONNX エクスポート

学習曲線グラフは実験終了時に自動生成されます。過去のCSVをまとめて再生成する場合:

```bash
scripts/plot_logs.sh   # log/log_metrics_*.csv を files/plots/ に一括グラフ化
python export_onnx.py
```

個別に実行する場合は `python plot_log.py log/log_metrics_1.csv` のようにします。

---

## 📊 実験知見

実験から得られた知見（政治レジームの分析・学習上の落とし穴など）は
[`docs/findings.md`](docs/findings.md) にまとめています。

実験はラウンド単位で実施し、ラウンドごとにレポートを作成しています。

| ラウンド | 主題 | レポート |
|---|---|---|
| 事前 | リファクタリング以前の33試行 | [`docs/reports/round0.md`](docs/reports/round0.md) |
| 第1 | 権力重みパターンの比較（試行6〜9） | [`docs/reports/round1.md`](docs/reports/round1.md) |
| 第2 | 事前学習と後継者の分離（試行10〜12） | [`docs/reports/round2.md`](docs/reports/round2.md) |
| 第3 | 再現性・要因計画の完成・L の効果（試行13〜16） | [`docs/reports/round3.md`](docs/reports/round3.md) |
| 第4 | L 軸の解明（試行17〜20） | [`docs/reports/round4.md`](docs/reports/round4.md) |
| 第5 | 再現・閾値の細分化・一般性（試行21〜24） | [`docs/reports/round5.md`](docs/reports/round5.md) |
| 第6 | 撤回した主張の決着と新仮説の検証（試行25〜28） | [`docs/reports/round6.md`](docs/reports/round6.md) |
| 第7 | 方策収束の直接測定（試行29〜32） | [`docs/reports/round7.md`](docs/reports/round7.md) |
| 第8 | 方策指標による L 曲線の完成（試行33〜36） | [`docs/reports/round8.md`](docs/reports/round8.md) |
| 第9 | 過剰賛成票ペナルティの検証（試行37〜40） | [`docs/reports/round9.md`](docs/reports/round9.md) |
| 第10 | 自己反対が現れない理由の切り分け（試行41〜45） | [`docs/reports/round10.md`](docs/reports/round10.md) |
| 第11 | 自己反対を生む条件の切り分け（試行46〜48） | [`docs/reports/round11.md`](docs/reports/round11.md) |
| 第12 | L と権力集中度の単独効果（試行49〜52） | [`docs/reports/round12.md`](docs/reports/round12.md) |
| 第13 | 自己反対率の再現性検証（試行53〜54） | [`docs/reports/round13.md`](docs/reports/round13.md) |
| 第14 | 解の分岐頻度の測定（試行55〜58） | [`docs/reports/round14.md`](docs/reports/round14.md) |
| 第15 | ルートが決まる時点の観測（試行59〜62） | [`docs/reports/round15.md`](docs/reports/round15.md) |
| 第16 | L の絶対値か L/宝石数の比か（試行63〜66） | [`docs/reports/round16.md`](docs/reports/round16.md) |
| 第17 | 機構は主要な実験群にも当てはまるか（試行67〜70） | [`docs/reports/round17.md`](docs/reports/round17.md) |
| 第18 | 票数観測と罰のどちらが効いたのか（試行71〜74） | [`docs/reports/round18.md`](docs/reports/round18.md) |
| 第19 | 票数観測が効くのは情報か次元数か（試行75〜78） | [`docs/reports/round19.md`](docs/reports/round19.md) |
| 第20 | 票数観測は独裁者を富ませるか（試行79〜82） | [`docs/reports/round20.md`](docs/reports/round20.md) |
| 第21 | 比の結論を n=4 で確認する（試行83〜86） | [`docs/reports/round21.md`](docs/reports/round21.md) |
| 第22 | 乱数セルの n=4 化と測定指標の刷新（試行87〜90） | [`docs/reports/round22.md`](docs/reports/round22.md) |
| 第23 | L=5 で票数観測の効果を判定する（試行91〜96） | [`docs/reports/round23.md`](docs/reports/round23.md) |
| 第24 | 票数観測の効果を n=8 で決着させる（試行97〜104） | [`docs/reports/round24.md`](docs/reports/round24.md) |
| 第25 | 比か絶対値かを非飽和域で決着させる（試行105〜111） | [`docs/reports/round25.md`](docs/reports/round25.md) |
| 第26 | エージェント数を6未満から測る（試行112〜123） | [`docs/reports/round26.md`](docs/reports/round26.md) |

各ラウンドの設定ファイルは [`configs/`](configs/) にあり、
`scripts/run_weight_experiments.sh <設定ファイル...>` で直列実行できます。
収束状況の分析は `scripts/analyze_convergence.sh` を使います。
