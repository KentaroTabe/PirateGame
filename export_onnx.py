import os
import json
import torch
from network import Net
from env import PirateGemEnv

def export_to_onnx(model_path="models/policy_1.pth", config_path="config.json", output_dir="models/onnx"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 環境設定の読み込みと次元の把握
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    raw_env = PirateGemEnv(config)
    agents = raw_env.possible_agents
    
    # ネットワークのダミー入力シェイプの用意
    obs_shape = raw_env.observation_spaces[agents[0]]["observation"].shape
    act_shape = raw_env.action_spaces[agents[0]].n
    
    # TianshouのMultiAgentPolicyManagerに合わせたステートディクトの読み込み
    state_dict = torch.load(model_path, map_location="cpu")
    
    print(f"📦 ONNXへのエクスポートを開始します... (ソース: {model_path})")
    
    for idx, agent in enumerate(agents):
        # 各エージェントの個別ネットワークを初期化
        net = Net(obs_shape, act_shape, hidden_sizes=[128, 128], device="cpu")
        
        # マネージャー内の各ポリシー（"policies.0.model.model.~" のようなキー）の重みを抽出
        prefix = f"policies.{idx}.model."
        agent_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                agent_state_dict[k[len(prefix):]] = v
                
        net.load_state_dict(agent_state_dict)
        net.eval()
        
        # ONNXエクスポートのためのダミー入力（ObservationベクトルとAction Mask）
        # network.py 内のフォワードパス構造に合わせるため、辞書型、またはBatchオブジェクトを模倣した形式にします
        dummy_obs = torch.randn(1, *obs_shape)
        dummy_mask = torch.ones(1, act_shape, dtype=torch.bool)
        
        # 辞書での入力がONNXトレーサーを通るよう、forwardへの引数の形を調整したダミーを用意
        # Tianshou内部のフォーマットに対応するため、辞書として直接渡します
        dummy_input = {"observation": dummy_obs, "action_mask": dummy_mask}
        
        onnx_file_path = os.path.join(output_dir, f"{agent}_policy.onnx")
        
        # エクスポート実行
        torch.onnx.export(
            net,
            (dummy_input,),
            onnx_file_path,
            export_params=True,
            opset_version=14, # 比較的新しく安定したopset
            do_constant_folding=True,
            input_names=["observation", "action_mask"],
            output_names=["logits", "state"],
            dynamic_axes={
                "observation": {0: "batch_size"},
                "action_mask": {0: "batch_size"}
            }
        )
        print(f"  └─ {agent} のモデルを保存しました -> {onnx_file_path}")

if __name__ == "__main__":
    # 最新のモデル番号などに合わせて適宜ファイルパスを書き換えて実行してください
    export_to_onnx()