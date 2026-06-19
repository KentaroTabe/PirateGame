import torch
import torch.nn as nn
import numpy as np

class Net(nn.Module):
    """
    Tianshouで利用するための、行動マスク機能付きMLP（多層パーセプトロン）ネットワーク
    """
    def __init__(self, state_shape, action_shape, hidden_sizes=[128, 128], device="cpu"):
        super().__init__()
        self.device = device
        
        # MLPの構築
        layers = []
        input_dim = np.prod(state_shape)
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, action_shape))
        
        self.model = nn.Sequential(*layers).to(device)

    def forward(self, obs, state=None, info={}):
        """
        obs: TianshouのBatchオブジェクト、または辞書型（observation, action_maskを含む）
        """
        # Observationベクトルの抽出
        if isinstance(obs, dict) and "observation" in obs:
            obs_tensor = torch.as_tensor(obs["observation"], dtype=torch.float32, device=self.device)
        elif hasattr(obs, "observation"):
            obs_tensor = torch.as_tensor(obs.observation, dtype=torch.float32, device=self.device)
        else:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
            
        logits = self.model(obs_tensor)
        
        # Action Maskの抽出と適用
        mask = None
        if hasattr(obs, "action_mask"):
            mask = torch.as_tensor(obs.action_mask, dtype=torch.bool, device=self.device)
        elif isinstance(obs, dict) and "action_mask" in obs:
            mask = torch.as_tensor(obs["action_mask"], dtype=torch.bool, device=self.device)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
                
        if mask is not None:
            # マスクされている（無効な）行動のQ値を非常に小さな値にして選ばれなくする
            logits = logits.masked_fill(~mask, -1e9)
            
        return logits, state