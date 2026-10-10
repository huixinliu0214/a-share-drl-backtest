"""
模型：LSTM + Temporal Attention 的 DQN 网络。

结构：
    输入 (B, T, F) -> LSTM -> 隐状态序列 -> 时间注意力加权 -> 上下文向量 -> Q(s, a)

所有标的共享同一个网络参数（参数共享），既降低过拟合风险，
也让模型能泛化到训练时权重配置未见过的组合状态。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLSTMQNet(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, n_actions: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, F)
        返回 (B, n_actions) 的 Q 值
        """
        seq, _ = self.lstm(x)               # (B, T, H)
        score = self.attn(seq)              # (B, T, 1)
        weight = torch.softmax(score, dim=1)
        context = (seq * weight).sum(dim=1)  # (B, H)
        return self.head(context)

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """返回注意力权重，用于可解释性分析。"""
        seq, _ = self.lstm(x)
        score = self.attn(seq)
        return torch.softmax(score, dim=1).squeeze(-1)


class AttentionLSTMPolicy(nn.Module):
    """
    策略网络（用于 REINFORCE / 策略梯度）。

    与 DQN 版的区别：DQN 让每只标的独立决策、再共享组合收益，
    这会让信用分配（credit assignment）彻底混乱——单只标的无法从组合级
    奖励中判断自己的贡献，最终 Q 值只能学到噪声。

    策略梯度把"选股组合"视为一个联合动作：网络给出每只标的的入选概率，
    采样得到联合动作，组合收益作为该动作的奖励，梯度可正确回传。
    """
    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.attn = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.score = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        seq, _ = self.lstm(x)
        weight = torch.softmax(self.attn(seq), dim=1)
        return (seq * weight).sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回每只标的的入选 logit，形状 (N,)。"""
        return self.score(self.encode(x)).squeeze(-1)

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        seq, _ = self.lstm(x)
        return torch.softmax(self.attn(seq), dim=1).squeeze(-1)


def dqn_loss(policy: AttentionLSTMQNet, target: AttentionLSTMQNet,
             batch: tuple, gamma: float = 0.99) -> torch.Tensor:
    states, actions, rewards, next_states, dones = batch
    q = policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        q_next = target(next_states).max(dim=1).values
        y = rewards + gamma * q_next * (1.0 - dones)
    return F.mse_loss(q, y)
