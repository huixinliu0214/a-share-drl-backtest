"""
组合交易环境：10 只标的的离散仓位配置任务。

动作定义：对每只标的独立选择仓位档位 {0: 空仓, 1: 半仓, 2: 满仓}，
组合权重按档位归一化，保证任意时刻资金完全配置（或全部持币）。

交易成本建模（A 股，偏保守）：
    佣金双边各 0.025% + 卖出印花税 0.05% + 滑点 0.05% ≈ 单次换手 0.15%
"""
from __future__ import annotations

import numpy as np

# 档位 -> 杠杆系数
LEVELS = np.array([0.0, 0.5, 1.0], dtype=np.float64)
COST_RATE = 0.0015  # 单位换手（单边）的成本
N_ACTIONS = 3


class PortfolioEnv:
    def __init__(self, X: np.ndarray, next_ret: np.ndarray,
                 daily_win: np.ndarray | None = None, cost: float = COST_RATE,
                 turn_penalty: float = 0.0):
        """
        X            : (T, N, L, F) 特征窗口
        next_ret     : (T, N)      决策后持有期实现的收益
        daily_win    : (T, H, N)   持有期内每天的个股收益，用于还原真实日频净值
        turn_penalty : 训练用的换手正则系数。真实交易成本已经计入收益，
                       但单期成本（约 0.15%）相对持有期收益波动（2%~3%）信号太弱，
                       强化学习几乎学不到"少换手"；额外的惩罚项用于放大这个信号。
                       评估阶段置 0，绩效指标仍按真实成本计算。
        """
        self.X = X
        self.next_ret = next_ret
        self.daily_win = daily_win
        self.cost = cost
        self.turn_penalty = turn_penalty
        self.T, self.N = next_ret.shape
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = 0
        self.w = np.zeros(self.N, dtype=np.float64)  # 初始空仓
        self.nav = 1.0
        self.history = []
        return self._obs(0)

    def _obs(self, t: int) -> np.ndarray:
        """
        观测 = 价格特征窗口 + 当前持仓权重。

        把当前权重拼进状态是必要的：否则 agent 不知道自己持有什么，
        无法学会"维持仓位"，会陷入每期全量换手、被交易成本吞噬的行为。
        """
        base = self.X[t]                                   # (N, L, F)
        w = np.broadcast_to(self.w[:, None, None],
                            (self.N, base.shape[1], 1)).copy()
        return np.concatenate([base, w], axis=2).astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """action: (N,) 每只标的的档位索引。"""
        lev = LEVELS[np.asarray(action, dtype=int)]
        total = lev.sum()
        # 多头组合策略不允许完全空仓：全空仓会让 agent 用"躺平"规避换手惩罚，
        # 且后续净值计算出现 0/0。因此退化时回退为等权持有。
        w_target = lev / total if total > 1e-8 else np.full(self.N, 1.0 / self.N)

        # 上一期权重经过当日价格变动后的漂移
        r = self.next_ret[self.t]
        if self.w.sum() > 1e-8:
            drifted = self.w * (1.0 + r)
            port_gross = drifted.sum()  # 组合总价值（相对）
            drifted = drifted / port_gross
        else:
            drifted = np.zeros(self.N)
            port_gross = 1.0

        turnover = float(np.abs(w_target - drifted).sum())
        cost = turnover * self.cost
        port_ret = float((w_target * r).sum()) - cost

        # 还原持有期内真实的每日净值波动：权重在这 h 天内保持不变
        if self.daily_win is not None:
            dw = self.daily_win[self.t]
            cum = np.cumprod(1.0 + dw, axis=0)          # (H, N)
            value = cum @ w_target                       # (H,)
            prev = np.concatenate([[1.0], value[:-1]])
            safe_prev = np.where(prev > 0, prev, 1.0)
            daily = np.where((value > 0) & (prev > 0), value / safe_prev - 1.0, 0.0)
            daily[0] -= cost
        else:
            daily = np.array([port_ret])

        self.nav *= (1.0 + port_ret)
        self.history.append(
            {"date_idx": self.t, "weights": w_target.copy(),
             "ret": port_ret, "turnover": turnover, "cost": cost,
             "daily": daily}
        )

        self.w = w_target
        self.t += 1
        done = self.t >= self.T
        obs = self._obs(self.t) if not done else np.zeros_like(self._obs(0))

        reward = float(np.log1p(max(port_ret, -0.99))) - self.turn_penalty * turnover
        info = {"ret": port_ret, "turnover": turnover, "nav": self.nav}
        return obs, reward, done, info
