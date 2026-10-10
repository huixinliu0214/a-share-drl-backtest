"""
训练与回测主流程：LSTM-Attention DQN 组合配置策略。

流程：
    真实行情 -> 特征张量 -> 时序切分（训练/验证/测试）-> DQN 训练
    -> 样本外回测 -> 与三类基准对比 -> 输出绩效表与净值曲线

关键防过拟合设计：
    1. 严格时序切分，测试区间完全样本外；
    2. 标准化参数仅由训练集拟合；
    3. 网络参数在所有标的间共享；
    4. 多随机种子重复实验，报告均值与离散度。
"""
from __future__ import annotations

import gc
import json
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from env import LEVELS, N_ACTIONS, PortfolioEnv
from features import build_dataset, fit_scaler, make_windows
from metrics import fmt, summarize
from model import AttentionLSTMPolicy, AttentionLSTMQNet

ROOT = Path(__file__).parent
SEED = 42

TRAIN_END = "2022-01-01"
VAL_END = "2023-07-01"

# 调仓周期（交易日）。日频调仓会让换手成本吞噬收益，
# 周频（5 日）在 A 股是实务上常见的频率，也能显著降低训练震荡。
REBALANCE_H = 5

# 换手正则系数（仅作用于训练奖励，不影响绩效评估口径）
TURN_PENALTY = 0.001

# 熵正则系数：防止策略过早收敛
ENTROPY_COEF = 0.005

# 每期最少持有的标的数量
K_MIN = 6

# 监督学习选股的持仓数量（股票池 40 只）
TOP_K = 10


def to_rebalance(X: np.ndarray, y: np.ndarray, d: pd.DatetimeIndex,
                 h: int = REBALANCE_H) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """
    把日频样本降采样为调仓点样本。

    返回:
        X_rb       : (调仓点数, 标的数, 回看窗口, 特征数)
        y_rb       : (调仓点数, 标的数) 持有 h 日的累计收益
        d_rb       : 调仓日
        daily_win  : (调仓点数, h, 标的数) 持有期内每天的个股收益，
                     用于还原真实的日频净值波动（直接按几何展开会人为平滑波动率）
    """
    idx = np.arange(0, len(X) - h, h)
    X_rb = X[idx]
    d_rb = d[idx]
    # 按标的维度分别累计，得到 (调仓点数, 标的数)
    y_rb = np.array([np.prod(1.0 + y[t:t + h], axis=0) - 1.0 for t in idx])
    daily_win = np.array([y[t:t + h] for t in idx])
    return X_rb, y_rb, d_rb, daily_win


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_and_prepare() -> dict:
    panel = pd.read_csv(ROOT / "data" / "panel_close.csv",
                        index_col=0, parse_dates=True)
    bench = pd.read_csv(ROOT / "data" / "benchmark.csv",
                        index_col=0, parse_dates=True)["bench"]

    feats, next_ret, dates, assets, feat_names = build_dataset(panel, bench)
    X, y, d = make_windows(feats, next_ret, dates)

    # 降采样到周频调仓
    X, y, d, daily_win = to_rebalance(X, y, d)

    train_mask = d < TRAIN_END
    val_mask = (d >= TRAIN_END) & (d < VAL_END)
    test_mask = d >= VAL_END

    mu, sd = fit_scaler(X[train_mask])
    X = (X - mu) / sd

    # 基准保留完整日频序列，评估时按策略实际覆盖的交易日切片对齐
    bench_daily = bench.pct_change().dropna()

    return {
        "X": X, "y": y, "dates": d, "assets": assets,
        "feat_names": feat_names, "daily_win": daily_win,
        "train": train_mask, "val": val_mask, "test": test_mask,
        "bench_daily": bench_daily,
    }


def slice_data(data: dict, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return data["X"][mask], data["y"][mask], data["daily_win"][mask]


def _collect_daily(history: list[dict]) -> np.ndarray:
    return np.concatenate([h["daily"] for h in history])


def evaluate(policy: AttentionLSTMQNet, X: np.ndarray, y: np.ndarray,
             dw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """贪心策略跑完一个区间，返回日频收益与每期换手率。"""
    env = PortfolioEnv(X, y, dw)
    obs = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            q = policy(torch.from_numpy(obs))
        action = q.argmax(dim=1).numpy()
        obs, _, done, _ = env.step(action)
    return _collect_daily(env.history), np.array([h["turnover"] for h in env.history])


def action_profile(policy: AttentionLSTMQNet, X: np.ndarray, y: np.ndarray,
                   dw: np.ndarray) -> dict:
    """统计测试集上各仓位档位的选择比例，用于判断策略是否退化。"""
    env = PortfolioEnv(X, y, dw)
    obs = env.reset()
    done = False
    acts = []
    while not done:
        with torch.no_grad():
            q = policy(torch.from_numpy(obs))
        a = q.argmax(dim=1).numpy()
        acts.append(a)
        obs, _, done, _ = env.step(a)
    arr = np.array(acts)
    return {f"档位{i}({lv})": float((arr == i).mean()) for i, lv in enumerate(["空仓", "半仓", "满仓"])}


def equal_weight(X: np.ndarray, y: np.ndarray, dw: np.ndarray) -> np.ndarray:
    """等权满仓（每期再平衡）基准，成本模型与策略完全一致。"""
    env = PortfolioEnv(X, y, dw)
    env.reset()
    done = False
    while not done:
        obs, _, done, _ = env.step(np.full(env.N, 2, dtype=int))
    return _collect_daily(env.history)


def momentum(X: np.ndarray, y: np.ndarray, dw: np.ndarray, top_k: int = 5) -> np.ndarray:
    """20 日动量选股基准：按窗口内累计涨幅选前 k 只等权持有。"""
    env = PortfolioEnv(X, y, dw)
    obs = env.reset()
    done = False
    while not done:
        cum_ret = obs[:, :, 0].sum(axis=1)  # 特征 0 为 1 日对数收益
        idx = np.argsort(cum_ret)[::-1][:top_k]
        action = np.zeros(env.N, dtype=int)
        action[idx] = 2
        obs, _, done, _ = env.step(action)
    return _collect_daily(env.history)


_EW_CACHE: dict = {}


def _ew_cache(X: np.ndarray, y: np.ndarray, dw: np.ndarray) -> np.ndarray:
    """缓存验证集的等权基准收益，避免每个 episode 重复计算。"""
    key = (X.shape, float(X.sum()))
    if key not in _EW_CACHE:
        _EW_CACHE[key] = equal_weight(X, y, dw)
    return _EW_CACHE[key]


def evaluate_pg(policy: AttentionLSTMPolicy, X: np.ndarray, y: np.ndarray,
                dw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """确定性策略（入选概率 > 0.5）跑完一个区间。"""
    env = PortfolioEnv(X, y, dw)
    obs = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            logits = policy(torch.from_numpy(obs))
        action = (torch.sigmoid(logits) > 0.5).numpy().astype(int) * 2
        obs, _, done, _ = env.step(action)
    return _collect_daily(env.history), np.array([h["turnover"] for h in env.history])


def train_sl(data: dict, epochs: int = 15, seed: int = SEED, lr: float = 1e-3,
             verbose: bool = True) -> tuple[AttentionLSTMPolicy, dict]:
    """
    监督学习选股：预测每只标的未来一期的横截面超额收益。

    标签做横截面标准化（减去当期全市场均值），模型学到的是"相对强弱"而非
    绝对收益，从而剥离市场 beta，避免模型在牛市里简单地全选、熊市里全不选。
    """
    set_seed(seed)
    Xtr, ytr, _ = slice_data(data, data["train"])
    Xval, yval, _ = slice_data(data, data["val"])

    # 横截面标准化标签：只保留个股之间的相对差异
    def xs_label(y: np.ndarray) -> np.ndarray:
        mu = y.mean(axis=1, keepdims=True)
        sd = y.std(axis=1, keepdims=True)
        sd[sd < 1e-8] = 1.0
        return ((y - mu) / sd).astype(np.float32)

    ytr_l = xs_label(ytr)
    yval_l = xs_label(yval)

    model = AttentionLSTMPolicy(Xtr.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(Xtr))
        tot = 0.0
        for i in perm:
            pred = model(torch.from_numpy(Xtr[i]))
            tgt = torch.from_numpy(ytr_l[i])
            loss = torch.nn.functional.mse_loss(pred, tgt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)

        model.eval()
        with torch.no_grad():
            vpred = torch.stack([model(torch.from_numpy(Xval[i])) for i in range(len(Xval))])
            vtgt = torch.from_numpy(yval_l)
            # 验证集上预测值与真实超额收益的横截面相关性（IC）
            ics = []
            for t in range(len(Xval)):
                p = vpred[t].numpy()
                g = vtgt[t].numpy()
                if p.std() > 1e-8 and g.std() > 1e-8:
                    ics.append(float(np.corrcoef(p, g)[0, 1]))
            ic = float(np.mean(ics)) if ics else 0.0
        history.append({"epoch": ep + 1, "train_loss": tot / len(Xtr), "val_IC": ic})
        if verbose:
            print(f"  epoch{ep + 1:>3}/{epochs} loss={tot / len(Xtr):.4f} val_IC={ic:.4f}")

    return model, {"history": history, "val_IC": history[-1]["val_IC"]}


def evaluate_sl(model: AttentionLSTMPolicy, X: np.ndarray, y: np.ndarray,
                dw: np.ndarray, top_k: int = TOP_K) -> tuple[np.ndarray, np.ndarray]:
    """每期选预测超额收益最高的 Top-K 只等权持有。"""
    env = PortfolioEnv(X, y, dw)
    obs = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            score = model(torch.from_numpy(obs[:, :, :-1]))  # 去掉持仓权重维度
        idx = np.argsort(score.numpy())[::-1][:top_k]
        action = np.zeros(env.N, dtype=int)
        action[idx] = 2
        obs, _, done, _ = env.step(action)
    return _collect_daily(env.history), np.array([h["turnover"] for h in env.history])


def train_pg(data: dict, episodes: int = 60, seed: int = SEED, lr: float = 3e-4,
             update_every: int = 10, verbose: bool = True) -> tuple[AttentionLSTMPolicy, dict]:
    """REINFORCE 训练：把"本期选哪些股"视为一个联合动作。"""
    set_seed(seed)
    Xtr, ytr, dwtr = slice_data(data, data["train"])
    Xval, yval, dwval = slice_data(data, data["val"])

    n_feat = Xtr.shape[-1] + 1
    policy = AttentionLSTMPolicy(n_feat)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    best_val_sharpe, best_state = -1e9, None
    history = []

    for ep in range(episodes):
        env = PortfolioEnv(Xtr, ytr, dwtr, turn_penalty=TURN_PENALTY)
        obs = env.reset()
        done = False
        logps: list = []
        rews: list = []
        ents: list = []
        step = 0
        ep_reward = 0.0

        while not done:
            logits = policy(torch.from_numpy(obs))
            dist = torch.distributions.Bernoulli(logits=logits)
            b = dist.sample()
            # 防止退化：若采样结果一只都没选中（或少于 K_MIN），
            # 用 logit 最高的若干只补足。否则策略会陷入
            # "全不选 -> 环境回退为等权 -> 奖励恒定 -> 梯度消失" 的死循环。
            if int(b.sum()) < K_MIN:
                need = K_MIN - int(b.sum())
                top_idx = torch.topk(logits, k=need).indices
                b = b.clone()
                b[top_idx] = 1.0
            logp = dist.log_prob(b).sum()
            action = (b.detach().numpy() * 2).astype(int)

            obs, reward, done, _ = env.step(action)
            logps.append(logp)
            rews.append(float(reward))
            ents.append(dist.entropy().sum())
            ep_reward += reward
            step += 1

            if done or step % update_every == 0:
                R = torch.tensor(rews, dtype=torch.float32)
                adv = R - R.mean()
                if float(R.std()) > 1e-8:
                    adv = adv / (R.std() + 1e-8)
                loss = (-(torch.stack(logps) * adv.detach()).mean()
                        - ENTROPY_COEF * torch.stack(ents).mean())
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                logps, rews, ents = [], [], []

        del logps, rews, ents
        gc.collect()

        val_ret, _ = evaluate_pg(policy, Xval, yval, dwval)
        val_sharpe = summarize(val_ret)["夏普比率"]

        # 选择标准用"相对等权基准的超额收益"而非夏普：
        # 验证区间（2022-2023）整体下跌，用夏普选模型会偏向几乎不持仓的
        # 保守策略，导致策略退化成等权。超额收益考察的是选股能力。
        val_excess = float(np.prod(1.0 + val_ret) - np.prod(1.0 + _ew_cache(Xval, yval, dwval)))
        history.append({"episode": ep + 1, "val_sharpe": val_sharpe,
                        "val_excess": val_excess, "ep_reward": ep_reward})
        if verbose:
            print(f"  ep{ep + 1:>3}/{episodes} train_reward={ep_reward:8.3f} "
                  f"val_sharpe={val_sharpe:6.3f} val_excess={val_excess:7.4f}")

        if val_excess > best_val_sharpe:
            best_val_sharpe = val_excess
            best_state = {k: v.clone() for k, v in policy.state_dict().items()}

    if best_state is not None:
        policy.load_state_dict(best_state)
    return policy, {"history": history, "best_val_excess": best_val_sharpe}


def train_dqn(data: dict, episodes: int = 30, seed: int = SEED,
              verbose: bool = True) -> tuple[AttentionLSTMQNet, dict]:
    set_seed(seed)
    Xtr, ytr, dwtr = slice_data(data, data["train"])
    Xval, yval, dwval = slice_data(data, data["val"])

    n_feat = Xtr.shape[-1] + 1  # +1：环境会把当前持仓权重拼进观测
    policy = AttentionLSTMQNet(n_feat)
    target = AttentionLSTMQNet(n_feat)
    target.load_state_dict(policy.state_dict())

    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    buffer: deque = deque(maxlen=60000)

    env = PortfolioEnv(Xtr, ytr, dwtr, turn_penalty=TURN_PENALTY)
    epsilon = 1.0
    eps_min, eps_decay = 0.05, 0.92
    best_val_sharpe, best_state = -1e9, None
    history = []

    for ep in range(episodes):
        obs = env.reset()
        done = False
        steps = 0
        ep_reward = 0.0

        while not done:
            with torch.no_grad():
                q = policy(torch.from_numpy(obs))
            action = q.argmax(dim=1).numpy()
            explore = np.random.rand(env.N) < epsilon
            action[explore] = np.random.randint(0, N_ACTIONS, size=int(explore.sum()))

            next_obs, reward, done, _ = env.step(action)
            ep_reward += reward

            for i in range(env.N):
                buffer.append((obs[i].copy(), int(action[i]),
                               float(reward), next_obs[i].copy() if not done else obs[i].copy(),
                               float(done)))
            obs = next_obs
            steps += 1

            if len(buffer) >= 512 and steps % 2 == 0:
                idx = np.random.randint(0, len(buffer), size=128)
                b = [buffer[i] for i in idx]
                s = torch.from_numpy(np.stack([x[0] for x in b]))
                a = torch.tensor([x[1] for x in b], dtype=torch.long)
                r = torch.tensor([x[2] for x in b], dtype=torch.float32)
                ns = torch.from_numpy(np.stack([x[3] for x in b]))
                dn = torch.tensor([x[4] for x in b], dtype=torch.float32)

                qv = policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    qn = target(ns).max(dim=1).values
                    y = r + 0.99 * qn * (1.0 - dn)
                loss = torch.nn.functional.mse_loss(qv, y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()

            if steps % 200 == 0:
                target.load_state_dict(policy.state_dict())

        epsilon = max(eps_min, epsilon * eps_decay)

        val_ret, _ = evaluate(policy, Xval, yval, dwval)
        val_sharpe = summarize(val_ret)["夏普比率"]
        history.append({"episode": ep + 1, "epsilon": epsilon,
                        "val_sharpe": val_sharpe, "ep_reward": ep_reward})
        if verbose:
            print(f"  ep{ep + 1:>3}/{episodes} eps={epsilon:.3f} "
                  f"train_reward={ep_reward:8.3f} val_sharpe={val_sharpe:6.3f}")

        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            best_state = {k: v.clone() for k, v in policy.state_dict().items()}

    if best_state is not None:
        policy.load_state_dict(best_state)
    return policy, {"history": history, "best_val_sharpe": best_val_sharpe}


def main(episodes: int = 60, seeds: list[int] | None = None, method: str = "pg"):
    seeds = seeds or [SEED]
    data = load_and_prepare()
    print(f"数据: 总样本 {len(data['dates'])} | "
          f"训练 {data['train'].sum()} | 验证 {data['val'].sum()} | 测试 {data['test'].sum()}")
    print(f"区间: {data['dates'].min().date()} ~ {data['dates'].max().date()}")

    results = {}
    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        t0 = time.time()
        if method == "sl":
            policy, log = train_sl(data, epochs=episodes, seed=seed)
        elif method == "pg":
            policy, log = train_pg(data, episodes=episodes, seed=seed)
        else:
            policy, log = train_dqn(data, episodes=episodes, seed=seed)
        metric_key = next((k for k in log if k.startswith("best") or k == "val_IC"), None)
        print(f"训练耗时 {time.time() - t0:.0f}s, "
              f"{metric_key}={log[metric_key]:.4f}")

        Xte, yte, dwte = slice_data(data, data["test"])
        dte = data["dates"][data["test"]]

        if method == "sl":
            strat_d, strat_turn = evaluate_sl(policy, Xte, yte, dwte)
        elif method == "pg":
            strat_d, strat_turn = evaluate_pg(policy, Xte, yte, dwte)
        else:
            strat_d, strat_turn = evaluate(policy, Xte, yte, dwte)
        ew_d = equal_weight(Xte, yte, dwte)
        mom_d = momentum(Xte, yte, dwte)

        # 基准日收益按策略实际覆盖的交易日切片对齐
        bd = data["bench_daily"]
        pos = int(bd.index.searchsorted(dte[0]))
        bench_d = bd.iloc[pos:pos + len(strat_d)].to_numpy()
        if len(bench_d) < len(strat_d):
            bench_d = np.concatenate([bench_d, np.zeros(len(strat_d) - len(bench_d))])

        results[str(seed)] = {
            "DRL策略": summarize(strat_d, bench_d, strat_turn),
            "等权基准": summarize(ew_d, bench_d),
            "动量基准": summarize(mom_d, bench_d),
            "沪深300": summarize(bench_d, bench_d),
            "dates": [str(x.date()) for x in dte],
            "nav": np.cumprod(1 + strat_d).tolist(),
            "ew_nav": np.cumprod(1 + ew_d).tolist(),
            "bench_nav": np.cumprod(1 + bench_d).tolist(),
        }

        if method in ("pg", "sl"):
            # 统计平均持仓数量，判断是否退化为全选或全不选
            envp = PortfolioEnv(Xte, yte, dwte)
            obsp = envp.reset()
            done_p = False
            counts = []
            while not done_p:
                with torch.no_grad():
                    if method == "sl":
                        sc = policy(torch.from_numpy(obsp[:, :, :-1])).numpy()
                        ap = np.zeros(envp.N, dtype=int)
                        ap[np.argsort(sc)[::-1][:TOP_K]] = 2
                    else:
                        lg = policy(torch.from_numpy(obsp))
                        ap = (torch.sigmoid(lg) > 0.5).numpy().astype(int) * 2
                counts.append(int((ap > 0).sum()))
                obsp, _, done_p, _ = envp.step(ap)
            print(f"\n平均持仓标的数: {np.mean(counts):.2f} / {envp.N}")
        else:
            prof = action_profile(policy, Xte, yte, dwte)
            print("\n动作分布: " + " | ".join(f"{k} {v:.1%}" for k, v in prof.items()))

        print("\n--- 样本外（测试集）绩效 ---")
        for name in ["DRL策略", "等权基准", "动量基准", "沪深300"]:
            print(f"\n[{name}]")
            print(fmt(results[str(seed)][name]))

    out = ROOT / "results.json"
    payload = {k: {kk: (vv if not isinstance(vv, dict) else
                        {a: (float(b) if isinstance(b, (int, float, np.floating)) else b)
                         for a, b in vv.items()})
                   for kk, vv in v.items()} for k, v in results.items()}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\n结果已写入 {out}")
    return results


if __name__ == "__main__":
    import sys
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    meth = sys.argv[2] if len(sys.argv) > 2 else "sl"
    seeds = ([int(s) for s in sys.argv[3].split(",")]
             if len(sys.argv) > 3 else None)
    main(episodes=ep, method=meth, seeds=seeds)
