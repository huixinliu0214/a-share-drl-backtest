"""
绩效评估：年化收益、夏普、最大回撤、Alpha/Beta、波动率、换手率等。
全部基于日频收益率序列计算，年化按 252 个交易日。
"""
from __future__ import annotations

import numpy as np

TRADING_DAYS = 252
RISK_FREE = 0.02  # 年化无风险利率


def max_drawdown(nav: np.ndarray) -> float:
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0
    return float(-dd.min())


def summarize(daily_ret: np.ndarray, bench_ret: np.ndarray | None = None,
              turnover: np.ndarray | None = None) -> dict:
    daily_ret = np.asarray(daily_ret, dtype=np.float64)
    nav = np.cumprod(1.0 + daily_ret)
    n = len(daily_ret)
    years = n / TRADING_DAYS

    total_ret = float(nav[-1] - 1.0)
    ann_ret = float(nav[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0
    ann_vol = float(daily_ret.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = (ann_ret - RISK_FREE) / ann_vol if ann_vol > 1e-8 else 0.0
    mdd = max_drawdown(nav)
    calmar = ann_ret / mdd if mdd > 1e-8 else 0.0
    win_rate = float((daily_ret > 0).mean())

    out = {
        "累计收益": total_ret,
        "年化收益": ann_ret,
        "年化波动": ann_vol,
        "夏普比率": sharpe,
        "最大回撤": mdd,
        "Calmar": calmar,
        "日胜率": win_rate,
        "样本天数": n,
    }

    if turnover is not None:
        out["期均换手"] = float(np.mean(turnover))

    if bench_ret is not None:
        b = np.asarray(bench_ret, dtype=np.float64)
        b = b[: len(daily_ret)]
        r = daily_ret[: len(b)]
        var_b = b.var(ddof=1)
        beta = float(np.cov(r, b, ddof=1)[0, 1] / var_b) if var_b > 1e-12 else 0.0
        b_nav = np.cumprod(1.0 + b)
        b_ann = float(b_nav[-1] ** (1.0 / (len(b) / TRADING_DAYS)) - 1.0)
        alpha = ann_ret - (RISK_FREE + beta * (b_ann - RISK_FREE))
        out["Beta"] = beta
        out["年化Alpha"] = alpha
        out["基准年化"] = b_ann

    return out


def fmt(d: dict) -> str:
    pct = {"累计收益", "年化收益", "年化波动", "最大回撤", "日胜率", "年化Alpha", "基准年化"}
    lines = []
    for k, v in d.items():
        if k == "样本天数":
            lines.append(f"{k:>10}: {int(v)}")
        elif k in pct:
            lines.append(f"{k:>10}: {v:7.2%}")
        elif isinstance(v, float):
            lines.append(f"{k:>10}: {v:7.3f}")
        else:
            lines.append(f"{k:>10}: {v}")
    return "\n".join(lines)
