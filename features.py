"""
特征工程：构造每只标的的时序特征张量与次日收益标签。

严格避免前视偏差（look-ahead bias）：
1. 所有技术指标仅使用 t 日及之前的收盘价；
2. 标准化参数（均值/标准差）仅在训练集区间上拟合，再应用到验证集与测试集；
3. 收益标签使用 t 日决策 -> t+1 日实现的收益。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACK = 20  # 每个样本回看的交易日数


def _asset_features(close: pd.Series) -> pd.DataFrame:
    """单只标的的价格类特征。"""
    logp = np.log(close)
    ret = logp.diff()

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()

    up = ret.clip(lower=0).rolling(14).mean()
    dn = (-ret.clip(upper=0)).rolling(14).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    hi20 = close.rolling(20).max()
    lo20 = close.rolling(20).min()

    feat = pd.DataFrame(
        {
            "ret1": ret,
            "ret5": logp.diff(5),
            "ret20": logp.diff(20),
            "vol20": ret.rolling(20).std(),
            "ma_ratio": ma5 / ma20 - 1.0,
            "px_ma20": close / ma20 - 1.0,
            "rsi14": 1.0 - 1.0 / (1.0 + up / (dn + 1e-8)),
            "macd": (ema12 - ema26) / close,
            "pos20": (close - lo20) / (hi20 - lo20 + 1e-8),
        },
        index=close.index,
    )
    return feat


def _market_features(bench: pd.Series) -> pd.DataFrame:
    """市场层面特征（沪深300），作为所有标的共享的宏观上下文。"""
    logb = np.log(bench)
    ret = logb.diff()
    return pd.DataFrame(
        {
            "mkt_ret1": ret,
            "mkt_ret20": logb.diff(20),
            "mkt_vol20": ret.rolling(20).std(),
        },
        index=bench.index,
    )


def build_dataset(
    panel: pd.DataFrame, bench: pd.Series
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list[str], list[str]]:
    """
    返回:
        X         : (T, N, F) 特征矩阵（已按训练集统计量标准化）
        next_ret  : (T, N)   每只标的 t->t+1 的简单收益率
        dates     : (T,)     对应的决策日
        asset_names / feature_names
    """
    mkt = _market_features(bench)

    per_asset = []
    for col in panel.columns:
        per_asset.append(_asset_features(panel[col]))

    # 对齐所有标的与市场的有效区间（去掉滚动窗口产生的 NaN 头部）
    valid = per_asset[0].notna().all(axis=1) & mkt.notna().all(axis=1)
    for fa in per_asset[1:]:
        valid &= fa.notna().all(axis=1)
    dates_all = panel.index[valid.to_numpy()]

    n_assets = len(per_asset)
    n_feat = per_asset[0].shape[1] + mkt.shape[1]

    # 原始特征张量：(T, N, F)
    feats = np.zeros((len(dates_all), n_assets, n_feat), dtype=np.float32)
    for i, fa in enumerate(per_asset):
        block = fa.loc[dates_all].to_numpy(dtype=np.float32)
        mblock = mkt.loc[dates_all].to_numpy(dtype=np.float32)
        feats[:, i, :] = np.concatenate([block, mblock], axis=1)

    # 次日收益：close(t+1)/close(t) - 1
    px = panel.loc[dates_all].to_numpy(dtype=np.float64)
    all_px = panel.to_numpy(dtype=np.float64)
    idx_map = {d: i for i, d in enumerate(panel.index)}
    # 决策日 t 的收益由 t 到 t+1 的收盘价变化决定
    next_ret = np.zeros_like(px)
    for t, d in enumerate(dates_all):
        j = idx_map[d]
        if j + 1 < len(panel.index):
            next_ret[t] = all_px[j + 1] / all_px[j] - 1.0
        else:
            next_ret[t] = 0.0

    feature_names = list(per_asset[0].columns) + list(mkt.columns)
    return feats, next_ret, dates_all, list(panel.columns), feature_names


def make_windows(
    feats: np.ndarray, next_ret: np.ndarray, dates: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """把 (T, N, F) 切成 (T-L, N, L, F) 的回看窗口。"""
    T, N, F = feats.shape
    X = np.zeros((T - LOOKBACK, N, LOOKBACK, F), dtype=np.float32)
    for t in range(LOOKBACK, T):
        X[t - LOOKBACK] = feats[t - LOOKBACK + 1 : t + 1].transpose(1, 0, 2)
    y = next_ret[LOOKBACK:]  # 决策发生在窗口末尾，收益在下一日实现
    d = dates[LOOKBACK:]
    return X, y, d


def fit_scaler(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """在训练集上拟合标准化参数（按特征维度）。"""
    mu = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    sd = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    sd[sd < 1e-8] = 1.0
    return mu, sd
