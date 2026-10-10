"""
数据获取模块：抓取真实 A 股日线行情（前复权）。

数据源：腾讯财经行情接口（稳定、无需 token）。
为什么自己写而不直接用 akshare：akshare 默认走东财接口，在当前网络环境下
被代理频繁限流；腾讯接口按年分段请求可稳定拿到完整历史。

设计原则：
1. 采用后复权（hfq）价格：长期高分红个股的前复权价格会变成负数（如茅台在
   2015 年前复权价为 -143），导致收益率完全失真；后复权价格恒为正，
   且收益率等价于含分红再投资的总收益；
2. 按年分段请求（单次上限约 250 条），合并后去重；
3. 原始数据落盘缓存，后续实验不重复请求网络。
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# 沪深300 成分股，覆盖金融/消费/医药/科技/资源/地产/公用等 8 个行业。
# 均为 2015 年前上市，保证回测区间数据完整。
# 股票池需要足够大：只有 10 只龙头股时，标的间差异有限，
# 选股任务本身可挖掘的信息量不足，策略容易退化成噪声交易。
UNIVERSE = {
    # 消费
    "sh600519": "贵州茅台", "sz000858": "五粮液", "sz000568": "泸州老窖",
    "sh600887": "伊利股份", "sh603288": "海天味业",
    # 金融
    "sh600036": "招商银行", "sh601318": "中国平安", "sh600000": "浦发银行",
    "sh601166": "兴业银行", "sz000001": "平安银行", "sh600030": "中信证券",
    "sh601601": "中国太保", "sh601688": "华泰证券",
    # 医药
    "sh600276": "恒瑞医药", "sh600196": "复星医药", "sz000538": "云南白药",
    "sh600436": "片仔癀",
    # 科技 / 电子
    "sz000725": "京东方A", "sz002415": "海康威视", "sh600703": "三安光电",
    "sz002475": "立讯精密", "sz000063": "中兴通讯",
    # 家电 / 汽车
    "sz000651": "格力电器", "sz000333": "美的集团", "sh600690": "海尔智家",
    "sz002594": "比亚迪", "sh600104": "上汽集团",
    # 资源 / 材料
    "sh601899": "紫金矿业", "sh600028": "中国石化", "sh601857": "中国石油",
    "sh600019": "宝钢股份", "sh603993": "洛阳钼业", "sh600585": "海螺水泥",
    # 地产 / 建筑 / 公用
    "sh600048": "保利发展", "sh601668": "中国建筑", "sh600009": "上海机场",
    "sh601006": "大秦铁路", "sh600900": "长江电力", "sh601012": "隆基绿能",
    # 机械
    "sh600031": "三一重工",
}

BENCHMARK_CODE = "sh510300"
BENCHMARK_NAME = "沪深300ETF"

START_YEAR = 2015
END_YEAR = 2026
END_DATE = "2026-09-30"

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def _tx_request(code: str, start: str, end: str, count: int = 320,
                adjust: str = "hfq") -> list | None:
    """请求单一年份的日线数据。"""
    url = (
        f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        f"?param={code},day,{start},{end},{count},{adjust}"
    )
    for attempt in range(4):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            payload = resp.json()
            if payload.get("code") != 0:
                return None
            node = payload["data"][code]
            return node.get(f"{adjust}day") or node.get("day")
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def fetch_symbol(code: str, name: str, adjust: str = "hfq") -> pd.DataFrame | None:
    """按年分段拉取单只标的的完整复权日线。"""
    cache = DATA_DIR / f"{code.replace('sh', '').replace('sz', '')}_{adjust}.csv"
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["date"])
        print(f"[cache] {code} {name}: {len(df)} rows")
        return df

    rows = []
    for year in range(START_YEAR, END_YEAR + 1):
        start = f"{year}-01-01"
        end = END_DATE if year == END_YEAR else f"{year}-12-31"
        chunk = _tx_request(code, start, end, adjust=adjust)
        if chunk:
            rows.extend(chunk)
        time.sleep(0.4)

    if not rows:
        print(f"[FAIL] {code} {name}: 无数据")
        return None

    df = pd.DataFrame(
        [r[:6] for r in rows],
        columns=["date", "open", "close", "high", "low", "volume"],
    )
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "close", "high", "low", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().drop_duplicates("date").sort_values("date").reset_index(drop=True)

    df.to_csv(cache, index=False)
    print(f"[fetch] {code} {name}: {len(df)} rows "
          f"({df['date'].min().date()} ~ {df['date'].max().date()})")
    return df


def build_panel() -> tuple[pd.DataFrame, pd.Series]:
    """对齐交易日，返回 (收盘价宽表, 基准收盘价序列)。"""
    frames = {}
    for code, name in UNIVERSE.items():
        df = fetch_symbol(code, name)
        if df is not None:
            frames[code] = df.set_index("date")["close"]

    if len(frames) < 5:
        raise RuntimeError(f"有效标的仅 {len(frames)} 只，数据不足，无法继续")

    bench_df = fetch_symbol(BENCHMARK_CODE, BENCHMARK_NAME)
    if bench_df is None:
        raise RuntimeError("基准数据获取失败，无法继续")

    panel = pd.DataFrame(frames).sort_index()
    panel = panel.dropna(how="any")  # 只在所有标的均有成交的交易日计算

    bench = bench_df.set_index("date")["close"].reindex(panel.index).ffill()
    print(f"\n[panel] shape={panel.shape} "
          f"{panel.index.min().date()} ~ {panel.index.max().date()}")
    return panel, bench


if __name__ == "__main__":
    panel, bench = build_panel()
    panel.to_csv(DATA_DIR / "panel_close.csv")
    bench.to_frame("bench").to_csv(DATA_DIR / "benchmark.csv")
    print("saved panel_close.csv / benchmark.csv")
    print(panel.tail(3).to_string())
