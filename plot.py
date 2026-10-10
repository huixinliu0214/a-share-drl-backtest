"""绘制样本外净值曲线对比图。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent

for font in ["PingFang HK", "PingFang SC", "Heiti TC", "Songti SC", "Arial Unicode MS"]:
    try:
        matplotlib.font_manager.findfont(font, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [font]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False


def main():
    data = json.loads((ROOT / "results.json").read_text())
    seeds = list(data.keys())
    seed = seeds[0]

    nav = np.array(data[seed]["nav"])
    ew = np.array(data[seed]["ew_nav"])
    bench = np.array(data[seed]["bench_nav"])

    # dates 是调仓日（周频），净值序列是日频，按调仓周期展开对齐
    dates = data[seed]["dates"]
    rep = max(1, round(len(nav) / max(len(dates), 1)))
    daily_dates = np.repeat(dates, rep)[: len(nav)]

    x = np.arange(len(nav))
    tick = np.linspace(0, len(nav) - 1, 8).astype(int)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    ax.plot(x, nav, label="LSTM-Attention 选股策略", color="#C0392B", linewidth=1.6)
    ax.plot(x, ew, label="等权基准（40 只成分股）", color="#2E86C1", linewidth=1.4,
            linestyle="--")
    ax.plot(x, bench, label="沪深300", color="#7F8C8D", linewidth=1.4, linestyle=":")

    ax.set_xticks(tick)
    ax.set_xticklabels([daily_dates[i] for i in tick], rotation=30, ha="right")
    ax.set_title("样本外净值曲线（2023-07 ~ 2026-09，周频调仓，已扣交易成本）",
                 fontsize=13, pad=14)
    ax.set_ylabel("净值（起点 = 1.0）")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25, linewidth=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    out = ROOT / "nav_curve.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")

    # 多种子年化收益散点
    ann = [data[s]["DRL策略"]["年化收益"] for s in seeds]
    fig2, ax2 = plt.subplots(figsize=(7, 4), dpi=150)
    ax2.bar(range(len(seeds)), ann, color="#C0392B", alpha=0.8)
    ax2.axhline(data[seeds[0]]["等权基准"]["年化收益"], color="#2E86C1",
                linestyle="--", label="等权基准")
    ax2.axhline(data[seeds[0]]["沪深300"]["年化收益"], color="#7F8C8D",
                linestyle=":", label="沪深300")
    ax2.set_xticks(range(len(seeds)))
    ax2.set_xticklabels([f"seed {s}" for s in seeds])
    ax2.set_ylabel("样本外年化收益")
    ax2.set_title("多种子重复实验：策略表现对随机种子高度敏感", fontsize=12, pad=12)
    ax2.legend(frameon=False)
    ax2.grid(axis="y", alpha=0.25, linewidth=0.6)
    for s in ["top", "right"]:
        ax2.spines[s].set_visible(False)
    fig2.tight_layout()
    out2 = ROOT / "seed_sensitivity.png"
    fig2.savefig(out2, bbox_inches="tight")
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
