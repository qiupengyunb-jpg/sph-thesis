"""Time-series figures for the resolution comparison.

1) r=32 morphology evolution across T (the standing per-round convention)
2) resolution x time morphology grid  -> shows "finer = worse" visually
3) extended metric time series (6 panels: delta, gap, min S/S0, n>0.03,
   h_max, connected blocks)
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, r"E:\哈哈\_pr_diag")
from compare_resolutions import RUNS, fast_load, frame_T_map

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SRC = Path(r"E:\哈哈\_pr_diag\out\res_compare")
OUT = Path(r"E:\sphmethod\SPH_results_center\recon_tapered_20260914"
           r"\resolution_r16_r24_r32_20260915")
STYLE = {"r16": "#1f77b4", "r24": "#ff7f0e", "r32": "#d62728"}
WANT_T = (0.0, 20.0, 40.0, 60.0, 80.0)


def frame_at(tag, want):
    d, dx = RUNS[tag]
    fs = frame_T_map(d)
    p, T = min(fs, key=lambda it: abs(it[1] - want))
    a = fast_load(p)
    fib = p.parent / p.name.replace("LiquidFilmHalf_", "RigidFiberHalf_")
    f = fast_load(fib) if fib.exists() else None
    return T, a["Position"][:, :2], (f["Position"][:, :2] if f else None)


def panel(ax, liq, fib, label, scale_bar=False):
    for sign in (1, -1):
        ax.scatter(liq[:, 0], sign * liq[:, 1], s=0.05, c="#4c8fd6", lw=0,
                   rasterized=True)
        if fib is not None:
            ax.scatter(fib[:, 0], sign * fib[:, 1], s=0.05, c="#8c8c8c", lw=0,
                       rasterized=True)
    ax.set_xlim(-14, 14)
    ax.set_ylim(-3.9, 3.9)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(-13.4, 2.6, label, fontsize=10)
    if scale_bar:
        ax.plot([6.0, 8.0], [3.3, 3.3], "-", color="k", lw=3)
        ax.text(6.1, 3.45, "2a", fontsize=8)


# ---------------------------------------------------------------- figure 1
fig, axes = plt.subplots(5, 1, figsize=(12, 11), dpi=150)
for ax, want in zip(axes, WANT_T):
    T, liq, fib = frame_at("r32", want)
    panel(ax, liq, fib, "r=32   T = %.0f" % T, scale_bar=(ax is axes[0]))
    ax.set_ylabel("r / a")
axes[-1].set_xlabel("轴向坐标 x / a")
fig.suptitle("细—粗—细无正则化基准 r=32 形貌演化（半侧解按 r→−r 镜像；每 20 个 T 一帧）",
             fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "morph_r32_T0_T20_T40_T60_T80.png")
plt.close(fig)
print("wrote morph_r32_T0_T20_T40_T60_T80.png")

# ---------------------------------------------------------------- figure 2
fig, axes = plt.subplots(len(RUNS), len(WANT_T), figsize=(16, 8), dpi=150,
                         sharex=True, sharey=True)
for i, tag in enumerate(("r16", "r24", "r32")):
    for j, want in enumerate(WANT_T):
        T, liq, fib = frame_at(tag, want)
        ax = axes[i][j]
        panel(ax, liq, fib, "T = %.0f" % T if i == 0 else "",
              scale_bar=(i == 0 and j == 0))
        if j == 0:
            ax.set_ylabel("r / a\n(%s)" % tag, fontsize=10)
        if i == len(RUNS) - 1:
            ax.set_xlabel("x / a")
fig.suptitle("同一时刻的形貌随分辨率变化：分辨率越高，鼓包之间的径向失联越早越强"
             "（行 = r16/r24/r32，列 = T）", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "morph_resolution_grid.png")
plt.close(fig)
print("wrote morph_resolution_grid.png")

# ---------------------------------------------------------------- figure 3
def rd(t):
    with open(SRC / ("%s.csv" % t), encoding="utf-8") as fh:
        recs = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in recs]) for k in recs[0]}


D = {t: rd(t) for t in STYLE}
fig, axes = plt.subplots(2, 3, figsize=(17, 8), dpi=150)
ax = axes[0][0]
for t, c in STYLE.items():
    ax.plot(D[t]["T"], D[t]["dmax"], "-", color=c, lw=1.6, label=t)
ax.axhline(0.03, color="k", ls=":", lw=1.2)
ax.text(2, 0.032, "δ_thr=0.03", fontsize=8)
ax.set_ylabel("δ 最大值")
ax.set_title("δ 最大值 vs T")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[0][1]
for t, c in STYLE.items():
    ax.plot(D[t]["T"], D[t]["gap_central"], "-", color=c, lw=1.6, label=t)
for y in (1.5, 2.0, 3.0):
    ax.axhline(y, ls=":", color="gray", lw=0.8)
ax.set_yscale("log")
ax.set_ylabel("中央区最大径向间隙 / dx")
ax.set_title("径向间隙 vs T")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

ax = axes[0][2]
for t, c in STYLE.items():
    ax.plot(D[t]["T"], D[t]["min_S"], "-", color=c, lw=1.6, label=t)
ax.set_ylabel("最小 S/S0")
ax.set_title("核支撑最低点 vs T")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1][0]
for t, c in STYLE.items():
    ax.plot(D[t]["T"], D[t]["n003"], "-", color=c, lw=1.6, label=t)
ax.set_yscale("symlog", linthresh=10)
ax.set_xlabel("无量纲时间 T"); ax.set_ylabel("δ>0.03 粒子数")
ax.set_title("越阈粒子数 vs T")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1][1]
for t, c in STYLE.items():
    ax.plot(D[t]["T"], D[t]["h_max"], "-", color=c, lw=1.6, label=t)
ax.set_xlabel("无量纲时间 T"); ax.set_ylabel("h_max / a")
ax.set_title("最大膜厚 vs T（物理量：r24 与 r32 基本重合）")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1][2]
for t, c in STYLE.items():
    ax.plot(D[t]["T"], D[t]["blocks"], "-", color=c, lw=1.6, label=t)
ax.set_yscale("log")
ax.set_xlabel("无量纲时间 T"); ax.set_ylabel("连通块数")
ax.set_title("连通块 vs T")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle("r=16 / r=24 / r=32 逐帧时间序列（同域长 70a、同扰动、同开关，唯一变量 = 分辨率）",
             fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "resolution_timeseries_6panel.png")
print("wrote resolution_timeseries_6panel.png")
