"""r=16 / r=24 / r=32 comparison figure."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SRC = Path(r"E:\哈哈\_pr_diag\out\res_compare")
OUT = Path(r"E:\sphmethod\SPH_results_center\recon_tapered_20260914"
           r"\resolution_r16_r24_r32_20260915")
STYLE = {"r16": ("#1f77b4", "o"), "r24": ("#ff7f0e", "s"), "r32": ("#d62728", "^")}


def rd(t):
    with open(SRC / ("%s.csv" % t), encoding="utf-8") as fh:
        recs = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in recs]) for k in recs[0]}


D = {t: rd(t) for t in STYLE}

fig, axes = plt.subplots(1, 4, figsize=(19, 4.4), dpi=150)

ax = axes[0]
for t, (c, m) in STYLE.items():
    ax.plot(D[t]["T"], D[t]["dmax"], "-", color=c, lw=1.6, label="%s  max δ" % t)
ax.axhline(0.03, color="k", ls=":", lw=1.2)
ax.text(2, 0.032, "δ_thr = 0.03", fontsize=8)
ax.set_xlabel("无量纲时间 T")
ax.set_ylabel("δ 最大值（|x₀|<6a 内部粒子）")
ax.set_title("图1  分辨率越高，δ 越早越大")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
for t, (c, m) in STYLE.items():
    ax.plot(D[t]["T"], D[t]["gap_central"], "-", color=c, lw=1.6,
            label="%s" % t)
for y, lab in ((1.5, "1.5dx"), (2.0, "2dx"), (3.0, "3dx")):
    ax.axhline(y, ls=":", color="gray", lw=1)
    ax.text(1, y * 1.05, lab, fontsize=8, color="gray")
ax.set_yscale("log")
ax.set_xlabel("无量纲时间 T")
ax.set_ylabel("中央区最大径向间隙 / dx")
ax.set_title("图2  间隙打开时刻随分辨率提前（60→52→46）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[2]
for t, (c, m) in STYLE.items():
    ax.plot(D[t]["T"], D[t]["min_S"], "-", color=c, lw=1.6, label="%s" % t)
ax.set_xlabel("无量纲时间 T")
ax.set_ylabel("最小 S/S0")
ax.set_title("图3  核支撑最低点随分辨率降低")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[3]
for t, (c, m) in STYLE.items():
    ax.plot(D[t]["T"], D[t]["n003"], "-", color=c, lw=1.6,
            label="%s" % t)
ax.set_yscale("symlog", linthresh=10)
ax.set_xlabel("无量纲时间 T")
ax.set_ylabel("δ>0.03 的粒子数")
ax.set_title("图4  越阈粒子数：16 对 30 对 1000+")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle("细—粗—细无正则化基准的分辨率对照（r=16 / r=24 / r=32，同域长 70a、同扰动、同开关）",
             fontsize=12)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "resolution_r16_r24_r32.png")
print("wrote", OUT / "resolution_r16_r24_r32.png")
