"""Deliverable figures: no-reg vs surface-only, plus the inert-operator proof."""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(r"E:\哈哈\_pr_diag\out")


def read(path):
    with open(path, encoding="utf-8") as fh:
        recs = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in recs]) for k in recs[0]}


base = read(OUT / "base52" / "metrics.csv")
srf = read(OUT / "surfreg" / "metrics.csv")

fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), dpi=150)

ax = axes[0, 0]
ax.plot(base["T"], base["gap_central_max"], "-", lw=4, color="#1f77b4",
        alpha=0.5, label="无正则化基准 |x|<4a")
ax.plot(srf["T"], srf["gap_central_max"], "o--", ms=5, lw=1, color="#d62728",
        label="surface-only |x|<4a")
for y, lab in ((1.5, "1.5dx"), (2.0, "2dx"), (3.0, "3dx")):
    ax.axhline(y, ls=":", lw=0.8, color="gray")
    ax.text(0.5, y + 0.05, lab, fontsize=8, color="gray")
ax.set_yscale("log")
ax.set_xlabel("无量纲时间 T")
ax.set_ylabel("中央区最大径向间隙 / dx")
ax.set_title("图A  间隙演化：两条曲线逐帧完全重合")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(base["T"], base["h_max"], "-", lw=4, color="#1f77b4", alpha=0.5,
        label="无正则化基准")
ax.plot(srf["T"], srf["h_max"], "o--", ms=5, lw=1, color="#d62728",
        label="surface-only")
ax.axhline(1.18, ls=":", color="gray", lw=1)
ax.text(0.5, 1.185, "基准 T≈40 的 1.18a", fontsize=8, color="gray")
ax.set_xlabel("无量纲时间 T")
ax.set_ylabel(r"最大膜厚 $h_{max}/a$")
ax.set_title("图B  最大膜厚")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(base["T"], base["max_speed"], "-", lw=4, color="#1f77b4", alpha=0.5,
        label="无正则化基准")
ax.plot(srf["T"], srf["max_speed"], "o--", ms=5, lw=1, color="#d62728",
        label="surface-only")
ax.set_xlabel("无量纲时间 T")
ax.set_ylabel("最大速度")
ax.set_title("图C  最大速度")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.plot(base["T"], base["blocks"], "-", lw=4, color="#1f77b4", alpha=0.5,
        label="无正则化基准")
ax.plot(srf["T"], srf["blocks"], "o--", ms=5, lw=1, color="#d62728",
        label="surface-only")
ax.set_yscale("log")
ax.set_xlabel("无量纲时间 T")
ax.set_ylabel("液体连通块数")
ax.set_title("图D  几何连通性")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle("surface-only 单变量对照：全部指标与基准逐帧完全相同（max|Δx| = 0）",
             fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "cmp_surfreg_vs_base.png")
plt.close(fig)

# ---- why it was inert
sys.path.insert(0, r"E:\哈哈\_pr_diag")
from run_metrics import load

SRF = Path(r"E:\sphmethod\SPH_results_center\recon_tapered_20260914"
           r"\r24_surfreg_T50\output_r24_surfreg_T50")
TAGS = ("ite_0000000000", "0001341640", "0020124611", "0033541019")
T_LA = ("T=0", "T=2", "T=30", "T=50")

topo, shift, normdir, resid = [], [], [], []
for tag in TAGS:
    a = load(SRF / ("LiquidFilmHalf_%s.vtp" % tag))
    ind = a["Indicator"] == 1
    topo.append(float(np.linalg.norm(a["TopologyNormal"][:, :2][ind],
                                     axis=1).max()))
    shift.append(float(np.linalg.norm(
        a["SurfaceRegularizationShift"][:, :2][ind], axis=1).max()))
    normdir.append(float(np.linalg.norm(a["NormDirection"][:, :2][ind],
                                        axis=1).max()))

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), dpi=150)
ax = axes[0]
x = np.arange(len(TAGS))
ax.bar(x - 0.26, topo, 0.26, color="#d62728", label="TopologyNormal（算子读的输入）")
ax.bar(x, shift, 0.26, color="#ff7f0e", label="写出的 SurfaceRegularizationShift")
ax.bar(x + 0.26, normdir, 0.26, color="#2ca02c",
       label="NormDirection（本来可用的法向）")
for xi, v in zip(x - 0.26, topo):
    ax.text(xi, 0.02, "0", ha="center", fontsize=9, color="#7f0000")
for xi, v in zip(x, shift):
    ax.text(xi, 0.02, "0", ha="center", fontsize=9, color="#7f4f00")
for xi, v in zip(x + 0.26, normdir):
    ax.text(xi, v + 0.02, "%.1f" % v, ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(T_LA)
ax.set_ylim(0, 1.25)
ax.set_ylabel("向量的最大模长")
ax.set_title("图E  算子为什么没起作用：它读的法向量恒为 0")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

ax = axes[1]
labels = ["残差最大值", "残差在自由表面的均值",
          "按系数换算的理想位移\n（dx 为单位）", "位移上限 surface_max_shift_dx"]
vals = [24.5137, 11.7880, 0.0511, 0.03]
colors = ["#1f77b4", "#1f77b4", "#ff7f0e", "#7f7f7f"]
bars = ax.barh(labels, vals, color=colors)
for b, v in zip(bars, vals):
    ax.text(v * 1.02, b.get_y() + b.get_height() / 2, "%.4g" % v,
            va="center", fontsize=9)
ax.set_xscale("log")
ax.set_xlabel("数值（对数轴）")
ax.set_title("图F  算子本来会给出的修正量并不小\n"
             "（残差 = -2·Σ dW_ij·V_j·e_ij，离线在同一帧复算）")
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(OUT / "cmp_why_inert.png")
plt.close(fig)

print("wrote", OUT / "cmp_surfreg_vs_base.png")
print("wrote", OUT / "cmp_why_inert.png")

print("\n逐帧最大绝对差（surface-only 减基准）")
for key in ("gap_central_max", "gap_interior_max", "h_max", "max_speed",
            "blocks", "biggest_block_frac", "rho_c_min"):
    n = min(len(base[key]), len(srf[key]))
    d = np.abs(base[key][:n] - srf[key][:n])
    print("  %-20s max|diff| = %.6g" % (key, np.nanmax(d)))
