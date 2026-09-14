"""Figures 1-5 plus a morphology panel, Chinese captions."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(r"E:\哈哈\_pr_diag\out")
z = np.load(OUT / "support_frames.npz")
tags = sorted({k.split("_")[0] for k in z.files}, key=lambda k: float(k[1:]))
T = np.array([float(t[1:]) for t in tags])

ref = {n: z["T0_%s" % n] for n in ("id", "s_num")}
ref_map = dict(zip(ref["id"], ref["s_num"]))


def frame(tag):
    return {n: z["%s_%s" % (tag, n)] for n in
            ("x", "r", "rho_c", "rho_s", "s_num", "nbr", "d_min", "p",
             "gap", "id", "mass", "indicator")}


def ratio(a):
    return a["s_num"] / np.array([ref_map[i] for i in a["id"]])


# ---------------------------------------------------------------- figure 1
gmax_all, gmax_cen, min_ratio, n_lt95, n_lt90, nbr_min, rho_c_min, rho_s_min = \
    ([] for _ in range(8))
for tag in tags:
    a = frame(tag)
    reg = np.abs(a["x"]) < 6.0
    cen = np.abs(a["x"]) < 4.0
    gmax_all.append(np.nanmax(a["gap"][np.abs(a["x"]) < 26.0]))
    gmax_cen.append(np.nanmax(a["gap"][cen]))
    min_ratio.append(ratio(a)[reg].min())
    n_lt95.append(int((ratio(a)[reg] < 0.95).sum()))
    n_lt90.append(int((ratio(a)[reg] < 0.90).sum()))
    nbr_min.append(a["nbr"][reg].min())
    rho_c_min.append(a["rho_c"][reg].min())
    rho_s_min.append(a["rho_s"][reg].min())

fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), dpi=150)
ax = axes[0, 0]
ax.plot(T, gmax_all, "o-", ms=4, color="#1f77b4", label="整个内部区 |x|<26a")
ax.plot(T, gmax_cen, "s-", ms=4, color="#d62728", label="中央病灶区 |x|<4a")
for y, lab in ((1.5, "1.5dx"), (2.0, "2dx"), (3.0, "3dx")):
    ax.axhline(y, ls=":", lw=1, color="gray")
    ax.text(1, y + 0.15, lab, fontsize=8, color="gray")
ax.set_yscale("log")
ax.set_xlabel("无量纲时间 T")
ax.set_ylabel("最大径向相邻间隙 / dx")
ax.set_title("图1a  径向间隙随时间：病灶在 T≈50 才打开")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(T, min_ratio, "o-", ms=4, color="#2ca02c")
ax.set_xlabel("无量纲时间 T")
ax.set_ylabel("最小核支撑比 S(T)/S(0)")
ax.set_title("图1b  核支撑在 T≈20 就开始流失")
ax.axhline(0.95, ls=":", color="gray", lw=1)
ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(T, n_lt95, "o-", ms=4, label="S/S0 < 0.95")
ax.plot(T, n_lt90, "s-", ms=4, label="S/S0 < 0.90")
ax.set_xlabel("无量纲时间 T")
ax.set_ylabel("粒子数（|x|<6a 内）")
ax.set_title("图1c  支撑受损粒子数：先缓慢增长，T≈56 后跳升")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.plot(T, rho_c_min, "o-", ms=4, color="#1f77b4", label="连续方程密度最小值")
ax.plot(T, rho_s_min, "s-", ms=4, color="#d62728", label="离线求和密度最小值")
ax.set_xlabel("无量纲时间 T")
ax.set_ylabel(r"密度 $\rho/\rho_0$（|x|<6a）")
ax.set_title("图1d  两种密度的分离")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig1_gap_support_density_vs_T.png")
plt.close(fig)

# ---------------------------------------------------------------- figure 2/3
a62 = frame("T62")
r62 = ratio(a62)
cen62 = np.abs(a62["x"]) < 6.0
gap62 = a62["gap"][cen62]

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=150)
ax = axes[0]
ax.scatter(gap62, r62[cen62], s=3, alpha=0.25, c=a62["r"][cen62],
           cmap="viridis")
ax.axhline(1.0, ls="--", color="gray", lw=1)
ax.set_xlabel("该粒子所在 2a 轴箱内的最大径向间隙 / dx")
ax.set_ylabel("核支撑比 S(T)/S(0)")
ax.set_title("图2  核支撑随几何间隙下降（T=62，|x|<6a）\n颜色=径向坐标 r")
ax.grid(alpha=0.3)
cb = fig.colorbar(ax.collections[0], ax=ax)
cb.set_label("r / a")

ax = axes[1]
ax.scatter(gap62, a62["nbr"][cen62], s=3, alpha=0.25, color="#d62728")
ax.axhline(21, ls="--", color="gray", lw=1)
ax.text(0.5, 21.4, "完整支撑约 21 个邻居", fontsize=8, color="gray")
ax.set_xlabel("该粒子所在 2a 轴箱内的最大径向间隙 / dx")
ax.set_ylabel("核半径内邻居数")
ax.set_title("图3  邻居数随几何间隙下降（T=62，|x|<6a）")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig2_fig3_support_nbr_vs_gap.png")
plt.close(fig)

# ---------------------------------------------------------------- figure 5
a = a62
sel = np.abs(a["x"]) < 8.0
bins = np.arange(-8.0, 8.01, 0.5)
idx = np.digitize(a["x"][sel], bins) - 1
bx, bgap, bsup, brc, brs, bnbr = ([] for _ in range(6))
for b in range(len(bins) - 1):
    m = idx == b
    if m.sum() < 6:
        continue
    bx.append(0.5 * (bins[b] + bins[b + 1]))
    bgap.append(np.nanmax(a["gap"][sel][m]))
    bsup.append(r62[sel][m].min())
    brc.append(a["rho_c"][sel][m].min())
    brs.append(a["rho_s"][sel][m].min())
    bnbr.append(a["nbr"][sel][m].min())

fig, axes = plt.subplots(4, 1, figsize=(11, 9), dpi=150, sharex=True)
axes[0].step(bx, bgap, where="mid", color="#d62728")
axes[0].set_ylabel("径向间隙 / dx")
axes[0].set_title("图5  T=62 时中央区域的轴向剖面（|x|<8a，0.5a 分箱取最不利值）")
axes[1].step(bx, bnbr, where="mid", color="#ff7f0e")
axes[1].set_ylabel("最小邻居数")
axes[2].step(bx, bsup, where="mid", color="#2ca02c")
axes[2].set_ylabel("最小支撑比")
axes[3].step(bx, brc, where="mid", color="#1f77b4", label="连续方程密度 ρ_c")
axes[3].step(bx, brs, where="mid", color="#d62728", label="离线求和密度 ρ_s")
axes[3].set_ylabel(r"$\rho/\rho_0$")
axes[3].set_xlabel("轴向坐标 x / a")
axes[3].legend(fontsize=8)
axes[3].set_ylim(0.5, 1.05)
for ax in axes:
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig5_axial_profile_T62.png")
plt.close(fig)

# ---------------------------------------------------------------- figure 4
late = frame("T62")
worst = np.abs(late["x"]) < 6.0
order = np.argsort(ratio(late)[worst])[:200]
ids = set(np.asarray(late["id"][worst])[order].tolist())

rows = []
for tag in tags:
    a = frame(tag)
    s = np.array([i in ids for i in a["id"]])
    rows.append((float(tag[1:]), a["rho_c"][s].mean(), a["rho_s"][s].mean(),
                 ratio(a)[s].mean(), a["nbr"][s].mean(), a["p"][s].mean()))
rows = np.array(rows)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=150)
ax = axes[0]
ax.plot(rows[:, 0], rows[:, 1], "o-", ms=4, label=r"连续方程密度 $\rho_c$")
ax.plot(rows[:, 0], rows[:, 2], "s-", ms=4, label=r"离线求和密度 $\rho_s$")
ax.set_xlabel("无量纲时间 T")
ax.set_ylabel(r"$\rho/\rho_0$")
ax.set_title("图4  同一批最受损粒子的两种密度（T=62 时选出最差 200 个）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axes[1]
ax.plot(rows[:, 0], rows[:, 3], "o-", ms=4, color="#2ca02c", label="支撑比")
ax.plot(rows[:, 0], rows[:, 4] / 21.0, "s-", ms=4, color="#ff7f0e",
        label="邻居数 / 21")
ax.set_xlabel("无量纲时间 T")
ax.set_ylabel("归一化支撑指标")
ax.set_title("图4b  同一批粒子的邻域完整性")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig4_two_densities_cohort.png")
plt.close(fig)

print("figures written to", OUT)
