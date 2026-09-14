"""2C calibration, part 2: overlap metrics, flip-delta distribution, figures."""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(r"E:\哈哈\_pr_diag\out\calib")
THR = [0.02, 0.03, 0.05, 0.08, 0.10]
LAMS = [10, 12, 14, 16, 18, 20, 22, 24, 26, 30]


def read_csv(p):
    with open(p, encoding="utf-8") as fh:
        recs = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in recs]) for k in recs[0]}


z = np.load(OUT / "equal_radius_rows.npz")
var = read_csv(OUT / "variable_rows.csv")

# ---- healthy baseline: worst case over every lambda and every time
healthy_max, healthy_p99, healthy_n_int = 0.0, 0.0, 0
healthy_frac = {th: 0.0 for th in THR}
for lam in LAMS:
    key = "lam%g_max" % lam
    if key not in z.files:
        continue
    healthy_max = max(healthy_max, float(np.nanmax(z[key])))
    healthy_p99 = max(healthy_p99, float(np.nanmax(z["lam%g_p99" % lam])))
    healthy_n_int = max(healthy_n_int, int(z["lam%g_n_0.02" % lam][0] * 0 +
                                           z["lam%g_mean" % lam].size))
    for th in THR:
        k = "lam%g_frac_%g" % (lam, th)
        if k in z.files:
            healthy_frac[th] = max(healthy_frac[th], float(np.nanmax(z[k])))

print("=== 正常等径 PR 的全部 λ、全部 T(=0~20)、全部初始内部粒子 ===")
print("  内部粒子 δ 的最大值（10 个 λ 取最坏）= %.6f" % healthy_max)
print("  内部粒子 δ 的 p99 最大值                = %.6f" % healthy_p99)
for th in THR:
    print("  δ>%.2f 的最大比例                       = %.6f" % (th, healthy_frac[th]))

print("\n=== 病态变半径算例（初始内部粒子，初始 |x|<6a）===")
n_int_var = None
for th in THR:
    col = "n_%g" % th
    idx = np.nonzero(var[col] > 0)[0]
    first_T = var["T"][idx[0]] if len(idx) else None
    print("  δ>%.2f 首次出现：T=%s（%d 个粒子）" % (th, first_T,
          int(var[col][idx[0]]) if len(idx) else 0))
print("  最大 δ 首次超过各阈值的时间（用 max 列）")
for th in THR:
    idx = np.nonzero(var["max"] > th)[0]
    if len(idx):
        print("    max δ > %.2f @ T=%.1f" % (th, var["T"][idx[0]]))

# first T at which p99 exceeds the whole healthy baseline's worst p99
idx = np.nonzero(var["p99"] > healthy_p99)[0]
print("\n  p99 首次超过健康基线最坏 p99(%.6f)：T=%.1f" %
      (healthy_p99, var["T"][idx[0]] if len(idx) else float("nan")))
idx = np.nonzero(var["max"] > healthy_max)[0]
print("  max 首次超过健康基线最坏 max(%.6f)：T=%.1f" %
      (healthy_max, var["T"][idx[0]] if len(idx) else float("nan")))

# fraction of lesion particles above the healthy maximum
print("\n  病灶粒子中 δ 超过健康基线最大值(%.6f) 的比例：" % healthy_max)
for T in (10, 20, 26, 30, 36, 42, 50):
    i = int(np.argmin(np.abs(var["T"] - T)))
    print("    T=%5.1f  p99=%.4f  max=%.4f  n(>0.05)=%d"
          % (var["T"][i], var["p99"][i], var["max"][i], int(var["n_0.05"][i])))

# ---- flip-delta distribution
vz = np.load(OUT / "variable_rows.csv.npz") if False else np.load(
    OUT / "variable_rows.npz")
fd = vz["flip_delta"]
ft = vz["flip_T"]
print("\n=== 初始内部粒子发生 Indicator 翻转的情况 ===")
print("  翻转粒子总数 = %d" % len(fd))
if len(fd):
    print("  首次翻转 T=%.1f，中位翻转 T=%.1f，最后翻转 T=%.1f"
          % (np.nanmin(ft), np.nanmedian(ft), np.nanmax(ft)))
    qs = [5, 25, 50, 75, 90, 95]
    print("  翻转前一刻的 δ："
          + "  ".join("p%d=%.4f" % (q, np.nanpercentile(fd, q)) for q in qs)
          + "  max=%.4f" % np.nanmax(fd))
    print("  其中翻转前 δ>0.05 的占 %.1f%%；δ>0.10 的占 %.1f%%；δ<0 的占 %.1f%%"
          % (100 * (fd > 0.05).mean(), 100 * (fd > 0.10).mean(),
             100 * (fd < 0).mean()))

# ---- figures
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), dpi=150)
ax = axes[0]
lam_best = 18
for key, lab, col in (("lam18_p95", "等径 λ=18", "#1f77b4"),
                      ("lam18_p99", None, "#1f77b4"),
                      ("lam18_max", None, "#1f77b4")):
    T = np.arange(len(z[key])) * 1.0
    ax.plot(T, z[key], "-", color=col, lw=1.4 if "max" in key else 1.0,
            label=("等径 λ=18：p95 / p99 / max" if "max" in key else None))
ax.plot(var["T"], var["p95"], "-", color="#d62728", lw=1.6,
        label="变半径：p95")
ax.plot(var["T"], var["p99"], "--", color="#d62728", lw=1.6,
        label="变半径：p99")
ax.plot(var["T"], var["max"], "-", color="#ff7f0e", lw=1.8,
        label="变半径：max")
for th in (0.02, 0.05):
    ax.axhline(th, ls=":", color="gray", lw=1)
    ax.text(30, th * 1.1, "δ=%.2f" % th, fontsize=8, color="gray")
ax.axvline(50, ls="-.", color="k", lw=1)
ax.text(50.5, 1e-4, "T=50 几何 gap 打开", fontsize=8)
ax.set_yscale("log")
ax.set_ylim(1e-5, 1)
ax.set_xlabel("无量纲时间 T")
ax.set_ylabel("自参照核完整性亏损 δ")
ax.set_title("图1  δ 的量级分离（初始内部粒子）")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3, which="both")

ax = axes[1]
for th, col in zip(THR, ("#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e",
                         "#d62728")):
    ax.plot(var["T"], var["frac_%g" % th], "-", color=col,
            label="病灶 δ>%.2f" % th)
ax.axhline(0.0, color="k", lw=0.5)
ax.set_xlabel("无量纲时间 T")
ax.set_ylabel("病灶内部粒子的触发比例")
ax.set_title("图2  病灶触发比例（等径算例在所有阈值上均为 0）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[2]
ax.hist(fd, bins=40, color="#d62728", alpha=0.8)
ax.axvline(np.nanmedian(fd), color="k", ls="--", lw=1,
           label="中位 %.3f" % np.nanmedian(fd))
ax.set_xlabel("Indicator 翻转前一刻的 δ")
ax.set_ylabel("粒子数")
ax.set_title("图3  翻转前 δ 的分布（n=%d）" % len(fd))
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "calib_delta.png")
print("\nwrote", OUT / "calib_delta.png")
