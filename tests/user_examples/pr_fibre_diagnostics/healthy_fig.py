"""Figure: healthy r24 control vs variable-radius lesion (delta), plus A/A0."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(r"E:\sphmethod\SPH_results_center\healthy_control_20260914")
CAL = Path(r"E:\哈哈\_pr_diag\out\calib\variable_rows.csv")
H = Path(r"E:\哈哈\_pr_diag\out\healthy")


def rd(p):
    with open(p, encoding="utf-8") as fh:
        recs = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in recs]) for k in recs[0]}


hd, les, amp = rd(H / "healthy_rows.csv"), rd(CAL), rd(H / "amplitude.csv")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), dpi=150)

ax = axes[0]
ax.plot(hd["T"], hd["max"], "-", color="#1f77b4", lw=1.8,
        label="健康等径 λ=18（同程序/r24/同开关）max")
ax.plot(hd["T"], hd["p99"], "--", color="#1f77b4", lw=1.0, label="健康 p99")
ax.plot(les["T"], les["max"], "-", color="#ff7f0e", lw=1.8,
        label="变半径病灶 max")
ax.plot(les["T"], les["p99"], "--", color="#ff7f0e", lw=1.0,
        label="病灶 p99")
ax.axhline(0.03, color="#d62728", ls=":", lw=1.5)
ax.text(30, 0.032, "δ_thr = 0.03", color="#d62728", fontsize=9)
ax.axvline(50, color="k", ls="-.", lw=1)
ax.text(50.5, 2e-5, "T=50 病灶 gap 打开", fontsize=8)
ax.set_yscale("log")
ax.set_ylim(1e-5, 0.5)
ax.set_xlabel("无量纲时间 T")
ax.set_ylabel("自参照核完整性亏损 δ")
ax.set_title("图1  健康对照（同程序同分辨率）与病灶的 δ 分离")
ax.legend(fontsize=7.5, loc="lower right")
ax.grid(alpha=0.3, which="both")

ax = axes[1]
ax.plot(amp["T"], amp["A_over_A0"], "-", color="#2ca02c", lw=1.8,
        label="界面幅度 A/A0（λ=18 单模）")
k = np.polyfit(amp["T"][amp["T"] > 5], np.log(amp["A_over_A0"][amp["T"] > 5]), 1)
ax.plot(amp["T"], np.exp(k[1] + k[0] * amp["T"]), ":", color="k", lw=1,
        label="指数拟合 σ = %.4f /T" % k[0])
ax.set_xlabel("无量纲时间 T")
ax.set_ylabel("A / A0")
ax.set_title("图2  健康对照确实在发生 PR 增长\n（不是静止膜，因此 δ 平坦不是平凡结果）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(les["T"], les["n_0.02"] / max(les["n_0.02"].max(), 1) * 0 + les["n_0.02"],
        "-", color="#7f7f7f", label="病灶 δ>0.02 粒子数")
ax.plot(les["T"], les["n_0.05"], "-", color="#2ca02c", label="病灶 δ>0.05")
ax.plot(les["T"], les["n_0.1"], "-", color="#d62728", label="病灶 δ>0.10")
ax.plot(hd["T"], hd["n_0.02"], "-", color="#1f77b4", lw=2,
        label="健康对照 δ>0.02（全程 0）")
ax.set_xlabel("无量纲时间 T")
ax.set_ylabel("粒子数（初始内部粒子中）")
ax.set_title("图3  触发粒子数：健康对照在所有阈值上恒为 0")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "healthy_vs_lesion.png")
print("wrote", OUT / "healthy_vs_lesion.png")
print("fit sigma = %.5f /T over T>5" % k[0])
