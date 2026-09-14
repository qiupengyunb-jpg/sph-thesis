"""Morphology panel: mirror the half solution into a full fibre."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, r"E:\哈哈\_pr_diag")
from diag_core import LIQUID_DIR, load, frames

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(r"E:\哈哈\_pr_diag\out")
allf = frames()


def fibre_path(liquid_path):
    name = liquid_path.name.replace("LiquidFilmHalf_", "RigidFiberHalf_")
    return LIQUID_DIR / name


panels = []
for want in (0.0, 20.0, 42.0, 62.0):
    p, T = min(allf, key=lambda it: abs(it[1] - want))
    liq = load(p)["Position"][:, :2]
    fp = fibre_path(p)
    fib = load(fp)["Position"][:, :2] if fp.exists() else np.zeros((0, 2))
    panels.append((T, liq, fib))

fig, axes = plt.subplots(4, 1, figsize=(13, 9.5), dpi=150, sharex=True)
for ax, (T, liq, fib) in zip(axes, panels):
    for sign in (1, -1):
        ax.scatter(liq[:, 0], sign * liq[:, 1], s=0.12, c="#4c8fd6", lw=0,
                   label="液体" if sign == 1 else None)
        if len(fib):
            ax.scatter(fib[:, 0], sign * fib[:, 1], s=0.12, c="#8c8c8c", lw=0,
                       label="刚性纤维" if sign == 1 else None)
    ax.set_ylim(-3.6, 3.6)
    ax.set_xlim(-14, 14)
    ax.set_aspect("equal")
    ax.set_ylabel("r / a")
    ax.text(-13.4, 2.4, "T = %.0f" % T, fontsize=11)
    ax.grid(alpha=0.2)
    if ax is axes[0]:
        ax.plot([6.0, 8.0], [3.0, 3.0], "-", color="k", lw=3)
        ax.text(6.1, 3.15, "2a", fontsize=9)
        ax.legend(loc="upper right", markerscale=18, fontsize=9)
axes[-1].set_xlabel("轴向坐标 x / a")
fig.suptitle("细—粗—细全包覆液膜形貌演化（r=24，无正则化，重构式；半侧解按 r→−r 镜像）",
             fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "morph_T0_T20_T42_T62.png")
print("wrote", OUT / "morph_T0_T20_T42_T62.png")
