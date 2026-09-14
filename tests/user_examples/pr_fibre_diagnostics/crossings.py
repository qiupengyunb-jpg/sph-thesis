"""Threshold crossing times and the axisymmetric summation-variant check."""
import sys

import numpy as np

sys.path.insert(0, r"E:\哈哈\_pr_diag")
from diag_core import DX, SIGMA0_NUM

OUT = r"E:\哈哈\_pr_diag\out"
import csv

with open(OUT + r"\frame_metrics.csv", encoding="utf-8") as fh:
    recs = list(csv.DictReader(fh))
rows = {k: np.array([float(r[k]) for r in recs]) for k in recs[0]}


def first_cross(col, level, region):
    for T, v in zip(rows["T"], rows[col]):
        if v > level:
            return T
    return None


print("=== 首次超过阈值的时刻（径向相邻间隙）===")
for col, name in (("max_gap_central", "中央区 |x|<4a"),
                  ("max_gap_interior", "内部区 |x|<26a")):
    for level in (1.5, 2.0, 3.0):
        T = first_cross(col, level, name)
        print("  %-16s %4.1fdx -> T = %s" % (name, level, T))

print("\n=== 首次超过阈值的时刻（核支撑损失）===")
print("  最小支撑比 < 0.95 : T =", first_cross("min_support_ratio_central", -0.95, "") * 0
      if False else None)
for level in (0.95, 0.90, 0.85):
    T = None
    for t, v in zip(rows["T"], rows["min_support_ratio_central"]):
        if v < level:
            T = t
            break
    print("  最小支撑比 < %.2f : T = %s" % (level, T))

print("\n=== 支撑受损粒子数首次出现（|x|<6a）===")
z = np.load(OUT + r"\support_frames.npz")
ref = dict(zip(z["T0_id"], z["T0_s_num"]))
tags = sorted({k.split("_")[0] for k in z.files}, key=lambda k: float(k[1:]))
for tag in tags:
    x, s, pid = z[tag + "_x"], z[tag + "_s_num"], z[tag + "_id"]
    reg = np.abs(x) < 6.0
    ratio = s[reg] / np.array([ref[i] for i in pid[reg]])
    for level in (0.95, 0.90, 0.80):
        if (ratio < level).any() and level not in globals().setdefault("seen", {}):
            seen[level] = float(tag[1:])
for level in sorted(seen):
    print("  首次出现 S/S0 < %.2f : T = %.1f" % (level, seen[level]))

print("\n=== 轴对称求和两种写法的差异（T=62）===")
a = {n: z["T62_%s" % n] for n in ("x", "r", "mass", "s_num")}
from diag_core import neighbor_sums
pos = np.column_stack([a["x"], a["r"]])
s_num, s_mass, _, _ = neighbor_sums(pos, a["mass"])
away = a["r"] > 0.25 * DX * 1.01          # away from the r=0.25dx mass floor
near = ~away
print("  远离轴线 (r > 0.25dx) 粒子数 %d 处 max|number-mass| = %.3e"
      % (away.sum(), np.abs((s_num / SIGMA0_NUM - s_mass / 1.010473e0)[away]).max()))
print("  靠近轴线 (r <= 0.25dx) 粒子数 %d 处 max|number-mass| = %.3e"
      % (near.sum(), np.abs((s_num / SIGMA0_NUM - s_mass / 1.010473e0)[near]).max()))
