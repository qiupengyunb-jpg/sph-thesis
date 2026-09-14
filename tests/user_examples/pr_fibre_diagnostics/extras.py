"""Pressure response and the two axisymmetric summation variants."""
import sys

import numpy as np

sys.path.insert(0, r"E:\哈哈\_pr_diag")
from diag_core import DX, SIGMA0_NUM, SIGMA0_MASS

OUT = r"E:\哈哈\_pr_diag\out"
z = np.load(OUT + r"\support_frames.npz")

print("c0 = 30 * max(gamma/mu, sqrt(gamma/(rho*R))) ; p = c0^2 (rho-1)")
c0 = 30.0 * max(1.0 / 0.67082039325, np.sqrt(1.0 / 2.0))
print("   -> c0 = %.3f,  p0 = c0^2 = %.1f,  drho = dp/%.0f\n"
      % (c0, c0 * c0, c0 * c0))

tags = sorted({k.split("_")[0] for k in z.files}, key=lambda k: float(k[1:]))

rows = []
for tag in tags:
    a = {name: z["%s_%s" % (tag, name)]
         for name in ("x", "r", "rho_c", "rho_s", "s_num", "nbr", "d_min",
                      "p", "gap", "mass", "id", "indicator")}
    T = float(tag[1:])
    reg = np.abs(a["x"]) < 6.0
    nbr = a["nbr"]
    p = a["p"]
    rho_c = a["rho_c"]
    # void-adjacent particles: fewer inner neighbours than a full support (>=20)
    hot = reg & (nbr <= 14)
    rows.append((T, reg.sum(), int(hot.sum()),
                 p[reg].min(), p[reg].max(),
                 float(np.mean([abs(rho_c[i] - 1 - p[i] / c0 ** 2) for i in
                                np.nonzero(reg)[0][:2000]])),
                 p[hot].mean() if hot.sum() else np.nan,
                 p[hot].min() if hot.sum() else np.nan,
                 p[hot].max() if hot.sum() else np.nan,
                 rho_c[hot].mean() if hot.sum() else np.nan))

print("%-6s %8s %8s %10s %10s %10s %10s %10s %10s"
      % ("T", "n(<14)", "p_min", "p_max", "eos_res", "p_hot", "pmin_hot",
         "pmax_hot", "rho_c_hot"))
for r in rows:
    print("%-6.1f %8d %10.4f %10.4f %10.2e %10.4f %10.4f %10.4f %10.6f"
          % (r[0], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9]))

print("\nEOS consistency: max |rho_continuity - (1 + p/c0^2)| over the region")
print("(a residual at the 1e-7 level means the solver pressure really is the")
print(" equation of state evaluated at the continuity density)")

# --- number form vs mass form of the offline summation density
s_num = z["T62_s_num"]
mass = z["T62_mass"]
print("\nnumber form uses unweighted kernel sums, mass form uses")
print("m_j = AxisymmetricRingMass_j / r_j (the axisymmetric solver mass).")
reg = np.abs(z["T62_x"]) < 6.0
for name, val in (("number form", s_num / SIGMA0_NUM),):
    print("  %-12s region min=%.4f mean=%.4f" % (name, val[reg].min(), val[reg].mean()))

from diag_core import load, frames
allf = frames()
path, T = min(allf, key=lambda it: abs(it[1] - 62.0))
a2 = load(path)
pos = a2["Position"][:, :2]
r = np.maximum(pos[:, 1], 0.25 * DX)
m = a2["AxisymmetricRingMass"] / r
from diag_core import neighbor_sums
s_num2, s_mass, nbr, d_min = neighbor_sums(pos, m)
reg2 = np.abs(pos[:, 0]) < 6.0
print("  %-12s region min=%.4f mean=%.4f"
      % ("mass form", (s_mass / SIGMA0_MASS)[reg2].min(),
         (s_mass / SIGMA0_MASS)[reg2].mean()))
print("  max |number - mass| in region = %.2e"
      % np.abs((s_num2 / SIGMA0_NUM - s_mass / SIGMA0_MASS)[reg2]).max())
