"""Why is SurfaceRegularizationShift identically zero?"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"E:\哈哈\_pr_diag")
from run_metrics import load

SRF = Path(r"E:\sphmethod\SPH_results_center\recon_tapered_20260914"
           r"\r24_surfreg_T50\output_r24_surfreg_T50")

print("norm of the normal-like fields that the operator could read")
print("%-24s %10s %12s %12s %12s %12s"
      % ("frame", "n_surf", "TopologyNorm", "JfmSmoothedN", "NormDirect",
         "ColorGrad"))
for name in ("ite_0000000000.vtp", "0001341640.vtp", "0020124611.vtp",
             "0033541019.vtp"):
    p = SRF / ("LiquidFilmHalf_" + name)
    a = load(p)
    ind = a["Indicator"]
    out = [int((ind == 1).sum())]
    for key in ("TopologyNormal", "JfmSmoothedNormal", "NormDirection",
                "ColorGradient"):
        if key in a:
            v = np.linalg.norm(a[key][:, :2], axis=1)
            out.append(v.max())
        else:
            out.append(float("nan"))
    print("%-24s %10d %12.4f %12.4f %12.4f %12.4f"
          % (name, *out))

# would the operator do anything if the normal were available?
a = load(SRF / "LiquidFilmHalf_0020124611.vtp")
pos = a["Position"][:, :2]
Vol = None
a2 = load(SRF / "LiquidFilmHalf_0001341640.vtp")
from scipy.spatial import cKDTree
H, DX = 1.3 / 24.0, 1.0 / 24.0
CUT = 2.0 * H


def w(r):
    q = r / H
    out = np.zeros_like(q)
    m = q < 2.0
    qq = q[m]
    out[m] = 7.0 / (4.0 * np.pi * H * H) * (1 - 0.5 * qq) ** 4 * (1 + 2 * qq)
    return out


def dw(r):
    q = r / H
    out = np.zeros_like(q)
    m = (q < 2.0) & (q > 0)
    qq = q[m]
    out[m] = (1.0 / H) * 7.0 / (4.0 * np.pi * H * H) * \
        (-2.0 * (1 - 0.5 * qq) ** 3 * (1 + 2 * qq) * 1.0
         + (1 - 0.5 * qq) ** 4 * 2.0)
    return -out


tree = cKDTree(pos)
pairs = tree.query_ball_point(pos, CUT, return_sorted=False)
vol = DX * DX
inc = np.zeros((len(pos), 2))
for i, js in enumerate(pairs):
    js = np.asarray(js, dtype=np.int64)
    js = js[js != i]
    if js.size == 0:
        continue
    d = pos[js] - pos[i]
    r = np.linalg.norm(d, axis=1)
    e = d / r[:, None]
    inc[i] = (-2.0 * dw(r)[:, None] * vol * e).sum(axis=0)

ind = a["Indicator"]
mag = np.linalg.norm(inc, axis=1)
print("\nraw consistency residual  -2*sum(dW_ij*V_j*e_ij)")
print("  all particles      max=%.4f  mean=%.4f" % (mag.max(), mag.mean()))
print("  free-surface only  max=%.4f  mean=%.4f"
      % (mag[ind == 1].max(), mag[ind == 1].mean()))
print("  would-be shift max = %.3e (coefficient 0.05 * dx^2 * |residual|)"
      % (0.05 * DX * DX * mag[ind == 1].max()))
