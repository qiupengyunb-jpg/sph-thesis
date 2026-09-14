"""Which way does the kernel-gradient residual point for bulk particles?"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, r"E:\哈哈\_pr_diag")
from run_metrics import load

H = 1.3 / 24.0
DX = 1.0 / 24.0
CUT = 2.0 * H
VOL = DX * DX

for tag in ("0001341640", "0020124611", "0033541019"):
    p = Path(r"E:\sphmethod\SPH_results_center\recon_tapered_20260914"
             r"\r24_surfreg_T50\output_r24_surfreg_T50")
    a = load(p / ("LiquidFilmHalf_%s.vtp" % tag))
    pos = a["Position"][:, :2]
    ind = a["Indicator"]
    tree = cKDTree(pos)
    pairs = tree.query_ball_point(pos, CUT, return_sorted=False)
    inc = np.zeros((len(pos), 2))
    for i, js in enumerate(pairs):
        js = np.asarray(js, dtype=np.int64)
        js = js[js != i]
        if js.size == 0:
            continue
        d = pos[js] - pos[i]
        r = np.linalg.norm(d, axis=1)
        e = d / r[:, None]
        q = r / H
        m = q < 2.0
        dW = np.zeros_like(r)
        qq = q[m]
        dW[m] = (7.0 / (4.0 * np.pi * H ** 3)) * (
            -2.0 * (1 - 0.5 * qq) ** 3 * (1 + 2 * qq) * 1.0
            + (1 - 0.5 * qq) ** 4 * 2.0)
        inc[i] = (-2.0 * dW[:, None] * VOL * e).sum(axis=0)

    mag = np.linalg.norm(inc, axis=1)
    bulk = ind == 0
    # local film extent in each particle's own axial bin
    er = np.column_stack([np.zeros(len(pos)), np.ones(len(pos))])  # radial unit
    ir = (inc * er).sum(axis=1)
    outer, inner = [], []
    for x0 in np.arange(-6.0, 6.0, 2.0):
        sel = (pos[:, 0] >= x0) & (pos[:, 0] < x0 + 2.0) & bulk & (mag > 1e-9)
        if sel.sum() < 20:
            continue
        rmid = 0.5 * (np.percentile(pos[sel, 1], 5) +
                      np.percentile(pos[sel, 1], 95))
        hi = sel & (pos[:, 1] > rmid)
        lo = sel & (pos[:, 1] <= rmid)
        if hi.sum():
            outer.append(ir[hi].mean())
        if lo.sum():
            inner.append(ir[lo].mean())
    print("frame T=%.0f  bulk particles=%d" % (float(tag) / 1e6 * 0 + 
          {0: 2, 1: 30, 2: 50}.get(["0001341640", "0020124611",
                                     "0033541019"].index(tag), -1), bulk.sum()))
    print("   mean I_r  outer-half of film = %+.4f   inner-half = %+.4f"
          % (np.mean(outer), np.mean(inner)))
    print("   |I| bulk: mean=%.3f  p95=%.3f  max=%.3f"
          % (mag[bulk].mean(), np.percentile(mag[bulk], 95), mag[bulk].max()))
