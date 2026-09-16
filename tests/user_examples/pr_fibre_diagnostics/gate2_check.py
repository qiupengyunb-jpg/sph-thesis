"""Eight-item comparison: late-window directional correction vs the r=24 baseline."""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, r"E:\哈哈\_pr_diag")
from compare_resolutions import fast_load, frame_T_map
from gate_check import metrics, s_of, DX

GATE = Path(r"E:\sphmethod\SPH_results_center\recon_tapered_20260914"
            r"\r24_gate2_T55\output_r24_gate2_T55")
BASE = Path(r"E:\sphmethod\SPH_results_center\recon_tapered_20260914"
            r"\r24_broadband_T80\output_recon_r24_T80")
OUT = Path(r"E:\哈哈\_pr_diag\out\gate2")
NAMES = ("Position", "Indicator", "OriginalID", "Velocity", "Density",
         "SupportDeficit", "SupportGateActive", "SupportShiftDose",
         "SupportGateState", "SupportGateShift", "SupportGapIncrease",
         "SupportGapBisector", "SupportFilmLayers", "SupportGateEligible")


def load(p):
    a = fast_load(p, NAMES)
    for k in ("Velocity", "SupportGateShift", "SupportGapBisector"):
        if k in a:
            a[k] = a[k].reshape(-1, 3)
    return a


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gfs = [(p, T) for p, T in frame_T_map(GATE) if T <= 55.5]
    bfs = [(p, T) for p, T in frame_T_map(BASE) if T <= 55.5]
    a0 = load(gfs[0][0])
    pos0 = a0["Position"][:, :2]
    id0 = a0["OriginalID"].astype(np.int64)
    ind0 = np.where(a0["Indicator"] > 0.5, 1, 0)
    s0, _ = s_of(pos0, DX)
    ref = dict(zip(id0, s0))
    n0lists = cKDTree(pos0).query_ball_point(pos0, 2.6 * DX,
                                             return_sorted=False)
    cohort = set(id0[(ind0 == 0) & (np.abs(pos0[:, 0]) < 6.0)].tolist())
    print("cohort(initial interior, |x0|<6a) = %d" % len(cohort))

    rows = []
    for p, T in gfs:
        a = load(p)
        pos = a["Position"][:, :2]
        ids = a["OriginalID"].astype(np.int64)
        fib_p = p.parent / p.name.replace("LiquidFilmHalf_", "RigidFiberHalf_")
        fib = fast_load(fib_p, ("Position",))["Position"][:, :2]
        s, nbrs = s_of(pos, DX)
        delta = np.array([1.0 - s[k] / ref[int(i)] for k, i in enumerate(ids)])
        m = metrics(pos, fib, DX)
        v = a["Velocity"][:, :2]
        act = a.get("SupportGateActive", np.zeros(len(ids)))
        dose = a.get("SupportShiftDose", np.zeros(len(ids)))
        dg = a.get("SupportGapIncrease", np.full(len(ids), np.nan))
        sel = np.array([int(i) in cohort for i in ids])

        # direction check against d_missing
        idx_of = {int(i): k for k, i in enumerate(ids)}
        cos = []
        for k in np.nonzero(act == 1)[0]:
            iid = int(ids[k])
            j0 = int(np.nonzero(id0 == iid)[0][0]) if (id0 == iid).any() else -1
            if j0 < 0:
                continue
            n0 = [x for x in n0lists[j0] if x != j0]
            if not n0:
                continue
            cur = np.array([idx_of.get(int(id0[x]), -1) for x in n0])
            vv = cur >= 0
            if not vv.any():
                continue
            pn = pos[cur[vv]]
            gone = pn[np.linalg.norm(pn - pos[k], axis=1) > 2.6 * DX]
            if len(gone) == 0:
                continue
            miss = (gone - pos[k]).sum(axis=0)
            nm = np.linalg.norm(miss)
            sh = a["SupportGateShift"][k, :2]
            nn = np.linalg.norm(sh)
            if nm > 0 and nn > 0:
                cos.append(float(sh @ miss / (nn * nm)))
        cos = np.array(cos) if cos else np.array([np.nan])

        bT = min((t for _, t in bfs), key=lambda t: abs(t - T))
        bp = [q for q, t in bfs if abs(t - bT) < 1e-6][0]
        ab = load(bp)
        bpos = ab["Position"][:, :2]
        bid = ab["OriginalID"].astype(np.int64)
        bfib_p = bp.parent / bp.name.replace("LiquidFilmHalf_", "RigidFiberHalf_")
        bfib = fast_load(bfib_p, ("Position",))["Position"][:, :2]
        bs, _ = s_of(bpos, DX)
        bdelta = np.array([1.0 - bs[k] / ref[int(i)] for k, i in enumerate(bid)])
        bsel = np.array([int(i) in cohort for i in bid])
        bm = metrics(bpos, bfib, DX)

        rows.append(dict(
            T=T, n_active=int((act == 1).sum()), n_dose=int((dose > 0).sum()),
            dose_max_dx=float(dose.max() / DX), n_exhausted=int((a.get("SupportGateState", np.zeros(len(ids))) == 2).sum()),
            cos_dir=float(np.nanmean(cos)),
            frac_dir_pos=float(np.nanmean(cos > 0)),
            frac_dir_07=float(np.nanmean(cos > 0.7)),
            gap_gate=m["gap_central"], gap_base=bm["gap_central"],
            hmax_gate=m["h_max"], hmax_base=bm["h_max"],
            blocks_gate=m["blocks"], blocks_base=bm["blocks"],
            minS_gate=1.0 - float(np.nanmax(delta[sel])),
            minS_base=1.0 - float(np.nanmax(bdelta[bsel])),
            dmax_gate=float(np.nanmax(delta[sel])),
            dmax_base=float(np.nanmax(bdelta[bsel])),
            vmax_gate=float(np.linalg.norm(v, axis=1).max()),
        ))
        r = rows[-1]
        if T % 2 == 0 or T >= 40:
            print("T=%5.1f act=%4d dose=%.3fdx dir_cos=%+.3f(%.0f%%>0.7) | "
                  "gap %.2f/%.2f | minS %.3f/%.3f | hmax %.3f/%.3f | "
                  "blk %d/%d | vmax %.4f"
                  % (T, r["n_active"], r["dose_max_dx"], r["cos_dir"],
                     100 * r["frac_dir_07"], r["gap_gate"], r["gap_base"],
                     r["minS_gate"], r["minS_base"], r["hmax_gate"],
                     r["hmax_base"], r["blocks_gate"], r["blocks_base"],
                     r["vmax_gate"]))
    keys = list(rows[0].keys())
    with open(OUT / "gate2_vs_base.csv", "w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for row in rows:
            fh.write(",".join("%.6g" % row[k] for k in keys) + "\n")
    print("\nwrote", OUT / "gate2_vs_base.csv")


if __name__ == "__main__":
    main()
