"""Compare the first gated run (r24_gate_T35) against the ungated r=24 baseline.

Checks, in the order the user asked for:
  1. does activation start after T ~ 26, and is the active count sensible
  2. does the APPLIED shift still point into the missing-support region
  3. max delta / min S/S0 vs the baseline
  4. is the radial gap amplified
  5. h_max difference
  6. max_speed sanity
  7. per-particle cumulative artificial displacement
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, r"E:\哈哈\_pr_diag")
from compare_resolutions import fast_load, frame_T_map, s_and_nbr as _unused

DX = 1.0 / 24.0
BIN = 2.0
GATE = Path(r"E:\sphmethod\SPH_results_center\recon_tapered_20260914"
            r"\r24_gate_T35\output_r24_gate_T35")
BASE = Path(r"E:\sphmethod\SPH_results_center\recon_tapered_20260914"
            r"\r24_broadband_T80\output_recon_r24_T80")
OUT = Path(r"E:\哈哈\_pr_diag\out\gate_check")


def load_full(p):
    names = ("Position", "Indicator", "OriginalID", "Velocity",
             "SupportDeficit", "SupportGateActive", "SupportShiftDose",
             "SupportFilmLayers", "SupportGateState", "SupportGateShift",
             "SupportGateEligible", "SupportGateWatch")
    a = fast_load(p, names)
    for k in ("Velocity",):
        if k in a:
            a[k] = a[k].reshape(-1, 3)
    return a


def s_of(pos, dx):
    h = 1.3 * dx
    w0 = 7.0 / (4.0 * np.pi * h * h)
    tree = cKDTree(pos)
    lists = tree.query_ball_point(pos, 2.0 * h, return_sorted=False)
    s = np.full(len(pos), w0)
    for i, js in enumerate(lists):
        js = np.asarray(js, dtype=np.int64)
        js = js[js != i]
        if js.size == 0:
            continue
        q = np.linalg.norm(pos[js] - pos[i], axis=1) / h
        m = q < 2.0
        s[i] += (7.0 / (4.0 * np.pi * h * h) *
                 (1 - 0.5 * q[m]) ** 4 * (1 + 2 * q[m])).sum()
    return s, lists


def metrics(pos, fib_pos, dx):
    gap, rmax = {}, {}
    for x0 in np.arange(-28.0, 28.0, BIN):
        sel = (pos[:, 0] >= x0) & (pos[:, 0] < x0 + BIN)
        fsel = (fib_pos[:, 0] >= x0) & (fib_pos[:, 0] < x0 + BIN)
        if sel.sum() < 8:
            continue
        rr = np.sort(pos[sel, 1])
        gap[round(x0, 1)] = float(np.diff(rr).max()) / dx
        rmax[round(x0, 1)] = float(rr.max())
    hmax = np.nan
    if fib_pos is not None:
        fmax = {}
        for x0 in np.arange(-28.0, 28.0, BIN):
            fsel = (fib_pos[:, 0] >= x0) & (fib_pos[:, 0] < x0 + BIN)
            if fsel.sum() >= 8:
                fmax[round(x0, 1)] = float(fib_pos[fsel, 1].max())
        hmax = max((rmax[k] - fmax[k] for k in rmax if k in fmax), default=np.nan)
    cen = [v for k, v in gap.items() if -4.1 <= k < 4.0]
    tree = cKDTree(pos)
    pairs = tree.query_pairs(1.5 * dx, output_type="ndarray")
    if len(pairs) == 0:
        nblk = len(pos)
    else:
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(pos), len(pos)))
        nblk = connected_components(g, directed=False)[0]
    return dict(gap_central=max(cen), gap_interior=max(gap.values()),
                h_max=hmax, blocks=int(nblk))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gfs = frame_T_map(GATE)
    gfs = [(p, T) for p, T in gfs if T <= 35.5]
    bmap = {round(T, 1): p for p, T in frame_T_map(BASE)}

    # reference for delta and for the missing-support direction
    a0 = load_full(gfs[0][0])
    pos0 = a0["Position"][:, :2]
    id0 = a0["OriginalID"].astype(np.int64)
    s0, nbr0 = s_of(pos0, DX)
    s0_by_id = dict(zip(id0, s0))
    pos0_index = {int(i): k for k, i in enumerate(id0)}
    idx_of_n0 = {int(i): nbr0[k] for k, i in enumerate(id0)}

    rows = []
    for p, T in gfs:
        a = load_full(p)
        pos = a["Position"][:, :2]
        ids = a["OriginalID"].astype(np.int64)
        fib_p = p.parent / p.name.replace("LiquidFilmHalf_", "RigidFiberHalf_")
        fib = fast_load(fib_p, ("Position",))["Position"][:, :2] if fib_p.exists() else None
        s, nbrs = s_of(pos, DX)
        delta = np.array([1.0 - s[k] / s0_by_id[int(i)] for k, i in enumerate(ids)])
        m = metrics(pos, fib, DX)

        active = a.get("SupportGateActive", np.zeros(len(ids)))
        dose = a.get("SupportShiftDose", np.zeros(len(ids)))
        deficit_file = a.get("SupportDeficit", np.full(len(ids), np.nan))
        vel = a.get("Velocity", np.zeros((len(ids), 3)))[:, :2]

        # direction check: does the APPLIED shift point into the missing region?
        idx_of = {int(i): k for k, i in enumerate(ids)}
        cos_list = []
        for k in np.nonzero(active == 1)[0]:
            iid = int(ids[k])
            j0 = pos0_index.get(iid)
            if j0 is None:
                continue
            n0_ids = id0[idx_of_n0[iid]]
            cur = np.array([idx_of.get(int(i), -1) for i in n0_ids])
            still = set(np.asarray(nbrs[k]).tolist())
            gone = [c for c, i in zip(cur, n0_ids) if c >= 0 and c not in still]
            if not gone:
                continue
            miss = (pos[np.asarray(gone)] - pos[k]).sum(axis=0)
            nm = np.linalg.norm(miss)
            sft = a["SupportGateShift"][k, :2] if "SupportGateShift" in a else None
            if sft is None or nm <= 0:
                continue
            nn = np.linalg.norm(sft)
            if nn > 0:
                cos_list.append(float(sft.dot(miss) / (nn * nm)))

        bT = min(bmap, key=lambda t: abs(t - T))
        bp = bmap[bT]
        ab = load_full(bp)
        bpos = ab["Position"][:, :2]
        bfib_p = bp.parent / bp.name.replace("LiquidFilmHalf_", "RigidFiberHalf_")
        bfib = fast_load(bfib_p, ("Position",))["Position"][:, :2]
        bid = ab["OriginalID"].astype(np.int64)
        bs, _ = s_of(bpos, DX)
        bdelta = np.array([
            1.0 - bs[k] / s0_by_id[int(i)]
            for k, i in enumerate(bid) if int(i) in s0_by_id]
            + [np.nan] * (len(bid) - sum(1 for i in bid if int(i) in s0_by_id)))
        bm = metrics(bpos, bfib, DX)

        rows.append(dict(
            T=T, n_active=int((active == 1).sum()),
            n_dose_gt0=int((dose > 0).sum()),
            dose_max_dx=float(dose.max() / DX),
            file_deficit_max=float(np.nanmax(deficit_file)),
            my_deficit_max=float(np.nanmax(delta)),
            cosmos=float(np.mean(cos_list)) if cos_list else np.nan,
            cos_pos=float(np.mean(np.array(cos_list) > 0)) if cos_list else np.nan,
            speed_max=float(np.linalg.norm(vel, axis=1).max()),
            gap_gate=m["gap_central"], gap_base=bm["gap_central"],
            hmax_gate=m["h_max"], hmax_base=bm["h_max"],
            blocks_gate=m["blocks"], blocks_base=bm["blocks"],
            minS_gate=1.0 - float(np.nanmax(delta)),
            minS_base=1.0 - float(np.nanmax(bdelta)),
            n003_gate=int((delta > 0.03).sum()),
            n003_base=int((bdelta > 0.03).sum()),
        ))
        r = rows[-1]
        print("T=%5.1f act=%4d doseMax=%.4fdx  cos(shift,miss)=%+.3f (%.0f%%+)  "
              "gap %.2f/%.2f  hmax %.3f/%.3f  minS %.3f/%.3f  n>0.03 %d/%d  vmax %.4f"
              % (T, r["n_active"], r["dose_max_dx"], r["cosmos"],
                 100 * (r["cos_pos"] if not np.isnan(r["cos_pos"]) else 0),
                 r["gap_gate"], r["gap_base"], r["hmax_gate"], r["hmax_base"],
                 r["minS_gate"], r["minS_base"], r["n003_gate"], r["n003_base"],
                 r["speed_max"]))

    keys = list(rows[0].keys())
    with open(OUT / "gate_vs_base.csv", "w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for row in rows:
            fh.write(",".join("%.6g" % row[k] for k in keys) + "\n")
    print("\nwrote", OUT / "gate_vs_base.csv")


if __name__ == "__main__":
    main()
