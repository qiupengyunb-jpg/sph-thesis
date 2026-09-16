r"""Offline pre-check for the delta-gated support-restoration scheme.

NO solver change, NO new run.  Uses existing r=24 / r=32 frames.

Gate (all must hold):
  1. interior at T=0                       (Indicator == 0 at T=0, matched by ID)
  2. delta_i(T) > 0.03                     delta = 1 - S(T)/S(0)
  3. local film thickness H >= 4*dx        H from 2a axial bins, liquid minus fibre
  4. dry_fraction == 0                     read from the run's own stdout log
  5. >= 2 neighbours above the threshold   neighbours within the kernel cutoff

Directions compared for every activated particle i:
  d_now   = kernel-weighted centroid of the CURRENT neighbours
            (the direction the user asked for; = +grad C, anti-diffusive)
  d_shift = -d_now                          (standard particle-shifting direction)
  d_ref   = kernel-weighted centroid over the T=0 NEIGHBOUR SET, evaluated at the
            neighbours' CURRENT positions  -> points to where the lost support went
  d_miss  = normalised sum over (N0 \ N1) of (x_j_now - x_i_now)
            -> the "missing support" direction, used as the objective reference
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, r"E:\哈哈\_pr_diag")
from compare_resolutions import RUNS, fast_load, frame_T_map

WANT_T = (20.0, 30.0, 40.0, 46.0, 50.0)
DELTA_THR = 0.03
NBR_MIN = 2
H_MIN_DX = 4.0
BIN = 2.0


def kernel_w(q):
    out = np.zeros_like(q)
    m = q < 2.0
    out[m] = 7.0 / (4.0 * np.pi) * (1 - 0.5 * q[m]) ** 4 * (1 + 2 * q[m])
    return out  # includes the 1/h^2 factor only through q scaling below


def s_and_neighbours(pos, dx):
    """S_i (self-inclusive, liquid-only) and the neighbour index lists."""
    h = 1.3 * dx
    w0 = 7.0 / (4.0 * np.pi * h * h)
    tree = cKDTree(pos)
    lists = tree.query_ball_point(pos, 2.0 * h, return_sorted=False)
    s = np.full(len(pos), w0)
    for i, js in enumerate(lists):
        js = np.asarray(js, dtype=np.int64)
        js = js[js != i]
        lists[i] = js
        if js.size == 0:
            continue
        q = np.linalg.norm(pos[js] - pos[i], axis=1) / h
        s[i] += (7.0 / (4.0 * np.pi * h * h) *
                 (1 - 0.5 * q[q < 2.0]) ** 4 * (1 + 2 * q[q < 2.0])).sum()
    return s, lists, h


def film_thickness(x, r, fx, fr, dx):
    """H per 2a axial bin (liquid outer radius minus fibre outer radius)."""
    H = {}
    for x0 in np.arange(-28.0, 28.0, BIN):
        sel = (x >= x0) & (x < x0 + BIN)
        fsel = (fx >= x0) & (fx < x0 + BIN)
        if sel.sum() < 8 or fsel.sum() < 8:
            continue
        H[round(x0, 1)] = r[sel].max() - fr[fsel].max()
    return H


def analyse(tag, run_dir, dx, dry_map):
    print("\n=== %s (dx=%.5f) ===" % (tag, dx))
    fs = frame_T_map(run_dir)
    p0, _ = fs[0]
    a0 = fast_load(p0)
    pos0 = a0["Position"][:, :2]
    id0 = a0["OriginalID"].astype(np.int64)
    ind0 = np.where(a0["Indicator"] > 0.5, 1, 0)
    s0, nbr0, h = s_and_neighbours(pos0, dx)
    pos0_of = {int(i): k for k, i in enumerate(id0)}
    s0_by_id = dict(zip(id0, s0))
    ind0_by_id = dict(zip(id0, ind0))

    rows = []
    for want in WANT_T:
        p, T = min(fs, key=lambda it: abs(it[1] - want))
        a = fast_load(p, ("Position", "Indicator", "OriginalID", "Velocity"))
        pos = a["Position"][:, :2]
        ids = a["OriginalID"].astype(np.int64)
        ind = np.where(a["Indicator"] > 0.5, 1, 0)
        s, nbrs, _ = s_and_neighbours(pos, dx)
        fib = p.parent / p.name.replace("LiquidFilmHalf_", "RigidFiberHalf_")
        f = fast_load(fib)
        H = film_thickness(pos[:, 0], pos[:, 1], f["Position"][:, 0],
                           f["Position"][:, 1], dx)
        dry = dry_map.get(round(T, 1), 0.0)

        idx_of = {int(i): k for k, i in enumerate(ids)}
        delta = np.full(len(ids), np.nan)
        for k, i in enumerate(ids):
            ref = s0_by_id.get(int(i))
            if ref:
                delta[k] = 1.0 - s[k] / ref
        above = delta > DELTA_THR

        # neighbour consistency: count neighbours also above the threshold
        n_above = np.zeros(len(ids), dtype=np.int64)
        for k in range(len(ids)):
            js = nbrs[k]
            if js.size:
                n_above[k] = above[js].sum()

        # masks
        was_interior = np.array([ind0_by_id.get(int(i), 1) == 0 for i in ids])
        h_bin = np.array([H.get(round(np.floor(pos[k, 0] / BIN) * BIN, 1), 0.0)
                          for k in range(len(ids))])
        thr_ok = h_bin >= H_MIN_DX * dx
        act = was_interior & above & thr_ok & (n_above >= NBR_MIN) & (dry == 0)
        print("  T=%5.1f  dry=%.3f  above=%d  interior0=%d  H>=4dx=%d  "
              "nbr>=2=%d  -> activated=%d"
              % (T, dry, above.sum(), was_interior.sum(), thr_ok.sum(),
                 (n_above >= NBR_MIN).sum(), act.sum()))

        if act.sum() == 0:
            continue

        # directions for the activated particles
        cos_now, cos_ref, cos_neg, rad_now, ax_now, len_now = [], [], [], [], [], []
        for k in np.nonzero(act)[0]:
            iid = int(ids[k])
            j0 = pos0_of.get(iid)
            if j0 is None:
                continue
            # current neighbours
            js = nbrs[k]
            d_now = np.zeros(2)
            if js.size:
                d = pos[js] - pos[k]
                q = np.linalg.norm(d, axis=1) / h
                w = kernel_w(q)
                d_now = (w[:, None] * d).sum(axis=0) / max(w.sum(), 1e-30)
            # T=0 neighbour set, at current positions
            n0 = nbr0[j0]
            n0_ids = id0[n0]
            cur = np.array([idx_of.get(int(i), -1) for i in n0_ids])
            d_ref = np.zeros(2)
            miss = np.zeros(2)
            if len(n0_ids):
                valid = cur >= 0
                if valid.any():
                    d = pos[cur[valid]] - pos[k]
                    q = np.linalg.norm(d, axis=1) / h
                    w = kernel_w(q)
                    d_ref = (w[:, None] * d).sum(axis=0) / max(w.sum(), 1e-30)
                # missing = in T=0 set but no longer within the cutoff now
                still = set(np.asarray(nbrs[k]).tolist())
                gone = [c for c, i in zip(cur, n0_ids)
                        if c >= 0 and c not in still]
                if gone:
                    miss = (pos[np.asarray(gone)] - pos[k]).sum(axis=0)
            nm = np.linalg.norm(miss)
            nn = np.linalg.norm(d_now)
            nr = np.linalg.norm(d_ref)
            if nm > 0 and nn > 0:
                cos_now.append(float(d_now.dot(miss) / (nn * nm)))
                cos_neg.append(float(-d_now.dot(miss) / (nn * nm)))
            if nm > 0 and nr > 0:
                cos_ref.append(float(d_ref.dot(miss) / (nr * nm)))
            if nn > 0:
                u = d_now / nn
                rad_now.append(float(u[1]))
                ax_now.append(float(abs(u[0])))
                len_now.append(float(nn))

        def stat(a):
            return (len(a), np.mean(a), np.median(a), np.min(a), np.max(a)) \
                if len(a) else (0, np.nan, np.nan, np.nan, np.nan)

        vel = a["Velocity"].reshape(-1, 3)[:, :2]
        v = vel[act]
        rows.append(dict(
            T=T, n=int(act.sum()),
            cos_dnow_miss=stat(cos_now)[1], cos_negnow_miss=stat(cos_neg)[1],
            cos_dref_miss=stat(cos_ref)[1],
            frac_cos_now_pos=float(np.mean(np.array(cos_now) > 0)) if cos_now else np.nan,
            frac_cos_neg_pos=float(np.mean(np.array(cos_neg) > 0)) if cos_neg else np.nan,
            frac_cos_ref_pos=float(np.mean(np.array(cos_ref) > 0)) if cos_ref else np.nan,
            mean_radial_u=float(np.mean(rad_now)) if rad_now else np.nan,
            frac_outward=float(np.mean(np.array(rad_now) > 0)) if rad_now else np.nan,
            mean_axial_frac=float(np.mean(ax_now)) if ax_now else np.nan,
            speed_mean=float(np.linalg.norm(v, axis=1).mean()),
            speed_p90=float(np.percentile(np.linalg.norm(v, axis=1), 90)),
            d_now_mean_len=float(np.mean(len_now)) if len_now else np.nan,
        ))
        r = rows[-1]
        print("     n=%4d  cos(d_now,d_miss)=%+.3f (%.0f%% 同向)   "
              "cos(-d_now,d_miss)=%+.3f (%.0f%% 同向)   cos(d_ref,d_miss)=%+.3f"
              % (r["n"], r["cos_dnow_miss"], 100 * r["frac_cos_now_pos"],
                 r["cos_negnow_miss"], 100 * r["frac_cos_neg_pos"],
                 r["cos_dref_miss"]))
        print("     径向分量 u_r 均值=%+.3f  向外占比=%.0f%%  轴向占比均值=%.2f  "
              "|v| 均值=%.4f p90=%.4f"
              % (r["mean_radial_u"], 100 * r["frac_outward"],
                 r["mean_axial_frac"], r["speed_mean"], r["speed_p90"]))
    return rows


def dry_map_of(run_dir):
    """dry_fraction per output time, from the run's own stdout log."""
    out = {}
    log = run_dir.parent / "stdout.log"
    if not log.exists():
        return out
    for line in log.read_text(errors="replace").splitlines():
        if not line.startswith("T="):
            continue
        parts = dict(kv.split("=") for kv in line.split() if "=" in kv)
        try:
            out[round(float(parts["T"]), 1)] = float(parts["dry"])
        except (KeyError, ValueError):
            pass
    return out


def main():
    out = Path(r"E:\哈哈\_pr_diag\out\gate_precheck")
    out.mkdir(parents=True, exist_ok=True)
    all_rows = {}
    for tag in ("r24", "r32"):
        d, dx = RUNS[tag]
        all_rows[tag] = analyse(tag, d, dx, dry_map_of(d))
        if all_rows[tag]:
            keys = list(all_rows[tag][0].keys())
            with open(out / ("%s.csv" % tag), "w", encoding="utf-8") as fh:
                fh.write(",".join(keys) + "\n")
                for row in all_rows[tag]:
                    fh.write(",".join("%.6g" % row[k] for k in keys) + "\n")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
