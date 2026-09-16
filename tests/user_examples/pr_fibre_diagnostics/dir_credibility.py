"""Calibrate the credibility of the M1 (angular gap) bisector direction.

Per cohort particle (interior at T=0, >= 3 dx off the wall):
  G1, G2      largest / second largest angular gap (rad)
  dG          G1(T) - G1(0)
  d1          unit bisector of the largest gap
  cos_truth   cos(d1, d_missing)      offline reference, never a gate
  cos_time    cos(d1(T), d1(T_prev))  temporal persistence (observable online)
  cos_space   mean_j cos(d1_i, d1_j)  spatial coherence  (observable online)

Fully vectorised.  No solver change, no SPH run.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, r"E:\哈哈\_pr_diag")
from compare_resolutions import fast_load, frame_T_map
from support_metrics import DATASETS, DX

OUT = Path(r"E:\哈哈\_pr_diag\out\dir_cred")
WANT_T = (10.0, 20.0, 30.0, 40.0, 50.0)


def gap_metrics(pos, dx):
    """Return n, G1, G2, unit bisector of G1, and the neighbour pair arrays."""
    h = 1.3 * dx
    cutoff = 2.0 * h
    tree = cKDTree(pos)
    lists = tree.query_ball_point(pos, cutoff, return_sorted=False)
    n = np.array([len(l) for l in lists], dtype=np.int64)
    src = np.repeat(np.arange(len(pos)), n)
    dst = (np.concatenate([np.asarray(l, dtype=np.int64) for l in lists])
           if n.sum() else np.zeros(0, dtype=np.int64))
    keep = src != dst
    src, dst = src[keep], dst[keep]
    d = pos[dst] - pos[src]
    r = np.linalg.norm(d, axis=1)
    u = d / np.maximum(r, 1e-30)[:, None]
    ang = np.arctan2(u[:, 1], u[:, 0])

    order = np.lexsort((ang, src))
    si, sa = src[order], ang[order]
    starts = np.nonzero(np.concatenate([[True], si[1:] != si[:-1]]))[0]
    ends = np.append(starts[1:], len(si))
    gap = np.empty(len(si))
    gap[:-1] = sa[1:] - sa[:-1]
    wrap = 2.0 * np.pi - (sa[ends - 1] - sa[starts])
    gap[ends - 1] = wrap
    bis = np.empty(len(si))
    bis[:-1] = 0.5 * (sa[1:] + sa[:-1])
    bis[ends - 1] = sa[ends - 1] + 0.5 * wrap

    N = len(pos)
    g1s = np.maximum.reduceat(gap, starts)
    owner = np.searchsorted(starts, np.arange(len(si)), side="right") - 1
    second = np.where(gap < g1s[owner] - 1e-12, gap, -np.inf)
    g2s = np.maximum.reduceat(second, starts)
    g1 = np.zeros(N)
    g2 = np.zeros(N)
    g1[si[starts]] = g1s
    g2[si[starts]] = np.where(np.isfinite(g2s), g2s, 0.0)

    on1 = gap >= (g1s[owner] - 1e-12)
    bx = np.bincount(si[on1], weights=np.cos(bis[on1]), minlength=N)
    by = np.bincount(si[on1], weights=np.sin(bis[on1]), minlength=N)
    d1 = np.stack([bx, by], axis=1)
    d1 = d1 / np.maximum(np.linalg.norm(d1, axis=1), 1e-30)[:, None]
    return n, g1, g2, d1, src, dst


def id_map(from_ids, to_ids):
    """cur_of_from[k] = index in to_ids whose OriginalID equals from_ids[k]."""
    order = np.argsort(to_ids)
    srt = to_ids[order]
    loc = np.searchsorted(srt, from_ids)
    loc_c = np.clip(loc, 0, len(srt) - 1)
    ok = srt[loc_c] == from_ids
    out = np.full(len(from_ids), -1, dtype=np.int64)
    out[ok] = order[loc_c[ok]]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    store = {}
    for tag in ("lesion_r24", "lesion_r32", "healthy_r24_lam18"):
        d = DATASETS[tag]
        dx = DX[tag]
        cutoff = 2.6 * dx
        fs = frame_T_map(d)
        a0 = fast_load(fs[0][0], ("Position", "Indicator", "OriginalID"))
        pos0 = a0["Position"][:, :2]
        id0 = a0["OriginalID"].astype(np.int64)
        ind0 = np.where(a0["Indicator"] > 0.5, 1, 0)
        fib0 = fs[0][0].parent / fs[0][0].name.replace("LiquidFilmHalf_",
                                                       "RigidFiberHalf_")
        fp = fast_load(fib0, ("Position",))["Position"][:, :2]
        cx, pr = [], []
        for x0 in np.arange(-30.0, 30.0, 0.5):
            s = (fp[:, 0] >= x0) & (fp[:, 0] < x0 + 0.5)
            if s.sum() >= 4:
                cx.append(x0 + 0.25)
                pr.append(fp[s, 1].max())
        depth0 = pos0[:, 1] - np.interp(pos0[:, 0], cx, pr)
        cohort = (ind0 == 0) & (depth0 >= 3.0 * dx)
        cohort_ids = set(id0[cohort].tolist())
        n0, g1_0, g2_0, d1_0, src0, dst0 = gap_metrics(pos0, dx)
        print("[%s] cohort=%d" % (tag, len(cohort_ids)), flush=True)

        prev_cache = {}
        for want in WANT_T:
            p, T = min(fs, key=lambda it: abs(it[1] - want))
            a = fast_load(p, ("Position", "Indicator", "OriginalID"))
            pos = a["Position"][:, :2]
            ids = a["OriginalID"].astype(np.int64)
            n, g1, g2, d1, src, dst = gap_metrics(pos, dx)
            on = np.array([int(i) in cohort_ids for i in ids])

            # reference G1(0) for the same particle
            m0 = id_map(ids, id0)
            g1_ref = np.where(m0 >= 0, g1_0[np.maximum(m0, 0)], g1)
            dG = g1 - g1_ref

            # temporal persistence against the previous available frame
            prev = [q for q, TT in fs if TT < T - 1e-6]
            ct = np.full(len(pos), np.nan)
            if prev:
                q = prev[-1]
                if q not in prev_cache:
                    ap = fast_load(q, ("Position", "OriginalID"))
                    prev_cache[q] = (ap["Position"][:, :2],
                                     ap["OriginalID"].astype(np.int64),
                                     gap_metrics(ap["Position"][:, :2], dx)[3])
                pp, idp, d1p = prev_cache[q]
                mp = id_map(ids, idp)
                ok = mp >= 0
                ct[ok] = np.einsum("ij,ij->i", d1[ok], d1p[mp[ok]])

            # spatial coherence among CURRENT neighbours
            dot = np.einsum("ij,ij->i", d1[src], d1[dst])
            npair = np.bincount(src, minlength=len(pos))
            ssum = np.bincount(src, weights=dot, minlength=len(pos))
            cs = np.where(npair > 0, ssum / np.maximum(npair, 1), np.nan)

            # offline truth: d_missing from the T=0 neighbour set
            ci = id_map(id0, ids)
            good_pair = (ci[src0] >= 0) & (ci[dst0] >= 0)
            pn = pos[ci[dst0[good_pair]]] - pos[ci[src0[good_pair]]]
            rr = np.linalg.norm(pn, axis=1)
            gone = rr > cutoff
            src_g = ci[src0[good_pair]][gone]
            mx = np.bincount(src_g, weights=pn[gone, 0], minlength=len(pos))
            my = np.bincount(src_g, weights=pn[gone, 1], minlength=len(pos))
            mm = np.hypot(mx, my)
            ctrue = np.full(len(pos), np.nan)
            ok = mm > 0
            ctrue[ok] = (d1[ok, 0] * mx[ok] + d1[ok, 1] * my[ok]) / mm[ok]

            store["%s_T%02d" % (tag, int(T))] = dict(
                tag=tag, T=T, on=on, dG=dG, g1=g1, g2=g2,
                ratio=np.where(g2 > 1e-6, g1 / np.maximum(g2, 1e-6), np.inf),
                diff=g1 - g2, cos_time=ct, cos_space=cs, cos_truth=ctrue)
            print("   T=%4.0f cohort=%d truth=%d"
                  % (T, int(on.sum()), int(np.isfinite(ctrue[on]).sum())),
                  flush=True)
    with open(OUT / "dir_cred.pkl", "wb") as fh:
        pickle.dump(store, fh)
    print("\nwrote", OUT / "dir_cred.pkl")


if __name__ == "__main__":
    main()
