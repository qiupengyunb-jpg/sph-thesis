"""Offline comparison of two disorder-decoupled "missing support" indicators.

Both indicators are purely local (neighbour geometry only): they do not use the
current x position and they do not use the current Indicator.

  M1  angular gap
      Neighbours inside the kernel cutoff are projected onto the local (x, r)
      plane as unit directions.  Sort their polar angles and take the largest
      angular gap G_i (radians) plus that gap's bisector direction.  A complete
      isotropic neighbourhood gives G ~ 2*pi/n; a one-sided one gives G -> pi.
      Also reported: the normalised gap G_i * n_i / (2*pi), 1 = uniform.

  M2  support tensor
      T_i = sum_j W_ij (u_j x u_j) / sum_j W_ij, u = unit neighbour direction.
      Eigenvalues l1 >= l2 with l1 + l2 = 1.  Report the anisotropy ratio
      l2/l1 (1 = isotropic, 0 = fully one-sided) and the eigenvector of l2,
      which points along the least-supported direction.

Direction reference: d_missing, built from the T=0 neighbour set of the same
particle (the neighbours that have since left the cutoff).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, r"E:\哈哈\_pr_diag")
from compare_resolutions import fast_load, frame_T_map

WANT_T = (5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0)
OUT = Path(r"E:\哈哈\_pr_diag\out\support_metrics")

DATASETS = {
    "healthy_r24_lam18": Path(
        r"E:\sphmethod\SPH_results_center\healthy_control_20260914"
        r"\uniform_r24_lam18_T80\output_uniform_r24_lam18_T80"),
    "lesion_r24": Path(
        r"E:\sphmethod\SPH_results_center\recon_tapered_20260914"
        r"\r24_broadband_T80\output_recon_r24_T80"),
    "lesion_r32": Path(
        r"E:\sphmethod\SPH_results_center\recon_tapered_20260914"
        r"\r32_broadband_T80\output_recon_r32_T80"),
}
DX = {"healthy_r24_lam18": 1.0 / 24, "lesion_r24": 1.0 / 24,
      "lesion_r32": 1.0 / 32}


def metrics_of_frame(pos, dx):
    """Per-particle (n, maxgap, normgap, gap_dir, ratio, tensor_dir)."""
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
    q = r / h
    w = np.zeros_like(r)
    m = q < 2.0
    w[m] = 7.0 / (4.0 * np.pi * h * h) * (1 - 0.5 * q[m]) ** 4 * (1 + 2 * q[m])

    N = len(pos)
    sw = np.bincount(src, weights=w, minlength=N)
    a = np.bincount(src, weights=w * u[:, 0] * u[:, 0], minlength=N)
    b = np.bincount(src, weights=w * u[:, 0] * u[:, 1], minlength=N)
    c = np.bincount(src, weights=w * u[:, 1] * u[:, 1], minlength=N)
    safe = np.maximum(sw, 1e-30)
    a, b, c = a / safe, b / safe, c / safe
    half = 0.5 * (a + c)
    rad = np.sqrt(np.maximum((0.5 * (a - c)) ** 2 + b * b, 0.0))
    l1, l2 = half + rad, half - rad
    ratio = np.where(l1 > 1e-30, l2 / np.maximum(l1, 1e-30), 1.0)
    tv = np.where(np.abs(b)[:, None] > 1e-12,
                  np.stack([b, l2 - a], axis=1),
                  np.stack([np.where(a <= c, 1.0, 0.0),
                            np.where(a <= c, 0.0, 1.0)], axis=1))
    tv = tv / np.maximum(np.linalg.norm(tv, axis=1), 1e-30)[:, None]

    ang = np.arctan2(u[:, 1], u[:, 0])
    order = np.lexsort((ang, src))
    si, sa = src[order], ang[order]
    starts = np.nonzero(np.concatenate([[True], si[1:] != si[:-1]]))[0]
    ends = np.append(starts[1:], len(si))
    gap_after = np.empty(len(si))
    gap_after[:-1] = sa[1:] - sa[:-1]
    wrap = 2.0 * np.pi - (sa[ends - 1] - sa[starts])
    gap_after[ends - 1] = wrap
    # NOTE: index ends[k]-1 == starts[k+1]-1 holds the wrap-around gap of
    # group k (the raw difference there is the cross-group jump and is
    # overwritten above).  It must stay in the group's segment, so nothing is
    # zeroed here -- doing that silently dropped every group's wrap gap and
    # produced a spurious pi for healthy particles.
    maxgap_sorted = np.maximum.reduceat(gap_after, starts)
    maxgap = np.zeros(N)
    maxgap[si[starts]] = maxgap_sorted
    bis = np.empty(len(si))
    bis[:-1] = 0.5 * (sa[1:] + sa[:-1])
    bis[ends - 1] = sa[ends - 1] + 0.5 * wrap
    owner = np.searchsorted(starts, np.arange(len(si)), side="right") - 1
    on_max = gap_after >= (maxgap_sorted[owner] - 1e-12)
    bx = np.bincount(si[on_max], weights=np.cos(bis[on_max]), minlength=N)
    by = np.bincount(si[on_max], weights=np.sin(bis[on_max]), minlength=N)
    gap_dir = np.stack([bx, by], axis=1)
    gap_dir = gap_dir / np.maximum(np.linalg.norm(gap_dir, axis=1),
                                   1e-30)[:, None]
    normgap = maxgap * np.maximum(n, 1) / (2.0 * np.pi)
    return n, maxgap, normgap, gap_dir, ratio, tv


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    store = {}
    healthy_g99 = None
    print("%-17s %-4s %6s %9s %9s %9s %9s %8s %8s %9s"
          % ("dataset", "T", "n_coh", "dG_p99", "dG_max", "dRho_p99", "dRho_max",
             "cos_gap", "cos_ten", "frac>health"))
    cohort_store = {}
    for tag, d in DATASETS.items():
        dx = DX[tag]
        fs = frame_T_map(d)
        a0 = fast_load(fs[0][0], ("Position", "Indicator", "OriginalID"))
        pos0 = a0["Position"][:, :2]
        id0 = a0["OriginalID"].astype(np.int64)
        ind0 = np.where(a0["Indicator"] > 0.5, 1, 0)
        h = 1.3 * dx
        cutoff = 2.0 * h
        n0lists = cKDTree(pos0).query_ball_point(pos0, cutoff,
                                                 return_sorted=False)
        # --- T=0 reference metrics and the cohort (interior, >=3dx off the wall)
        n_r, mg_r, ng_r, gd_r, rt_r, td_r = metrics_of_frame(pos0, dx)
        fib0 = fs[0][0].parent / fs[0][0].name.replace("LiquidFilmHalf_",
                                                       "RigidFiberHalf_")
        fp = fast_load(fib0, ("Position",))["Position"][:, :2]
        centres, prof = [], []
        for x0 in np.arange(-30.0, 30.0, 0.5):
            s = (fp[:, 0] >= x0) & (fp[:, 0] < x0 + 0.5)
            if s.sum() >= 4:
                centres.append(x0 + 0.25)
                prof.append(fp[s, 1].max())
        r_fib = np.interp(pos0[:, 0], centres, prof)
        depth0 = pos0[:, 1] - r_fib
        cohort0 = (ind0 == 0) & (depth0 >= 3.0 * dx)
        ref = {int(i): k for k, i in enumerate(id0)}
        print("[%s] cohort = %d of %d interior (depth>=3dx), n0 particle-count "
              "median=%.0f, gap0 median=%.3f, ratio0 median=%.3f"
              % (tag, int(cohort0.sum()), int((ind0 == 0).sum()),
                 np.median(n_r[cohort0]), np.median(mg_r[cohort0]),
                 np.median(rt_r[cohort0])))
        cohort_ids = set(id0[cohort0].tolist())
        gap_ref = {int(i): (mg_r[k], rt_r[k]) for k, i in enumerate(id0)
                   if int(i) in cohort_ids}
        for want in WANT_T:
            p, T = min(fs, key=lambda it: abs(it[1] - want))
            a = fast_load(p, ("Position", "Indicator", "OriginalID"))
            pos = a["Position"][:, :2]
            ids = a["OriginalID"].astype(np.int64)
            ind = np.where(a["Indicator"] > 0.5, 1, 0)
            n, mg, ng, gdir, ratio, tdir = metrics_of_frame(pos, dx)
            idx_of = {int(i): k for k, i in enumerate(ids)}
            cos_gap, cos_ten = [], []
            cos_gap_pp = np.full(len(pos), np.nan)
            cos_ten_pp = np.full(len(pos), np.nan)
            for k in np.nonzero(ind0 == 0)[0]:
                j = idx_of.get(int(id0[k]))
                if j is None:
                    continue
                n0 = [x for x in n0lists[k] if x != k]
                if not n0:
                    continue
                cur = np.array([idx_of.get(int(id0[x]), -1) for x in n0])
                v = cur >= 0
                if not v.any():
                    continue
                pn = pos[cur[v]]
                gone = pn[np.linalg.norm(pn - pos[j], axis=1) > cutoff]
                if len(gone) == 0:
                    continue
                miss = (gone - pos[j]).sum(axis=0)
                nm = np.linalg.norm(miss)
                if nm <= 0:
                    continue
                if np.linalg.norm(gdir[j]) > 1e-9:
                    v = float(gdir[j] @ miss / nm)
                    cos_gap.append(v)
                    cos_gap_pp[j] = v
                if np.linalg.norm(tdir[j]) > 1e-9:
                    v = float(tdir[j] @ miss / nm)
                    cos_ten.append(v)
                    cos_ten_pp[j] = v
            on = np.array([int(i) in cohort_ids for i in ids])
            kk = np.array([ref.get(int(i), -1) for i in ids])
            g0 = np.where(on, mg_r[np.maximum(kk, 0)], np.nan)
            r0 = np.where(on, rt_r[np.maximum(kk, 0)], np.nan)
            dG = mg - g0
            dRho = r0 - ratio              # positive = support got MORE one-sided
            store["%s_T%02d" % (tag, int(T))] = dict(
                tag=tag, T=T, n=n, maxgap=mg, ratio=ratio, dG=dG, dRho=dRho,
                gap_dir=gdir, tensor_dir=tdir, ids=ids, pos=pos, on=on,
                cos_gap=np.array(cos_gap), cos_ten=np.array(cos_ten),
                cos_gap_pp=cos_gap_pp, cos_ten_pp=cos_ten_pp)
            cg = np.array(cos_gap) if cos_gap else np.array([np.nan])
            ct = np.array(cos_ten) if cos_ten else np.array([np.nan])
            g99 = float(np.nanpercentile(dG[on], 99))
            if tag == "healthy_r24_lam18":
                healthy_g99 = g99 if healthy_g99 is None else max(healthy_g99, g99)
            cohort_store["%s_T%02d" % (tag, int(T))] = g99
            fgt = np.nan
            if healthy_g99 is not None and tag != "healthy_r24_lam18":
                fgt = float(np.nanmean(dG[on] > healthy_g99))
            print("%-17s %-4.0f %6d %9.4f %9.4f %9.4f %9.4f %8.3f %8.3f %9s"
                  % (tag, T, int(on.sum()), g99, np.nanmax(dG[on]),
                     np.nanpercentile(dRho[on], 99), np.nanmax(dRho[on]),
                     np.nanmean(cg), np.nanmean(ct),
                     ("%.3f" % fgt) if not np.isnan(fgt) else "-"))
    with open(OUT / "support_metrics.pkl", "wb") as fh:
        pickle.dump(store, fh)
    print("\nwrote", OUT / "support_metrics.pkl")


if __name__ == "__main__":
    main()
