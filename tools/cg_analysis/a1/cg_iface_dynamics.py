"""Experiment C: is the condensed body liquid-like or frozen?

Measures, inside the free aggregate (no fibre):
  * MSD(dt) with unwrapped displacements (minimum-image per frame pair)
  * the long-time self-diffusion coefficient D (2D: MSD = 4 D dt)
  * the neighbour-exchange fraction: share of the neighbours of a particle at
    time t that are no longer neighbours at t + dt (connectivity 1.5 sigma)
  * the caged plateau of the MSD at short times
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cg_iface import load_vtp, read_params

R_CUT = 1.5


def frames_of(run, name):
    out = run / ("output_" + name)
    rows = []
    with open(run / "selfcheck.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            p = out / ("CGParticles_ite_%010d.vtp" % int(r["steps"]))
            if p.exists():
                rows.append((float(r["time"]), p))
    return rows


def _frame_step(runs):
    return (runs[-1][0] - runs[0][0]) / max(len(runs) - 1, 1)


def msd_curve(runs, lx, lags, n_origin=12):
    """runs: list of (time, pos, ids) sorted by time; lags in time units."""
    dt_frame = _frame_step(runs)
    out = {lag: [] for lag in lags}
    for lag in lags:
        nf = int(round(lag / dt_frame))
        if nf < 1 or nf >= len(runs):
            continue
        origins = range(0, len(runs) - nf)
        step = max(1, len(origins) // n_origin)
        for i in list(origins)[::step]:
            _, pos0, ids0 = runs[i]
            _, pos1, ids1 = runs[i + nf]
            common, i0, i1 = np.intersect1d(ids0, ids1, return_indices=True)
            d = pos1[i1] - pos0[i0]
            d[:, 0] -= lx * np.round(d[:, 0] / lx)
            out[lag].append(float(np.mean(np.sum(d * d, axis=1))))
    return {k: float(np.mean(v)) if v else np.nan for k, v in out.items()}


def neighbour_exchange(runs, lags, n_origin=8):
    dt_frame = _frame_step(runs)
    res = {}
    for lag in lags:
        fracs = []
        nf = int(round(lag / dt_frame))
        if nf < 1 or nf >= len(runs):
            res[lag] = np.nan
            continue
        origins = list(range(0, len(runs) - nf))
        step = max(1, len(origins) // n_origin)
        for i in origins[::step]:
            _, pos0, ids0 = runs[i]
            _, pos1, ids1 = runs[i + nf]
            common, i0, i1 = np.intersect1d(ids0, ids1, return_indices=True)
            p0, p1 = pos0[i0], pos1[i1]

            def nbrs(p):
                t = cKDTree(p)
                return t.query_ball_point(p, R_CUT)

            n0, n1 = nbrs(p0), nbrs(p1)
            keep = []
            for k in range(len(common)):
                s0, s1 = set(n0[k]), set(n1[k])
                s0.discard(k); s1.discard(k)
                if len(s0) == 0:
                    continue
                keep.append(1.0 - len(s0 & s1) / len(s0))
            if keep:
                fracs.append(float(np.mean(keep)))
        res[lag] = float(np.mean(fracs)) if fracs else np.nan
    return res


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("usage: python cg_iface_dynamics.py <results-root>")
        print("       reads the VTP frames of every case in <results-root>")
        raise SystemExit(2)
    root = Path(sys.argv[1])
    if not root.is_dir():
        raise SystemExit("not a directory: %s" % root)
    lags_msd = [20, 60, 100, 200, 400, 800, 1200, 2000, 3000]
    lags_nb = [60, 100, 200, 500, 1000]
    rows_out = []
    print("%-18s %-8s | %-28s | %-26s | %s"
          % ("case", "N", "MSD at dt=100/400/1200 (sig^2)", "neighbour replaced %",
             "D (sig^2/t)"))
    for d in sorted([x for x in root.iterdir() if x.is_dir()]):
        name = d.name
        p = read_params(d)
        lx = float(p["domain_length"])
        fr = frames_of(d, name)
        if len(fr) < 50:
            continue
        runs = []
        for t, path in fr[::2]:          # every 20 time units
            v = load_vtp(path, ("Position", "OriginalID"))
            runs.append((t, v["Position"], v["OriginalID"]))
        msd = msd_curve(runs, lx, lags_msd)
        nb = neighbour_exchange(runs, lags_nb)
        # D from the largest lag; alpha = log-log slope of the MSD (1 = diffusive)
        d_est = msd.get(1200, np.nan) / (4.0 * 1200.0)
        m1, m2 = msd.get(400, np.nan), msd.get(2000, np.nan)
        alpha = (np.log(m2 / m1) / np.log(2000.0 / 400.0)
                 if (m1 and m2 and m1 > 0 and m2 > 0) else np.nan)
        print("%-18s %-8s | %8.2f %8.2f %8.2f | %6.1f %6.1f %6.1f %6.1f %6.1f | %.4f  a=%.2f"
              % (name, p["particle_number"], msd.get(100, np.nan), msd.get(400, np.nan),
                 msd.get(1200, np.nan), 100 * nb.get(60, np.nan), 100 * nb.get(100, np.nan),
                 100 * nb.get(200, np.nan), 100 * nb.get(500, np.nan),
                 100 * nb.get(1000, np.nan), d_est, alpha))
        for lag in lags_msd:
            rows_out.append(dict(case=name, kind="msd", lag=lag, value=msd[lag]))
        for lag in lags_nb:
            rows_out.append(dict(case=name, kind="neighbour_replaced", lag=lag,
                                 value=nb[lag]))
    with open(root / "iface_dynamics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["case", "kind", "lag", "value"])
        w.writeheader(); w.writerows(rows_out)
    print("wrote", root / "iface_dynamics.csv")


if __name__ == "__main__":
    main()
