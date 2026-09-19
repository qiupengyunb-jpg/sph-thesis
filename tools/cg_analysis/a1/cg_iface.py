"""Interface-tension verification for a FREE coarse-grained aggregate.

No fibre.  One pre-condensed cluster (triangular lattice, prescribed shape) is
released and followed.  Everything is measured on a COARSE-GRAINED DENSITY
FIELD, never on the outermost particle positions:

    rho(x,y)  : counts on a 0.5 sigma grid, Gaussian-smoothed with 1.0 sigma
    interface : the 0.5 * rho_bulk isodensity contour
    A, P      : area and perimeter of that contour (marching squares)
    C = 4 pi A / P^2      circularity (1 for a perfect circle)
    AR                    major/minor axis ratio of the enclosed region
    rho(r)                radial profile -> plateau, interface width w

The same script also measures the internal dynamics (MSD, neighbour exchange)
so that "no shape recovery" can be told apart from "the aggregate is frozen".
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CELL = 0.5
SMOOTH = 1.0


def load_vtp(path, wanted=("Position", "OriginalID")):
    txt = Path(path).read_text(errors="replace")
    out = {}
    for name in wanted:
        key = 'Name="%s"' % name
        i = txt.find(key)
        if i < 0:
            continue
        s = txt.index(">", i) + 1
        e = txt.index("<", s)
        vals = np.array(txt[s:e].split(), dtype=float)
        out[name] = vals.reshape(-1, 3)[:, :2] if name == "Position" else vals.astype(int)
    return out


def read_params(run):
    d = {}
    with open(run / "parameters.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            d[r["parameter"]] = r["value"]
    return d


def density_field(pos, lx, ly, cell=CELL, smooth=SMOOTH):
    nx, ny = int(round(lx / cell)), int(round(ly / cell))
    ix = np.floor(pos[:, 0] / lx * nx).astype(int) % nx
    iy = np.clip(np.floor(pos[:, 1] / ly * ny).astype(int), 0, ny - 1)
    cnt = np.zeros((ny, nx))
    np.add.at(cnt, (iy, ix), 1.0)
    rho = cnt / (cell * cell)
    # periodic in x, reflecting (mirror) in y
    pad = int(np.ceil(4 * smooth / cell))
    ext = np.concatenate([rho[:, -pad:], rho, rho[:, :pad]], axis=1)
    ext = np.concatenate([ext[:pad][::-1], ext, ext[-pad:][::-1]], axis=0)
    ext = gaussian_filter(ext, smooth / cell, mode="nearest")
    rho_s = ext[pad:pad + ny, pad:pad + nx]
    xs = (np.arange(nx) + 0.5) * cell
    ys = (np.arange(ny) + 0.5) * cell
    return xs, ys, rho_s


def bulk_density(rho):
    """Robust core density of the condensed body.

    Uses the 99.9th percentile as a reference (insensitive to the fraction of
    the box occupied by the body, unlike a fixed top-decile rule which fails
    once the body covers less than 10 % of the box) and returns the median of
    the cells above 0.7 of it.
    """
    flat = rho.ravel()
    ref = float(np.quantile(flat, 0.999))
    sel = flat[flat > 0.7 * ref]
    return float(np.median(sel)) if len(sel) else 0.0


def contour_metrics(xs, ys, rho, level):
    """Area, perimeter, circularity, axis ratio and edge roughness."""
    cs = plt.contour(xs, ys, rho, levels=[level])
    segs = [s for s in cs.allsegs[0] if len(s) >= 8]
    plt.close("all")
    if not segs:
        return None
    seg = max(segs, key=lambda s: len(s))
    x, y = seg[:, 0], seg[:, 1]
    if abs(x[0] - x[-1]) + abs(y[0] - y[-1]) > 1e-9:   # close it
        x, y = np.append(x, x[0]), np.append(y, y[0])
    area = 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
    per = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    if area <= 0 or per <= 0:
        return None
    cx, cy = x.mean(), y.mean()
    # second moments
    x0, y0 = x - cx, y - cy
    mxx, myy, mxy = np.mean(x0 * x0), np.mean(y0 * y0), np.mean(x0 * y0)
    tr, det = mxx + myy, mxx * myy - mxy * mxy
    l1 = 0.5 * tr + np.sqrt(max(0.25 * tr * tr - det, 0.0))
    l2 = 0.5 * tr - np.sqrt(max(0.25 * tr * tr - det, 0.0))
    ar = float(np.sqrt(l1 / max(l2, 1e-12)))
    return dict(A=float(area), P=per, C=float(4 * np.pi * area / per ** 2),
                AR=ar, R_eq=float(np.sqrt(area / np.pi)),
                cx=float(cx), cy=float(cy), contour=(x, y))


def radial_profile(pos, lx, ly, cxy, dr=0.5, rmax=None):
    dx = pos[:, 0] - cxy[0]
    dx -= lx * np.round(dx / lx)
    dy = pos[:, 1] - cxy[1]
    r = np.hypot(dx, dy)
    rmax = rmax if rmax else 0.5 * min(lx, ly)
    edges = np.arange(0.0, rmax + dr, dr)
    cnt, _ = np.histogram(r, bins=edges)
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    return 0.5 * (edges[:-1] + edges[1:]), cnt / area


def interface_width(r, prof, rho_bulk, rho_dil):
    """10-90 % width and the tanh-fitted width."""
    hi = rho_dil + 0.9 * (rho_bulk - rho_dil)
    lo = rho_dil + 0.1 * (rho_bulk - rho_dil)
    valid = r > 1.0                       # skip the noisy first bins at r ~ 0
    below_hi = valid & (prof < hi)
    below_lo = valid & (prof < lo)
    i_hi = int(np.argmax(below_hi)) if below_hi.any() else -1
    i_lo = int(np.argmax(below_lo)) if below_lo.any() else -1
    w1090 = float(r[i_lo] - r[i_hi]) if i_lo > i_hi > 0 else np.nan
    # tanh fit rho = mid + amp tanh((R - r)/w)
    mid = 0.5 * (rho_bulk + rho_dil)
    amp = 0.5 * (rho_bulk - rho_dil)
    m = (r > 1.0) & (prof < rho_bulk * 1.05) & (prof > -1)
    best = (np.inf, np.nan, np.nan)
    for R in np.arange(2.0, r[m].max() if m.any() else 10.0, 0.5):
        for w in np.arange(0.2, 4.01, 0.2):
            model = mid + amp * np.tanh((R - r) / w)
            err = float(np.mean((prof[m] - model[m]) ** 2))
            if err < best[0]:
                best = (err, R, w)
    return w1090, float(best[2]), float(best[1])


def analyse_case(run_dir, name, t_end=None, n_win=12):
    p = read_params(run_dir)
    lx, ly = float(p["domain_length"]), float(p["domain_height"])
    out = run_dir / ("output_" + name)
    rows = []
    with open(run_dir / "selfcheck.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append((float(r["time"]), int(r["steps"])))
    tmax = t_end if t_end is not None else rows[-1][0]
    out_rows = []
    for t, s in rows:
        f = out / ("CGParticles_ite_%010d.vtp" % s)
        if not f.exists():
            continue
        d = load_vtp(f, ("Position",))
        pos = d["Position"]
        xs, ys, rho = density_field(pos, lx, ly)
        rb = bulk_density(rho)
        m = contour_metrics(xs, ys, rho, 0.5 * rb)
        if m is None:
            continue
        r, prof = radial_profile(pos, lx, ly, (m["cx"], m["cy"]))
        far = prof[r > 0.8 * r.max()]
        rho_dil = float(np.median(far)) if len(far) else 0.0
        w1090, wfit, Rfit = interface_width(r, prof, rb, rho_dil)
        # radial roughness of the contour about its own best circle
        x, y = m.pop("contour")
        rr = np.hypot(x - m["cx"], y - m["cy"])
        rough = float(np.std(rr))
        out_rows.append(dict(time=t, rho_bulk=rb, rho_dilute=rho_dil,
                             w_1090=w1090, w_fit=wfit, R_fit=Rfit,
                             rough=rough, **m))
    return out_rows, lx, ly


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("usage: python cg_iface.py <results-root>")
        print("       <results-root> holds one sub-directory per case, each with")
        print("       parameters.csv, selfcheck.csv and output_<case>/*.vtp")
        raise SystemExit(2)
    root = Path(sys.argv[1])
    if not root.is_dir():
        raise SystemExit("not a directory: %s" % root)
    names = sorted([d.name for d in root.iterdir() if d.is_dir()])
    all_rows = []
    for nm in names:
        rows, lx, ly = analyse_case(root / nm, nm)
        if not rows:
            print("skip", nm)
            continue
        for r in rows:
            r["case"] = nm
        all_rows += rows
        first, last = rows[0], rows[-1]
        late = [r for r in rows if r["time"] > 0.75 * rows[-1]["time"]]
        g = lambda k, rr=late: float(np.mean([q[k] for q in rr]))
        print("%-16s t=%4.0f→%4.0f | A %6.0f→%6.0f (%+.1f%%) | P %6.1f→%6.1f (%+.1f%%) | "
              "C %.3f→%.3f | AR %.3f→%.3f | ρ_bulk %.3f | w %.2fσ"
              % (nm, first["time"], last["time"], first["A"], g("A"),
                 100 * (g("A") / first["A"] - 1), first["P"], g("P"),
                 100 * (g("P") / first["P"] - 1), first["C"], g("C"),
                 first["AR"], g("AR"), g("rho_bulk"), g("w_fit")))
        par = read_params(root / nm)
        n_par = float(par["particle_number"])
        print("                 sanity: N=%d, A·rho_bulk=%.0f (%.1f%% of N), "
              "R_eq=%.1fσ, 轮廓粗糙度 %.2fσ (粒子直径=1σ)"
              % (int(n_par), g("A") * g("rho_bulk"),
                 100 * g("A") * g("rho_bulk") / n_par, g("R_eq"), g("rough")))
    fields = ["case", "time", "A", "P", "C", "AR", "R_eq", "cx", "cy",
              "rho_bulk", "rho_dilute", "w_1090", "w_fit", "R_fit", "rough"]
    with open(root / "iface_series.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(all_rows)
    print("wrote", root / "iface_series.csv")


if __name__ == "__main__":
    main()
