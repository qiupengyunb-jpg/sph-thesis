"""A-2 contact-angle analysis on the ISODENSITY contour (flat wall / strip).

For every late frame of each flat-wall case:
  * build the coarse-grained density field rho(x,y)  (0.5 sigma cells, 1 sigma
    Gaussian smoothing)   -- the SAME field definition as cg_analysis/a1
  * take the largest 0.5*rho_bulk contour
  * measure H (height of the contour above the wall contact plane h = wall_re)
    and l (x-extent of the contour where it crosses h = 1 sigma, i.e. the
    isodensity contact width),  theta = 2 atan(2H/l)
  * independent check: least-squares circle through the free part of the
    contour (h > --exclude-h, default 2 sigma) and the angle implied by it.
The near-wall particle layer is therefore excluded from the geometry.

usage:
    python cg_contact_angle.py <results-root> [--exclude-h 2.0]
                               [--width-h 1.0] [--out contact_angle.csv]

<results-root> holds one sub-directory per case (parameters.csv, selfcheck.csv,
output_<case>/CGParticles_ite_*.vtp).  The wall geometry (fibre_half_width,
wall_re, domain_height) is read from each case's parameters.csv.
"""
import csv
import sys
from pathlib import Path

import numpy as np

# the density field / contour definitions live in the sibling a1 tool-set
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "a1"))
from cg_iface import bulk_density, contour_metrics, density_field, load_vtp, read_params


def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc = sol[0] / 2.0, sol[1] / 2.0
    r = np.sqrt(sol[2] + xc * xc + yc * yc)
    return float(xc), float(yc), float(r)


def main():
    args, opts = [], {}
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            if "=" in a:
                k, v = a.split("=", 1)
            elif i + 1 < len(argv):
                k, v = a, argv[i + 1]
                i += 1
            else:
                k, v = a, ""
            opts[k] = v
        else:
            args.append(a)
        i += 1
    if not args:
        print(__doc__)
        raise SystemExit(2)
    ROOT = Path(args[0])
    if not ROOT.is_dir():
        raise SystemExit("not a directory: %s" % ROOT)
    # near-wall band excluded from the circle fit, in sigma above the contact plane
    exclude_h = float(opts.get("--exclude-h", 2.0))
    # h at which the isodensity "contact width" of the droplet is measured
    width_h = float(opts.get("--width-h", 1.0))
    # averaging window: frames with t > window * t_end
    window = float(opts.get("--window", 0.85))
    out_name = opts.get("--out", "contact_angle.csv")
    band = 0.6                                # +/- sigma around width_h (fixed)

    print("%-8s %-7s %-8s %-8s %-8s %-9s %-9s %s"
          % ("case", "eps_pf", "H(σ)", "l(σ)", "θ_iso(°)", "R_fit(σ)", "H_c(σ)", "θ_fit(°)"))
    rows = []
    for d in sorted([x for x in ROOT.iterdir() if x.is_dir()]):
        name = d.name
        p = read_params(d)
        lx, ly = float(p["domain_length"]), float(p["domain_height"])
        hw = float(p["fibre_half_width"])
        # geometric contact plane (wall well position); 0.5 sigma by default
        h_contact = float(p.get("wall_re", 0.5))
        out = d / ("output_" + name)
        fr = []
        with open(d / "selfcheck.csv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                f = out / ("CGParticles_ite_%010d.vtp" % int(r["steps"]))
                if f.exists():
                    fr.append((float(r["time"]), f))
        t_end = fr[-1][0]
        fr = [f for f in fr if f[0] > window * t_end]
        H_list, l_list, th_list, R_list, Hc_list, thfit_list = [], [], [], [], [], []
        for t, f in fr[::max(1, len(fr) // 12)]:
            pos = load_vtp(f, ("Position",))["Position"]
            xs, ys, rho = density_field(pos, lx, ly)
            rb = bulk_density(rho)
            m = contour_metrics(xs, ys, rho, 0.5 * rb)
            if m is None:
                continue
            x, y = m["contour"]
            h = np.abs(y - 0.5 * ly) - hw            # contour points in h
            H = float(h.max()) - h_contact
            # contact width: where the contour crosses h = width_h
            msk = np.abs(h - width_h) < band
            if msk.sum() < 2:
                continue
            l = float(x[msk].max() - x[msk].min())
            if H <= 0 or l <= 0:
                continue
            th = 2.0 * np.degrees(np.arctan2(2.0 * H, l))
            H_list.append(H); l_list.append(l); th_list.append(th)
            # circle fit on the free interface only (h > exclude_h)
            free = np.abs(y - 0.5 * ly) - hw > exclude_h
            if free.sum() >= 8:
                xc, yc, R = fit_circle(x[free], y[free])
                hc = abs(yc - 0.5 * ly) - hw - h_contact  # centre height above contact plane
                Hc_list.append(hc); R_list.append(R)
                if R > abs(hc):
                    Hfit = hc + R
                    lfit = 2.0 * np.sqrt(R * R - hc * hc)
                    thfit_list.append(2.0 * np.degrees(np.arctan2(2.0 * Hfit, lfit)))
        if not H_list:
            print("%-8s  (no usable frame)" % name)
            continue
        row = dict(case=name, eps_pf=float(p["wall_depth_D0"]),
                   H=float(np.mean(H_list)), l=float(np.mean(l_list)),
                   theta_iso=float(np.mean(th_list)), theta_iso_std=float(np.std(th_list)),
                   R_fit=float(np.mean(R_list)) if R_list else np.nan,
                   H_c=float(np.mean(Hc_list)) if Hc_list else np.nan,
                   theta_fit=float(np.mean(thfit_list)) if thfit_list else np.nan,
                   n_frames=len(H_list), window_lo=window * t_end, t_end=t_end,
                   exclude_h=exclude_h, width_h=width_h)
        rows.append(row)
        print("%-8s %-7.1f %-8.2f %-8.1f %-8.1f %-9.2f %-9.2f %.1f"
              % (name, row["eps_pf"], row["H"], row["l"], row["theta_iso"],
                 row["R_fit"], row["H_c"], row["theta_fit"]))
    if not rows:
        raise SystemExit("no case produced a contact angle")
    with open(ROOT / out_name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", ROOT / out_name)


if __name__ == "__main__":
    main()
