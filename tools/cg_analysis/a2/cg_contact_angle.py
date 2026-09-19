"""Contact-angle measurement for the A-2 flat-wall wetting cases (A-1 model).

Frozen method (A-2 brief, section 5):
  1. late-time averaged density field
  2. rho = 0.5 * rho_bulk iso-density contour
  3. drop every contour point closer than 2 sigma to the wall SURFACE
  4. circle fit (Kasa least squares) to what is left
  5. tangent contact angle at the wall

The density field and rho_bulk are imported from the frozen A-1 definitions
(tools/cg_analysis/a1/cg_iface.py), so the A-1 and A-2 numbers cannot drift
apart.  Nothing here re-defines them.

Wall model (solver: strip fibre).  h is the particle-centre to wall-SURFACE
distance, h = |y - Ly/2| - fibre_half_width, and h = 0.5 sigma is geometric
contact.  The strip has a top face at y = Ly/2 + w (outward normal +y) and a
bottom face at y = Ly/2 - w (normal -y); h uses whichever face is nearer.  A
case run with --cluster-offset-y puts the droplet entirely on one face, so one
half space carries the liquid and the other is empty (reported as N_other).

The contact angle is the tangent of the fitted circle at the wall:
    theta = 90 deg + asin(yc / R),   yc = circle centre height above the face
(90 deg -> centre on the wall, 180 deg -> centre one radius above it).

Two quantities are deliberately NOT the A-1 definition, because the A-1 radial
definition about the contour centroid is degenerate for a droplet sitting on a
wall (the profile crosses the wall):
  * interface_width: 10-90 % width of rho(r) measured along the radius of the
    fitted circle, using shells that stay in the apex sector
    (|angle - 90 deg| <= 35 deg) so neither the wall nor the foot contaminates
    it.  w_fit_tanh_apex (tanh fit to the same profile) is reported alongside.
  * equilibrium_time: first time after which the droplet gyration radius Rg(t)
    never leaves +/-2 % of its tail mean.  It says when the droplet stopped
    changing SIZE; theta_converged below is the stricter shape test.
theta_drift_per_100t (deg per 100 time units, linear fit over the last
--conv-frames frames) and theta_converged (|drift| <= 0.30 deg/100t) say
whether the reported theta can be used at all.

output: <root>/contact_angle_A1.csv
        <root>/analysis_log_contact_angle.txt
usage : python cg_contact_angle.py <results-root> [--avg N] [--theta-scatter K]
                                   [--conv-frames K] [--drift-tol X]
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_A1 = _HERE.parent / "a1"
if str(_A1) not in sys.path:
    sys.path.insert(0, str(_A1))
from cg_iface import (  # noqa: E402  (sys.path set just above)
    load_vtp,
    read_params,
    density_field,
    bulk_density,
    contour_metrics,
    radial_profile,
    interface_width,
)

H_EXCLUDE = 2.0     # sigma; near-wall band excluded from the fit (frozen)
LEVEL_FRAC = 0.5    # iso-density level as a fraction of rho_bulk (frozen)
EQ_TOL = 0.02       # 2 % band for the Rg equilibrium-time detector
APEX_HALF_ANGLE = 35.0
DRIFT_TOL = 0.30    # deg per 100 time units


def circular_mean(x, lx):
    ang = 2.0 * np.pi * np.asarray(x, dtype=float) / lx
    return (np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang))) / (2.0 * np.pi)) * lx


def unwrap_x(x, lx, centre):
    return centre + ((np.asarray(x) - centre + 0.5 * lx) % lx) - 0.5 * lx


def read_times(case_dir, dt):
    """steps -> physical time, from selfcheck.csv (dt is the fallback)."""
    f = case_dir / "selfcheck.csv"
    if f.exists():
        try:
            return {int(r["steps"]): float(r["time"]) for r in csv.DictReader(open(f, encoding="utf-8"))}
        except Exception:
            pass
    return {}


def droplet_side_stats(pos, lx, ly, w):
    """Per half-space particle count, centre-of-mass height and gyration radius."""
    out = {}
    for side in (+1, -1):
        face = 0.5 * ly + side * w
        y = pos[:, 1]
        sel = (y > 0.5 * ly) if side > 0 else (y <= 0.5 * ly)
        p = pos[sel]
        st = dict(n=int(sel.sum()), side=side, face=float(face))
        if st["n"] >= 8:
            cx = circular_mean(p[:, 0], lx)
            dx = unwrap_x(p[:, 0], lx, cx)
            dy = p[:, 1] - face
            st["h_com"] = float(np.mean(dy) * side)
            st["Rg"] = float(np.sqrt(np.mean(dx * dx + dy * dy)))
        out[side] = st
    return out


def equilibrium_time(times, values, tol=EQ_TOL, tail_frac=0.2):
    """First time after which the series never leaves the tol band of its tail mean."""
    v = np.asarray(values, dtype=float)
    t = np.asarray(times, dtype=float)
    ok = np.isfinite(v) & np.isfinite(t)
    if ok.sum() < 8:
        return float("nan"), float("nan")
    v, t = v[ok], t[ok]
    n_tail = max(5, int(tail_frac * len(v)))
    ref = float(np.mean(v[-n_tail:]))
    if ref <= 0:
        return float("nan"), ref
    bad = np.abs(v - ref) > tol * ref
    if not bad.any():
        return float(t[0]), ref
    last = int(np.max(np.nonzero(bad)[0]))
    if last + 1 >= len(t):
        return float("nan"), ref
    return float(t[last + 1]), ref


def drift_rate(times, values):
    """Least-squares slope of values vs time, in units per 100 time units."""
    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(t) & np.isfinite(v)
    if ok.sum() < 5 or np.ptp(t[ok]) <= 0:
        return float("nan")
    return float(np.polyfit(t[ok], v[ok], 1)[0] * 100.0)


def masked_density(xs, ys, rho, ly, w, side):
    """Zero the wall interior and the half space that carries no liquid."""
    dy = ys - 0.5 * ly
    keep = np.abs(dy) > w
    keep = keep & ((dy > 0) if side > 0 else (dy <= 0))
    return np.where(keep[:, None], rho, 0.0)


def fit_circle(x, y):
    a_mat = np.c_[2.0 * x, 2.0 * y, np.ones(len(x))]
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(a_mat, b, rcond=None)
    xc, yc, c = sol
    return float(xc), float(yc), float(np.sqrt(c + xc * xc + yc * yc))


def measure_angle(xs, ys, rho, rho_bulk, ly, w, side, h_excl=H_EXCLUDE):
    """Circle-fit contact angle of the liquid on the chosen wall face."""
    level = LEVEL_FRAC * rho_bulk
    field = masked_density(xs, ys, rho, ly, w, side)
    m = contour_metrics(xs, ys, field, level)
    cs = plt.contour(xs, ys, field, levels=[level])
    segs = [s for s in cs.allsegs[0] if len(s) >= 8]
    plt.close("all")
    if not segs:
        return None
    seg = max(segs, key=len)
    x, y = seg[:, 0], seg[:, 1]
    face = 0.5 * ly + side * w
    h = (y - face) if side > 0 else (face - y)
    sel = h >= h_excl
    if m is None:
        area = 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
        per = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
        m = dict(A=float(area), P=per, C=float(4 * np.pi * area / per ** 2),
                 AR=float("nan"), cx=float(x.mean()), cy=float(y.mean()))
    res = dict(A=m["A"], P=m["P"], C=m["C"], AR=m["AR"], cx=m["cx"], cy=m["cy"],
               n_pts=len(x), n_fit=int(sel.sum()),
               footprint=float(np.nanmax(x) - np.nanmin(x)),
               height=float(np.nanmax(h)), h_wall=float(np.nanmin(h)))
    if res["n_fit"] < 12:
        res.update(theta=float("nan"), R_fit=float("nan"), yc=float("nan"),
                   fit_rms=float("nan"), cap_ratio=float("nan"))
        return res
    lx = float(xs[-1] + xs[1] - xs[0])
    X = unwrap_x(x[sel], lx, circular_mean(x[sel], lx))
    Y = h[sel]
    xc, yc, R = fit_circle(X, Y)
    res["fit_rms"] = float(np.sqrt(np.mean((np.hypot(X - xc, Y - yc) - R) ** 2)))
    res["R_fit"] = R
    res["yc"] = yc
    res["theta"] = float(90.0 + np.degrees(np.arcsin(np.clip(yc / R, -1.0, 1.0))))
    chord = 2.0 * np.sqrt(max(R * R - yc * yc, 0.0))
    res["cap_ratio"] = float(res["footprint"] / chord) if chord > 1e-9 else float("nan")
    return res


def apex_interface_width(pos_list, lx, face, xcen, ycen, rho_bulk, rho_dil,
                         R_fit, half_angle=APEX_HALF_ANGLE, dr=0.4):
    """10-90 % and tanh width of rho(r) along the fitted circle radius, apex sector."""
    p = np.vstack(pos_list)
    dx = p[:, 0] - xcen
    dx -= lx * np.round(dx / lx)
    dy = p[:, 1] - ycen
    r = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dy, dx))
    keep = (ang > 90.0 - half_angle) & (ang < 90.0 + half_angle)
    if keep.sum() < 50:
        return float("nan"), float("nan"), float("nan"), float("nan")
    edges = np.arange(0.0, R_fit + 8.0 + dr, dr)
    cnt, _ = np.histogram(r[keep], bins=edges)
    frac = 2.0 * half_angle / 180.0
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2) * frac
    prof = cnt / np.maximum(area, 1e-12)
    rr = 0.5 * (edges[:-1] + edges[1:])
    win = (rr > R_fit - 6.0) & (rr < R_fit + 6.0)

    def cross(level):
        """Radius where the profile first drops below level (interpolated)."""
        idx = np.nonzero(win & (prof < level))[0]
        if not len(idx):
            return float("nan")
        i = int(idx[0])
        if i == 0:
            return float(rr[0])
        p0, p1, r0, r1 = prof[i - 1], prof[i], rr[i - 1], rr[i]
        if p0 == p1:
            return float(r1)
        f = (p0 - level) / (p0 - p1)
        return float(r0 + min(max(f, 0.0), 1.0) * (r1 - r0))

    r_hi, r_lo = cross(0.9 * rho_bulk), cross(0.1 * rho_bulk)
    w1090 = r_lo - r_hi if np.isfinite(r_hi) and np.isfinite(r_lo) else float("nan")
    best = (np.inf, float("nan"), float("nan"))
    if win.any():
        for R0 in np.arange(R_fit - 4.0, R_fit + 4.01, 0.25):
            for ww in np.arange(0.2, 6.01, 0.1):
                model = 0.5 * (rho_bulk + rho_dil) + 0.5 * (rho_bulk - rho_dil) * np.tanh((R0 - rr) / ww)
                err = float(np.mean((prof[win] - model[win]) ** 2))
                if err < best[0]:
                    best = (err, R0, ww)
    return w1090, float(best[2]), r_hi, r_lo


def find_frames(case_dir):
    tag_dir = case_dir / ("output_" + case_dir.name)
    if tag_dir.is_dir():
        return sorted(tag_dir.glob("CGParticles_ite_*.vtp"))
    return sorted(case_dir.glob("output_*/CGParticles_ite_*.vtp"))


def analyse_case(case_dir, n_avg=12, n_scatter=12, n_conv=120, drift_tol=DRIFT_TOL):
    p = read_params(case_dir)
    lx, ly = float(p["domain_length"]), float(p["domain_height"])
    w = float(p["fibre_half_width"])
    dt = float(p["dt"])
    eps_pf = float(p["wall_depth_D0"])
    if p.get("fibre_shape") != "strip" or eps_pf <= 0.0:
        return None, "no strip wall with eps_pf > 0 (fibre_shape=%s, D0=%s)" % (
            p.get("fibre_shape"), p.get("wall_depth_D0"))
    frames = find_frames(case_dir)
    if len(frames) < 4:
        return None, "fewer than 4 frames"

    steps, pos_all = [], []
    for f in frames:
        try:
            steps.append(int(f.stem.split("_ite_")[-1]))
            pos_all.append(load_vtp(f, ("Position",))["Position"])
        except Exception:
            continue
    if len(pos_all) < 4:
        return None, "no readable frames"
    tmap = read_times(case_dir, dt)
    times = [tmap.get(s, s * dt) for s in steps]

    # per-frame pass: which face carries the liquid, when did the size settle
    stats = [droplet_side_stats(pp, lx, ly, w) for pp in pos_all]
    side = +1 if stats[-1][+1]["n"] >= stats[-1][-1]["n"] else -1
    other = -1 if side > 0 else +1
    rg = [s[side].get("Rg", float("nan")) for s in stats]
    t_eq, rg_ref = equilibrium_time(times, rg)

    # late-time averaged density field
    acc = None
    for pp in pos_all[-n_avg:]:
        xs, ys, rho = density_field(pp, lx, ly)
        acc = rho if acc is None else acc + rho
    rho_avg = acc / float(min(n_avg, len(pos_all)))
    rb = bulk_density(rho_avg)

    m = measure_angle(xs, ys, rho_avg, rb, ly, w, side)
    if m is None:
        return None, "no iso-density contour"
    m_other = measure_angle(xs, ys, rho_avg, rb, ly, w, other)

    face = 0.5 * ly + side * w
    r, prof = radial_profile(pos_all[-1], lx, ly, (m["cx"], m["cy"]))
    tail = prof[r > 0.8 * r.max()]
    rho_dil = float(np.median(tail)) if len(tail) else 0.0
    w1090_c, w_fit_c, _ = interface_width(r, prof, rb, rho_dil)
    w1090, w_fit, r_hi, r_lo = apex_interface_width(
        pos_all[-n_avg:], lx, face, m["cx"], face + m["yc"], rb, rho_dil, m["R_fit"])

    # per-frame theta scatter inside the averaging window
    th = []
    for pp in pos_all[-n_scatter:]:
        _, _, rr_ = density_field(pp, lx, ly)
        mm = measure_angle(xs, ys, rr_, bulk_density(rr_), ly, w, side)
        if mm and np.isfinite(mm["theta"]):
            th.append(mm["theta"])
    th = np.asarray(th, dtype=float)

    # convergence: per-frame theta over the last n_conv frames
    conv = []
    for pp in pos_all[-n_conv:]:
        _, _, rr_ = density_field(pp, lx, ly)
        mm = measure_angle(xs, ys, rr_, bulk_density(rr_), ly, w, side)
        conv.append((mm or {}).get("theta", float("nan")))
    conv = np.asarray(conv, dtype=float)
    t_conv = np.asarray(times[-n_conv:], dtype=float)
    drift = drift_rate(t_conv, conv)
    span = np.ptp(t_conv[np.isfinite(t_conv)]) if len(t_conv) else float("nan")
    converged = bool(np.isfinite(drift) and abs(drift) <= drift_tol)
    if np.isfinite(drift) and np.isfinite(span):
        resid = drift / 100.0 * span
    else:
        resid = float("nan")

    sel_core = (np.abs(ys - 0.5 * ly) > (w + H_EXCLUDE))[:, None] & (rho_avg > 0.7 * rb)
    rb_core = float(np.median(rho_avg[sel_core])) if sel_core.any() else float("nan")

    row = dict(case=case_dir.name,
               epsilon_pf=eps_pf,
               lambda_=float(p["interface_gradient_lambda"]),
               h_rho=float(p["interface_gradient_h"]),
               theta=m["theta"],
               rho_bulk=rb,
               interface_width=w1090 if np.isfinite(w1090) else w_fit,
               C=m["C"], AR=m["AR"],
               equilibrium_time=t_eq,
               side="top" if side > 0 else "bottom",
               n_frames=len(pos_all), n_avg=min(n_avg, len(pos_all)),
               N_side=int(stats[-1][side]["n"]), N_other=int(stats[-1][other]["n"]),
               R_fit=m["R_fit"], yc=m["yc"], fit_rms=m["fit_rms"],
               footprint=m["footprint"], height=m["height"],
               cap_ratio=m["cap_ratio"], h_min_face=m["h_wall"],
               n_fit=m["n_fit"], n_pts=m["n_pts"],
               w_fit_tanh_apex=w_fit, r_1090_hi=r_hi, r_1090_lo=r_lo,
               w_1090_centroid=w1090_c, w_fit_centroid=w_fit_c,
               rho_bulk_core=rb_core, rho_dilute=rho_dil,
               Rg_ref=rg_ref, Rg_last=rg[-1],
               theta_scatter_std=float(np.std(th)) if len(th) > 1 else float("nan"),
               theta_scatter_n=len(th),
               theta_conv_first=float(conv[0]) if np.isfinite(conv[0]) else float("nan"),
               theta_conv_last=float(conv[-1]) if np.isfinite(conv[-1]) else float("nan"),
               theta_drift_per_100t=drift,
               theta_drift_resid=resid,
               theta_converged=converged,
               theta_other=(m_other or {}).get("theta", float("nan")))
    return row, None


COLS = ["case", "epsilon_pf", "lambda", "h_rho", "theta", "rho_bulk",
        "interface_width", "C", "AR", "equilibrium_time",
        "side", "n_frames", "n_avg", "N_side", "N_other", "R_fit", "yc",
        "fit_rms", "footprint", "height", "cap_ratio", "h_min_face", "n_fit",
        "n_pts", "w_fit_tanh_apex", "r_1090_hi", "r_1090_lo",
        "w_1090_centroid", "w_fit_centroid", "rho_bulk_core", "rho_dilute",
        "Rg_ref", "Rg_last", "theta_scatter_std", "theta_scatter_n",
        "theta_conv_first", "theta_conv_last", "theta_drift_per_100t",
        "theta_drift_resid", "theta_converged", "theta_other"]


def main():
    args = list(sys.argv[1:])
    n_avg, n_scatter, n_conv, drift_tol = 12, 12, 120, DRIFT_TOL
    for key in ("--avg", "--theta-scatter", "--conv-frames"):
        if key in args:
            i = args.index(key)
            val = int(args[i + 1])
            del args[i:i + 2]
            if key == "--avg":
                n_avg = val
            elif key == "--theta-scatter":
                n_scatter = val
            else:
                n_conv = val
    if "--drift-tol" in args:
        i = args.index("--drift-tol")
        drift_tol = float(args[i + 1])
        del args[i:i + 2]
    if not args:
        print(__doc__)
        raise SystemExit(2)
    root = Path(args[0])
    if not root.is_dir():
        raise SystemExit("not a directory: %s" % root)

    rows, log = [], []
    for d in sorted([x for x in root.iterdir() if x.is_dir()]):
        if not (d / "parameters.csv").exists():
            log.append("skip %-26s no parameters.csv" % d.name)
            continue
        try:
            row, why = analyse_case(d, n_avg=n_avg, n_scatter=n_scatter,
                                    n_conv=n_conv, drift_tol=drift_tol)
        except Exception as exc:                       # keep going on a bad case
            row, why = None, "%s: %s" % (type(exc).__name__, exc)
        if row is None:
            log.append("skip %-26s %s" % (d.name, why))
            continue
        rows.append(row)
        log.append(
            "%-24s eps_pf=%-4g lam=%-4g h=%-5.1g N=%d/%d theta=%7.2f deg  R=%6.2f "
            "yc=%+7.3f rms=%.2f cap=%.3f C=%.3f AR=%.3f foot=%5.1f height=%5.2f "
            "rho_b=%.4f w_apex=%.2f t_eq=%-8s drift=%+6.2f deg/100t%2s conv=%s"
            % (row["case"], row["epsilon_pf"], row["lambda_"], row["h_rho"],
               row["N_side"], row["N_other"], row["theta"], row["R_fit"], row["yc"],
               row["fit_rms"], row["cap_ratio"], row["C"], row["AR"],
               row["footprint"], row["height"], row["rho_bulk"],
               row["interface_width"],
               ("%.1f" % row["equilibrium_time"]) if np.isfinite(row["equilibrium_time"]) else "nan",
               row["theta_drift_per_100t"], "", "YES" if row["theta_converged"] else "NO"))

    out = root / "contact_angle_A1.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLS)
        wr.writeheader()
        for r in rows:
            rr = dict(r)
            rr["lambda"] = rr.pop("lambda_")
            wr.writerow(rr)
    with open(root / "analysis_log_contact_angle.txt", "w", encoding="utf-8") as fh:
        fh.write("A-2 contact angle -- frozen method: 0.5*rho_bulk contour, h >= %g sigma "
                 "kept, Kasa circle fit, theta = 90 + asin(yc/R)\n" % H_EXCLUDE)
        fh.write("density field and rho_bulk imported from "
                 "tools/cg_analysis/a1/cg_iface.py (frozen definitions)\n")
        fh.write("theta is usable only when theta_converged=YES "
                 "(|drift| <= %g deg per 100 time units over the last frames)\n\n" % drift_tol)
        fh.write("\n".join(log) + "\n")
    print("\n".join(log))
    print("\nwrote %s (%d cases)" % (out, len(rows)))


if __name__ == "__main__":
    main()