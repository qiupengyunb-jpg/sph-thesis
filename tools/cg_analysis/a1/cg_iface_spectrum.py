"""Interface line-tension test from the shape fluctuation spectrum.

For a 2D droplet with line tension gamma the capillary-wave spectrum of the
interface radius r(theta) obeys equipartition,
        <|r_q|^2>  ~  kT / (gamma q^2),
so a log-log plot of <|r_q|^2> against q should be a straight line of slope -2
and its amplitude fixes gamma.  A frozen aggregate has no such spectrum.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cg_iface import bulk_density, contour_metrics, density_field, load_vtp, read_params


def polar_spectrum(x, y, ntheta=512, qmax=32):
    cx, cy = x.mean(), y.mean()
    th = np.arctan2(y - cy, x - cx)
    r = np.hypot(x - cx, y - cy)
    order = np.argsort(th)
    th, r = th[order], r[order]
    grid = np.linspace(-np.pi, np.pi, ntheta, endpoint=False)
    th2 = np.unwrap(np.append(th, th[0] + 2.0 * np.pi))
    r2 = np.append(r, r[0])
    rg = np.interp(grid, th2, r2)
    a = np.fft.rfft(rg - rg.mean()) / ntheta
    return np.abs(a[:qmax + 1]) ** 2 * 2.0, rg.mean()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("usage: python cg_iface_spectrum.py <results-root>")
        print("       needs <results-root>/iface_series.csv from cg_iface.py")
        raise SystemExit(2)
    root = Path(sys.argv[1])
    if not root.is_dir():
        raise SystemExit("not a directory: %s" % root)
    out_rows = []
    print("%-18s %-9s %-11s %-9s %-11s %-9s %s"
          % ("case", "slope", "gamma_est", "std(AR)", "std(P)/P", "std(A)/A", "note"))
    for d in sorted([x for x in root.iterdir() if x.is_dir()]):
        name = d.name
        p = read_params(d)
        lx, ly = float(p["domain_length"]), float(p["domain_height"])
        out = d / ("output_" + name)
        fr = []
        with open(d / "selfcheck.csv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                f = out / ("CGParticles_ite_%010d.vtp" % int(r["steps"]))
                if f.exists():
                    fr.append((float(r["time"]), f))
        late = [f for f in fr if f[0] > 0.75 * fr[-1][0]]
        late = late[::max(1, len(late) // 30)]
        specs, ARs, Ps, As = [], [], [], []
        for t, f in late:
            pos = load_vtp(f, ("Position",))["Position"]
            xs, ys, rho = density_field(pos, lx, ly)
            rb = bulk_density(rho)
            m = contour_metrics(xs, ys, rho, 0.5 * rb)
            if m is None:
                continue
            x, y = m["contour"]
            s, rmean = polar_spectrum(x, y)
            specs.append(s); ARs.append(m["AR"]); Ps.append(m["P"]); As.append(m["A"])
        if len(specs) < 5:
            continue
        S = np.mean(np.array(specs), axis=0)
        q = np.arange(len(S))
        msk = (q >= 4) & (q <= 20)
        slope, intercept = np.polyfit(np.log(q[msk]), np.log(S[msk]), 1)
        kT = 1.0
        gamma = kT / (2.0 * np.pi * np.exp(intercept))
        note = ""
        if abs(slope + 2.0) > 0.5:
            note = "slope far from -2: no clean capillary spectrum"
        print("%-18s %-9.2f %-11.3f %-9.3f %-11.3f %-9.3f %s"
              % (name, slope, gamma, np.std(ARs), np.std(Ps) / np.mean(Ps),
                 np.std(As) / np.mean(As), note))
        for qq in q[1:25]:
            out_rows.append(dict(case=name, q=int(qq), S=float(S[qq])))
    with open(root / "iface_spectrum.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["case", "q", "S"])
        w.writeheader(); w.writerows(out_rows)
    print("wrote", root / "iface_spectrum.csv")


if __name__ == "__main__":
    main()
