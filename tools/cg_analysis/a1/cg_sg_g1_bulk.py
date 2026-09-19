"""Gate 1 bulk analysis: compares lambda = 0 against lambda > 0 in a
homogeneous periodic-in-x slab, using ONLY the interior (3 sigma away from the
reflecting y-walls) so that no free surface enters the statistics.

Reports: true number density, kernel density, T_kin, self-diffusion D from the
MSD (x minimum image), and the long-time neighbour-exchange fraction.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))

MARGIN = 3.0          # sigma excluded next to each reflecting wall
R_CUT = 1.5
ROOT = Path(".")      # set from the command line in main()


def load_vtp(path):
    txt = Path(path).read_text(errors="replace")
    out = {}
    for name in ("Position", "OriginalID"):
        key = 'Name="%s"' % name
        i = txt.find(key)
        s = txt.index(">", i) + 1
        e = txt.index("<", s)
        vals = np.array(txt[s:e].split(), dtype=float)
        out[name] = vals.reshape(-1, 3)[:, :2] if name == "Position" else vals.astype(int)
    return out


def frames(name):
    d = ROOT / name
    out = d / ("output_" + name)
    rows = []
    with open(d / "selfcheck.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            p = out / ("CGParticles_ite_%010d.vtp" % int(r["steps"]))
            if p.exists():
                rows.append((float(r["time"]), int(r["steps"]), p, r))
    return rows


def analyse(name, lx, ly):
    fr = frames(name)
    if len(fr) < 20:
        return None
    dt_frame = (fr[-1][0] - fr[0][0]) / max(len(fr) - 1, 1)
    data = []
    for t, s, p, r in fr:
        d = load_vtp(p)
        data.append((t, d["Position"], d["OriginalID"], r))
    # UNBIASED interior density: average the instantaneous count in the band
    # (a "particles that never touch the wall" mask biases the density low).
    area_int = lx * (ly - 2 * MARGIN)
    late_pos = [p for t, p, _, _ in data[len(data) // 2:]]
    counts = [int(((p[:, 1] > MARGIN) & (p[:, 1] < ly - MARGIN)).sum()) for p in late_pos]
    n_int = int(np.mean(counts))
    rho_true = float(np.mean(counts)) / area_int
    # for the dynamics use ALL particles (identical treatment for every case)
    inside = np.ones(len(data[0][2]), dtype=bool)
    late = data[-1][3]
    rho_k = float(late["sg_rho_mean"]) if late.get("sg_rho_mean") else np.nan

    def msd(lag):
        nf = int(round(lag / dt_frame))
        if nf < 1 or nf >= len(data):
            return np.nan
        vals = []
        for i in range(0, len(data) - nf, max(1, (len(data) - nf) // 10)):
            _, p0, _, _ = data[i]
            _, p1, _, _ = data[i + nf]
            d = p1[inside] - p0[inside]
            d[:, 0] -= lx * np.round(d[:, 0] / lx)
            vals.append(np.mean((d ** 2).sum(1)))
        return float(np.mean(vals))

    def nb_exchange(lag):
        nf = int(round(lag / dt_frame))
        if nf < 1 or nf >= len(data):
            return np.nan
        fr_ = []
        for i in range(0, len(data) - nf, max(1, (len(data) - nf) // 8)):
            _, p0, _, _ = data[i]
            _, p1, _, _ = data[i + nf]
            a, b = p0[inside], p1[inside]
            n0 = cKDTree(a).query_ball_point(a, R_CUT)
            n1 = cKDTree(b).query_ball_point(b, R_CUT)
            rel = []
            for k in range(len(a)):
                s0, s1 = set(n0[k]), set(n1[k])
                s0.discard(k); s1.discard(k)
                if s0:
                    rel.append(1.0 - len(s0 & s1) / len(s0))
            fr_.append(float(np.mean(rel)) if rel else np.nan)
        return float(np.nanmean(fr_))

    tk = float(np.mean([float(r["kinetic_temperature"]) for _, _, _, r in data[len(data) // 2:]]))
    vmax = max(float(r["max_speed"]) for _, _, _, r in data[len(data) // 2:])
    res = dict(case=name, n_interior=n_int, rho_true=rho_true, rho_kernel=rho_k,
               T_kin=tk, max_speed=vmax,
               msd200=msd(200), msd600=msd(600), msd1500=msd(1500),
               nb200=nb_exchange(200), nb600=nb_exchange(600), nb1500=nb_exchange(1500),
               t_end=fr[-1][0])
    res["D_eff"] = res["msd1500"] / (4.0 * 1500.0) if np.isfinite(res["msd1500"]) else np.nan
    return res


def main():
    global ROOT
    if len(sys.argv) < 2:
        print(__doc__)
        print("usage: python cg_sg_g1_bulk.py <results-root> [case ...]")
        print("       <results-root> holds the homogeneous-bulk cases "
              "(one sub-directory per case)")
        raise SystemExit(2)
    ROOT = Path(sys.argv[1])
    if not ROOT.is_dir():
        raise SystemExit("not a directory: %s" % ROOT)
    names = sys.argv[2:] or ["G1b_lat_lam0", "G1b_lat_lam25", "G1b_lat_lam25_h3"]
    print("%-20s %-7s %-9s %-9s %-7s %-8s %-8s %-8s %-7s %-7s %-7s %s"
          % ("case", "rho_true", "rho_kernel", "T_kin", "vmax", "msd200",
             "msd600", "msd1500", "D_eff", "nb(200)", "nb(1500)", "N_int"))
    rows = []
    for nm in names:
        p = {}
        try:
            with open(ROOT / nm / "parameters.csv", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    p[row["parameter"]] = row["value"]
        except OSError:
            print("%-20s (no parameters.csv)" % nm)
            continue
        lx = float(p.get("domain_length", 45.0))
        ly = float(p.get("domain_height", 30.0))
        r = analyse(nm, lx, ly)
        if r is None:
            print("%-20s (not ready)" % nm); continue
        rows.append(r)
        print("%-20s %-7.4f %-9.4f %-9.4f %-7.2f %-8.3f %-8.3f %-8.3f %-7.4f %-7.3f %-7.3f %d"
              % (nm, r["rho_true"], r["rho_kernel"], r["T_kin"], r["max_speed"],
                 r["msd200"], r["msd600"], r["msd1500"], r["D_eff"],
                 100 * r["nb200"], 100 * r["nb1500"], r["n_interior"]))
    if rows:
        with open(ROOT / "g1_bulk.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        base = [r for r in rows if r["case"] == "G1b_lat_lam0"]
        if base:
            b = base[0]
            for r in rows:
                print("  %-20s  rho/rho0=%.3f   D/D0=%.3f"
                      % (r["case"], r["rho_true"] / b["rho_true"],
                         r["D_eff"] / b["D_eff"] if b["D_eff"] else np.nan))
        print("wrote", ROOT / "g1_bulk.csv")


if __name__ == "__main__":
    main()
