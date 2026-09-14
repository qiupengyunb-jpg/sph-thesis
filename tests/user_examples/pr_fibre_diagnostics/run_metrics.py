"""Reusable per-frame metric table for one run directory.

Metrics kept deliberately to the A/B/C deliverable list:
  radial gap per axial bin, h_max, connectivity, continuity density,
  plus solver max_speed parsed from stdout.log.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

DX = 1.0 / 24.0
TIME_SCALE = 0.67082039325
BIN = 2.0


def load(path):
    root = ET.parse(path).getroot()
    data = {}
    for da in root.iter("DataArray"):
        name = da.get("Name")
        if not name or not da.text:
            continue
        vals = np.asarray(da.text.split(), dtype=float)
        ncomp = int(da.get("NumberOfComponents", "1"))
        data[name] = vals.reshape(-1, ncomp) if ncomp > 1 else vals
    return data


def frames(run_dir, t_max):
    out = []
    for p in sorted(run_dir.glob("LiquidFilmHalf_*.vtp")):
        t = float(ET.parse(p).getroot()
                  .find('.//FieldData/DataArray[@Name="TimeValue"]').text)
        T = t / TIME_SCALE
        if T <= t_max + 1e-6:
            out.append((p, T))
    out.sort(key=lambda it: it[1])
    return out


def fibre_of(liquid_path):
    return liquid_path.parent / liquid_path.name.replace(
        "LiquidFilmHalf_", "RigidFiberHalf_")


def bin_table(x, r, lo=-30.0, hi=30.0):
    """max radial gap (dx) and outer radius per axial bin."""
    gap, rmax, x0s = {}, {}, []
    for x0 in np.arange(lo, hi, BIN):
        sel = (x >= x0) & (x < x0 + BIN)
        if sel.sum() < 8:
            continue
        rr = np.sort(r[sel])
        gap[round(x0, 1)] = float(np.diff(rr).max() / DX)
        rmax[round(x0, 1)] = float(rr.max())
        x0s.append(round(x0, 1))
    return gap, rmax, x0s


def connectivity(pos, thr):
    cell = thr
    key = np.floor(pos / cell).astype(np.int64)
    buckets = {}
    for i, k in enumerate(map(tuple, key)):
        buckets.setdefault(k, []).append(i)
    parent = np.arange(len(pos))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    thr2 = thr * thr
    for (kx, ky), idxs in buckets.items():
        for dkx in (-1, 0, 1):
            for dky in (-1, 0, 1):
                other = buckets.get((kx + dkx, ky + dky))
                if not other:
                    continue
                for a_i in idxs:
                    a = pos[a_i]
                    ra = find(a_i)
                    for b_i in other:
                        if b_i <= a_i:
                            continue
                        d = pos[b_i] - a
                        if d[0] * d[0] + d[1] * d[1] <= thr2:
                            rb = find(b_i)
                            if ra != rb:
                                parent[rb] = ra
    roots = np.array([find(i) for i in range(len(pos))])
    _, counts = np.unique(roots, return_counts=True)
    counts = np.sort(counts)[::-1]
    return len(counts), float(counts[0] / len(pos))


def read_max_speed(stdout):
    speeds = {}
    with open(stdout, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("T="):
                continue
            parts = dict(kv.split("=") for kv in line.split() if "=" in kv)
            try:
                # stdout already prints the dimensionless T (solver time is the
                # VTP TimeValue); do NOT divide again here.
                speeds[round(float(parts["T"]), 3)] = float(parts["max_speed"])
            except (KeyError, ValueError):
                continue
    return speeds


def run_metrics(run_root, out_dir, t_max=52.0):
    out = Path(out_dir)
    candidates = [d for d in Path(run_root).glob("output*") if d.is_dir()]
    liquid_dir = max(candidates,
                     key=lambda d: len(list(d.glob("LiquidFilmHalf_*.vtp"))))
    speeds = read_max_speed(Path(run_root) / "stdout.log")
    rows = []
    for path, T in frames(liquid_dir, t_max):
        a = load(path)
        pos = a["Position"][:, :2]
        x, r, rho = pos[:, 0], pos[:, 1], a["Density"]
        gap, rmax, _ = bin_table(x, r)
        fb = fibre_of(path)
        fib_rmax = {}
        if fb.exists():
            fp = load(fb)["Position"][:, :2]
            _, frm, _ = bin_table(fp[:, 0], fp[:, 1])
            fib_rmax = frm
        hmax = max((rmax[k] - fib_rmax[k] for k in rmax if k in fib_rmax),
                   default=np.nan)
        central = {k: v for k, v in gap.items() if -4.1 <= k < 4.0}
        nblk, big = connectivity(pos, 1.5 * DX)
        reg = np.abs(x) < 6.0
        rows.append(dict(
            T=T,
            gap_central_max=max(central.values()) if central else np.nan,
            gap_interior_max=max(gap.values()),
            gap_bin_m4_m2=gap.get(-4.0, np.nan),
            gap_bin_m2_0=gap.get(-2.0, np.nan),
            gap_bin_0_2=gap.get(0.0, np.nan),
            gap_bin_2_4=gap.get(2.0, np.nan),
            h_max=hmax,
            blocks=nblk,
            biggest_block_frac=big,
            rho_c_min=float(rho[reg].min()),
            rho_c_max=float(rho[reg].max()),
            max_speed=speeds.get(round(T, 3), np.nan),
        ))
    keys = list(rows[0].keys())
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.csv", "w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for row in rows:
            fh.write(",".join("%.10g" % row[k] for k in keys) + "\n")
    return rows


if __name__ == "__main__":
    import sys
    run_root, out_dir = sys.argv[1], sys.argv[2]
    rows = run_metrics(run_root, out_dir)
    print("%-6s %8s %8s %8s %8s %8s %8s %6s %8s"
          % ("T", "gapC", "gap26", "b-4..-2", "b-2..0", "b0..2", "b2..4",
             "h_max", "maxspd"))
    for row in rows:
        print("%-6.1f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %6.3f %8.5f"
              % (row["T"], row["gap_central_max"], row["gap_interior_max"],
                 row["gap_bin_m4_m2"], row["gap_bin_m2_0"], row["gap_bin_0_2"],
                 row["gap_bin_2_4"], row["h_max"], row["max_speed"]))
