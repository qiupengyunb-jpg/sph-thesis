"""r=16 / r=24 / r=32 comparison on the tapered broadband case.

Same metric definitions everywhere:
  S_i = W(0) + sum over LIQUID neighbours within 2h of W(|x_i-x_j|)
  delta_i(T) = 1 - S_i(T)/S_i(0)          (matched by OriginalID)
  cohort = particles that are interior (Indicator==0) AND |x|<6a at T=0
  gap    = max radial gap inside a 2a axial bin, in units of dx
  h_max  = max over axial bins of (max liquid r - max fibre r)
  blocks = connected components at 1.5*dx adjacency (mirror not applied)
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

CENTRAL = 4.0
REGION = 6.0
INTERIOR = 26.0
BIN = 2.0
THR = (0.02, 0.03, 0.05)

RUNS = {
    "r16": (Path(r"E:\sphmethod\SPH_results_center\recon_tapered_20260912"
                 r"\broadband_T120\output_recon_broadband_T120"), 1.0 / 16.0),
    "r24": (Path(r"E:\sphmethod\SPH_results_center\recon_tapered_20260914"
                 r"\r24_broadband_T80\output_recon_r24_T80"), 1.0 / 24.0),
    "r32": (Path(r"E:\sphmethod\SPH_results_center\recon_tapered_20260914"
                 r"\r32_broadband_T80\output_recon_r32_T80"), 1.0 / 32.0),
}


def fast_load(path, wanted=("Position", "Indicator", "OriginalID")):
    """Cheap VTP reader: locate each DataArray by name and parse its text."""
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
        out[name] = vals.reshape(-1, 3) if name == "Position" else vals
    return out


def frame_T_map(run_dir):
    """Pair each liquid frame with its T from the run's own stdout log."""
    # NOTE: sort by the numeric tag, NOT by filename.  The initial frame is
    # named "..._ite_0000000000.vtp", and 'i' sorts AFTER any digit, so a plain
    # sorted() would put the T=0 frame last and shift every label by one.
    def tag_time(p):
        t = p.stem.replace("LiquidFilmHalf_", "")
        if t.startswith("ite_"):
            return 0.0
        d = "".join(ch for ch in t if ch.isdigit())
        return float(d) / 1e6 if d else 0.0

    frames = sorted(run_dir.glob("LiquidFilmHalf_*.vtp"), key=tag_time)
    log = run_dir.parent / "stdout.log"
    Ts = []
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith("T="):
                Ts.append(float(line.split()[0][2:]))
    if len(Ts) != len(frames):
        # fall back to TimeValue / 0.67082
        Ts = []
        for p in frames:
            t = float(ET.parse(p).getroot()
                      .find('.//FieldData/DataArray[@Name="TimeValue"]').text)
            Ts.append(t / 0.67082)
    return list(zip(frames, Ts))


def s_and_nbr(pos, dx):
    h = 1.3 * dx
    w0 = 7.0 / (4.0 * np.pi * h * h)
    tree = cKDTree(pos)
    pairs = tree.query_ball_point(pos, 2.0 * h, return_sorted=False)
    s = np.full(len(pos), w0)
    for i, js in enumerate(pairs):
        js = np.asarray(js, dtype=np.int64)
        js = js[js != i]
        if js.size == 0:
            continue
        q = np.linalg.norm(pos[js] - pos[i], axis=1) / h
        m = q < 2.0
        s[i] += (7.0 / (4.0 * np.pi * h * h) *
                 (1 - 0.5 * q[m]) ** 4 * (1 + 2 * q[m])).sum()
    return s


def blocks_of(pos, dx):
    """Connected components at 1.5dx, via KD-tree pairs + csgraph."""
    tree = cKDTree(pos)
    pairs = tree.query_pairs(1.5 * dx, output_type="ndarray")
    n = len(pos)
    if len(pairs) == 0:
        return n, 1.0
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                   shape=(n, n))
    ncomp, labels = connected_components(g, directed=False)
    counts = np.bincount(labels)
    return int(ncomp), float(counts.max() / n)


def bin_stats(x, r, dx):
    """Max radial gap (in units of dx) and outer radius per axial bin."""
    gap, rmax = {}, {}
    for x0 in np.arange(-28.0, 28.0, BIN):
        sel = (x >= x0) & (x < x0 + BIN)
        if sel.sum() < 8:
            continue
        rr = np.sort(r[sel])
        gap[round(x0, 1)] = float(np.diff(rr).max()) / dx
        rmax[round(x0, 1)] = float(rr.max())
    return gap, rmax


def analyse(tag, run_dir, dx, tmax=80.0):
    fs = [(p, T) for p, T in frame_T_map(run_dir) if T <= tmax + 1e-6]
    print("[%s] %d frames, T=%.1f..%.1f, dx=%.5f"
          % (tag, len(fs), fs[0][1], fs[-1][1], dx))
    a0 = fast_load(fs[0][0])
    pos0 = a0["Position"][:, :2]
    id0 = a0["OriginalID"].astype(np.int64)
    ind0 = np.where(a0["Indicator"] > 0.5, 1, 0)
    s0 = s_and_nbr(pos0, dx)
    cohort = (ind0 == 0) & (np.abs(pos0[:, 0]) < REGION)
    ref = dict(zip(id0[cohort], s0[cohort]))
    print("[%s] cohort = %d initial-interior particles with |x0|<6a"
          % (tag, cohort.sum()))

    rows = []
    for p, T in fs:
        a = fast_load(p)
        pos = a["Position"][:, :2]
        x, r = pos[:, 0], pos[:, 1]
        ids = a["OriginalID"].astype(np.int64)
        s = s_and_nbr(pos, dx)
        gap, rmax = bin_stats(x, r, dx)
        fib = p.parent / p.name.replace("LiquidFilmHalf_", "RigidFiberHalf_")
        hmax = np.nan
        if fib.exists():
            f = fast_load(fib)
            _, fmax = bin_stats(f["Position"][:, 0], f["Position"][:, 1], dx)
            hmax = max((rmax[k] - fmax[k] for k in rmax if k in fmax),
                       default=np.nan)
        nblk, big = blocks_of(pos, dx)
        nbr = {int(i): sv for i, sv in zip(ids, s)}
        d = np.array([1.0 - nbr[i] / ref[i] for i in ref])
        cen = [v for k, v in gap.items() if -4.1 <= k < 4.0]
        rows.append(dict(
            T=T, gap_central=max(cen), gap_interior=max(gap.values()),
            h_max=hmax, blocks=nblk, biggest=big,
            min_S=1.0 - float(np.nanmax(d)),
            p95=float(np.nanpercentile(d, 95)),
            p99=float(np.nanpercentile(d, 99)),
            dmax=float(np.nanmax(d)),
            n002=int((d > 0.02).sum()), n003=int((d > 0.03).sum()),
            n005=int((d > 0.05).sum())))
        print("   T=%5.1f gapC=%7.2f gap26=%7.2f hmax=%6.3f blocks=%2d "
              "minS/S0=%.3f dmax=%.4f n>0.03=%d"
              % (T, rows[-1]["gap_central"], rows[-1]["gap_interior"],
                 hmax, nblk, rows[-1]["min_S"], rows[-1]["dmax"],
                 rows[-1]["n003"]))
    return rows


def first_cross(rows, key, level):
    for r in rows:
        if r[key] > level:
            return r["T"]
    return None


def main():
    out = Path(r"E:\哈哈\_pr_diag\out\res_compare")
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for tag, (d, dx) in RUNS.items():
        res[tag] = analyse(tag, d, dx)
        keys = list(res[tag][0].keys())
        with open(out / ("%s.csv" % tag), "w", encoding="utf-8") as fh:
            fh.write(",".join(keys) + "\n")
            for row in res[tag]:
                fh.write(",".join("%.8g" % row[k] for k in keys) + "\n")

    print("\n=== 关键帧汇总 ===")
    hdr = "%-5s" % "T"
    for tag in RUNS:
        hdr += " | %-28s" % tag
    print(hdr)
    keys = ("gap_central", "min_S", "dmax", "n003")
    for T in (0.0, 20.0, 30.0, 40.0, 50.0, 62.0, 70.0, 80.0):
        line = "%-5.0f" % T
        for tag in RUNS:
            r = min(res[tag], key=lambda q: abs(q["T"] - T))
            line += " | gapC=%7.2f minS=%.3f dmax=%.4f n3=%4d" % (
                r["gap_central"], r["min_S"], r["dmax"], r["n003"])
        print(line)

    print("\n=== 首次超过 1.5/2/3 dx 的时间（中央区） ===")
    for tag in RUNS:
        print("  %s: 1.5dx T=%s  2dx T=%s  3dx T=%s"
              % (tag, first_cross(res[tag], "gap_central", 1.5),
                 first_cross(res[tag], "gap_central", 2.0),
                 first_cross(res[tag], "gap_central", 3.0)))

    print("\n=== h_max 与连通块 ===")
    for T in (0.0, 40.0, 50.0, 62.0, 80.0):
        line = "  T=%-4.0f" % T
        for tag in RUNS:
            r = min(res[tag], key=lambda q: abs(q["T"] - T))
            line += "  %s: hmax=%.3f blocks=%d" % (tag, r["h_max"], r["blocks"])
        print(line)


if __name__ == "__main__":
    main()
