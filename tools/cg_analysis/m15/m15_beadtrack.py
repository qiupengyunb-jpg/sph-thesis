#!/usr/bin/env python3
"""Stage 2: bead trajectory tracking (birth / migration / merge).

Turns the frame-wise bead detection of m15_beads.py into bead IDs that persist
across frames, so that "bead_count drops" can be attributed to a real
migration-and-merge event rather than to a threshold flicker.

Outputs (next to the run):
  m15_track_beads.csv    one row per tracked bead
  m15_track_events.csv   birth / death / merge events
  m15_track_summary.csv  the single-row unified table used by the sweep report

Matching is done on the ATOMICALLY periodic z axis: |dz| is taken modulo Lz, so
a bead crossing z=0 is not reported as death + birth.
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "m14"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m14_axisym_modes as M  # noqa: E402
import m15_beads as B  # noqa: E402

MIN_LIFETIME_FRAMES = 2      # a bead must survive this many frames to count
MERGE_DIST_SIGMA = 3.0       # parents closer than this before a death => merge


def pdist(a, b, Lz):
    d = abs(a - b) % Lz
    return min(d, Lz - d)


def size_proxy(h, peak, z, R0, Lz):
    """Excess liquid around a peak: Int (h - <h>) dz over +-quarter spacing."""
    n = h.size
    m = float(np.mean(h))
    # half width = half the distance to the neighbouring minima (robust)
    d = min(6, n // 4)
    seg = h[(np.arange(peak - d, peak + d + 1)) % n]
    return float(np.sum(np.maximum(seg - m, 0.0)) * (Lz / n))


def track(run_dir, R0, Lz, nbins, mmax, t_lo, t_hi, tag="A",
          smooth=2.0, prom_frac=0.25):
    files = M.frame_list(run_dir)
    times = M.frame_times(run_dir, files)
    prof = []
    for f, tf in zip(files, times):
        if tf < t_lo or tf > t_hi:
            continue
        _t, z, r = M.read_vtp(f)
        h = (M.h_profile_slabmax if tag == "A" else M.h_profile_outerlayer)(z, r, R0, Lz, nbins)
        if np.all(np.isnan(h)):
            continue
        # extra smoothing: the raw reconstruction carries a ~0.1-0.3 sigma
        # artefact floor, which otherwise produces spurious peak flicker
        h = M._fill_and_smooth(h, nbins, smooth)
        m = float(np.mean(h))
        amp = float(np.max(h) - np.min(h))
        peaks = B.find_peaks_periodic(h, max(1, int(2.0 / (Lz / nbins))),
                                      max(0.02, prom_frac * amp))
        dz = Lz / nbins
        recs = [dict(z=p * dz, size=size_proxy(h, p, z, R0, Lz)) for p in peaks]
        prof.append(dict(t=float(tf), beads=recs, mean_h=m, amp=amp))
    if len(prof) < 3:
        raise SystemExit("need at least 3 usable frames in the window")

    beads = []          # dict(id, birth, death, frames=[(t,z,size)])
    events = []         # birth/death/merge
    next_id = 0
    for rec in prof[0]["beads"]:
        beads.append(dict(id=next_id, birth=prof[0]["t"], death=None,
                          frames=[(prof[0]["t"], rec["z"], rec["size"])],
                          partner=None, merge_time=None))
        events.append(dict(time=prof[0]["t"], kind="birth", bead=next_id,
                           z=rec["z"], partner=-1, d_pair=np.nan))
        next_id += 1

    for k in range(1, len(prof)):
        prev, cur = prof[k - 1], prof[k]
        dt = cur["t"] - prev["t"]
        alive = [b for b in beads if b["death"] is None]
        cand = []
        for b in alive:
            zp = b["frames"][-1][1]
            for j, nb in enumerate(cur["beads"]):
                cand.append((pdist(zp, nb["z"], Lz), b, j))
        cand.sort(key=lambda c: c[0])
        used_b, used_j = set(), set()
        match = {}
        max_move = max(1.5, 0.4 * (Lz / max(1, len(cur["beads"]))))
        for d, b, j in cand:
            if d > max_move or b["id"] in used_b or j in used_j:
                continue
            used_b.add(b["id"])
            used_j.add(j)
            match[b["id"]] = j
        # update / die
        for b in alive:
            if b["id"] in match:
                j = match[b["id"]]
                b["frames"].append((cur["t"], cur["beads"][j]["z"], cur["beads"][j]["size"]))
            else:
                b["death"] = cur["t"]
        # A bead that dies while a SURVIVING bead sits nearby is a MERGE
        # (the survivor is the one the matcher kept); otherwise it is a death.
        for b in [x for x in alive if x["id"] not in match]:
            zlast = b["frames"][-1][1]
            best, bd = None, np.inf
            for b2 in alive:
                if b2["id"] == b["id"] or b2["id"] not in match:
                    continue
                z2 = cur["beads"][match[b2["id"]]]["z"]
                d = pdist(zlast, z2, Lz)
                if d < bd:
                    best, bd = b2, d
            if best is not None and bd <= MERGE_DIST_SIGMA:
                b["partner"], b["merge_time"] = best["id"], cur["t"]
                events.append(dict(time=cur["t"], kind="merge", bead=b["id"],
                                   z=zlast, partner=best["id"], d_pair=bd))
            else:
                events.append(dict(time=cur["t"], kind="death", bead=b["id"],
                                   z=zlast, partner=-1, d_pair=np.nan))
        # genuinely new peaks are births
        for j, nb in enumerate(cur["beads"]):
            if j in used_j:
                continue
            beads.append(dict(id=next_id, birth=cur["t"], death=None,
                              frames=[(cur["t"], nb["z"], nb["size"])],
                              partner=None, merge_time=None))
            events.append(dict(time=cur["t"], kind="birth", bead=next_id,
                               z=nb["z"], partner=-1, d_pair=np.nan))
            next_id += 1
    # close surviving beads
    for b in beads:
        if b["death"] is None:
            b["death"] = prof[-1]["t"]

    # ---- per-bead table ----
    rows, lifetimes, speeds = [], [], []
    for b in beads:
        fr = b["frames"]
        life = b["death"] - b["birth"]
        v = []
        for i in range(1, len(fr)):
            dt = fr[i][0] - fr[i - 1][0]
            if dt > 0:
                v.append(pdist(fr[i][1], fr[i - 1][1], Lz) / dt)
        mv = float(np.mean(v)) if v else np.nan
        if life > 0:
            lifetimes.append(life)
            if np.isfinite(mv):
                speeds.append(mv)
        rows.append(dict(bead=b["id"], birth=b["birth"], death=b["death"],
                         lifetime=life, n_frames=len(fr),
                         z_birth=fr[0][1], z_death=fr[-1][1],
                         mean_vz=mv, max_vz=float(np.max(v)) if v else np.nan,
                         size_mean=float(np.mean([x[2] for x in fr])),
                         size_first=fr[0][2], size_last=fr[-1][2],
                         merged_into=b["partner"] if b["partner"] is not None else -1,
                         merge_time=b["merge_time"] if b["merge_time"] is not None else -1))
    T = prof[-1]["t"] - prof[0]["t"]
    # Flicker filter: a peak that survives fewer than MIN_LIFETIME_FRAMES
    # frames is a threshold artefact, not a bead.  It is excluded from every
    # count (and reported separately as n_flicker).
    life_of = {b["id"]: (b["death"] - b["birth"]) for b in beads}
    min_life = MIN_LIFETIME_FRAMES * (prof[1]["t"] - prof[0]["t"])
    stable = {i for i, l in life_of.items() if l >= min_life}
    births = [e for e in events if e["kind"] == "birth" and e["bead"] in stable]
    merges = [e for e in events if e["kind"] == "merge" and e["bead"] in stable]
    n_flicker = sum(1 for e in events if e["kind"] == "birth" and e["bead"] not in stable)
    deaths = [b for b in beads if b["death"] < prof[-1]["t"]]
    out = dict(t_start=prof[0]["t"], t_end=prof[-1]["t"], T=T,
               n_frames=len(prof),
               birth_count=len(births), R_birth=len(births) / T if T > 0 else np.nan,
               merge_events=len(merges), R_merge=len(merges) / T if T > 0 else np.nan,
               death_count=len(deaths),
               n_flicker=n_flicker,
               Q_balance=(len(births) / T) / (len(merges) / T + 1e-9) if T > 0 else np.nan,
               median_bead_lifetime=float(np.median(lifetimes)) if lifetimes else np.nan,
               mean_migration_speed=float(np.mean(speeds)) if speeds else np.nan,
               median_migration_speed=float(np.median(speeds)) if speeds else np.nan,
               max_bead_count=max(len(p["beads"]) for p in prof),
               merge_d_pair_mean=float(np.mean([e["d_pair"] for e in merges])) if merges else np.nan)
    # ---- merge verification: separation history of the two parents ----
    verify = []
    for e in merges:
        dead = next((b for b in beads if b["id"] == e["bead"]), None)
        surv = next((b for b in beads if b["id"] == e["partner"]), None)
        if dead is None or surv is None:
            continue
        hist = []
        for (t1, z1, _s1) in dead["frames"][-6:]:
            z2 = None
            for (t2, zz2, _s2) in surv["frames"]:
                if abs(t2 - t1) < 1e-9:
                    z2 = zz2
                    break
            if z2 is not None:
                hist.append(pdist(z1, z2, Lz))
        verify.append(dict(merge_time=e["time"], dead=dead["id"], survivor=surv["id"],
                           d_pair_at_merge=e["d_pair"],
                           d_pair_history=";".join("%.2f" % d for d in hist),
                           monotone_decreasing=bool(len(hist) >= 3 and
                                                    all(hist[i] >= hist[i + 1] - 0.5
                                                        for i in range(len(hist) - 1)))))
    return out, rows, events, verify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--R0", type=float, required=True)
    ap.add_argument("--Lz", type=float, default=80.0)
    ap.add_argument("--nbins", type=int, default=0)
    ap.add_argument("--t-lo", type=float, default=20.0)
    ap.add_argument("--t-hi", type=float, default=1e9)
    ap.add_argument("--method", default="A")
    ap.add_argument("--label", default="")
    ap.add_argument("--smooth", type=float, default=2.0)
    ap.add_argument("--prom-frac", type=float, default=0.25)
    a = ap.parse_args()
    nbins = a.nbins or max(48, int(round(a.Lz * 2.0)))
    summ, rows, events, verify = track(a.run_dir, a.R0, a.Lz, nbins, 12, a.t_lo, a.t_hi,
                                       a.method, a.smooth, a.prom_frac)
    if a.label:
        summ["parameter"] = a.label
    with open(os.path.join(a.run_dir, "m15_track_beads.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(a.run_dir, "m15_track_events.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["time", "kind", "bead", "partner", "z", "d_pair"])
        w.writeheader()
        for e in events:
            w.writerow(e)
    with open(os.path.join(a.run_dir, "m15_track_summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ.keys())); w.writeheader(); w.writerow(summ)
    if verify:
        with open(os.path.join(a.run_dir, "m15_track_merge_verify.csv"), "w",
                  newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(verify[0].keys())); w.writeheader()
            for v in verify:
                w.writerow(v)
    for k, v in summ.items():
        print("%-24s %s" % (k, v))
    for v in verify[:10]:
        print("   merge@%.1f  %d+%d  d_pair=%.2f  hist=[%s]  %s"
              % (v["merge_time"], v["dead"], v["survivor"], v["d_pair_at_merge"],
                 v["d_pair_history"], "monotone" if v["monotone_decreasing"] else "noisy"))


if __name__ == "__main__":
    main()
