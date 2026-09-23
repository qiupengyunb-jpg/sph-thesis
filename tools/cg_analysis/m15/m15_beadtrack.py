#!/usr/bin/env python3
"""Stage 2 (v2): persistent bead tracking with gap bridging and strict merges.

v1 of this tool produced the BASELINE that Stage 2 must be compared against
(commit 4f7f67162, outputs m15_track_*.csv of that run).  v2 adds:

  * PERSISTENCE: a peak only becomes a registered bead after it has been seen
    in --persist consecutive frames; shorter-lived peaks are flicker.
  * GAP BRIDGING: an unmatched bead may be missing for up to --gap frames and
    re-appear within a physically plausible distance (velocity * dt * (gap+1));
    that is a DETECTION_GAP, not a death + birth.
  * PREDICTION: matching uses the previous position AND velocity, so a bead
    crossing the periodic boundary z=0/Lz is matched, never split.
  * STRICT MERGE (task section 10): the label TRUE_MERGE requires two stable
    tracks, a monotonically shrinking pair distance, a surviving track whose
    size proxy is compatible with the sum of the parents, and the disappearance
    not being a threshold crossing.  Everything else is DETECTION_GAP,
    UNRESOLVED or TRUE_DEATH.

Outputs: m15_track_beads.csv, m15_track_events.csv, m15_track_merge_verify.csv,
m15_track_summary.csv (single row, the unified table).
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

MIN_STABLE_FRAMES = 4        # a track must live this long to be "stable"
MERGE_DIST_SIGMA = 6.0       # parents must be this close when the merge happens


def pdist(a, b, Lz):
    d = abs(a - b) % Lz
    return min(d, Lz - d)


def profile_frames(run_dir, R0, Lz, nbins, t_lo, t_hi, tag, smooth,
                   prom_frac, min_dist_sigma, legacy):
    files = M.frame_list(run_dir)
    times = M.frame_times(run_dir, files)
    out = []
    dz = Lz / nbins
    for f, tf in zip(files, times):
        if tf < t_lo or tf > t_hi:
            continue
        _t, z, r = M.read_vtp(f)
        h = (M.h_profile_slabmax if tag == "A" else M.h_profile_outerlayer)(
            z, r, R0, Lz, nbins)
        if np.all(np.isnan(h)):
            continue
        if smooth:
            h = M._fill_and_smooth(h, nbins, smooth)
        mean_h = float(np.mean(h))
        amp = float(np.max(h) - np.min(h))
        peaks = B.find_peaks_periodic(h, max(1, int(min_dist_sigma / dz)),
                                      max(B.PROM_ABS_SIGMA, prom_frac * amp), legacy)
        beads = []
        for p in peaks:
            d = min(6, nbins // 4)
            seg = h[(np.arange(p - d, p + d + 1)) % nbins]
            beads.append(dict(z=p * dz,
                              size=float(np.sum(np.maximum(seg - mean_h, 0.0)) * dz)))
        out.append(dict(t=float(tf), beads=beads, mean_h=mean_h, amp=amp))
    return out


def track(prof, Lz, persist, gap, dt_hint=1.0):
    tracks = []           # dict(id, frames=[(t,z,size,gap)], birth, death, kind)
    events = []
    nid = 0
    for k, rec in enumerate(prof):
        t = rec["t"]
        live = [b for b in tracks if b["death"] is None]
        # predict
        preds = {}
        for b in live:
            fr = [x for x in b["frames"] if not x[3]]
            if len(fr) >= 2 and fr[-1][0] > fr[-2][0]:
                v = (fr[-1][1] - fr[-2][1]) / (fr[-1][0] - fr[-2][0])
                preds[b["id"]] = fr[-1][1] + v * (t - fr[-1][0])
            elif fr:
                preds[b["id"]] = fr[-1][1]
            else:
                preds[b["id"]] = 0.0
        cand = []
        for b in live:
            for j, nb in enumerate(rec["beads"]):
                cand.append((pdist(preds[b["id"]], nb["z"], Lz), b, j))
        cand.sort(key=lambda c: c[0])
        used_b, used_j = set(), set()
        matched = {}
        for d, b, j in cand:
            if b["id"] in used_b or j in used_j:
                continue
            budget = (gap + 1) * dt_hint * 3.0 + 2.0
            if d > budget:
                continue
            used_b.add(b["id"])
            used_j.add(j)
            matched[b["id"]] = j
        for b in live:
            if b["id"] in matched:
                j = matched[b["id"]]
                missed = sum(1 for x in b["frames"][-gap - 1:] if x[3]) if gap else 0
                if missed and b["birth"] is not None:
                    events.append(dict(time=t, kind="detection_gap", bead=b["id"],
                                       partner=-1, z=rec["beads"][j]["z"],
                                       d_pair=np.nan, note="reacquired after %d missed frames" % missed))
                b["frames"].append((t, rec["beads"][j]["z"], rec["beads"][j]["size"], False))
                if b["birth"] is None and len([x for x in b["frames"] if not x[3]]) >= persist:
                    first = [x for x in b["frames"] if not x[3]][0]
                    b["birth"] = first[0]
                    events.append(dict(time=first[0], kind="birth", bead=b["id"],
                                       partner=-1, z=first[1], d_pair=np.nan,
                                       note="registered after %d frames" % persist))
            else:
                b["frames"].append((t, np.nan, np.nan, True))
                if sum(1 for x in b["frames"][-gap - 1:] if x[3]) > gap:
                    b["death"] = b["frames"][-gap - 2][0] if gap else b["frames"][-1][0]
        for j, nb in enumerate(rec["beads"]):
            if j in used_j:
                continue
            tracks.append(dict(id=nid, frames=[(t, nb["z"], nb["size"], False)],
                               birth=None, death=None))
            nid += 1
    last_t = prof[-1]["t"]
    for b in tracks:
        if b["death"] is None:
            b["death"] = last_t
    # stable set
    stable = [b for b in tracks
              if b["birth"] is not None and (b["death"] - b["birth"]) > 0.5 and
              len([x for x in b["frames"] if not x[3]]) >= MIN_STABLE_FRAMES]
    # ---- merge / death attribution for stable tracks ----
    verify = []
    by_id = {b["id"]: b for b in stable}
    for b in stable:
        if b["death"] >= last_t - 1e-9:
            continue
        zlast = [x for x in b["frames"] if not x[3]][-1][1]
        # candidate survivor: another stable track alive at the death time
        best, bd = None, np.inf
        for c in stable:
            if c["id"] == b["id"]:
                continue
            fr = [x for x in c["frames"] if not x[3] and abs(x[0] - b["death"]) < 2.0]
            if not fr:
                continue
            d = pdist(zlast, fr[-1][1], Lz)
            if d < bd:
                best, bd = c, d
        kind, note = "true_death", "no survivor within %.1f sigma" % MERGE_DIST_SIGMA
        if best is not None and bd <= MERGE_DIST_SIGMA:
            hist = []
            for (t1, z1, _s1, g1) in [x for x in b["frames"] if not x[3]][-6:]:
                z2 = None
                for (t2, zz2, _s2, g2) in best["frames"]:
                    if not g2 and abs(t2 - t1) < 1e-9:
                        z2 = zz2
                        break
                if z2 is not None:
                    hist.append(pdist(z1, z2, Lz))
            shrinking = bool(len(hist) >= 3 and
                             all(hist[i] >= hist[i + 1] - 0.5 for i in range(len(hist) - 1)))
            size_dead = [x for x in b["frames"] if not x[3]][-1][2]
            size_surv_before = best["frames"][-1][2]
            size_surv_at = None
            for (t2, _z2, s2, g2) in best["frames"]:
                if not g2 and t2 > b["death"]:
                    size_surv_at = s2
                    break
            compatible = (size_surv_at is None or
                          size_surv_at >= 0.5 * (size_dead + size_surv_before))
            if shrinking and compatible:
                kind, note = "true_merge", "shrinking pair + compatible size"
            else:
                kind = "unresolved"
                note = "near survivor but shrinking=%s size_compatible=%s" % (shrinking, compatible)
            verify.append(dict(death_time=b["death"], dead=b["id"], survivor=best["id"],
                               d_pair=bd, d_pair_history=";".join("%.2f" % x for x in hist),
                               shrinking=shrinking, size_compatible=compatible,
                               verdict=kind))
        events.append(dict(time=b["death"], kind=kind, bead=b["id"],
                           partner=best["id"] if best is not None else -1,
                           z=zlast, d_pair=bd if best is not None else np.nan,
                           note=note))
    # ---- tables ----
    rows = []
    for b in stable:
        fr = [x for x in b["frames"] if not x[3]]
        v = []
        for i in range(1, len(fr)):
            dt = fr[i][0] - fr[i - 1][0]
            if dt > 0:
                v.append(pdist(fr[i][1], fr[i - 1][1], Lz) / dt)
        rows.append(dict(bead=b["id"], birth=b["birth"], death=b["death"],
                         lifetime=b["death"] - b["birth"], n_frames=len(fr),
                         z_birth=fr[0][1], z_death=fr[-1][1],
                         mean_vz=float(np.mean(v)) if v else np.nan,
                         max_vz=float(np.max(v)) if v else np.nan,
                         size_mean=float(np.mean([x[2] for x in fr])),
                         size_first=fr[0][2], size_last=fr[-1][2],
                         n_gap_frames=sum(1 for x in b["frames"] if x[3])))
    births = [e for e in events if e["kind"] == "birth" and e["bead"] in by_id]
    merges = [e for e in events if e["kind"] == "true_merge"]
    gaps = [e for e in events if e["kind"] == "detection_gap" and e["bead"] in by_id]
    deaths = [e for e in events if e["kind"] == "true_death"]
    unres = [e for e in events if e["kind"] == "unresolved"]
    T = prof[-1]["t"] - prof[0]["t"]
    life = [r["lifetime"] for r in rows]
    spd = [r["mean_vz"] for r in rows if np.isfinite(r["mean_vz"])]
    summ = dict(t_start=prof[0]["t"], t_end=prof[-1]["t"], T=T,
                n_frames=len(prof), n_flicker=len(tracks) - len(stable),
                stable_beads=len(stable),
                birth_count=len(births), R_birth=len(births) / T if T else np.nan,
                true_merge=len(merges), unresolved_events=len(unres),
                true_death=len(deaths), R_merge=len(merges) / T if T else np.nan,
                detection_gap=len(gaps), gap_frames_total=int(sum(r["n_gap_frames"] for r in rows)),
                median_bead_lifetime=float(np.median(life)) if life else np.nan,
                mean_migration_speed=float(np.mean(spd)) if spd else np.nan,
                median_migration_speed=float(np.median(spd)) if spd else np.nan,
                max_bead_count=max(len(p["beads"]) for p in prof))
    return summ, rows, events, verify


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
    ap.add_argument("--min-dist-sigma", type=float, default=2.0)
    ap.add_argument("--persist", type=int, default=2)
    ap.add_argument("--gap", type=int, default=2)
    ap.add_argument("--legacy", action="store_true")
    a = ap.parse_args()
    nbins = a.nbins or max(48, int(round(a.Lz * 2.0)))
    prof = profile_frames(a.run_dir, a.R0, a.Lz, nbins, a.t_lo, a.t_hi,
                          a.method, a.smooth, a.prom_frac, a.min_dist_sigma, a.legacy)
    if len(prof) < 5:
        raise SystemExit("need >=5 frames")
    summ, rows, events, verify = track(prof, a.Lz, a.persist, a.gap)
    if a.label:
        summ["parameter"] = a.label
    with open(os.path.join(a.run_dir, "m15_track_beads.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(a.run_dir, "m15_track_events.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["time", "kind", "bead", "partner", "z", "d_pair", "note"])
        w.writeheader()
        for e in events:
            w.writerow(e)
    with open(os.path.join(a.run_dir, "m15_track_summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ.keys())); w.writeheader(); w.writerow(summ)
    if verify:
        with open(os.path.join(a.run_dir, "m15_track_merge_verify.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(verify[0].keys())); w.writeheader()
            for v in verify:
                w.writerow(v)
    for k, v in summ.items():
        print("%-22s %s" % (k, v))


if __name__ == "__main__":
    main()
