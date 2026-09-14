"""Reconcile the two cohort-based rho_s series."""
import numpy as np

OUT = r"E:\哈哈\_pr_diag\out"
z = np.load(OUT + r"\support_frames.npz")
tags = sorted({k.split("_")[0] for k in z.files}, key=lambda k: float(k[1:]))
ref = dict(zip(z["T0_id"], z["T0_s_num"]))


def get(tag, name):
    return z["%s_%s" % (tag, name)]


late = "T62"
reg = np.abs(get(late, "x")) < 6.0
ratio_late = get(late, "s_num")[reg] / np.array(
    [ref[i] for i in get(late, "id")[reg]])
order = np.argsort(ratio_late)[:200]
ids = set(np.asarray(get(late, "id")[reg])[order].tolist())
print("cohort size", len(ids), " min ratio", ratio_late[order].min())

print("\n%-6s %10s %10s %10s %10s %10s"
      % ("T", "rho_s_coh", "rho_c_coh", "support", "gap_mean", "n<=14"))
rows = []
for tag in tags:
    sel = np.array([i in ids for i in get(tag, "id")])
    s = get(tag, "s_num")
    rr = s[sel] / np.array([ref[i] for i in get(tag, "id")[sel]])
    rows.append((float(tag[1:]), get(tag, "rho_s")[sel].mean(),
                 get(tag, "rho_c")[sel].mean(), rr.mean(),
                 np.nanmean(get(tag, "gap")[sel]),
                 get(tag, "nbr")[sel].mean(),
                 (get(tag, "d_min")[sel] / (1.0 / 24.0)).mean(),
                 int((get(tag, "nbr")[sel] <= 14).sum())))
    print("%-6s %10.5f %10.6f %10.4f %10.2f %10d"
          % (tag[1:], get(tag, "rho_s")[sel].mean(),
             get(tag, "rho_c")[sel].mean(), rr.mean(),
             np.nanmean(get(tag, "gap")[sel]),
             int((get(tag, "nbr")[sel] <= 14).sum())))

with open(OUT + r"\cohort.csv", "w", encoding="utf-8") as fh:
    fh.write("T,cohort_nbr_mean,cohort_support_ratio_mean,cohort_dmin_mean,"
             "cohort_rho_s_mean,cohort_rho_c_mean,cohort_gap_mean,n_le_14nbr\n")
    for r in rows:
        fh.write("%.10g,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n"
                 % (r[0], r[5], r[3], r[6], r[1], r[2], r[4], r[7]))
print("\ncohort.csv rewritten with the corrected cohort (200 worst at T=62)")

print("\nsanity: rho_s of T0 frame equals s_num/SIGMA0 for the same ids")
print("  T0 mean rho_s (cohort) = %.5f"
      % get("T0", "rho_s")[np.array([i in ids for i in get("T0", "id")])].mean())
