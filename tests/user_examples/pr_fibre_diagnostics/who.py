"""Who loses support early: free-surface particles or interior ones?"""
import numpy as np

OUT = r"E:\哈哈\_pr_diag\out"
z = np.load(OUT + r"\support_frames.npz")
ref = dict(zip(z["T0_id"], z["T0_s_num"]))

print("%-6s %8s %8s %10s %10s %10s" %
      ("T", "n<0.95", "n<0.90", "r_low", "r_high", "surf_frac"))
for tag in ("T20", "T28", "T36", "T42", "T50", "T56", "T62"):
    x, r, s, pid = (z[tag + "_x"], z[tag + "_r"], z[tag + "_s_num"],
                    z[tag + "_id"])
    ind = z[tag + "_indicator"]
    reg = np.abs(x) < 6.0
    ratio = s[reg] / np.array([ref[i] for i in pid[reg]])
    bad = ratio < 0.95
    b90 = ratio < 0.90
    rr = r[reg][bad]
    print("%-6s %8d %8d %10s %10s %10.3f"
          % (tag, bad.sum(), b90.sum(),
             ("%.3f" % rr.min()) if bad.sum() else "-",
             ("%.3f" % rr.max()) if bad.sum() else "-",
             ind[reg][bad].mean() if bad.sum() else float("nan")))

    # radial position of each affected particle relative to the local film
    if bad.sum() and tag in ("T42", "T62"):
        print("      affected r quantiles: "
              + ", ".join("%.2f" % q for q in np.percentile(rr, [5, 50, 95])))
