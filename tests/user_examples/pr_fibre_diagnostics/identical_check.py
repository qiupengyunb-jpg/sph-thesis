"""Are the no-reg and surface-only runs bit-identical?"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"E:\哈哈\_pr_diag")
from run_metrics import load

BASE = Path(r"E:\sphmethod\SPH_results_center\recon_tapered_20260914"
            r"\r24_broadband_T80\output_recon_r24_T80")
SRF = Path(r"E:\sphmethod\SPH_results_center\recon_tapered_20260914"
           r"\r24_surfreg_T50\output_r24_surfreg_T50")


def pick(d, iter_tag):
    hits = list(d.glob("LiquidFilmHalf_*%s.vtp" % iter_tag))
    return hits[0] if hits else None


base_files = {p.name.replace("LiquidFilmHalf_", ""): p
              for p in BASE.glob("LiquidFilmHalf_*.vtp")}

print("%-22s %10s %14s %14s %14s"
      % ("frame", "N", "max|dpos|", "max|dvel|", "max|drho|"))
for srf in sorted(SRF.glob("LiquidFilmHalf_*.vtp")):
    tag = srf.name.replace("LiquidFilmHalf_", "")
    b = base_files.get(tag)
    if b is None:
        print("%-22s  (no matching baseline frame)" % tag)
        continue
    A, B = load(b), load(srf)
    if len(A["Position"]) != len(B["Position"]):
        print("%-22s  N differs: %d vs %d"
              % (tag, len(A["Position"]), len(B["Position"])))
        continue
    dpos = np.abs(A["Position"] - B["Position"]).max()
    dvel = np.abs(A["Velocity"] - B["Velocity"]).max()
    drho = np.abs(A["Density"] - B["Density"]).max()
    print("%-22s %10d %14.3e %14.3e %14.3e" % (tag, len(A["Position"]),
                                                dpos, dvel, drho))

print("\n--- SurfaceRegularizationShift magnitude in the surface-only run ---")
for srf in sorted(SRF.glob("LiquidFilmHalf_*.vtp"))[::5]:
    B = load(srf)
    if "SurfaceRegularizationShift" not in B:
        print("  field missing")
        break
    s = np.linalg.norm(B["SurfaceRegularizationShift"][:, :2], axis=1)
    ind = B["Indicator"]
    print("  %-24s max=%.3e  mean=%.3e  max(surface only)=%.3e  n_surface=%d"
          % (srf.name, s.max(), s.mean(), s[ind == 1].max() if (ind == 1).any()
             else float("nan"), int((ind == 1).sum())))
