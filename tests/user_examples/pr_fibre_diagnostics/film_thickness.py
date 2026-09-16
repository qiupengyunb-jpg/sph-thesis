"""Local film thickness per axial bin for the r24 variable-radius case.

Answers: is the lesion a THIN film (physical necking) or a SPACED-OUT film
(numerical disconnection with the overall thickness still intact)?
Needed to design a safe exit criterion for any support-restoring correction.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"E:\哈哈\_pr_diag")
from run_metrics import load, frames, fibre_of, BIN, DX

CASE = Path(r"E:\sphmethod\SPH_results_center\recon_tapered_20260914"
            r"\r24_broadband_T80")
LIQ = CASE / "output_recon_r24_T80"
WANT = (0.0, 20.0, 30.0, 40.0, 50.0, 55.0, 60.0, 62.0, 70.0, 80.0)


def main():
    fs = frames(LIQ, 1e9)
    print("r=24 variable-radius case: film thickness per axial bin")
    print("(H = max liquid r - max fibre r in the bin; dx = %.5f, 4dx = %.4f)"
          % (DX, 4 * DX))
    print()
    hdr = "%-6s" % "T"
    for x0 in (-6.0, -4.0, -2.0, 0.0, 2.0, 4.0):
        hdr += "  x=[%+.0f,%+.0f]" % (x0, x0 + BIN)
    print(hdr + "   H_min_over_all_bins")
    for want in WANT:
        p, T = min(fs, key=lambda it: abs(it[1] - want))
        a = load(p)
        x, r = a["Position"][:, 0], a["Position"][:, 1]
        fb = fibre_of(p)
        fx, fr = load(fb)["Position"][:, 0], load(fb)["Position"][:, 1]
        row = "%-6.0f" % T
        vals = []
        for x0 in (-6.0, -4.0, -2.0, 0.0, 2.0, 4.0):
            sel = (x >= x0) & (x < x0 + BIN)
            fsel = (fx >= x0) & (fx < x0 + BIN)
            if sel.sum() < 8 or fsel.sum() < 8:
                row += "  %9s" % "-"
                continue
            H = r[sel].max() - fr[fsel].max()
            vals.append(H)
            row += "  %9.4f" % H
        # worst resolved bin over the whole studied section |x|<26
        allH = []
        for x0 in np.arange(-26.0, 26.0, BIN):
            sel = (x >= x0) & (x < x0 + BIN)
            fsel = (fx >= x0) & (fx < x0 + BIN)
            if sel.sum() < 8 or fsel.sum() < 8:
                continue
            allH.append(r[sel].max() - fr[fsel].max())
        print(row + "   %.4f a" % np.min(allH))


if __name__ == "__main__":
    main()
