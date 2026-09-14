"""Quick inspection of available diagnostics in the r24 broadband case."""
import sys

import numpy as np

sys.path.insert(0, r"E:\哈哈\_pr_diag")
from v2io import load, liquid_frames, DX

frames = liquid_frames()
print("frames:", len(frames), "T from %.2f to %.2f" % (frames[0][1], frames[-1][1]))
print("T list:", ", ".join("%.0f" % t for _, t in frames))

for idx in (0, 7, 14, 21, 25, 40):
    path, T = frames[idx]
    a = load(path)
    print("--- frame#%02d  T=%6.2f  N=%d" % (idx, T, len(a["Position"])))
    for key in ("JfmShepardSum", "SurfaceSupportNeighbors", "Density",
                "Pressure", "AxisymmetricRingMass", "Indicator"):
        if key in a:
            x = a[key]
            print("    %-22s min=%12.6f max=%12.6f mean=%12.6f"
                  % (key, x.min(), x.max(), x.mean()))
