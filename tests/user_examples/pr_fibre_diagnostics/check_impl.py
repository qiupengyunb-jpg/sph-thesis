"""Self-check: does the offline kernel sum give rho=1 on the pristine T=0 film?"""
import sys

import numpy as np

sys.path.insert(0, r"E:\哈哈\_pr_diag")
from diag_core import (DX, SIGMA0_NUM, SIGMA0_MASS, CUTOFF, H, W0_FACTOR,
                       frames, particle_fields, neighbor_sums)

print("dx=%.6f  h=%.6f  cutoff=%.6f  W0=%.4f" % (DX, H, CUTOFF, W0_FACTOR))
print("sigma0 (number form) = %.6f" % SIGMA0_NUM)
print("sigma0 (mass form)   = %.6e" % SIGMA0_MASS)

fs = frames()
path, T = fs[0]
print("frame T=%.2f  %s" % (T, path.name))
f = particle_fields(path)
s_num, s_mass, nbr, d_min = neighbor_sums(f["pos"], f["mass"])

rho_num = s_num / SIGMA0_NUM * 1.0
rho_mass = s_mass / SIGMA0_MASS

print("\n--- offline rho_summation on the initial frame ---")
for name, val in (("number form", rho_num), ("mass  form", rho_mass)):
    print("  %-12s min=%.5f  p1=%.5f  mean=%.5f  p99=%.5f  max=%.5f"
          % (name, val.min(), np.percentile(val, 1), val.mean(),
             np.percentile(val, 99), val.max()))
print("  rho_continuity  min=%.5f max=%.5f" % (f["rho_c"].min(), f["rho_c"].max()))

# interior particles only (indicator==0 => not free surface)
core = f["indicator"] == 0
print("\n--- interior particles only (%d of %d) ---" % (core.sum(), len(core)))
for name, val in (("number form", rho_num), ("mass  form", rho_mass)):
    v = val[core]
    print("  %-12s min=%.5f  mean=%.5f  max=%.5f  std=%.2e"
          % (name, v.min(), v.mean(), v.max(), v.std()))

print("\nneighbor count: min=%d mean=%.1f max=%d"
      % (nbr.min(), nbr.mean(), nbr.max()))
print("nearest-neighbour distance / dx: min=%.3f mean=%.3f max=%.3f"
      % (d_min.min() / DX, d_min.mean() / DX, d_min.max() / DX))
