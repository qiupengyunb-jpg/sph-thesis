"""Offline geometry / kernel-support / density diagnostics.

Reads only existing VTP output of the r24 no-regularisation broadband case.
Nothing is written back to the solver and no SPH run is started.

Kernel and axisymmetric conventions replicated from the solver sources:
  KernelWendlandC2 with h = 1.3 * dx          (SPHAdaptation default ratio)
  W2D(q) = 7/(4*pi*h^2) * (1-q/2)^4 * (1+2q),  q = r/h, cutoff q < 2
  sigma0 = lattice number density (includes the self term W0)
  axisymmetric particle mass  m_j = AxisymmetricRingMass_j / max(r_j, 0.25*dx)
                                (UpdateAxisymmetricWeights in the case source)
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

CASE = Path(r"E:\sphmethod\SPH_results_center\recon_tapered_20260914\r24_broadband_T80")
LIQUID_DIR = CASE / "output_recon_r24_T80"
TIME_SCALE = 0.67082039325
DX = 1.0 / 24.0
H_SPACING_RATIO = 1.3
H = H_SPACING_RATIO * DX
CUTOFF = 2.0 * H
W0_FACTOR = 7.0 / (4.0 * np.pi * H * H)
RHO0 = 1.0


def w2d(r):
    q = r / H
    out = np.zeros_like(q)
    m = q < 2.0
    qq = q[m]
    out[m] = W0_FACTOR * (1.0 - 0.5 * qq) ** 4 * (1.0 + 2.0 * qq)
    return out


def lattice_sums():
    """Self-inclusive lattice sums of W and of W*mass over a perfect dx lattice."""
    n = int(np.ceil(CUTOFF / DX)) + 1
    xs = np.arange(-n, n + 1) * DX
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    d = np.sqrt(gx ** 2 + gy ** 2)
    inside = d < CUTOFF
    sigma0_num = float(w2d(d[inside]).sum())
    sigma0_mass = float((w2d(d[inside]) * (RHO0 * DX * DX)).sum())
    return sigma0_num, sigma0_mass


SIGMA0_NUM, SIGMA0_MASS = lattice_sums()


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


def frames():
    """All liquid frames including the initial one, as (path, T)."""
    out = []
    for p in sorted(LIQUID_DIR.glob("LiquidFilmHalf_*.vtp")):
        root = ET.parse(p).getroot()
        t = float(root.find('.//FieldData/DataArray[@Name="TimeValue"]').text) / TIME_SCALE
        out.append((p, t))
    out.sort(key=lambda item: item[1])
    return out


def particle_fields(path):
    a = load(path)
    pos = a["Position"][:, :2]
    ring = a["AxisymmetricRingMass"]
    r = np.maximum(pos[:, 1], 0.25 * DX)
    return {
        "x": pos[:, 0],
        "r": pos[:, 1],
        "pos": pos,
        "id": a["OriginalID"].astype(np.int64),
        "rho_c": a["Density"],
        "p": a["Pressure"],
        "indicator": a["Indicator"],
        "mass": ring / r,
        "ring": ring,
        "vol": None,
    }


def neighbor_sums(pos, mass):
    """Self-inclusive kernel sums: number form and mass-weighted form."""
    tree = cKDTree(pos)
    pairs = tree.query_ball_point(pos, CUTOFF, return_sorted=False)
    n = len(pos)
    s_num = np.full(n, W0_FACTOR)          # self term W(0)
    s_mass = W0_FACTOR * mass              # self term m_i * W(0)
    nbr_count = np.zeros(n, dtype=np.int64)
    d_min = np.full(n, np.inf)
    for i, js in enumerate(pairs):
        js = np.asarray(js, dtype=np.int64)
        js = js[js != i]
        if js.size == 0:
            continue
        d = np.linalg.norm(pos[js] - pos[i], axis=1)
        w = w2d(d)
        s_num[i] += w.sum()
        s_mass[i] += (w * mass[js]).sum()
        nbr_count[i] = js.size
        d_min[i] = d.min()
    return s_num, s_mass, nbr_count, d_min
