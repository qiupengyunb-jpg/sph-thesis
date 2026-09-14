"""Shared readers for the r24 no-regularisation broadband case."""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

CASE = Path(r"E:\sphmethod\SPH_results_center\recon_tapered_20260914\r24_broadband_T80")
LIQUID_DIR = CASE / "output_recon_r24_T80"
TIME_SCALE = 0.67082039325          # solver_end_time / end_T  (mu*a/gamma)
DX = 1.0 / 24.0                     # particle spacing at resolution 24


def load(path):
    """Return {DataArray name: ndarray} for one VTP file."""
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


def liquid_frames():
    """Sorted list of (path, T) for the liquid body."""
    out = []
    for p in sorted(LIQUID_DIR.glob("LiquidFilmHalf_*.vtp")):
        if "ite_" in p.stem:
            continue
        out.append(p)
    frames = []
    for p in out:
        root = ET.parse(p).getroot()
        t = float(root.find('.//FieldData/DataArray[@Name="TimeValue"]').text)
        frames.append((p, t / TIME_SCALE))
    frames.sort(key=lambda item: item[1])
    return frames
