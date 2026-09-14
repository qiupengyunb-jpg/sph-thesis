"""Readers for the equal-radius periodic dispersion cases (first-version solver)."""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(r"E:\sphmethod\SPH_results_center\dispersion_uniform_20260912\v1")
TIME_SCALE = 0.70710678118      # solver_end_time / end_T for h/a=1, Oh=0.5


def case_dir(lam):
    """Case directory for lambda/a = lam (e.g. 18 -> v1_lambda1800)."""
    name = "v1_lambda%04d" % int(round(lam * 100))
    d = ROOT / name
    outs = [p for p in d.glob("output_dominant*") if p.is_dir()]
    if not outs:
        raise FileNotFoundError("no output dir for lambda=%s" % lam)
    return outs[0]


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


def frames(lam):
    """Sorted (path, T) list for the liquid body of one wavelength case."""
    d = case_dir(lam)
    out = []
    for p in d.glob("LiquidFilmHalf_*.vtp"):
        t = float(ET.parse(p).getroot()
                  .find('.//FieldData/DataArray[@Name="TimeValue"]').text)
        out.append((p, t / TIME_SCALE))
    out.sort(key=lambda it: it[1])
    return out


def fields(lam):
    p, _ = frames(lam)[1]
    return sorted(load(p).keys())
