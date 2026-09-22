"""N2: manufactured-profile validation of the axisymmetric square-gradient operator.

Independent of the production integrator: builds deterministic point sets whose
local areal number density equals a prescribed rho_target(r), then checks whether
the discrete operator reproduces

    L_axi    = rho_zz + rho_rr + rho_r / r      (axisymmetric, J = r/R0)
    L_planar = rho_zz + rho_rr                  (J = 1)

Kernel (identical to cg_self_assembly_pri.cpp, Wendland C2, 2D normalised):
    W(r)  = (7/(pi h^2)) (1-q)^4 (4q+1),  q = r/h < 1
    W'(r) = -(140/(pi h^3)) q (1-q)^3

Point-set construction (deterministic, no RNG, no dynamics):
    triangular areal density   rho_A(r) = 2 / (sqrt(3) d(r)^2)
    => nearest-neighbour spacing d(r) = sqrt( 2 / (sqrt(3) rho_A(r)) )
    row spacing  dr_i = (sqrt(3)/2) d(r_i)
    rows at      r_{i+1} = r_i + dr_i , starting at r_0 = R0 + h_in
    points per row  n_i = max(3, round(Lz / d(r_i))),  dz_i = Lz / n_i
    z positions  z_k = ((i%2)*0.5 + k) dz_i  (mod Lz)   -> odd rows offset half a spacing

Usage:  python n2_manufactured_operator_check.py --outdir <dir> [--quick]
"""

import argparse
import math
import os

import numpy as np


# ---------------------------------------------------------------- kernel ----
def W(r, h):
    q = r / h
    out = np.zeros_like(q)
    m = q < 1.0
    t = 1.0 - q[m]
    out[m] = (7.0 / (math.pi * h * h)) * t**4 * (4.0 * q[m] + 1.0)
    return out


def dW(r, h):
    q = r / h
    out = np.zeros_like(q)
    m = q < 1.0
    t = 1.0 - q[m]
    out[m] = -(140.0 / (math.pi * h**3)) * q[m] * t**3
    return out


def ddW(r, h):
    q = r / h
    out = np.zeros_like(q)
    m = q < 1.0
    t = 1.0 - q[m]
    out[m] = -(140.0 / (math.pi * h**4)) * t**2 * (1.0 - 4.0 * q[m])
    return out


# -------------------------------------------------------------- profiles ----
class Profile:
    def __init__(self, name, rho0, a, b, kind):
        self.name, self.rho0, self.a, self.b, self.kind = name, rho0, a, b, kind

    def rho(self, r, Rc):
        s = r - Rc
        if self.kind == "uniform":
            return np.full_like(r, self.rho0)
        return self.rho0 + self.a * s + self.b * s * s

    def rho_r(self, r, Rc):
        s = r - Rc
        if self.kind == "uniform":
            return np.zeros_like(r)
        return self.a + 2.0 * self.b * s

    def rho_rr(self, r, Rc):
        if self.kind == "uniform":
            return np.zeros_like(r)
        return np.full_like(r, 2.0 * self.b)


PROFILES = {
    "A_linear": Profile("A_linear", 0.90, 0.0225, 0.0, "linear"),
    "B_quadratic": Profile("B_quadratic", 0.90, 0.0225, 0.0015, "quadratic"),
    "C_pure_quadratic": Profile("C_pure_quadratic", 0.90, 0.0, 0.0015, "quadratic"),
    "U_uniform": Profile("U_uniform", 0.90, 0.0, 0.0, "uniform"),
}


# ----------------------------------------------------------- point set ------
def build_pointset(rho, R0, h_in, H, Lz):
    """Deterministic rows realising the areal density rho(r)."""
    r_rows, z_rows, rowid = [], [], []
    r = R0 + h_in
    r_top = R0 + h_in + H
    row = 0
    guard = 0
    while r <= r_top + 1e-12:
        d = math.sqrt(2.0 / (math.sqrt(3.0) * float(rho(np.array([r]))[0])))
        n = max(3, int(round(Lz / d)))
        dz = Lz / n
        shift = 0.5 * dz if (row % 2) else 0.0
        for k in range(n):
            z = (shift + k * dz) % Lz
            z_rows.append(z)
            r_rows.append(r)
            rowid.append(row)
        row += 1
        guard += 1
        if guard > 100000:
            raise RuntimeError("row construction did not terminate")
        r += 0.5 * math.sqrt(3.0) * d
    return np.array(z_rows), np.array(r_rows), np.array(rowid)


# --------------------------------------------------------- operator ---------
def analyse(z, r, rowid, rho_fn, R0, h, Lz, lam, rho_ref, force_diag=False):
    """Returns a dict of per-particle quantities using the model's kernel."""
    n = len(z)
    idx = np.arange(n)
    rho_num = np.zeros(n)
    gz = np.zeros(n)
    gr = np.zeros(n)
    lap = np.zeros(n)
    nbr = np.zeros(n, dtype=int)
    edge = np.full(n, np.inf)
    # local volume per particle = 1 / areal density target
    Vj = 1.0 / rho_fn(r)
    # Pass 1: kernel sums of rho and grad rho (must be complete before any
    # Laplacian estimator that references rho_j).
    for i in range(n):
        dz = z - z[i]
        dz -= Lz * np.round(dz / Lz)
        dr = r - r[i]
        d = np.sqrt(dz * dz + dr * dr)
        m = (d > 0.0) & (d < h)
        if not np.any(m):
            continue
        dd = d[m]
        ez = np.where(dd > 0, dz[m] / dd, 0.0)
        er = np.where(dd > 0, dr[m] / dd, 0.0)
        rho_num[i] = np.sum(W(dd, h))
        gz[i] = np.sum(dW(dd, h) * ez)
        gr[i] = np.sum(dW(dd, h) * er)
        nbr[i] = int(np.count_nonzero(m))
        edge[i] = float(np.min(np.abs(dd - h)))
    # Pass 2: Laplacian estimators, now that every rho_num is available.
    #   planar (no measure weight)  -> rho_zz + rho_rr
    #   axi    (measure weight r_j/r_i) -> rho_zz + rho_rr + rho_r/r
    lap = np.zeros(n)
    lap_axi = np.zeros(n)
    for i in range(n):
        dz = z - z[i]
        dz -= Lz * np.round(dz / Lz)
        dr = r - r[i]
        d = np.sqrt(dz * dz + dr * dr)
        m = (d > 0.0) & (d < h)
        if not np.any(m):
            continue
        dd = d[m]
        base = Vj[m] * (rho_num[i] - rho_num[m]) * dW(dd, h) / dd
        lap[i] = 2.0 * np.sum(base)
        lap_axi[i] = 2.0 * np.sum(base * (r[m] / r[i]))
    out = dict(rho_num=rho_num, gz=gz, gr=gr, lap=lap, lap_axi=lap_axi,
               nbr=nbr, edge=edge,
               n=n, idx=idx)
    if force_diag:
        out["force"] = force_pair(z, r, rho_num, gz, gr, h, Lz, lam, rho_ref, R0)
    return out


def force_pair(z, r, rho_num, gz, gr, h, Lz, lam, rho_ref, R0):
    """N1 production force (both J cases) on the same coordinates.
    Returns (Fr_axi, Fr_planar) using the *numeric* gradients."""
    n = len(z)
    coeff = -lam / rho_ref
    G = np.stack([gz, gr], axis=1)
    F_axi = np.zeros((n, 2))
    F_pl = np.zeros((n, 2))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dz = z[i] - z[j]
            dz -= Lz * math.floor(dz / Lz + 0.5)
            dr = r[i] - r[j]
            d = math.hypot(dz, dr)
            if d <= 0 or d >= h:
                continue
            e = np.array([dz, dr]) / d
            w1 = float(dW(np.array([d]), h)[0])
            w2 = float(ddW(np.array([d]), h)[0])
            for mode in (0, 1):  # 0 = AXI (J=r/R0), 1 = PLANAR (J=1)
                Ji = r[i] / R0 if mode == 0 else 1.0
                Jj = r[j] / R0 if mode == 0 else 1.0
                dG = Ji * G[i] - Jj * G[j]
                dGe = float(np.dot(dG, e))
                t = w2 * dGe * e + (w1 / d) * (dG - dGe * e)
                (F_axi if mode == 0 else F_pl)[i] += coeff * t
        F_axi[i] += coeff * (0.5 / R0) * float(np.dot(G[i], G[i])) * np.array([0.0, 1.0])
    return F_axi[:, 1], F_pl[:, 1]


def metrics(num, target):
    num = np.asarray(num, float)
    target = np.asarray(target, float)
    err = num - target
    rmse = float(np.sqrt(np.mean(err**2)))
    rms_t = float(np.sqrt(np.mean(target**2)))
    nrmse = rmse / rms_t if rms_t > 0 else float("nan")
    bias = float(np.mean(err))
    if np.std(num) > 0 and np.std(target) > 0:
        corr = float(np.corrcoef(num, target)[0, 1])
    else:
        corr = float("nan")
    return dict(rmse=rmse, nrmse=nrmse, bias=bias, corr=corr,
                mean_num=float(np.mean(num)), mean_target=float(np.mean(target)),
                std_num=float(np.std(num)), n=len(num))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--Lz", type=float, default=40.0)
    ap.add_argument("--H", type=float, default=16.0)
    ap.add_argument("--h_in", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=12.0)
    ap.add_argument("--rho_ref", type=float, default=0.88)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    resolutions = [("coarse", 2.0), ("medium", 2.8), ("fine", 3.6)]
    if args.quick:
        resolutions = [("medium", 2.8)]
    r0_list = [5.0, 10.0, 20.0]

    rows = []
    for pname, prof in PROFILES.items():
        for R0 in r0_list:
            h_ref = 2.8
            Rc = R0 + args.h_in + 0.5 * args.H
            z, r, rowid = build_pointset(
                lambda rr, p=prof, c=Rc: p.rho(rr, c), R0, args.h_in, args.H, args.Lz)
            for resname, h in resolutions:
                res = analyse(z, r, rowid, lambda rr, p=prof: p.rho(rr, Rc),
                              R0, h, args.Lz, args.lam, args.rho_ref)
                r_in_i = R0 + args.h_in
                r_out_i = R0 + args.h_in + args.H
                interior = (r > r_in_i + 2 * h) & (r < r_out_i - 2 * h)
                if not np.any(interior):
                    continue
                sel = interior
                lp = res["lap"][sel]
                geom = res["gr"][sel] / r[sel]
                op_axi = lp + geom
                op_pl = lp
                t_pl = prof.rho_rr(r[sel], Rc) + 0.0 * r[sel]
                t_ax = t_pl + prof.rho_r(r[sel], Rc) / r[sel]
                m_lp = metrics(lp, t_pl)
                m_ax = metrics(op_axi, t_ax)
                m_gm = metrics(geom, prof.rho_r(r[sel], Rc) / r[sel])
                if resname == "medium" and R0 == 5.0:
                    print("   [debug] %-18s mean(lap)=%+.4e mean(geom)=%+.4e "
                          "mean(lap_axi)=%+.4e target_axi=%+.4e target_planar=%+.4e"
                          % (pname, np.mean(lp), np.mean(geom),
                             np.mean(res["lap_axi"][sel]), np.mean(t_ax),
                             np.mean(t_pl)))
                rows.append(dict(profile=pname, R0=R0, resolution=resname, h=h,
                                 N=res["n"], N_interior=int(np.count_nonzero(sel)),
                                 excluded_frac=1.0 - np.count_nonzero(sel) / res["n"],
                                 mean_nbr=float(np.mean(res["nbr"][sel])),
                                 min_edge=float(np.min(res["edge"])),
                                 rmse_lap=m_lp["rmse"], nrmse_lap=m_lp["nrmse"],
                                 rmse_axi=m_ax["rmse"], nrmse_axi=m_ax["nrmse"],
                                 rmse_geom=m_gm["rmse"], nrmse_geom=m_gm["nrmse"],
                                 bias_axi=m_ax["bias"], corr_axi=m_ax["corr"],
                                 corr_lap=m_lp["corr"]))

    keys = sorted(rows[0].keys())
    with open(os.path.join(args.outdir, "n2_summary.csv"), "w") as fh:
        fh.write(",".join(keys) + "\n")
        for rw in rows:
            fh.write(",".join(str(rw[k]) for k in keys) + "\n")
    print("wrote n2_summary.csv with %d rows" % len(rows))
    for rw in rows:
        print("  %-18s R0=%4.1f %-7s N=%5d interior=%5d nbr=%5.2f  "
              "nrmse_axi=%.3e corr_axi=%.4f" %
              (rw["profile"], rw["R0"], rw["resolution"], rw["N"],
               rw["N_interior"], rw["mean_nbr"], rw["nrmse_axi"], rw["corr_axi"]))


if __name__ == "__main__":
    main()
