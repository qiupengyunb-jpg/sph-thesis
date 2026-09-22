#!/usr/bin/env python3
"""M1.3R stage 2 -- Scheme E (pairwise-difference interface energy), full gates.

Discrete energy (A-1 compatible normalisation):

    F_E = (lambda/2) V0^2 Sum_{i<j} G_ij Kt_ij (rho_i - rho_j)^2
          Kt_ij = c W(r_ij),   c = 144 / (5 h^2)
          G_ij  = J_ij = (r_i + r_j) / (2 R0)      [axisymmetric]
          G_ij  = 1                                [planar / flat limit]

The planar branch equals the production 2D A-1 energy (lambda/2) Int |grad rho|^2 dA,
and the axisymmetric branch equals (lambda/2) Int (r/R0) |grad rho|^2 dr dz, i.e.
lambda = 12 keeps its A-1 meaning.  The full physical measure 2 pi r is obtained
by multiplying by 2 pi R0 (lambda_physical = 2 pi R0 * lambda) -- reported, not
silently dropped.

Covered by this file:
    T1  uniform regular lattice, systematic radial force
    T2  uniform lattice + deterministic disorder (3 fixed patterns)
    T3  energy-force FD at 4 deltas (analytic force vs central difference)
    T5  small smooth perturbation (quadratic response + FD on perturbed state)
    T6  geometry: manufactured smooth field, discrete pair sum vs continuum
    T9  operation count
    F_E/F_A comparison on identical configurations

Diagnostics only; nothing here touches the production solver.
"""

import argparse
import json
import math
import os

import numpy as np

# ------------------------------------------------------------------ frozen ----
H_RHO = 2.8
RHO_REF = 0.88
V0 = 1.0 / RHO_REF
LAM = 12.0
C_K = 144.0 / (5.0 * H_RHO * H_RHO)
J_HAT = np.array([0.0, 1.0])
TINY = 1e-12


def W(r, h=H_RHO):
    q = np.asarray(r, float) / h
    out = np.zeros_like(q)
    m = q < 1.0
    t = 1.0 - q[m]
    out[m] = (7.0 / (math.pi * h * h)) * t ** 4 * (4.0 * q[m] + 1.0)
    return out


def dW(r, h=H_RHO):
    q = np.asarray(r, float) / h
    out = np.zeros_like(q)
    m = q < 1.0
    t = 1.0 - q[m]
    out[m] = -(140.0 / (math.pi * h ** 3)) * q[m] * t ** 3
    return out


def ddW(r, h=H_RHO):
    q = np.asarray(r, float) / h
    out = np.zeros_like(q)
    m = q < 1.0
    t = 1.0 - q[m]
    out[m] = -(140.0 / (math.pi * h ** 4)) * t ** 2 * (1.0 - 4.0 * q[m])
    return out


# -------------------------------------------------------------- geometry ----
def flat_lattice(n0, R0, Ly, Lz, r_lo=0.5):
    d = math.sqrt(2.0 / (math.sqrt(3.0) * n0))
    zs, rs = [], []
    row, rr = 0, R0 + r_lo
    while rr <= Ly - 1e-9:
        ncol = max(3, int(round(Lz / d)))
        dz = Lz / ncol
        shift = 0.5 * dz if row % 2 else 0.0
        for k in range(ncol):
            zs.append((shift + k * dz) % Lz)
            rs.append(rr)
        row += 1
        rr += 0.5 * math.sqrt(3.0) * d
    return np.array(zs), np.array(rs), d


def disturb(r, d, eps, freq=(1.7, 0.9, 0.5)):
    idx = np.arange(len(r), dtype=float)
    return r + eps * d * (np.sin(freq[0] * idx) + freq[2] * np.cos(freq[1] * idx))


def pair_arrays(z, r, Lz, h=H_RHO, Lr=None):
    """Unordered pair list (i<j).  Lr != None wraps the radial direction as well
    (an artificial torus used to remove free surfaces)."""
    n = len(z)
    ii, jj, rr = [], [], []
    for i in range(n):
        dz = z - z[i]
        dz -= Lz * np.round(dz / Lz)
        dr = r - r[i]
        if Lr is not None:
            dr -= Lr * np.round(dr / Lr)
        dist = np.sqrt(dz * dz + dr * dr)
        sel = np.where(dist < h)[0]
        sel = sel[sel > i]
        for j in sel:
            if dist[j] > TINY:
                ii.append(i)
                jj.append(int(j))
                rr.append(float(dist[j]))
    return np.array(ii, int), np.array(jj, int), np.array(rr)


def geometry(z, r, ii, jj, Lz, Lr=None):
    dz = z[ii] - z[jj]
    dz -= Lz * np.round(dz / Lz)
    dr = r[ii] - r[jj]
    if Lr is not None:
        dr -= Lr * np.round(dr / Lr)
    dist = np.sqrt(dz * dz + dr * dr)
    e = np.stack([dz / dist, dr / dist], axis=1)
    return dist, e


def margin_to_cutoff(z, r, Lz, h=H_RHO, Lr=None):
    """Smallest (h - r_max) over the particles: topology cannot change below it."""
    best = np.inf
    for i in range(len(z)):
        dz = z - z[i]
        dz -= Lz * np.round(dz / Lz)
        dr = r - r[i]
        if Lr is not None:
            dr -= Lr * np.round(dr / Lr)
        dist = np.sqrt(dz * dz + dr * dr)
        dist = dist[(dist > TINY) & (dist < h)]
        if len(dist):
            best = min(best, float(h - dist.max()))
    return best


# ---------------------------------------------------------------- kernels ----
def rho_grad(z, r, ii, jj, Lz, Lr=None):
    n = len(z)
    dist, e = geometry(z, r, ii, jj, Lz, Lr)
    dw = dW(dist)
    rho = np.zeros(n)
    np.add.at(rho, ii, W(dist))
    np.add.at(rho, jj, W(dist))
    grad = np.zeros((n, 2))
    np.add.at(grad, ii, dw[:, None] * e)
    np.add.at(grad, jj, -dw[:, None] * e)
    return rho, grad, dist, e


def _geo(r_i, r_j, R0, axi):
    return ((r_i + r_j) / (2.0 * R0)) if axi else 1.0


def energy_E(z, r, ii, jj, Lz, R0, axi=True, lam=LAM, rho=None, Lr=None):
    if rho is None:
        dist, _ = geometry(z, r, ii, jj, Lz, Lr)
        rho = np.zeros(len(z))
        np.add.at(rho, ii, W(dist))
        np.add.at(rho, jj, W(dist))
    dist, _ = geometry(z, r, ii, jj, Lz, Lr)
    geo = ((r[ii] + r[jj]) / (2.0 * R0)) if axi else np.ones(len(ii))
    acc = float(np.sum(geo * C_K * W(dist) * (rho[ii] - rho[jj]) ** 2))
    return 0.5 * lam * V0 * V0 * acc, rho


def energy_E_analytic_rho(z, r, ii, jj, Lz, R0, rho, axi=True, lam=LAM, Lr=None):
    dist, _ = geometry(z, r, ii, jj, Lz, Lr)
    geo = ((r[ii] + r[jj]) / (2.0 * R0)) if axi else np.ones(len(ii))
    acc = float(np.sum(geo * C_K * W(dist) * (rho[ii] - rho[jj]) ** 2))
    return 0.5 * lam * V0 * V0 * acc


def force_E_analytic(z, r, ii, jj, Lz, R0, axi=True, lam=LAM, rho_in=None,
                     grad_in=None, Lr=None):
    rho, grad, dist, e = rho_grad(z, r, ii, jj, Lz, Lr) if rho_in is None else (
        rho_in, grad_in, *geometry(z, r, ii, jj, Lz, Lr))
    n = len(z)
    pref = 0.5 * lam * V0 * V0 * C_K
    geo = ((r[ii] + r[jj]) / (2.0 * R0)) if axi else np.ones(len(ii))
    w = W(dist)
    dw = dW(dist)
    diff2 = (rho[ii] - rho[jj]) ** 2

    A = pref * geo * w                      # pair coefficient A_ij
    # conjugate field S_i = Sum_{j!=i} 2 A_ij (rho_i - rho_j)
    S = np.zeros(n)
    np.add.at(S, ii, 2.0 * A * (rho[ii] - rho[jj]))
    np.add.at(S, jj, -2.0 * A * (rho[ii] - rho[jj]))

    # self term: d rho_k / d x_k = +G_k  (verified against FD), hence -S_k G_k
    F = -S[:, None] * grad
    # implicit part: -Sum_i S_i dW(r_ik) e_ik
    np.add.at(F, ii, -(S[jj] * dw)[:, None] * e)
    np.add.at(F, jj, +(S[ii] * dw)[:, None] * e)
    # explicit part: dA_ij/dx_k (kernel + measure factor)
    gf = (pref * diff2)[:, None]
    if axi:
        term = gf * ((0.5 / R0) * w)[:, None] * J_HAT[None, :] + \
               (gf * (geo * dw)[:, None]) * e
        np.add.at(F, ii, -term)
        term_m = gf * ((0.5 / R0) * w)[:, None] * J_HAT[None, :] - \
                 (gf * (geo * dw)[:, None]) * e
        np.add.at(F, jj, -term_m)
    else:
        term = (gf * dw[:, None]) * e
        np.add.at(F, ii, -term)
        np.add.at(F, jj, +term)
    return F, rho, grad


def force_E_fd(z, r, ii, jj, Lz, R0, axi=True, lam=LAM, delta=1e-5, rho0=None,
               Lr=None):
    n = len(z)
    F = np.zeros((n, 2))
    for k in range(n):
        for c, arr in ((0, z), (1, r)):
            ap = arr.copy()
            am = arr.copy()
            ap[k] += delta
            am[k] -= delta
            if c == 0:
                ep, _ = energy_E(ap, r, ii, jj, Lz, R0, axi, lam, Lr=Lr)
                em, _ = energy_E(am, r, ii, jj, Lz, R0, axi, lam, Lr=Lr)
            else:
                ep, _ = energy_E(z, ap, ii, jj, Lz, R0, axi, lam, Lr=Lr)
                em, _ = energy_E(z, am, ii, jj, Lz, R0, axi, lam, Lr=Lr)
            F[k, c] = -(ep - em) / (2.0 * delta)
    return F


# ------------------------------------------------------- Scheme A (bench) ----
def energy_A(z, r, ii, jj, Lz, R0, axi=True, lam=LAM, j_one=False, Lr=None):
    rho, grad, _, _ = rho_grad(z, r, ii, jj, Lz, Lr)
    J = (r / R0) if (axi and not j_one) else np.ones_like(r)
    return 0.5 * lam * V0 * float(np.sum(J * np.sum(grad ** 2, axis=1))), rho, grad


def force_A_analytic(z, r, ii, jj, Lz, R0, axi=True, lam=LAM, j_one=False,
                     Lr=None):
    rho, grad, dist, e = rho_grad(z, r, ii, jj, Lz, Lr)
    jw = axi and not j_one
    r0 = R0 if jw else 1.0
    Jn = (r / r0) if jw else np.ones_like(r)
    Jp = Jn[None, :]
    coeff = -lam * V0
    ddw = ddW(dist)
    dw = dW(dist)
    dG = (Jn[ii][:, None] * grad[ii] - Jn[jj][:, None] * grad[jj])
    dGe = np.sum(dG * e, axis=1)
    fpair = coeff * (ddw[:, None] * dGe[:, None] * e +
                     (dw / dist)[:, None] * (dG - dGe[:, None] * e))
    F = np.zeros((len(z), 2))
    np.add.at(F, ii, fpair)
    np.add.at(F, jj, -fpair)
    if jw:
        F += coeff * (0.5 / r0) * np.sum(grad ** 2, axis=1)[:, None] * J_HAT[None, :]
    return F, rho, grad


def force_A_fd(z, r, ii, jj, Lz, R0, axi=True, lam=LAM, j_one=False, delta=1e-5):
    n = len(z)
    F = np.zeros((n, 2))
    for k in range(n):
        for c in (0, 1):
            zp, rp, zm, rm = z.copy(), r.copy(), z.copy(), r.copy()
            if c == 0:
                zp[k] += delta
                zm[k] -= delta
            else:
                rp[k] += delta
                rm[k] -= delta
            ep = energy_A(zp, rp, ii, jj, Lz, R0, axi, lam, j_one)[0]
            em = energy_A(zm, rm, ii, jj, Lz, R0, axi, lam, j_one)[0]
            F[k, c] = -(ep - em) / (2.0 * delta)
    return F


# ------------------------------------------------------------------ helpers ----
DELTAS = (1e-4, 3e-5, 1e-5, 3e-6)


def interior(z, r, R0, Ly, pad=2.0 * H_RHO):
    return (r > R0 + 0.5 + pad) & (r < Ly - pad)


def rel_err(a, b):
    den = float(np.max(np.abs(b)))
    return float(np.max(np.abs(a - b)) / den) if den > 0 else float(np.max(np.abs(a - b)))


def kernel_constant_check():
    """Independent quadrature check of  Int |u|^2 K(u) dA = 4."""
    rr = np.linspace(1e-6, H_RHO, 200000)
    val = 2.0 * math.pi * C_K * float(np.trapz(rr ** 3 * W(rr), rr))
    return val


def build(n0, R0, Ly, Lz, eps=0.0, freq=(1.7, 0.9, 0.5)):
    z, r, d = flat_lattice(n0, R0, Ly, Lz)
    if eps:
        r = disturb(r, d, eps, freq)
    ii, jj, rr = pair_arrays(z, r, Lz)
    return z, r, d, ii, jj, rr


# ------------------------------------------------------------------ gates ----
def gate_T1(R0, axi, Ly, Lz, n0):
    z, r, _d, ii, jj, _rr = build(n0, R0, Ly, Lz)
    sel = interior(z, r, R0, Ly)
    eE, _ = energy_E(z, r, ii, jj, Lz, R0, axi)
    FE, rhoE, _ = force_E_analytic(z, r, ii, jj, Lz, R0, axi)
    eA, _, _ = energy_A(z, r, ii, jj, Lz, R0, axi)
    FA, _, _ = force_A_analytic(z, r, ii, jj, Lz, R0, axi)
    return dict(R0=R0, axi=bool(axi), N=int(len(z)), n_int=int(sel.sum()),
                E_E=float(eE), E_A=float(eA),
                E_mean_Fr=float(np.mean(FE[sel, 1])),
                E_RMS_Fr=float(np.sqrt(np.mean(FE[sel, 1] ** 2))),
                E_mean_Fz=float(np.mean(FE[sel, 0])),
                E_max_Fz=float(np.max(np.abs(FE[sel, 0]))),
                A_mean_Fr=float(np.mean(FA[sel, 1])),
                A_RMS_Fr=float(np.sqrt(np.mean(FA[sel, 1] ** 2))),
                rho_std=float(np.std(rhoE[sel])))


def gate_T2(R0, Ly, Lz, n0, eps_list=(0.0, 0.01, 0.02, 0.05), patterns=None):
    rows = []
    for eps in eps_list:
        for pidx, freq in enumerate(patterns):
            z, r, _d, ii, jj, _rr = build(n0, R0, Ly, Lz, eps, freq)
            sel = interior(z, r, R0, Ly)
            F, _, _ = force_E_analytic(z, r, ii, jj, Lz, R0, True)
            Fr = F[sel, 1]
            rows.append(dict(eps=eps, pattern=pidx, n_int=int(sel.sum()),
                             mean_Fr=float(np.mean(Fr)),
                             rms_Fr=float(np.sqrt(np.mean(Fr ** 2))),
                             max_abs_Fz=float(np.max(np.abs(F[sel, 0])))))
    return rows


def gate_T3(z, r, ii, jj, Lz, R0, axi=True, deltas=DELTAS):
    Fa, _, _ = force_E_analytic(z, r, ii, jj, Lz, R0, axi)
    rows = []
    for dl in deltas:
        Ff = force_E_fd(z, r, ii, jj, Lz, R0, axi, delta=dl)
        rows.append(dict(delta=dl, rel_err=rel_err(Fa, Ff),
                         abs_err=float(np.max(np.abs(Fa - Ff))),
                         force_scale=float(np.max(np.abs(Ff)))))
    return rows


def gate_T5(R0, Ly, Lz, n0, axi=True):
    z, r0, d = flat_lattice(n0, R0, Ly, Lz)
    ii, jj, _ = pair_arrays(z, r0, Lz)
    e0, _ = energy_E(z, r0, ii, jj, Lz, R0, axi)
    rows = []
    for eps in (0.005, 0.01, 0.02, 0.04):
        r = r0 + eps * d * np.cos(2.0 * math.pi * z / Lz)
        e, _ = energy_E(z, r, ii, jj, Lz, R0, axi)
        rows.append(dict(eps=eps, dE=float(e - e0),
                         dE_over_eps2=float((e - e0) / eps ** 2)))
    eps = 0.02
    r = r0 + eps * d * np.cos(2.0 * math.pi * z / Lz)
    fd_rows = gate_T3(z, r, ii, jj, Lz, R0, axi)
    return rows, fd_rows


def manufactured_cont(amp, kz, R0, r_lo, r_hi, Lz, axi):
    """Exact continuum value of (lambda/2) Int w |grad rho|^2 for
    rho = 1 + amp cos(kz z), over one period in z (kz*Lz = 2 pi n)."""
    base = 0.5 * LAM * amp ** 2 * kz ** 2 * (Lz / 2.0)
    if axi:
        return base * (r_hi ** 2 - r_lo ** 2) / (2.0 * R0)
    return base * (r_hi - r_lo)


def gate_T6(R0, n0=0.88, amp=0.2, Lz=60.0, kz_mult=(1, 2, 3)):
    rows = []
    r_lo_geo, r_hi_geo = R0 + 0.5, R0 + 12.0
    z, r, d = flat_lattice(n0, R0, r_hi_geo, Lz, r_lo=0.5)
    ii, jj, _ = pair_arrays(z, r, Lz)
    dist, _e = geometry(z, r, ii, jj, Lz)
    for m in kz_mult:
        kz = 2.0 * math.pi * m / Lz
        rho = 1.0 + amp * np.cos(kz * z)
        acc_a = float(np.sum(((r[ii] + r[jj]) / (2.0 * R0)) * C_K * W(dist) *
                             (rho[ii] - rho[jj]) ** 2))
        acc_p = float(np.sum(C_K * W(dist) * (rho[ii] - rho[jj]) ** 2))
        e_disc_a = 0.5 * LAM * V0 * V0 * acc_a
        e_disc_p = 0.5 * LAM * V0 * V0 * acc_p
        e_cont_a = manufactured_cont(amp, kz, R0, r.min(), r.max(), Lz, True)
        e_cont_p = manufactured_cont(amp, kz, R0, r.min(), r.max(), Lz, False)
        rows.append(dict(R0=R0, m=m, kz=kz, kh=kz * H_RHO,
                         e_disc_axi=e_disc_a, e_cont_axi=e_cont_a,
                         ratio_axi=e_disc_a / e_cont_a,
                         e_disc_planar=e_disc_p, e_cont_planar=e_cont_p,
                         ratio_planar=e_disc_p / e_cont_p,
                         axi_over_planar_disc=e_disc_a / e_disc_p,
                         axi_over_planar_cont=e_cont_a / e_cont_p))
    return rows


def gate_ratio(R0, axi, Ly, Lz, n0, eps=0.05, freq=(2.3, 1.3, 0.25)):
    z, r, _d, ii, jj, _rr = build(n0, R0, Ly, Lz, eps, freq)
    eE, _ = energy_E(z, r, ii, jj, Lz, R0, axi)
    eA, _, _ = energy_A(z, r, ii, jj, Lz, R0, axi)
    return dict(R0=R0, axi=bool(axi), E_E=float(eE), E_A=float(eA),
                ratio=float(eE / eA) if eA else float("nan"))


def gate_N3(R0, Ly, Lz, n0, eps=0.0, freq=(1.7, 0.9, 0.5)):
    """Reproduce the N3 mechanism in the toy and check whether Scheme E removes it.

    On a state whose coarse-grained density is uniform, the exact continuum
    interface energy is zero, so ANY mean radial force is spurious.  Scheme A
    produces one through the explicit measure-derivative branch
        <Fr>  ~  -(lambda V0 / 2 R0) <|G_i|^2>
    because <|G_i|^2> != 0 off a perfectly centrosymmetric neighbour set.
    """
    z, r, _d, ii, jj, _rr = build(n0, R0, Ly, Lz, eps, freq)
    rho, grad, _dist, _e = rho_grad(z, r, ii, jj, Lz)
    g2 = np.sum(grad ** 2, axis=1)
    FE, _, _ = force_E_analytic(z, r, ii, jj, Lz, R0, True)
    FA, _, _ = force_A_analytic(z, r, ii, jj, Lz, R0, True)
    pred = -(LAM * V0 / (2.0 * R0)) * float(np.mean(g2))
    return dict(R0=R0, eps=eps, N=int(len(z)),
                mean_G2=float(np.mean(g2)),
                mean_Fr_A=float(np.mean(FA[:, 1])),
                rms_Fr_A=float(np.sqrt(np.mean(FA[:, 1] ** 2))),
                mean_Fr_E=float(np.mean(FE[:, 1])),
                rms_Fr_E=float(np.sqrt(np.mean(FE[:, 1] ** 2))),
                mean_Fr_A_explicit=pred,
                ratio_A_to_pred=float(np.mean(FA[:, 1]) / pred) if pred else float("nan"),
                ratio_E_over_A=float(np.mean(FE[:, 1]) / np.mean(FA[:, 1]))
                if np.mean(FA[:, 1]) else float("nan"))


# -------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "_m13r_out")
    os.makedirs(out_dir, exist_ok=True)
    report = {}

    print("=" * 108)
    print("Scheme E gates | F_E = (lam/2) V0^2 Sum_{i<j} G_ij Kt_ij (rho_i-rho_j)^2")
    print("  G_ij = J_ij = (r_i+r_j)/(2R0) [axi] or 1 [planar];  Kt = c W, c = 144/(5h^2)")
    print("  h = %.2f  V0 = %.5f  lam = %.1f  c = %.5f   (A-1 compatible normalisation)"
          % (H_RHO, V0, LAM, C_K))
    print("=" * 108)

    kc = kernel_constant_check()
    print("\n[K] independent quadrature of Int |u|^2 K(u) dA = %.12f   (must be 4)" % kc)
    report["kernel_constant"] = kc

    # ---- T1 -------------------------------------------------------------
    print("\n[T1] uniform regular lattice, T=0 static force, Ly=60, Lz=20")
    print("     R0  mode      N  n_int        E_E          E_A     E:meanFr     "
          "E:RMSFr     A:meanFr      A:RMSFr   max|Fz|(E)     std(rho)")
    t1 = []
    for R0 in (5.0, 10.0, 20.0):
        for axi in (True, False):
            a = gate_T1(R0, axi, 60.0, 20.0, 0.88)
            t1.append(a)
            print("   %5.1f %-7s %5d %6d %12.4e %12.4e %12.4e %11.3e %12.4e %11.3e %12.3e %12.3e"
                  % (R0, "axi" if axi else "planar", a["N"], a["n_int"], a["E_E"], a["E_A"],
                     a["E_mean_Fr"], a["E_RMS_Fr"], a["A_mean_Fr"], a["A_RMS_Fr"],
                     a["E_max_Fz"], a["rho_std"]))
    report["T1"] = t1
    print("   -> interior rho is EXACTLY uniform (std ~ 5e-16) for both schemes, so")
    print("      interior pair differences vanish; the residual E_E comes from the")
    print("      free surface.  On a perfectly centrosymmetric lattice G_i = 0 exactly,")
    print("      so T1 alone CANNOT discriminate A from E -- see [N3-mechanism] below.")

    # ---- N3 mechanism ----------------------------------------------------
    print("\n[N3-mechanism] uniform / disordered, ALL particles (no interior cut)")
    print("     The uniform-density continuum target has zero interface energy, so any")
    print("     mean radial force here is spurious.")
    print("     R0  eps    N     <|G|^2>     A:<Fr>      A:RMSFr   A:explicit   A/pred   "
          "E:<Fr>      E/A")
    n3 = []
    for eps in (0.0, 0.05):
        for R0 in (5.0, 10.0, 20.0):
            d = gate_N3(R0, 60.0, 20.0, 0.88, eps)
            n3.append(d)
            print("   %5.1f %5.3f %5d %11.4e %11.4e %11.4e %11.4e %8.4f %11.4e %8.4f"
                  % (d["R0"], d["eps"], d["N"], d["mean_G2"], d["mean_Fr_A"],
                     d["rms_Fr_A"], d["mean_Fr_A_explicit"], d["ratio_A_to_pred"],
                     d["mean_Fr_E"], d["ratio_E_over_A"]))
    report["N3_mechanism"] = n3

    # ---- T2 -------------------------------------------------------------
    patterns = [(1.7, 0.9, 0.5), (2.3, 1.3, 0.25), (0.9, 2.1, 0.75)]
    t2 = gate_T2(5.0, 60.0, 20.0, 0.88, patterns=patterns)
    print("\n[T2] uniform lattice + deterministic disorder (3 fixed patterns)")
    print("     eps/d  pattern  n_int      mean Fr        RMS Fr     max|Fz|")
    for row in t2:
        print("   %7.3f %7d %6d %13.4e %13.4e %11.3e"
              % (row["eps"], row["pattern"], row["n_int"], row["mean_Fr"],
                 row["rms_Fr"], row["max_abs_Fz"]))
    worst = max(abs(r["mean_Fr"]) for r in t2 if r["eps"] > 0.0)
    print("   -> worst |mean Fr| over disorder = %.4e   (Scheme A @R0=5: 2.4e-03)"
          % worst)
    report["T2"] = t2

    # ---- T3 -------------------------------------------------------------
    print("\n[T3] energy-force FD, analytic vs central difference")
    t3 = {}
    z0, r0, d0, ii0, jj0, _ = build(0.88, 5.0, 60.0, 20.0)
    mg = margin_to_cutoff(z0, r0, 20.0)
    print("     neighbour-list margin h - r_max = %.4f  (max delta 1e-4)" % mg)
    zd, rd, _dd, iid, jjd, _rrd = build(0.88, 5.0, 60.0, 20.0, 0.05, patterns[1])
    cases = {
        "uniform": (z0, r0, ii0, jj0),
        "disordered": (zd, rd, iid, jjd),
        "perturbed": (z0, r0 + 0.02 * d0 * np.cos(2.0 * np.pi * z0 / 20.0), ii0, jj0),
    }
    print("     case         delta      rel_err      abs_err   force_scale")
    for name, (zz, rr, ii, jj) in cases.items():
        rows = gate_T3(zz, rr, ii, jj, 20.0, 5.0, True)
        t3[name] = rows
        for row in rows:
            print("   %-12s %8.1e %12.3e %12.3e %12.3e"
                  % (name, row["delta"], row["rel_err"], row["abs_err"], row["force_scale"]))
    report["T3"] = t3

    # ---- Scheme A FD cross-check (control) -------------------------------
    print("\n[T3-control] Scheme A analytic vs FD on the same uniform lattice")
    tac = []
    for dl in DELTAS:
        Fa, _, _ = force_A_analytic(z0, r0, ii0, jj0, 20.0, 5.0, True)
        Ff = force_A_fd(z0, r0, ii0, jj0, 20.0, 5.0, True, delta=dl)
        tac.append(dict(delta=dl, rel_err=rel_err(Fa, Ff),
                        abs_err=float(np.max(np.abs(Fa - Ff)))))
        print("     delta=%8.1e  rel_err=%12.3e  abs_err=%12.3e"
              % (dl, tac[-1]["rel_err"], tac[-1]["abs_err"]))
    report["T3_A_control"] = tac

    # ---- T5 -------------------------------------------------------------
    print("\n[T5] small smooth radial perturbation r -> r + eps*dx*cos(2 pi z/Lz)")
    t5e, t5f = gate_T5(5.0, 60.0, 20.0, 0.88)
    print("        eps          dE        dE/eps^2")
    for row in t5e:
        print("   %8.4f %12.5e %15.5e" % (row["eps"], row["dE"], row["dE_over_eps2"]))
    print("     FD check on the eps=0.02 perturbed state:")
    for row in t5f:
        print("     delta=%8.1e  rel_err=%12.3e  abs_err=%12.3e"
              % (row["delta"], row["rel_err"], row["abs_err"]))
    report["T5"] = dict(energy=t5e, fd=t5f)

    # ---- T6 -------------------------------------------------------------
    print("\n[T6] geometry: manufactured smooth field rho = 1 + A cos(kz z)")
    print("     R0   m     kz       kh     disc(axi)    cont(axi)   ratio_axi  "
          "disc(pl)   cont(pl)  ratio_pl  axi/pl(disc) axi/pl(cont)")
    t6 = []
    for R0 in (5.0, 10.0, 20.0):
        rows = gate_T6(R0, n0=0.88, amp=0.2, Lz=60.0, kz_mult=(1, 2, 4))
        t6 += rows
        for row in rows:
            print("   %5.1f %3d %7.4f %7.4f %12.5e %12.5e %9.5f %11.4e %11.4e %9.5f %12.4f %13.4f"
                  % (row["R0"], row["m"], row["kz"], row["kh"], row["e_disc_axi"],
                     row["e_cont_axi"], row["ratio_axi"], row["e_disc_planar"],
                     row["e_cont_planar"], row["ratio_planar"],
                     row["axi_over_planar_disc"], row["axi_over_planar_cont"]))
    report["T6"] = t6

    # ---- F_E / F_A -------------------------------------------------------
    print("\n[F_E/F_A] identical (disordered, free-surface) configuration")
    print("     R0  mode        E_E          E_A        ratio")
    rat = []
    for R0 in (5.0, 10.0, 20.0):
        for axi in (True, False):
            row = gate_ratio(R0, axi, 60.0, 20.0, 0.88)
            rat.append(row)
            print("   %5.1f %-7s %12.5e %12.5e %10.4f"
                  % (R0, "axi" if axi else "planar", row["E_E"], row["E_A"], row["ratio"]))
    report["ratio_E_over_A"] = rat

    report["T9"] = dict(sweeps_scheme_E=3, sweeps_scheme_A=2,
                        note="rho/G, conjugate S, force vs rho/G, force; both O(N*nbr);"
                             " E needs no ddW kernel",
                        neighbour_ops_ratio=1.5,
                        extra_storage="one Real per particle (S_i)")
    print("\n[T9] Scheme E = 3 neighbour sweeps (rho/G, S_i, force) vs 2 for A-1 "
          "(ratio 1.5); no ddW; one extra Real per particle.")

    path = os.path.join(out_dir, "m13r_scheme_e_gates.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print("\n-> %s" % path)


if __name__ == "__main__":
    main()
