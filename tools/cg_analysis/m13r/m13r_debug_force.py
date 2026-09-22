#!/usr/bin/env python3
"""Scratch diagnostic: per-term decomposition of the Scheme E force vs FD.

Not part of the gate suite; kept so the M1.3R debugging trail is reproducible.
"""
import numpy as np

import m13r_scheme_e as M

np.set_printoptions(precision=6, linewidth=200)

R0, Ly, Lz, n0 = 5.0, 14.0, 8.0, 0.88
z, r, d, ii, jj, rr = M.build(n0, R0, Ly, Lz)
print("N=%d pairs=%d  d=%.4f  margin=%.4f" % (len(z), len(ii), d, M.margin_to_cutoff(z, r, Lz)))

rho, grad, dist, e = M.rho_grad(z, r, ii, jj, Lz)
print("std(rho)=%.3e  max|G|=%.3e  mean|G|^2=%.3e"
      % (np.std(rho), np.max(np.linalg.norm(grad, axis=1)),
         np.mean(np.sum(grad ** 2, axis=1))))

E, _ = M.energy_E(z, r, ii, jj, Lz, R0, True)
Fa, _, _ = M.force_E_analytic(z, r, ii, jj, Lz, R0, True)
Ff = M.force_E_fd(z, r, ii, jj, Lz, R0, True, delta=1e-5)
print("E=%.8f  max|Fa|=%.4e  max|Ff|=%.4e  max|dF|=%.4e"
      % (E, np.max(np.linalg.norm(Fa, axis=1)), np.max(np.linalg.norm(Ff, axis=1)),
         np.max(np.linalg.norm(Fa - Ff, axis=1))))
print("Fa[:6]=\n", Fa[:6])
print("Ff[:6]=\n", Ff[:6])
print("diff[:6]=\n", (Fa - Ff)[:6])

pref = 0.5 * M.LAM * M.V0 * M.V0 * M.C_K
geo = (r[ii] + r[jj]) / (2.0 * R0)
w = M.W(dist)
dw = M.dW(dist)
diff2 = (rho[ii] - rho[jj]) ** 2
A = pref * geo * w
S = np.zeros(len(z))
np.add.at(S, ii, 2.0 * A * (rho[ii] - rho[jj]))
np.add.at(S, jj, -2.0 * A * (rho[ii] - rho[jj]))

k = int(np.argmin(np.abs(r - (R0 + 6.0))))
print("\nk=%d z=%.4f r=%.4f rho=%.6f" % (k, z[k], r[k], rho[k]))
print("  G=%s   S=%.6e" % (grad[k], S[k]))

Fc = -S[:, None] * grad
np.add.at(Fc, ii, -(S[jj] * dw)[:, None] * e)
np.add.at(Fc, jj, +(S[ii] * dw)[:, None] * e)

gf = (pref * diff2)[:, None]
term = gf * ((0.5 / R0) * w)[:, None] * M.J_HAT[None, :] + (gf * (geo * dw)[:, None]) * e
term_m = gf * ((0.5 / R0) * w)[:, None] * M.J_HAT[None, :] - (gf * (geo * dw)[:, None]) * e
Fex = np.zeros_like(Fc)
np.add.at(Fex, ii, -term)
np.add.at(Fex, jj, -term_m)
print("  chain   =%s" % (Fc[k],))
print("  explicit=%s" % (Fex[k],))
print("  sum     =%s" % ((Fc + Fex)[k],))
print("  FD      =%s" % (Ff[k],))

for kk in (0, 1, 6, 7, 13):
    print("\nk=%d z=%.4f r=%.4f rho=%.6f  G=%s S=%.4e  nbrs=%d"
          % (kk, z[kk], r[kk], rho[kk], grad[kk], S[kk],
             int(np.sum(ii == kk) + np.sum(jj == kk))))
    print("  chain=%s explicit=%s sum=%s FD=%s"
          % (Fc[kk], Fex[kk], (Fc + Fex)[kk], Ff[kk]))
    # per-neighbour breakdown of the chain term for this particle
    on_i = np.where(ii == kk)[0]
    on_j = np.where(jj == kk)[0]
    print("  S*G = %s" % ((S[kk] * grad[kk]),))
    tot = np.zeros(2)
    for p in on_i:
        v = -(S[jj[p]] * dw[p]) * e[p]
        tot += v
    for p in on_j:
        v = +(S[ii[p]] * dw[p]) * e[p]
        tot += v
    print("  neighbour sum check = %s   (chain - S*G = %s)"
          % (tot, Fc[kk] - S[kk] * grad[kk]))

# ---- Scheme A control (production implementation, per-particle loop) ----
def force_A_production(z, r, Lz, R0, lam=M.LAM):
    ii, jj, rr = M.pair_arrays(z, r, Lz)
    rho, grad, dist, e = M.rho_grad(z, r, ii, jj, Lz)
    n = len(z)
    F = np.zeros((n, 2))
    ddw = M.ddW(dist)
    dw = M.dW(dist)
    for p in range(len(ii)):
        i, j = int(ii[p]), int(jj[p])
        rp = dist[p]
        ep = e[p]
        for a, b, sgn in ((i, j, +1.0), (j, i, +1.0)):
            dG = (r[a] / R0) * grad[a] - (r[b] / R0) * grad[b]
            dGe = float(dG @ ep)
            f = -lam * M.V0 * (ddw[p] * dGe * ep +
                               (dw[p] / rp) * (dG - dGe * ep))
            F[a] += f
    F -= (lam * M.V0 / (2.0 * R0)) * np.sum(grad ** 2, axis=1)[:, None] * M.J_HAT[None, :]
    return F, rho, grad


FA_prod, rhoA, gradA = force_A_production(z, r, Lz, R0)
FA_toy, _, _ = M.force_A_analytic(z, r, ii, jj, Lz, R0, True)
FA_fd = M.force_A_fd(z, r, ii, jj, Lz, R0, True, delta=1e-5)
sel = (r > R0 + 0.5 + 2 * M.H_RHO)
print("\nScheme A control (uniform lattice, R0=%.1f):" % R0)
print("  mean Fr  production-impl = %+.6e" % np.mean(FA_prod[sel, 1]))
print("  mean Fr  toy(antisym)    = %+.6e" % np.mean(FA_toy[sel, 1]))
print("  mean Fr  FD              = %+.6e" % np.mean(FA_fd[sel, 1]))
print("  predicted -(lam V0/2R0)<|G|^2> = %+.6e"
      % (-(M.LAM * M.V0 / (2.0 * R0)) * np.mean(np.sum(gradA ** 2, axis=1)[sel])))
print("  max|prod-toy| = %.3e" % np.max(np.abs(FA_prod - FA_toy)))
print("  max|prod-fd|  = %.3e" % np.max(np.abs(FA_prod - FA_fd)))

# ---- piece-by-piece numerical derivative at a boundary particle --------
print("\nPiece-by-piece numerical derivative (delta=1e-6) for k=0:")


def rho_from(z_, r_, ii_, jj_):
    dd, _ = M.geometry(z_, r_, ii_, jj_, Lz)
    rho_ = np.zeros(len(z_))
    np.add.at(rho_, ii_, M.W(dd))
    np.add.at(rho_, jj_, M.W(dd))
    return rho_


dl = 1e-6
k0 = 0
rp = r.copy()
rp[k0] += dl
rm = r.copy()
rm[k0] -= dl
rho_p, rho_m = rho_from(z, rp, ii, jj), rho_from(z, rm, ii, jj)
d_rho = (rho_p - rho_m) / (2 * dl)
print("  d rho_0 / d r_0   (FD) = %+.6e     -G_0,r = %+.6e" % (d_rho[k0], -grad[k0, 1]))
_near = np.argsort(np.abs(r - r[k0]) + np.abs(z - z[k0]))[1:4]
print("  d rho_j / d r_0   (FD) at the 3 nearest particles %s = %s"
      % (_near, d_rho[_near]))

def parts(rr_):
    dd, _ = M.geometry(z, rr_, ii, jj, Lz)
    geo_ = (rr_[ii] + rr_[jj]) / (2.0 * R0)
    rho_ = rho_from(z, rr_, ii, jj)
    A_ = pref * geo_ * M.W(dd)
    return A_, rho_

Ap, rhop = parts(rp)
Am, rhom = parts(rm)
dA = (Ap - Am) / (2 * dl)
dD = (rhop[ii] - rhop[jj]) - (rhom[ii] - rhom[jj])
dD = dD / (2 * dl)
dE_expl = float(np.sum(dA * (rho[ii] - rho[jj]) ** 2))
dE_impl = float(np.sum(2.0 * A * (rho[ii] - rho[jj]) * dD))
print("  sum_p dA_p/dr_0 * dRho^2      = %+.6e   -> force piece %+.6e" % (dE_expl, -dE_expl))
print("  sum_p 2 A_p dRho d(dRho)/dr_0 = %+.6e   -> force piece %+.6e" % (dE_impl, -dE_impl))
print("  sum of pieces                 = %+.6e   -> force %+.6e" % (dE_expl + dE_impl, -(dE_expl + dE_impl)))
print("  analytic chain=%+.6e  explicit=%+.6e  sum=%+.6e"
      % (Fc[k0, 1], Fex[k0, 1], (Fc + Fex)[k0, 1]))
print("  FD force      =%+.6e" % Ff[k0, 1])

# ---- delta convergence for a boundary particle -------------------------
print("\nScheme E, boundary particle k=0 (r=%.4f): delta convergence" % r[0])
print("     delta        F_r(FD)          F_r(analytic)      sum_a F_a,r (analytic)  -dE/dr_total (FD)")
for dl in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
    Ff1 = M.force_E_fd(z, r, ii, jj, Lz, R0, True, delta=dl)
    print("   %8.1e  %+.8e  %+.8e" % (dl, Ff1[0, 1], Fa[0, 1]))
for dl in (1e-4, 1e-5, 1e-6):
    tot_a = -float(np.sum(Fa[:, 1]))
    rp = r + dl
    rm = r - dl
    ep, _ = M.energy_E(z, rp, ii, jj, Lz, R0, True)
    em, _ = M.energy_E(z, rm, ii, jj, Lz, R0, True)
    print("   uniform-r shift delta=%8.1e : analytic sum F_r=%+.8e   FD -dE/dr=%+.8e"
          % (dl, tot_a, -(ep - em) / (2.0 * dl)))
