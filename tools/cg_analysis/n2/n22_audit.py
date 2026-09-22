"""N2.2 part B (kernel-normalisation audit) and part C/D (exact production-kernel
derivatives validated by finite differences of the production density).

Production kernel (identical to cg_self_assembly_pri.cpp):
    W(s)  = (7/(pi h^2)) (1-q)^4 (4q+1) ,  W'(s) = -(140/(pi h^3)) q (1-q)^3
    W''(s) = -(140/(pi h^4)) (1-q)^2 (1-4q) ,   q = s/h,  W = 0 for q >= 1

Production density:  rho_i = Sum_{j != i} W(s_ij)
Exact derivatives of THAT field w.r.t. x_i (with (z,r) treated as flat coords):
    grad_i  = Sum_j W'(s) e_ij                       (e_ij = (x_i - x_j)/s)
    lap_i   = Sum_j [ W''(s) + W'(s)/s ]             (2D radial Laplacian)
Both are checked against position finite differences of rho_i itself.
"""
import math
import numpy as np

HQ = 60          # quadrature nodes

def W(s, h):
    q = np.asarray(s, float) / h
    out = np.zeros_like(q); m = q < 1.0; t = 1.0 - q[m]
    out[m] = (7.0/(math.pi*h*h))*t**4*(4.0*q[m]+1.0); return out
def dW(s, h):
    q = np.asarray(s, float) / h
    out = np.zeros_like(q); m = q < 1.0; t = 1.0 - q[m]
    out[m] = -(140.0/(math.pi*h**3))*q[m]*t**3; return out
def ddW(s, h):
    q = np.asarray(s, float) / h
    out = np.zeros_like(q); m = q < 1.0; t = 1.0 - q[m]
    out[m] = -(140.0/(math.pi*h**4))*t**2*(1.0-4.0*q[m]); return out

def I_W(h):
    """2 pi Integral_0^h W(r) r dr by Gauss-Legendre (exact to machine precision
    for this polynomial kernel)."""
    x, w = np.polynomial.legendre.leggauss(HQ)
    r = 0.5*(x+1.0)*h; wr = 0.5*h*w
    return float(2.0*math.pi*np.sum(W(r, h)*r*wr))

def triangular(Lz, r_lo, r_hi, d):
    z=[]; r=[]; row=0; rr=r_lo
    while rr <= r_hi + 1e-12:
        n = max(3, int(round(Lz/d))); dz = Lz/n
        sh = 0.5*dz if row % 2 else 0.0
        for k in range(n):
            z.append((sh+k*dz) % Lz); r.append(rr)
        row += 1; rr += 0.5*math.sqrt(3.0)*d
    return np.array(z), np.array(r)

print("="*74)
print("B1. kernel normalisation:  I_W = 2 pi Integral_0^h W(r) r dr")
print("="*74)
for h in (2.0, 2.8, 3.6):
    print("   h = %.1f   I_W = %.15f   (2D normalised kernel requires 1)" % (h, I_W(h)))

print()
print("="*74)
print("B2. lattice quadrature: sum_j W over an interior particle of a uniform")
print("    triangular lattice, versus the true areal number density rho_A")
print("="*74)
print("   rho_A      d      h=%.1f       h=%.1f       h=%.1f   (values: sum W / rho_A)"
      % (2.0, 2.8, 3.6))
Lz = 60.0
for rhoA in (0.90, 0.50, 0.30, 0.15):
    d = math.sqrt(2.0/(math.sqrt(3.0)*rhoA))
    rowsp = 0.5*math.sqrt(3.0)*d
    nrow = int(round(Lz/d)); dz = Lz/nrow
    # one interior particle at (0, R) with R large enough that all neighbours
    # with s < h are inside the point set
    line = "   %.3f  %.4f " % (rhoA, d)
    for h in (2.0, 2.8, 3.6):
        R = 20.0
        zz = (np.arange(-60, 61)[:, None] * dz).ravel()
        zz = np.concatenate([zz - (0.5*dz if (j % 2) else 0.0) for j in (0, 1)])
        rr_off = np.array([0.0, rowsp] * 121)
        # build a proper local lattice: rows j around R
        zs=[]; rs=[]
        for j in range(-int(h/rowsp)-1, int(h/rowsp)+2):
            y = R + j*rowsp
            sh = 0.5*dz if (j % 2) else 0.0
            for k in range(nrow):
                zs.append((k*dz + sh) % Lz); rs.append(y)
        zs = np.array(zs); rs = np.array(rs)
        dzv = zs - 0.0; dzv -= Lz*np.round(dzv/Lz); drv = rs - R
        s = np.sqrt(dzv*dzv + drv*drv); m = (s > 0) & (s < h)
        line += "   %8.5f" % (float(np.sum(W(s[m], h)))/rhoA)
    print(line)

print()
print("="*74)
print("C/D. FD validation of the EXACT production-kernel derivatives")
print("     configuration: uniform triangular lattice, interior particle")
print("="*74)
h = 2.8
rhoA = 0.90
d = math.sqrt(2.0/(math.sqrt(3.0)*rhoA))
Lz, H = 60.0, 30.0
zs, rs = triangular(Lz, 0.0, H, d)

def rho_of(z, r, h, Lz, i, all_z, all_r):
    """rho_i = sum_j W, with particle i at (z,r); z periodic."""
    dz = all_z - z; dz -= Lz*np.round(dz/Lz); dr = all_r - r
    s = np.sqrt(dz*dz + dr*dr); m = (s > 0) & (s < h); m[i] = False
    return float(np.sum(W(s[m], h)))

sel = (rs > 2*h) & (rs < H - 2*h)
idx = np.where(sel)[0]
print("   lattice N=%d  interior=%d  h/d=%.3f" % (len(zs), len(idx), h/d))

Gz=[]; Gr=[]; Lp=[]; nb=[]
for i in idx:
    dz = zs - zs[i]; dz -= Lz*np.round(dz/Lz); dr = rs - rs[i]
    s = np.sqrt(dz*dz + dr*dr); m = (s > 0) & (s < h)
    ss = s[m]; ez = dz[m]/ss; er = dr[m]/ss
    Gz.append(float(np.sum(dW(ss, h)*ez)))
    Gr.append(float(np.sum(dW(ss, h)*er)))
    Lp.append(float(np.sum(ddW(ss, h) + dW(ss, h)/ss)))
    nb.append(int(np.count_nonzero(m)))
Gz=np.array(Gz); Gr=np.array(Gr); Lp=np.array(Lp); nb=np.array(nb)
print("   mean neighbour count = %.2f" % nb.mean())

# finite differences on a few interior particles
print()
print("   FD check on the first 3 interior particles (delta = 1e-5):")
print("   i    drho/dz(FD)      Gz(sum W' e)     drho/dr(FD)      Gr(sum W' e)"
      "     lap(FD)          lap(sum W''+W'/s)")
delta = 1.0e-5
for i in idx[:3]:
    z0, r0 = zs[i], rs[i]
    allz = zs.copy(); allr = rs.copy()
    gz_fd = -(rho_of(z0+delta, r0, h, Lz, i, allz, allr)
              - rho_of(z0-delta, r0, h, Lz, i, allz, allr))/(2*delta)
    gr_fd = -(rho_of(z0, r0+delta, h, Lz, i, allz, allr)
              - rho_of(z0, r0-delta, h, Lz, i, allz, allr))/(2*delta)
    lap_fd = (rho_of(z0+delta, r0, h, Lz, i, allz, allr)
              - 2.0*rho_of(z0, r0, h, Lz, i, allz, allr)
              + rho_of(z0-delta, r0, h, Lz, i, allz, allr))/delta**2 \
           + (rho_of(z0, r0+delta, h, Lz, i, allz, allr)
              - 2.0*rho_of(z0, r0, h, Lz, i, allz, allr)
              + rho_of(z0, r0-delta, h, Lz, i, allz, allr))/delta**2
    k = np.where(idx == i)[0][0]
    print("   %3d  %+.10e  %+.10e  %+.10e  %+.10e  %+.8e  %+.8e"
          % (i, gz_fd, Gz[k], gr_fd, Gr[k], lap_fd, Lp[k]))
