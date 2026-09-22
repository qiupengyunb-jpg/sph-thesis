"""N2.1 step 1: (F) density realisation check and (E) polynomial reproduction.

F: does the deterministic point set really carry the prescribed areal density?
E: on a UNIFORM triangular lattice, far from boundaries, do the raw and
   moment-corrected Laplacian estimators reproduce known polynomials?
   Fields are fed EXACTLY (analytic values at the particle positions), so this
   isolates the estimator from the point-set construction.
"""
import math
import numpy as np

def W(r, h):
    q = r / h; m = q < 1.0; out = np.zeros_like(q); t = 1.0 - q[m]
    out[m] = (7.0/(math.pi*h*h))*t**4*(4.0*q[m]+1.0); return out
def dW(r, h):
    q = r / h; m = q < 1.0; out = np.zeros_like(q); t = 1.0 - q[m]
    out[m] = -(140.0/(math.pi*h**3))*q[m]*t**3; return out

def triangular(Lz, Ly, d):
    z=[]; r=[]; row=0; rr=0.0
    while rr <= Ly:
        n = max(3, int(round(Lz/d)))
        dz = Lz/n; sh = 0.5*dz if row % 2 else 0.0
        for k in range(n):
            z.append((sh+k*dz) % Lz); r.append(rr)
        row += 1; rr += 0.5*math.sqrt(3.0)*d
    return np.array(z), np.array(r)

def kernel_sums(z, r, h, Lz, field=None):
    """Returns rho (=kernel sum of ones) or the given analytic field values."""
    n=len(z); rho=np.zeros(n); gz=np.zeros(n); gr=np.zeros(n); nb=np.zeros(n,int)
    for i in range(n):
        dz=z-z[i]; dz-=Lz*np.round(dz/Lz); dr=r-r[i]
        d=np.sqrt(dz*dz+dr*dr); m=(d>0)&(d<h)
        if not np.any(m): continue
        dd=d[m]; ez=np.where(dd>0,dz[m]/dd,0.0); er=np.where(dd>0,dr[m]/dd,0.0)
        val = np.ones_like(dd) if field is None else field(z[m], r[m])
        rho[i]=np.sum(W(dd,h)*val)          # kernel-weighted field average
        gz[i]=np.sum(dW(dd,h)*ez); gr[i]=np.sum(dW(dd,h)*er)
        nb[i]=int(np.count_nonzero(m))
    return rho,gz,gr,nb

def laplacians(z, r, f, h, Lz, V=None):
    """raw and moment-corrected Laplacian of the EXACT field f(z,r)."""
    n=len(z)
    if V is None: V=np.ones(n)
    raw=np.zeros(n); Z=np.zeros(n); gz=np.zeros(n); gr=np.zeros(n)
    for i in range(n):
        dz=z-z[i]; dz-=Lz*np.round(dz/Lz); dr=r-r[i]
        d=np.sqrt(dz*dz+dr*dr); m=(d>0)&(d<h)
        if not np.any(m): continue
        dd=d[m]
        raw[i]=2.0*np.sum(V[m]*(f(z[i],r[i])-f(z[m],r[m]))*dW(dd,h)/dd)
        Z[i]  =-0.5*np.sum(V[m]*(dd**2)*dW(dd,h)/dd)
        gz[i]=np.sum(dW(dd,h)*(dz[m]/dd)); gr[i]=np.sum(dW(dd,h)*(dr[m]/dd))
    return raw, Z, gz, gr

print("="*70)
print("E. POLYNOMIAL REPRODUCTION on a uniform triangular lattice (h=2.8)")
print("="*70)
Lz, Ly, h, rhoA = 60.0, 40.0, 2.8, 0.90
d = math.sqrt(2.0/(math.sqrt(3.0)*rhoA))
z, r = triangular(Lz, Ly, d)
V = np.full(len(z), 1.0/rhoA)
interior = (r > 2*h) & (r < Ly - 2*h)
print(f"lattice: d={d:.6f} N={len(z)} interior={int(np.count_nonzero(interior))} "
      f"h/d={h/d:.3f}")

tests = [
    ("P0 const      rho=c",        lambda z,r: np.full_like(r, 0.9),      0.0, 0.0),
    ("P1 linear     rho=a*r+c",    lambda z,r: 0.06*r + 0.2,              0.06, 0.0),
    ("P2 quadratic  rho=b*r^2+c",  lambda z,r: 0.004*r*r + 0.2,           0.0, 2*0.004),
]
for name, f, a_target, lap_target in tests:
    raw,Z,gz,gr = laplacians(z,r,f,h,Lz,V)
    corr = raw/Z
    print(f"\n{name}")
    print(f"   grad_r raw mean = {np.mean(gr[interior]):+.6e}   "
          f"(target grad_r at mean r = {a_target if a_target else 2*0.004*np.mean(r[interior]):+.6e})")
    print(f"   lap raw  mean = {np.mean(raw[interior]):+.6e}  target={lap_target:+.6e}")
    print(f"   Z    mean = {np.mean(Z[interior]):+.6f}  (should be ~1 for a uniform set)")
    print(f"   lap corr mean = {np.mean(corr[interior]):+.6e}  target={lap_target:+.6e}")

print()
print("="*70)
print("F. DENSITY REALISATION of the manufactured point set (R0=5, h=2.8)")
print("="*70)

def build(rho_fn, R0, h_in, H, Lz):
    zs=[]; rs=[]; row=0; rr=R0+h_in; top=R0+h_in+H
    while rr <= top + 1e-12:
        dd_ = math.sqrt(2.0/(math.sqrt(3.0)*float(rho_fn(np.array([rr]))[0])))
        n = max(3, int(round(Lz/dd_))); dz = Lz/n
        sh = 0.5*dz if row % 2 else 0.0
        for k in range(n):
            zs.append((sh+k*dz) % Lz); rs.append(rr)
        row += 1; rr += 0.5*math.sqrt(3.0)*dd_
    return np.array(zs), np.array(rs)

R0, h_in, H, Lz = 5.0, 0.5, 16.0, 40.0
Rc = R0 + h_in + 0.5*H
for label, fn in [("A_linear", lambda rr: 0.90 + 0.0225*(rr-Rc)),
                  ("U_uniform", lambda rr: np.full_like(rr, 0.90))]:
    z, r = build(fn, R0, h_in, H, Lz)
    rho_num,_,_,_ = kernel_sums(z, r, h, Lz)
    tgt = fn(r)
    sel = (r > R0+h_in+2*h) & (r < R0+h_in+H-2*h)
    rel = (rho_num[sel]-tgt[sel])/tgt[sel]
    print(f"{label}: N={len(z)} interior={int(np.count_nonzero(sel))}")
    print(f"   rho_num   mean={np.mean(rho_num[sel]):.6f}  std={np.std(rho_num[sel]):.3e}")
    print(f"   rho_target mean={np.mean(tgt[sel]):.6f}  std={np.std(tgt[sel]):.3e}")
    print(f"   relative deviation: mean={np.mean(rel):+.3e}  std={np.std(rel):.3e}  max|.|={np.max(np.abs(rel)):.3e}")
    # local area per particle implied by construction
    print(f"   construction local area/particle = 1/rho_target, V_j used in diagnostics = 1/rho_target(r_j)")
