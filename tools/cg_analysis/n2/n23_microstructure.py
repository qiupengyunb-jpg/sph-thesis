"""N2.3 lattice-microstructure / resolution-limit diagnostics (diagnostic only).

Three objects kept strictly apart:
  1. particle sampled density         rho_i = Sum_j W(|x_i - x_j|)
  2. particle displacement derivative d rho_i / d r_i = Sum_j W'(s) e_r,ij
  3. Eulerian reconstructed field     rho_probe(r), d rho_probe/dr
     (the probe is a POINT: no source particle is moved; z-averaged over one period)

Nothing here touches production code or production parameters.
"""
import math
import numpy as np

def W(s, h):
    q=np.asarray(s,float)/h; out=np.zeros_like(q); m=q<1.0; t=1.0-q[m]
    out[m]=(7.0/(math.pi*h*h))*t**4*(4.0*q[m]+1.0); return out
def dW(s, h):
    q=np.asarray(s,float)/h; out=np.zeros_like(q); m=q<1.0; t=1.0-q[m]
    out[m]=-(140.0/(math.pi*h**3))*q[m]*t**3; return out

def build(rho_fn, R0, h_in, H, Lz, phase=0.0, disorder=0.0, z_mode="alt"):
    """Deterministic rows realising the target areal density.
    phase    : radial shift of every row, in units of the local row spacing
    disorder : r_i -> r_i + disorder * d * f_i with a fixed deterministic pattern
    z_mode   : "alt"  = odd rows shifted by half a z-spacing (triangular, as in M1.2)
               "none" = every row at the same z phase (square-like lattice)
               "gold" = deterministic irrational per-row z phase (breaks the
                        alternating row-phase locking without any RNG)
    """
    zs=[]; rs=[]; row=0; rr=R0+h_in; top=R0+h_in+H
    while rr <= top+1e-12:
        dd_=math.sqrt(2.0/(math.sqrt(3.0)*float(rho_fn(np.array([rr]))[0])))
        rowsp=0.5*math.sqrt(3.0)*dd_
        n=max(3,int(round(Lz/dd_))); dz=Lz/n
        if z_mode=="alt":   sh=0.5*dz if row%2 else 0.0
        elif z_mode=="none":sh=0.0
        else:               sh=(row*0.6180339887498949 % 1.0)*dz
        rr_eff = rr + phase*rowsp
        for k in range(n):
            zs.append((sh+k*dz)%Lz); rs.append(rr_eff)
        row+=1; rr+=rowsp
    z=np.array(zs); r=np.array(rs)
    if disorder:
        i=np.arange(len(r))
        f=np.sin(1.7*i)+0.5*np.cos(0.9*i)
        d_loc=math.sqrt(2.0/(math.sqrt(3.0)*float(np.mean(rho_fn(r)))))
        r=r+disorder*d_loc*f
    return z,r

def particle_deriv(z, r, h, Lz):
    """d rho_i / d r_i = Sum_j W'(s) (r_i - r_j)/s   (exact).
    NOTE the sign convention: the production G uses e_ij = (x_i - x_j)/s, so the
    correct particle derivative carries (r_i - r_j), NOT (r_j - r_i).  An earlier
    revision of this file used (r - r[i]) and therefore reported the NEGATIVE of
    the derivative, which produced the spurious Q_particle ~ -1."""
    n=len(z); out=np.zeros(n)
    for i in range(n):
        dz=z-z[i]; dz-=Lz*np.round(dz/Lz); dr=r-r[i]
        s=np.sqrt(dz*dz+dr*dr); m=(s>0)&(s<h)
        out[i]=float(np.sum(dW(s[m],h)*(-dr[m]/s[m])))
    return out

def probe_field(z, r, rg, h, Lz, nz=16):
    """z-averaged Eulerian density and its analytic r-derivative at probe radii.
    No source particle is moved."""
    zp=np.linspace(0.0,Lz,nz,endpoint=False)
    rho_p=np.zeros(len(rg)); drho_p=np.zeros(len(rg))
    for z0 in zp:
        for a,rp in enumerate(rg):
            dz=z-z0; dz-=Lz*np.round(dz/Lz); dr=r-rp
            s=np.sqrt(dz*dz+dr*dr); m=(s>0)&(s<h)
            if not np.any(m): continue
            ss=s[m]
            rho_p[a]+=float(np.sum(W(ss,h)))
            drho_p[a]+=float(np.sum(dW(ss,h)*((rp-r[m])/ss)))
    return rho_p/nz, drho_p/nz

def gauss_smooth(y, x, sigma):
    """Fixed low-pass rule (declared a priori): Gaussian with sigma = 0.5 d."""
    out=np.zeros_like(y)
    for i in range(len(y)):
        w=np.exp(-0.5*((x-x[i])/sigma)**2)
        out[i]=np.sum(w*y)/np.sum(w)
    return out

PROFILES={"A_linear":(0.0225,0.0),"B_quadratic":(0.0225,0.0015),
          "C_pure_quadratic":(0.0,0.0015)}

Lz,H,h_in,R0=40.0,16.0,0.5,5.0
d0=math.sqrt(2.0/(math.sqrt(3.0)*0.90))

def profile_fn(pname,Rc,rho_scale=0.90):
    a,b=PROFILES[pname]
    return lambda rr,Rc=Rc,a=a,b=b,s=rho_scale: s*(1.0+ (a/0.90)*(rr-Rc) + (b/0.90)*(rr-Rc)**2)

def run_case(pname, R0, h, phase=0.0, disorder=0.0, rho_scale=0.90, nz=16,
             nprobe=220, z_mode="alt"):
    # the film band must be wide enough that the 2h_rho interior band survives
    H_use=max(H, 6.0*h)
    Rc=R0+h_in+0.5*H_use
    g=profile_fn(pname,Rc,rho_scale)
    z,r=build(g,R0,h_in,H_use,Lz,phase=phase,disorder=disorder,z_mode=z_mode)
    d=math.sqrt(2.0/(math.sqrt(3.0)*rho_scale))
    rg=np.linspace(R0+h_in+0.2*h, R0+h_in+H_use-0.2*h, nprobe)
    rho_p,drho_p=probe_field(z,r,rg,h,Lz,nz)
    eps=1e-4
    rp1,_=probe_field(z,r,rg+eps,h,Lz,nz); rp2,_=probe_field(z,r,rg-eps,h,Lz,nz)
    fd_err=float(np.max(np.abs((rp1-rp2)/(2*eps)-drho_p))/np.max(np.abs(drho_p)))
    env=gauss_smooth(rho_p,rg,0.5*d); denv=gauss_smooth(drho_p,rg,0.5*d)
    ripple=rho_p-env
    sel=(r>R0+h_in+2*h)&(r<R0+h_in+H_use-2*h)
    if not np.any(sel):
        return None
    pr=r[sel]; dpart=particle_deriv(z,r,h,Lz)[sel]
    rho_at=np.interp(pr,rg,rho_p)
    C=float(np.mean(rho_at/g(pr)))
    a,b=PROFILES[pname]
    tgt=rho_scale*((a/0.90)+2*(b/0.90)*(pr-Rc))
    f=lambda v: float(np.mean(v/(C*tgt))) if np.any(tgt!=0) else float('nan')
    return dict(profile=pname,h=h,h_over_d=h/d,N=len(z),n_int=int(np.count_nonzero(sel)),
                C_kernel=C,Q_particle=f(dpart),
                Q_probe=f(np.interp(pr,rg,drho_p)),Q_envelope=f(np.interp(pr,rg,denv)),
                ripple=float(np.sqrt(np.mean(ripple**2))/
                              max(np.sqrt(np.mean((env-env.mean())**2)),1e-30)),
                probe_fd_err=fd_err)

print("="*78)
print("N2.3  lattice microstructure / resolution diagnostics  (Profile A unless noted)")
print("="*78)
print("reference lattice d = %.4f sigma (rho_A = 0.90);  production h_rho = 2.8 -> h/d = %.3f"
      % (d0, 2.8/d0))

o=run_case("A_linear",R0,2.8)
print()
print("[1] Eulerian probe derivative: FD vs analytic, relative error = %.3e" % o["probe_fd_err"])
print("    (no source particle is moved in either evaluation)")

print()
print("[2] h/d sweep at FIXED lattice (d = %.4f): raise h_rho only" % d0)
print("      h/d    h_rho    Q_particle      Q_probe    Q_envelope      ripple")
for hd in (2.0,2.5,3.0,4.0,5.0,6.0,8.0):
    o=run_case("A_linear",R0,hd*d0)
    print("    %5.1f %8.3f %13.4f %13.4f %13.4f %11.4f"
          % (hd,hd*d0,o["Q_particle"],o["Q_probe"],o["Q_envelope"],o["ripple"]))

print()
print("[3] h/d sweep by REFINING the lattice (h_rho fixed at 2.8)")
print("      h/d        d    rho_A    Q_particle      Q_probe    Q_envelope")
for hd in (2.0,2.5,3.0,4.0,5.0,6.0):
    d_=2.8/hd; rho_A=2.0/(math.sqrt(3.0)*d_*d_)
    o=run_case("A_linear",R0,2.8,rho_scale=rho_A)
    if o is None:
        print("    %5.1f %8.4f %8.3f   (no interior band)" % (hd,d_,rho_A)); continue
    print("    %5.1f %8.4f %8.3f %13.4f %13.4f %13.4f"
          % (hd,d_,rho_A,o["Q_particle"],o["Q_probe"],o["Q_envelope"]))

print()
print("[4] radial ROW-PHASE test (h/d = 2.5), identical target density")
print("      phase    Q_particle      Q_probe      ripple")
for ph in (0.0,0.25,0.5,0.75):
    o=run_case("A_linear",R0,2.5*d0,phase=ph)
    print("    %7.2f %13.4f %13.4f %11.4f" % (ph,o["Q_particle"],o["Q_probe"],o["ripple"]))

print()
print("[5] deterministic small disorder (h/d = 2.5); fixed pattern, no RNG, no dynamics")
print("       eps/d    Q_particle      Q_probe")
for dsp in (0.0,0.02,0.05,0.10):
    o=run_case("A_linear",R0,2.5*d0,disorder=dsp)
    print("    %9.2f %13.4f %13.4f" % (dsp,o["Q_particle"],o["Q_probe"]))

print()
print("[6] representative B / C at production h_rho = 2.8 (and at h/d = 6 for contrast)")
for pn in ("B_quadratic","C_pure_quadratic"):
    for hd in (2.46,6.0):
        o=run_case(pn,R0,hd*d0)
        print("    %-18s h/d=%4.2f  Q_particle=%9.4f  Q_probe=%9.4f  Q_env=%9.4f"
              % (pn,hd,o["Q_particle"],o["Q_probe"],o["Q_envelope"]))

print()
print("[7] *** decisive: is the sign flip caused by the ROW z-phase structure? ***")
print("    same target density, same h/d, only the per-row z phase is changed")
print("      z_mode      h/d    Q_particle      Q_probe      ripple")
for zm in ("alt","none","gold"):
    for hd in (2.47, 4.0, 6.0):
        o=run_case("A_linear",R0,hd*d0,z_mode=zm)
        print("    %-8s %8.2f %13.4f %13.4f %11.4f"
              % (zm,hd,o["Q_particle"],o["Q_probe"],o["ripple"]))
