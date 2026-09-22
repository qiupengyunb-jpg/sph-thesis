"""M1.3R toy prototype: Scheme E (pairwise-difference interface energy).

F_E = (lambda/4) V0^2 Sum_i Sum_j J_ij Kt_ij (rho_i - rho_j)^2
      J_ij  = (r_i + r_j) / (2 R0)          (pair-midpoint measure factor)
      Kt_ij = c * W(r_ij),   c = 144 / (5 h_rho^2)

Kernel W is the production Wendland C2 (2D normalised, support h_rho).  The
constant c follows from the continuum identity
    (1/2) Int Int (rho-rho')^2 K(u) dA dA'  =  Int |grad rho|^2 dA
    provided  Int |u|^2 K(u) dA = 4 .
For W:  2 pi Int_0^h r^3 W(r) dr = 14 h^2 * (5/504) = 5 h^2 / 36   =>  c = 144/(5 h^2).

Everything below is diagnostic; nothing here touches the production solver.
"""
import math
import numpy as np

H = 2.8
RHO_REF = 0.88
LAM = 12.0
V0 = 1.0 / RHO_REF
C_K = 144.0 / (5.0 * H * H)

def W(r, h=H):
    q = np.asarray(r, float) / h
    o = np.zeros_like(q); m = q < 1.0; t = 1.0 - q[m]
    o[m] = (7.0/(math.pi*h*h))*t**4*(4.0*q[m]+1.0); return o
def dW(r, h=H):
    q = np.asarray(r, float) / h
    o = np.zeros_like(q); m = q < 1.0; t = 1.0 - q[m]
    o[m] = -(140.0/(math.pi*h**3))*q[m]*t**3; return o

def lattice(n0, R0, Ly, Lz):
    d = math.sqrt(2.0/(math.sqrt(3.0)*n0))
    zs=[];rs=[];row=0;rr=R0+0.5
    while rr <= Ly-1e-9:
        n=max(3,int(round(Lz/d))); dz=Lz/n; sh=0.5*dz if row%2 else 0.0
        for k in range(n): zs.append((sh+k*dz)%Lz); rs.append(rr)
        row+=1; rr+=0.5*math.sqrt(3.0)*d
    return np.array(zs), np.array(rs)

def pairs(z, r, Lz):
    n=len(z); out=[]
    for i in range(n):
        dz=z-z[i]; dz-=Lz*np.round(dz/Lz); dr=r-r[i]
        d=np.sqrt(dz*dz+dr*dr)
        for j in np.where((d>1e-12)&(d<H))[0]:
            if j>i: out.append((i,int(j),float(d[j])))
    return out

def density_and_grad(z, r, Lz):
    n=len(z); rho=np.zeros(n); gz=np.zeros(n); gr=np.zeros(n)
    for i in range(n):
        dz=z-z[i]; dz-=Lz*np.round(dz/Lz); dr=r-r[i]
        d=np.sqrt(dz*dz+dr*dr); m=(d>1e-12)&(d<H)
        if not np.any(m): continue
        ss=d[m]; e_r=(dr[m]/ss)
        rho[i]=float(np.sum(W(ss)))
        gz[i]=float(np.sum(dW(ss)*(-dz[m]/ss)))
        gr[i]=float(np.sum(dW(ss)*(-e_r)))
    return rho,gz,gr

def energy(z, r, Lz, R0, jmode="axi"):
    rho,_,_=density_and_grad(z,r,Lz)
    e=0.0
    for i,j,dij in pairs(z,r,Lz):
        J = (r[i]+r[j])/(2.0*R0) if jmode=="axi" else 1.0
        e += J*C_K*float(W(np.array([dij]))[0])*(rho[i]-rho[j])**2
    return 0.25*LAM*V0*V0*e, rho

def force_fd(z, r, Lz, R0, jmode="axi", delta=1e-6):
    """Central-difference force, used as the reference for the toy."""
    n=len(z); F=np.zeros((n,2))
    for k in range(n):
        for c in (0,1):
            zp=z.copy(); rp=r.copy()
            if c==0: zp[k]+=delta
            else:    rp[k]+=delta
            ep,_=energy(zp,rp,Lz,R0,jmode)
            zm=z.copy(); rm=r.copy()
            if c==0: zm[k]-=delta
            else:    rm[k]-=delta
            em,_=energy(zm,rm,Lz,R0,jmode)
            F[k,c]=-(ep-em)/(2*delta)
    return F

# ---------------------------------------------------------------- tests ----
print("="*100)
print("Scheme E toy  |  F_E = (lam/4) V0^2 Sum_ij J_ij Kt_ij (rho_i-rho_j)^2")
print("   Kt = c W,  c = 144/(5 h^2) = %.4f  (h = %.1f)   V0 = %.4f   lam = %.1f"
      % (C_K, H, V0, LAM))
print("="*100)

def T1(R0=5.0, Lz=20.0, Ly=30.0, n0=0.90, jmode="axi", pad=2*H):
    z,r=lattice(n0,R0,Ly,Lz)
    sel=(r>R0+0.5+pad)&(r<Ly-pad)
    e,rho=energy(z,r,Lz,R0,jmode)
    F=force_fd(z,r,Lz,R0,jmode)
    Fr=F[sel,1]; Fz=F[sel,0]
    return dict(N=len(z),nint=int(sel.sum()),E=e,
                meanFr=float(np.mean(Fr)),rmsFr=float(np.sqrt(np.mean(Fr**2))),
                maxFz=float(np.max(np.abs(Fz))),rho_std=float(np.std(rho[sel])))

print("\n[T1] uniform regular lattice, T=0 static force   (must have NO systematic Fr)")
print("     R0   J      N    n_int      energy      mean Fr        RMS Fr       max|Fz|   std(rho)_interior")
for R0 in (5.0,10.0,20.0):
    for jm in ("axi","planar"):
        a=T1(R0=R0,jmode=jm)
        print("   %5.1f %-7s %5d %6d %12.4e %13.4e %13.4e %11.3e %14.3e"
              % (R0,jm,a["N"],a["nint"],a["E"],a["meanFr"],a["rmsFr"],a["maxFz"],a["rho_std"]))

print("\n[T2] uniform lattice + deterministic small disorder (no RNG)")
def T2(R0=5.0,eps=0.02,jmode="axi",Lz=20.0,Ly=30.0,n0=0.90):
    z,r=lattice(n0,R0,Ly,Lz)
    i=np.arange(len(r)); d=math.sqrt(2.0/(math.sqrt(3.0)*n0))
    r=r+eps*d*(np.sin(1.7*i)+0.5*np.cos(0.9*i))
    sel=(r>R0+0.5+2*H)&(r<Ly-2*H)
    e,_=energy(z,r,Lz,R0,jmode); F=force_fd(z,r,Lz,R0,jmode)
    Fr=F[sel,1]
    return float(np.mean(Fr)),float(np.sqrt(np.mean(Fr**2))),int(sel.sum())
print("      eps/d    J        n_int      mean Fr        RMS Fr")
for eps in (0.0,0.01,0.02,0.05):
    for jm in ("axi",):
        m,rm,ni=T2(eps=eps,jmode=jm)
        print("   %8.3f %-7s %6d %13.4e %13.4e"%(eps,jm,ni,m,rm))

print("\n[T6] R0 scaling of the residual (axi), Ly=60 so every R0 has an interior:")
for R0 in (5.0,10.0,20.0):
    a=T1(R0=R0,jmode="axi",Ly=60.0)
    print("   R0=%5.1f  n_int=%4d  mean Fr=%+.4e  RMS=%+.4e  R0*mean Fr=%+.4e"
          % (R0,a["nint"],a["meanFr"],a["rmsFr"],R0*a["meanFr"]))

print("\n[T-BULK] domain-size scan: is the residual a BULK effect or a BOUNDARY effect?")
print("         (in an infinite uniform lattice Scheme E gives EXACTLY zero, because")
print("          rho_i = rho_j identically; any residual must be a free-surface effect)")
print("        Ly    n_int   boundary_frac    energy      mean Fr       RMS Fr    mean/RMS")
for Ly in (20.0,30.0,45.0,60.0,90.0):
    z,r=lattice(0.90,5.0,Ly,20.0)
    sel=(r>5.0+0.5+2*H)&(r<Ly-2*H)
    e,rho=energy(z,r,20.0,5.0,"axi")
    F=force_fd(z,r,20.0,5.0,"axi")
    Fr=F[sel,1]
    bfrac=1.0-len(Fr)/len(z)
    m=float(np.mean(Fr)); rm=float(np.sqrt(np.mean(Fr**2)))
    print("   %6.1f %6d %14.3f %12.4e %13.4e %13.4e %10.2f"
          % (Ly,int(sel.sum()),bfrac,e,m,rm,abs(m)/rm if rm>0 else 0.0))
