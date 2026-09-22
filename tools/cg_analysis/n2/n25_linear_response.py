"""N2.5 linear-response test of the A-1 second-order operator (diagnostic only).

CONVENTIONS (frozen, see N2.4):
    e_ij = (x_i - x_j)/r_ij ,  G_i = Sum_j W'(r_ij) e_ij = grad_{x_i} rho_i
    rho_i = Sum_j W(r_ij)          particle-sampled field
    L_i   = Sum_j [W''(s) + W'(s)/s]   exact kernel Laplacian of rho_i in (z,r)

---------------------------------------------------------------
COORDINATE MAPPING AND CONTINUOUS TARGET  (declared a priori)
---------------------------------------------------------------
Base: uniform lattice of areal number density n0 in the (z,r) plane.
Map : r -> r' = r + eps f(r)            (z untouched; particle identity kept)

Number conservation in the meridional plane:
    n'(r') dr' dz = n0 dr dz   =>   n'(r') = n0 / M'(r),   M'(r) = 1 + eps f'(r)
    =>  n'(r') = n0 [ 1 - eps f'(r') + O(eps^2) ]
    =>  delta n(r) := n'(r) - n0 = -eps n0 f'(r) + O(eps^2)

Choose the perturbation SHAPE directly instead of choosing f:
    delta n(r) = eps * a_p * (r - Rc)^2          (quadratic in r)
    =>  f'(r) = -a_p (r - Rc)^2 / n0
    =>  f(r)  = -a_p (r - Rc)^3 / (3 n0)

Targets (the "d/d eps at eps = 0" of the continuum operators):
    delta n''      = 2 a_p                              (planar Laplacian response)
    delta n'       = 2 a_p (r - Rc)                     (gradient response)
    delta (n'/r)   = 2 a_p (r - Rc)/r                   (geometry response)

The static lattice background L(0) is eps-independent, so the SYMMETRIC
difference  deltaL = [L(+eps) - L(-eps)]/(2 eps)  removes it to first order.
"""
import math
import numpy as np

def W(s,h):
    q=np.asarray(s,float)/h; o=np.zeros_like(q); m=q<1.0; t=1.0-q[m]
    o[m]=(7.0/(math.pi*h*h))*t**4*(4.0*q[m]+1.0); return o
def dW(s,h):
    q=np.asarray(s,float)/h; o=np.zeros_like(q); m=q<1.0; t=1.0-q[m]
    o[m]=-(140.0/(math.pi*h**3))*q[m]*t**3; return o
def ddW(s,h):
    q=np.asarray(s,float)/h; o=np.zeros_like(q); m=q<1.0; t=1.0-q[m]
    o[m]=-(140.0/(math.pi*h**4))*t**2*(1.0-4.0*q[m]); return o

def base_lattice(n0,R0,h_in,H,Lz):
    d=math.sqrt(2.0/(math.sqrt(3.0)*n0))
    zs=[];rs=[];row=0;rr=R0+h_in;top=R0+h_in+H
    while rr<=top+1e-12:
        n=max(3,int(round(Lz/d)));dz=Lz/n;sh=0.5*dz if row%2 else 0.0
        for k in range(n): zs.append((sh+k*dz)%Lz);rs.append(rr)
        row+=1;rr+=0.5*math.sqrt(3.0)*d
    return np.array(zs),np.array(rs),d

def apply_map(r,eps,a_p,n0,Rc):
    return r + eps*(-a_p*(r-Rc)**3/(3.0*n0))

def operators(z,r,h,Lz):
    """rho, grad_r (production sign), L = Sum[W''+W'/s]."""
    n=len(z);rho=np.zeros(n);gr=np.zeros(n);L=np.zeros(n)
    for k in range(n):
        dz=z-z[k];dz-=Lz*np.round(dz/Lz);dr=r-r[k]
        s=np.sqrt(dz*dz+dr*dr);m=(s>0)&(s<h)
        if not np.any(m): continue
        ss=s[m];er=(dr[m]/ss)              # (r_j - r_k)/s
        rho[k]=float(np.sum(W(ss,h)))
        gr[k]=float(np.sum(dW(ss,h)*(-er))) # e_ij=(x_k-x_j)/s
        L[k]=float(np.sum(ddW(ss,h)+dW(ss,h)/ss))
    return rho,gr,L

def stat(num,tgt):
    num=np.asarray(num,float);tgt=np.asarray(tgt,float);e=num-tgt
    rmse=float(np.sqrt(np.mean(e**2)));rms=float(np.sqrt(np.mean(tgt**2)))
    corr=float(np.corrcoef(num,tgt)[0,1]) if np.std(num)>0 and np.std(tgt)>0 else float('nan')
    return rmse,(rmse/rms if rms>0 else float('nan')),corr

Lz,h_in=40.0,0.5
n0=0.90
a_p=0.05
EPS=(0.005,0.01,0.02,0.04)

print("="*96)
print("N2.5  linear response of the A-1 second-order operator")
print("="*96)
print("mapping  r' = r + eps f(r),  f(r) = -a_p (r-Rc)^3/(3 n0);  delta n = eps a_p (r-Rc)^2")
print("targets  delta n'' = 2 a_p = %.4f   delta (n'/r) = 2 a_p (r-Rc)/r"%(2*a_p))

def run(R0,h,H=None,n0=n0,a_p=a_p,interior_pad=None):
    H = H if H else max(16.0,6.0*h)
    Rc = R0+h_in+0.5*H
    z,r0,d = base_lattice(n0,R0,h_in,H,Lz)
    pad = interior_pad if interior_pad else 2*h
    sel = (r0>R0+h_in+pad)&(r0<R0+h_in+H-pad)
    out={}
    for eps in (0.0,)+tuple(-e for e in EPS)+EPS:
        r=apply_map(r0,eps,a_p,n0,Rc)
        rho,gr,L=operators(z,r,h,Lz)
        out[eps]=(rho,gr,L,r)
    return out,sel,Rc,d

def analyse(R0,h,label):
    out,sel,Rc,d=run(R0,h)
    res={}
    L0=out[0.0][2][sel]
    res["L_bg_mean"]=float(np.mean(L0)); res["L_bg_rms"]=float(np.sqrt(np.mean(L0**2)))
    for eps in EPS:
        Lp=out[eps][2][sel]; Lm=out[-eps][2][sel]
        dL=(Lp-Lm)/(2*eps)
        res["dL_%.3f"%eps]=dL
    res["sel"]=sel; res["Rc"]=Rc; res["d"]=d; res["h"]=h; res["label"]=label
    return res

import os
os.makedirs(r"E:\哈哈\_tmp_probe\n25",exist_ok=True)
rows=[]

def report(R0,h,label,full=True):
    res=analyse(R0,h,label)
    tgt=2.0*a_p
    print("\n"+"-"*96)
    print("CASE %s :  R0=%.1f  h_rho=%.3f  d=%.4f  h/d=%.2f"%(label,R0,h,res["d"],h/res["d"]))
    print("  raw background L(0): mean=%+.4f  rms=%.4f   (theory: 0)"%(res["L_bg_mean"],res["L_bg_rms"]))
    print("     eps     mean(dL)    RMSE(dL)   NRMSE(dL)   corr     SNR=RMS(dL)/RMS(L_bg)")
    for eps in EPS:
        dL=res["dL_%.3f"%eps]
        rm,nr,co=stat(dL,np.full_like(dL,tgt))
        snr=float(np.sqrt(np.mean(dL**2)))/res["L_bg_rms"] if res["L_bg_rms"]>0 else float('nan')
        print("   %6.3f  %+9.5f  %9.5f  %9.4f  %7.4f   %9.5f"%(eps,float(np.mean(dL)),rm,nr,co,snr))
        rows.append(dict(case=label,R0=R0,h=h,d=res["d"],h_over_d=h/res["d"],eps=eps,
                         L_bg_mean=res["L_bg_mean"],L_bg_rms=res["L_bg_rms"],
                         dL_mean=float(np.mean(dL)),dL_target=tgt,rmse=rm,nrmse=nr,
                         corr=co,snr=snr))
    return res

print("\n"+"="*96); print("[1] PRODUCTION POINT  R0=5, h_rho=2.8 (h/d=2.46)  -- J=1 (planar) only")
base=report(5.0,2.8,"production")

print("\n"+"="*96); print("[2] R0 scaling (same local perturbation): R0 = 5 / 10 / 20")
for R0 in (5.0,10.0,20.0):
    r_=report(R0,2.8,"R0=%.0f"%R0,full=False)

print("\n"+"="*96)
print("[3] RESOLUTION A (PRIMARY): fix h_rho = 2.8, REFINE the lattice (d down)")
for hd in (2.0,2.5,3.0,4.0,5.0,6.0):
    d_=2.8/hd; n0_=2.0/(math.sqrt(3.0)*d_*d_)
    H=max(16.0,6.0*2.8); Rc=5.0+h_in+0.5*H
    z=apply_map(base_lattice(n0_,5.0,h_in,H,Lz)[1],0.0,a_p,n0_,Rc)
    z0,r0,_=base_lattice(n0_,5.0,h_in,H,Lz)
    sel=(r0>5.0+h_in+2*2.8)&(r0<5.0+h_in+H-2*2.8)
    if not np.any(sel): continue
    Ls={}
    for eps in (0.0,)+tuple(-e for e in EPS)+EPS:
        r=apply_map(r0,eps,a_p,n0_,Rc)
        Ls[eps]=operators(z0,r,2.8,Lz)[2][sel]
    print("  h/d=%.1f  d=%.4f  n0=%.3f  L_bg_rms=%.4f"%(hd,d_,n0_,float(np.sqrt(np.mean(Ls[0.0]**2)))))
    for eps in EPS:
        dL=(Ls[eps]-Ls[-eps])/(2*eps); rm,nr,co=stat(dL,np.full_like(dL,2*a_p))
        snr=float(np.sqrt(np.mean(dL**2)))/float(np.sqrt(np.mean(Ls[0.0]**2)))
        print("     eps=%.3f  mean(dL)=%+.5f  NRMSE=%.4f  corr=%+.4f  SNR=%.4f"%(eps,float(np.mean(dL)),nr,co,snr))
        rows.append(dict(case="resA_h%.1f"%hd,R0=5.0,h=2.8,d=d_,h_over_d=hd,eps=eps,
                         L_bg_mean=float(np.mean(Ls[0.0])),L_bg_rms=float(np.sqrt(np.mean(Ls[0.0]**2))),
                         dL_mean=float(np.mean(dL)),dL_target=2*a_p,rmse=rm,nrmse=nr,corr=co,snr=snr))

print("\n"+"="*96)
print("[4] RESOLUTION B (SECONDARY): fix d, INCREASE h_rho (pure smoothing)")
for hd in (2.0,2.5,3.0,4.0,5.0,6.0):
    hh=hd*1.1327
    r_=report(5.0,hh,"resB_h%.1f"%hd,full=False)

keys=["case","R0","h","d","h_over_d","eps","L_bg_mean","L_bg_rms","dL_mean",
      "dL_target","rmse","nrmse","corr","snr"]
with open(r"E:\哈哈\_tmp_probe\n25\n25_summary.csv","w") as fh:
    fh.write(",".join(keys)+"\n")
    for o in rows: fh.write(",".join(str(o[k]) for k in keys)+"\n")
print("\nwrote n25_summary.csv (%d rows)"%len(rows))
