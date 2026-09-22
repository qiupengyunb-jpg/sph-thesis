"""N2.2 Track P: production kernel-density operator, validated by finite
differences of the production density itself.

Step 1  sign of grad_rho_prod  : FD of rho_i w.r.t. position on a NON-uniform
                                 point set (a uniform lattice cannot decide it)
Step 2  36 cases (4 profiles x 3 R0 x 3 resolutions) re-analysed with the
        production-kernel observable only.
"""
import math
import numpy as np

def W(s,h):
    q=np.asarray(s,float)/h; out=np.zeros_like(q); m=q<1.0; t=1.0-q[m]
    out[m]=(7.0/(math.pi*h*h))*t**4*(4.0*q[m]+1.0); return out
def dW(s,h):
    q=np.asarray(s,float)/h; out=np.zeros_like(q); m=q<1.0; t=1.0-q[m]
    out[m]=-(140.0/(math.pi*h**3))*q[m]*t**3; return out
def ddW(s,h):
    q=np.asarray(s,float)/h; out=np.zeros_like(q); m=q<1.0; t=1.0-q[m]
    out[m]=-(140.0/(math.pi*h**4))*t**2*(1.0-4.0*q[m]); return out

def build(rho_fn, R0, h_in, H, Lz):
    zs=[]; rs=[]; row=0; rr=R0+h_in; top=R0+h_in+H
    while rr <= top+1e-12:
        dd_=math.sqrt(2.0/(math.sqrt(3.0)*float(rho_fn(np.array([rr]))[0])))
        n=max(3,int(round(Lz/dd_))); dz=Lz/n; sh=0.5*dz if row%2 else 0.0
        for k in range(n):
            zs.append((sh+k*dz)%Lz); rs.append(rr)
        row+=1; rr+=0.5*math.sqrt(3.0)*dd_
    return np.array(zs), np.array(rs)

def prod_fields(z, r, h, Lz, i=None, zi=None, ri=None):
    """rho, grad_z, grad_r, lap for every particle.  If i is given, particle i
    is evaluated at (zi, ri) instead of its stored position (for FD)."""
    n=len(z); rho=np.zeros(n); gz=np.zeros(n); gr=np.zeros(n); lap=np.zeros(n)
    zz=z.copy(); rr=r.copy()
    if i is not None: zz[i]=zi; rr[i]=ri
    for k in range(n):
        dz=zz-zz[k]; dz-=Lz*np.round(dz/Lz); dr=rr-rr[k]
        s=np.sqrt(dz*dz+dr*dr); m=(s>0)&(s<h)
        if not np.any(m): continue
        ss=s[m]; ez=dz[m]/ss; er=dr[m]/ss
        rho[k]=float(np.sum(W(ss,h)))
        gz[k]=float(np.sum(dW(ss,h)*ez))
        gr[k]=float(np.sum(dW(ss,h)*er))
        lap[k]=float(np.sum(ddW(ss,h)+dW(ss,h)/ss))
    return rho,gz,gr,lap

# ---------------------------------------------------------------- step 1 ---
print("="*74)
print("STEP 1. sign of grad_rho_prod from FD on a NON-uniform point set")
print("="*74)
R0,h_in,H,Lz,h = 5.0,0.5,16.0,40.0,2.8
Rc=R0+h_in+0.5*H
prof=lambda rr: 0.90+0.0225*(rr-Rc)
z,r=build(prof,R0,h_in,H,Lz)
sel=(r>R0+h_in+2*h)&(r<R0+h_in+H-2*h); idx=np.where(sel)[0]
rho0,gz0,gr0,lap0=prod_fields(z,r,h,Lz)
delta=1.0e-5
print("  i     drho/dr (FD)        Gr = sum W' e_r     ratio FD/Gr")
for i in idx[:4]:
    _,_,_,_ = 0,0,0,0
    rp,gzp,grp,_=prod_fields(z,r,h,Lz,i=i,zi=z[i],ri=r[i]+delta)
    rm,gzm,grm,_=prod_fields(z,r,h,Lz,i=i,zi=z[i],ri=r[i]-delta)
    fd=-(rp[i]-rm[i])/(2*delta)
    print("  %3d  %+.10e   %+.10e   %+.6f" % (i, fd, gr0[i], fd/gr0[i] if gr0[i]!=0 else float('nan')))
    rp,gzp,grp,_=prod_fields(z,r,h,Lz,i=i,zi=z[i]+delta,ri=r[i])
    rm,gzm,grm,_=prod_fields(z,r,h,Lz,i=i,zi=z[i]-delta,ri=r[i])
    print("       drho/dz (FD)=%+.10e   Gz=%+.10e" % (-(rp[i]-rm[i])/(2*delta), gz0[i]))

# ---------------------------------------------------------------- step 2 ---
PROFILES={
 "A_linear":       (lambda rr: 0.90+0.0225*(rr-Rc),       0.0225, 0.0),
 "B_quadratic":    (lambda rr: 0.90+0.0225*(rr-Rc)+0.0015*(rr-Rc)**2, 0.0225, 0.0015),
 "C_pure_quadratic":(lambda rr: 0.90+0.0015*(rr-Rc)**2,   0.0,    0.0015),
 "U_uniform":      (lambda rr: np.full_like(rr,0.90),     0.0,    0.0),
}
def metrics(num,tgt):
    num=np.asarray(num,float); tgt=np.asarray(tgt,float); e=num-tgt
    rmse=float(np.sqrt(np.mean(e**2))); rms=float(np.sqrt(np.mean(tgt**2)))
    out=dict(rmse=rmse, nrmse=(rmse/rms if rms>0 else float('nan')),
             bias=float(np.mean(e)), mean_num=float(np.mean(num)),
             mean_tgt=float(np.mean(tgt)))
    if np.std(num)>0 and np.std(tgt)>0: out['corr']=float(np.corrcoef(num,tgt)[0,1])
    else: out['corr']=float('nan')
    return out

print()
print("="*74)
print("STEP 2. 36 cases, Track P (production kernel-density operator)")
print("="*74)
rows=[]
for pname,(fn,a,b) in PROFILES.items():
    for R0 in (5.0,10.0,20.0):
        Rc=R0+h_in+0.5*H
        f=lambda rr,fn=fn,Rc=Rc: fn(rr-Rc) if False else fn(rr)
        # rebuild the profile with the current Rc
        if pname=="A_linear": g=lambda rr,Rc=Rc: 0.90+0.0225*(rr-Rc)
        elif pname=="B_quadratic": g=lambda rr,Rc=Rc: 0.90+0.0225*(rr-Rc)+0.0015*(rr-Rc)**2
        elif pname=="C_pure_quadratic": g=lambda rr,Rc=Rc: 0.90+0.0015*(rr-Rc)**2
        else: g=lambda rr,Rc=Rc: np.full_like(rr,0.90)
        z,r=build(g,R0,h_in,H,Lz)
        for resname,h in (("coarse",2.0),("medium",2.8),("fine",3.6)):
            rho,gz,gr,lap=prod_fields(z,r,h,Lz)
            sel=(r>R0+h_in+2*h)&(r<R0+h_in+H-2*h)
            if not np.any(sel): continue
            tgt=g(r[sel]); C=rho[sel]/tgt
            Cm=float(np.mean(C)); Cs=float(np.std(C))
            geom=gr[sel]/r[sel]
            geom_t=Cm*(a+2*b*(r[sel]-Rc))/r[sel]
            Lp_t=Cm*2*b*np.ones(np.count_nonzero(sel))
            Lax_t=Lp_t+geom_t
            m_lp=metrics(lap[sel],Lp_t); m_ax=metrics(lap[sel]+geom,Lax_t)
            E_ax=m_ax['rmse']; E_pl=m_lp['rmse']
            Q=float(np.mean(geom/geom_t)) if np.any(geom_t!=0) else float('nan')
            rows.append(dict(profile=pname,R0=R0,res=resname,h=h,N=len(z),
                             n_int=int(np.count_nonzero(sel)),
                             C_mean=Cm,C_std=Cs,
                             lap_mean=float(np.mean(lap[sel])),lap_tgt=float(np.mean(Lp_t)),
                             geom_mean=float(np.mean(geom)),geom_tgt=float(np.mean(geom_t)),
                             Q=Q,rmse_axi=E_ax,nrmse_axi=m_ax['nrmse'],
                             rmse_plan=E_pl,corr=m_ax['corr'],neg_ctrl=float(np.max(np.abs(geom)))))
keys=["profile","R0","res","h","N","n_int","C_mean","C_std","lap_mean","lap_tgt",
      "geom_mean","geom_tgt","Q","rmse_axi","nrmse_axi","rmse_plan","corr","neg_ctrl"]
import os
os.makedirs(r"E:\哈哈\_tmp_probe\n22", exist_ok=True)
with open(r"E:\哈哈\_tmp_probe\n22\trackP_summary.csv","w") as fh:
    fh.write(",".join(keys)+"\n")
    for w in rows: fh.write(",".join(str(w[k]) for k in keys)+"\n")
print("  wrote trackP_summary.csv (%d rows)" % len(rows))
print()
print("  %-18s %4s %-7s %7s %8s %9s %9s %9s %8s" %
      ("profile","R0","res","C_mean","C_std","Q","lap/C2b","geom/tgt","corr"))
for w in rows:
    lp_ratio = (w["lap_mean"]/w["lap_tgt"]) if w["lap_tgt"] not in (0.0,) else float('nan')
    gm_ratio = (w["geom_mean"]/w["geom_tgt"]) if w["geom_tgt"] not in (0.0,) else float('nan')
    print("  %-18s %4.0f %-7s %7.4f %8.1e %9.4f %9.4f %9.4f %8.4f" %
          (w["profile"],w["R0"],w["res"],w["C_mean"],w["C_std"],w["Q"],lp_ratio,gm_ratio,w["corr"]))
