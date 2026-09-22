"""N2.4 final clean re-run and verdict.

=================================================================
FROZEN CONVENTION 1 -- SIGN
    production :  e_ij = (x_i - x_j) / r_ij        (C++ neighborhood.e_ij_)
                  G_i  = Sum_j W'(r_ij) e_ij
    verified by particle-position FD :  G_i = grad_{x_i} rho_i
    EVERY diagnostic below uses that same direction.  A sign flip of e_ij is
    a bug, not a physical finding (that is what caused the N2/N2.2 retraction).
=================================================================
FROZEN CONVENTION 2 -- NORMALISATION
    A derivative may only be normalised by a constant belonging to the SAME
    reconstructed field.  Three fields are reported in separate columns:
        particle-sampled      rho_i      = Sum_j W        (C ~ 0.68)
        Eulerian reconstructed rho_probe(r)                (C ~ 1.00)
        continuous target     rho_target(r)  (given)
    The particle-sampled C is NEVER used to normalise an Eulerian slope.
    PRIMARY metric: the derivative compared against the ENVELOPE derivative of
    its own field (fixed low-pass rule, declared a priori: Gaussian, sigma=0.5d).
=================================================================
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

def build(rho_fn,R0,h_in,H,Lz):
    zs=[];rs=[];row=0;rr=R0+h_in;top=R0+h_in+H
    while rr<=top+1e-12:
        dd=math.sqrt(2.0/(math.sqrt(3.0)*float(rho_fn(np.array([rr]))[0])))
        n=max(3,int(round(Lz/dd)));dz=Lz/n;sh=0.5*dz if row%2 else 0.0
        for k in range(n): zs.append((sh+k*dz)%Lz);rs.append(rr)
        row+=1;rr+=0.5*math.sqrt(3.0)*dd
    return np.array(zs),np.array(rs)

def fields(z,r,h,Lz):
    """rho (particle), grad rho (correct sign), planar Laplacian. All exact
    kernel derivatives; e_ij = (x_i - x_j)/s as frozen above."""
    n=len(z);rho=np.zeros(n);gr=np.zeros(n);lap=np.zeros(n)
    for k in range(n):
        dz=z-z[k];dz-=Lz*np.round(dz/Lz);dr=r-r[k]
        s=np.sqrt(dz*dz+dr*dr);m=(s>0)&(s<h)
        if not np.any(m): continue
        ss=s[m];er=(dr[m]/ss)               # (r_j-r_k)/s
        rho[k]=float(np.sum(W(ss,h)))
        gr[k]=float(np.sum(dW(ss,h)*(-er)))  # e_ij=(x_k-x_j)/s  -> -er
        lap[k]=float(np.sum(ddW(ss,h)+dW(ss,h)/ss))
    return rho,gr,lap

def probe(z,r,rg,h,Lz,nz=16):
    zp=np.linspace(0.0,Lz,nz,endpoint=False)
    rp=np.zeros(len(rg));dp=np.zeros(len(rg));lp=np.zeros(len(rg))
    for z0 in zp:
        for a,x in enumerate(rg):
            dz=z-z0;dz-=Lz*np.round(dz/Lz);dr=r-x
            s=np.sqrt(dz*dz+dr*dr);m=(s>0)&(s<h)
            if not np.any(m): continue
            ss=s[m];er=(dr[m]/ss)            # (r_j-x)/s
            rp[a]+=float(np.sum(W(ss,h)))
            dp[a]+=float(np.sum(dW(ss,h)*(-er)))
            lp[a]+=float(np.sum(ddW(ss,h)+dW(ss,h)/ss))
    return rp/nz,dp/nz,lp/nz

def smooth(y,x,sig):
    o=np.zeros_like(y)
    for i in range(len(y)):
        w=np.exp(-0.5*((x-x[i])/sig)**2);o[i]=np.sum(w*y)/np.sum(w)
    return o

def stat(num,tgt):
    num=np.asarray(num,float);tgt=np.asarray(tgt,float);e=num-tgt
    rmse=float(np.sqrt(np.mean(e**2)));rms=float(np.sqrt(np.mean(tgt**2)))
    corr=float(np.corrcoef(num,tgt)[0,1]) if np.std(num)>0 and np.std(tgt)>0 else float('nan')
    return rmse,(rmse/rms if rms>0 else float('nan')),corr

PROF={"U_uniform":(0.0,0.0),"A_linear":(0.0225,0.0),
      "B_quadratic":(0.0225,0.0015),"C_pure_quadratic":(0.0,0.0015)}
Lz,H,h_in=40.0,16.0,0.5

def case(pname,R0,h):
    a,b=PROF[pname]; H_use=max(H,6.0*h); Rc=R0+h_in+0.5*H_use
    tgt_fn=lambda rr,Rc=Rc,a=a,b=b: 0.90+a*(rr-Rc)+b*(rr-Rc)**2
    z,r=build(tgt_fn,R0,h_in,H_use,Lz)
    d=math.sqrt(2.0/(math.sqrt(3.0)*0.90))
    rg=np.linspace(R0+h_in+0.2*h,R0+h_in+H_use-0.2*h,240)
    rho_p,drho_p,lap_p=probe(z,r,rg,h,Lz)
    rho,gr, lap=fields(z,r,h,Lz)
    sel=(r>R0+h_in+2*h)&(r<R0+h_in+H_use-2*h)
    if not np.any(sel): return None
    pr=r[sel]
    # envelopes of the two reconstructed fields (fixed rule: Gaussian sigma=0.5d)
    env_p=smooth(rho_p,rg,0.5*d); denv_p=smooth(drho_p,rg,0.5*d)
    # --- particle field, ROW-AVERAGED -------------------------------------
    # Every particle in one radial row shares exactly the same r, and every row
    # is a full z-line, so averaging within a row is the exact analogue of the
    # Eulerian probe's z-average.  Without it np.gradient sees duplicate abscissae.
    uniq,inv=np.unique(np.round(r,9),return_inverse=True)
    row_cnt=np.bincount(inv,minlength=len(uniq))
    row_rho=np.bincount(inv,weights=rho,minlength=len(uniq))/row_cnt
    row_gr =np.bincount(inv,weights=gr ,minlength=len(uniq))/row_cnt
    row_lap=np.bincount(inv,weights=lap,minlength=len(uniq))/row_cnt
    row_sel=((uniq>R0+h_in+2*h)&(uniq<R0+h_in+H_use-2*h))&(row_cnt>10)
    ur=uniq[row_sel]
    denv_particle=np.gradient(smooth(row_rho,uniq,0.5*d),uniq)[row_sel]
    gr_sel=row_gr[row_sel]; lap_sel=row_lap[row_sel]
    rho_sel=row_rho[row_sel]
    drho_sel=np.interp(ur,rg,drho_p); dprobe_env=np.interp(ur,rg,denv_p)
    pr=ur
    # normalisations of the two fields (reported separately, never mixed)
    C_p=float(np.mean(rho_sel/tgt_fn(pr))); C_e=float(np.mean(np.interp(pr,rg,rho_p)/tgt_fn(pr)))
    tg= tgt_fn(pr); tgt_grad=a+2*b*(pr-Rc); tgt_lap=2*b*np.ones_like(pr)
    Q_part=float(np.mean(gr_sel/denv_particle)) if np.any(denv_particle!=0) else float('nan')
    Q_probe=float(np.mean(drho_sel/dprobe_env)) if np.any(dprobe_env!=0) else float('nan')
    # gradient vs CONTINUOUS target, each field normalised by its own C
    rmseA,nrmseA,corrA=stat(gr_sel, C_p*tgt_grad)
    rmseB,nrmseB,corrB=stat(drho_sel, C_e*tgt_grad)
    # planar / geometry / total (B and C)
    geo_sel=gr_sel/pr; geo_t=C_p*tgt_grad/pr
    rmL,nrmL,corrL=stat(lap_sel, C_p*tgt_lap)
    rmG,nrmG,corrG=stat(geo_sel, geo_t)
    rmT,nrmT,corrT=stat(lap_sel+geo_sel, C_p*(tgt_lap+tgt_grad/pr))
    # --- J control in MODEL-SELECTION form --------------------------------
    # J = r/R0 -> the model's operator is L_axi = L_planar + rho_r/r
    # J = 1    -> the model's operator is L_planar
    # For each operator, ask WHICH target it fits better.
    rms=lambda v: float(np.sqrt(np.mean(np.asarray(v,float)**2)))
    L_ax=lap_sel+geo_sel; L_pl=lap_sel
    T_ax=C_p*(tgt_lap+tgt_grad/pr); T_pl=C_p*tgt_lap
    E_ax_correct=rms(L_ax-T_ax); E_ax_wrong=rms(L_ax-T_pl)
    E_pl_correct=rms(L_pl-T_pl); E_pl_wrong=rms(L_pl-T_ax)
    # --- Profile C: same-sign segmented statistics -------------------------
    # Threshold DECLARED A PRIORI: keep points with |target| > 10% of max|target|
    ZERO_FRAC=0.10
    tvec=tgt_grad/pr
    tmax=float(np.max(np.abs(tvec))) if len(tvec) else 0.0
    seg_ok=(np.abs(tvec)>ZERO_FRAC*tmax) if tmax>0 else np.zeros(len(tvec),bool)
    if np.count_nonzero(seg_ok)>2:
        seg_rmse,seg_nrmse,seg_corr=stat(geo_sel[seg_ok],geo_t[seg_ok])
    else:
        seg_rmse=seg_nrmse=seg_corr=float('nan')
    seg_frac=float(np.count_nonzero(seg_ok))/max(len(tvec),1)
    return dict(profile=pname,R0=R0,h=h,h_over_d=h/d,N=len(z),n_int=int(np.count_nonzero(row_sel)),
                C_particle=C_p,C_euler=C_e,
                Q_particle=Q_part,Q_probe=Q_probe,
                rmse_grad_part=rmseA,nrmse_grad_part=nrmseA,corr_grad_part=corrA,
                rmse_grad_probe=rmseB,nrmse_grad_probe=nrmseB,corr_grad_probe=corrB,
                planar_num=float(np.mean(lap_sel)),planar_tgt=float(np.mean(C_p*tgt_lap)),
                geom_num=float(np.mean(geo_sel)),geom_tgt=float(np.mean(geo_t)),
                rmse_planar=rmL,nrmse_planar=nrmL,rmse_geom=rmG,nrmse_geom=nrmG,
                rmse_total=rmT,nrmse_total=nrmT,corr_total=corrT,
                E_axi=E_ax_correct,E_plan=E_pl_correct,
                E_axi_wrong=E_ax_wrong,E_plan_wrong=E_pl_wrong,
                ratio_axi=E_ax_wrong/E_ax_correct if E_ax_correct>0 else float('nan'),
                ratio_plan=E_pl_wrong/E_pl_correct if E_pl_correct>0 else float('nan'),
                seg_rmse=seg_rmse,seg_nrmse=seg_nrmse,seg_corr=seg_corr,
                seg_frac=seg_frac,
                abs_grad=float(np.sqrt(np.mean(gr_sel**2))),
                abs_lap=float(np.sqrt(np.mean(lap_sel**2))),
                abs_geom=float(np.sqrt(np.mean(geo_sel**2))))

def show(rows,title):
    print("\n"+"="*100); print(title); print("="*100)
    for o in rows:
        if o is None: continue
        print("  %-18s R0=%4.1f h/d=%5.2f  n=%4d | C_part=%.3f C_eul=%.3f | "
              "Q_part=%7.4f Q_probe=%7.4f | nr_gradP=%.3f nr_gradE=%.3f"
              % (o["profile"],o["R0"],o["h_over_d"],o["n_int"],o["C_particle"],
                 o["C_euler"],o["Q_particle"],o["Q_probe"],
                 o["nrmse_grad_part"],o["nrmse_grad_probe"]))

d0=math.sqrt(2.0/(math.sqrt(3.0)*0.90))
res={"coarse":2.0*d0,"medium":2.8*d0,"fine":3.6*d0}
import os
os.makedirs(r"E:\哈哈\_tmp_probe\n24",exist_ok=True)
allrows=[]
for pn in ("U_uniform","A_linear","B_quadratic","C_pure_quadratic"):
    for R0 in (5.0,10.0,20.0):
        for rn,h in res.items():
            o=case(pn,R0,h)
            if o is None: continue
            o["res"]=rn; allrows.append(o)
keys=["profile","R0","res","h","h_over_d","N","n_int","C_particle","C_euler",
      "Q_particle","Q_probe","rmse_grad_part","nrmse_grad_part","corr_grad_part",
      "rmse_grad_probe","nrmse_grad_probe","corr_grad_probe",
      "planar_num","planar_tgt","geom_num","geom_tgt","rmse_planar","nrmse_planar",
      "rmse_geom","nrmse_geom","rmse_total","nrmse_total","corr_total",
      "E_axi","E_plan","E_axi_wrong","E_plan_wrong","ratio_axi","ratio_plan",
      "seg_rmse","seg_nrmse","seg_corr","seg_frac","abs_grad","abs_lap","abs_geom"]
with open(r"E:\哈哈\_tmp_probe\n24\final_summary.csv","w") as fh:
    fh.write(",".join(keys)+"\n")
    for o in allrows: fh.write(",".join(str(o[k]) for k in keys)+"\n")
print("wrote final_summary.csv: %d rows"%len(allrows))

show([o for o in allrows if o["profile"]=="A_linear"],
     "PROFILE A (linear): sign, 1/r trend, convergence  [PRIMARY]")
show([o for o in allrows if o["profile"]=="U_uniform"],
     "PROFILE U (uniform negative control): must be ~0")
print("  (uniform has zero target, so RATIOS are meaningless; report ABSOLUTE values)")
for o in allrows:
    if o["profile"]!="U_uniform": continue
    print("    R0=%4.1f %-7s |grad|rms=%.3e |lap|rms=%.3e |geom|rms=%.3e  (C_part=%.3f C_eul=%.3f)"
          % (o["R0"],o["res"],o["abs_grad"],o["abs_lap"],o["abs_geom"],
             o["C_particle"],o["C_euler"]))
print("\n"+"="*100); print("J CONTROL (formal)"); print("="*100)
for o in allrows:
    if o["profile"]=="U_uniform": continue
    print("  %-18s R0=%4.1f %-7s | J=r/R0: E_correct=%9.3e E_wrong=%9.3e ratio=%5.3f"
          " | J=1: E_correct=%9.3e E_wrong=%9.3e ratio=%5.3f"
          % (o["profile"],o["R0"],o["res"],o["E_axi"],o["E_axi_wrong"],o["ratio_axi"],
             o["E_plan"],o["E_plan_wrong"],o["ratio_plan"]))
print("\n"+"="*100); print("PROFILE B: planar / geometry / total"); print("="*100)
for o in allrows:
    if o["profile"]!="B_quadratic": continue
    print("  R0=%4.1f %-7s planar %.5f/%.5f  geom %+.5f/%+.5f  total nrmse=%.3f"
          % (o["R0"],o["res"],o["planar_num"],o["planar_tgt"],
             o["geom_num"],o["geom_tgt"],o["nrmse_total"]))
