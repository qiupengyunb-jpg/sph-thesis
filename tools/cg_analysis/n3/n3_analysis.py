"""N3 analysis: axisymmetric Level-B ground-state stability, thermal balance,
spurious radial drive.  Reads only files the solver already wrote.

Interior band (away from the fibre wall and the outer boundary):
    R0 + 0.5 + 2 h_rho  <  r  <  Ly - 2 h_rho
"""
import glob
import math
import os
import re

import numpy as np

HP = 2.8           # production h_rho
OUT = r"E:\哈哈\_tmp_probe"

def vtp(path):
    t = open(path).read()
    p = re.search(r'<DataArray Name="Position"[^>]*>(.*?)</DataArray>', t, re.S)
    v = re.search(r'<DataArray Name="Velocity"[^>]*>(.*?)</DataArray>', t, re.S)
    P = np.fromstring(p.group(1), sep=" ").reshape(-1, 3)
    V = np.fromstring(v.group(1), sep=" ").reshape(-1, 3)
    return P, V

def frames(run, tag):
    fs = glob.glob(os.path.join(OUT, run, "output_"+tag, "CGParticles_ite_*.vtp"))
    return sorted(fs, key=lambda f: int(re.search(r"_ite_(\d+)", f).group(1)))

def stats(x):
    x = np.asarray(x, float)
    return (float(np.mean(x)), float(np.median(x)), float(np.sqrt(np.mean(x**2))),
            float(np.std(x)))

print("="*100)
print("N3-A/B  UNIFORM POINT SET, T = 0, STATIC FORCE  (lambda=12, no Morse, no wall)")
print("="*100)
for run, R0 in (("n3_static_R5",5.0),("n3_static_R5_J1",5.0),
                ("n3_static_R10",10.0),("n3_static_R20",20.0)):
    p = os.path.join(OUT, run, "sg_force_split.csv")
    if not os.path.exists(p):
        print("  %-18s (no dump)" % run); continue
    D = np.genfromtxt(p, delimiter=",", names=True)
    z, r = D["z"], D["r"]
    Ly = 30.0 if "R5" in run else 60.0
    sel = (r > R0+0.5+2*HP) & (r < Ly-2*HP)
    Frc, Fre, Frt, Fzt = (D["Fr_chain"][sel], D["Fr_explicit_J"][sel],
                          D["Fr_total"][sel], D["Fz_total"][sel])
    g2 = D["grad2"][sel]
    slope = np.polyfit(r[sel], Frt, 1)[0]
    print("  %-18s N=%5d interior=%5d | mean Fr_chain=%+.3e Fr_expl=%+.3e "
          "Fr_tot=%+.3e" % (run, len(z), int(sel.sum()),
                            float(np.mean(Frc)), float(np.mean(Fre)),
                            float(np.mean(Frt))))
    print("      RMS: Fr_chain=%.3e Fr_expl=%.3e Fr_tot=%.3e Fz_tot=%.3e | "
          "max|Fz|=%.3e | Fr(r) slope=%+.3e | mean|G|^2=%.3e"
          % (float(np.sqrt(np.mean(Frc**2))), float(np.sqrt(np.mean(Fre**2))),
             float(np.sqrt(np.mean(Frt**2))), float(np.sqrt(np.mean(Fzt**2))),
             float(np.max(np.abs(Fzt))), slope, float(np.mean(g2))))

print()
print("="*100)
print("N3-C  T = 0 SHORT-TIME DRIFT  (only the Level-B interface term active)")
print("="*100)
for run, tag, R0, Ly in (("n3_drift_T0","n3_drift_T0",5.0,30.0),):
    fs = frames(run, tag)
    print("  frames=%d" % len(fs))
    print("     t     <r>      median r   rad.var    inner  mid   outer   N")
    inner=(R0+0.5, R0+0.5+(Ly-R0-0.5)/3)
    mid=(inner[1], inner[1]+(Ly-R0-0.5)/3)
    outer=(mid[1], Ly)
    for f in fs[::max(1,len(fs)//10)]:
        P,_ = vtp(f)
        r = P[:,1]
        ni=int(np.count_nonzero((r>=inner[0])&(r<inner[1])))
        nm=int(np.count_nonzero((r>=mid[0])&(r<mid[1])))
        no=int(np.count_nonzero((r>=outer[0])&(r<=outer[1])))
        t = float(re.search(r"_ite_(\d+)", f).group(1))
        print("  %6d %9.5f %10.5f %9.5f  %5d %5d %5d  %5d"
              % (t, float(np.mean(r)), float(np.median(r)),
                 float(np.var(r)), ni, nm, no, len(r)))

print()
print("="*100)
print("N3-D/E  T = 1  THERMAL BALANCE, T(r), vr^2, vz^2   (J=r/R0 vs J=1)")
print("="*100)
for run, tag in (("n3_therm_T1","n3_therm_T1"),("n3_therm_T1_J1","n3_therm_T1_J1")):
    fs = frames(run, tag)
    half = fs[len(fs)//2:]
    R=[];V=[]
    for f in half:
        P,Vv = vtp(f); R.append(P[:,1]); V.append(Vv)
    r=np.concatenate(R); v=np.concatenate(V)
    R0=5.0; Ly=30.0
    lo,hi = R0+0.5+2*HP, Ly-2*HP
    edges=np.linspace(lo,hi,9)
    v2=v[:,0]**2+v[:,1]**2
    Tk=v2/(2*1.0)          # 2D, m=1, k_B=1
    overall=float(np.mean(Tk))
    print("  --- %s : frames used=%d  N=%d  overall T_kin=%.4f" % (run,len(half),len(r),overall))
    print("       r-bin            count   T_kin    <vr^2>    <vz^2>   T/Tbar-1")
    cs=[];ts=[]
    for k in range(8):
        m=(r>=edges[k])&(r<edges[k+1])
        if np.count_nonzero(m)<50: continue
        cs.append(0.5*(edges[k]+edges[k+1])); ts.append(float(np.mean(Tk[m])))
        print("     [%5.2f,%5.2f)   %6d %8.4f %9.4f %9.4f %+8.2f%%"
              % (edges[k],edges[k+1],int(np.count_nonzero(m)),float(np.mean(Tk[m])),
                 float(np.mean(v[m,0]**2)),float(np.mean(v[m,1]**2)),
                 100*(float(np.mean(Tk[m]))/overall-1)))
    if len(cs)>2:
        A=np.polyfit(cs,ts,1)
        pred=np.polyval(A,cs)
        r2=1-((np.array(ts)-pred)**2).sum()/((np.array(ts)-np.mean(ts))**2).sum()
        print("     T(r) linear slope=%+.3e /sigma (%.2f%%/sigma)  R2=%.3f"
              % (A[0], 100*A[0]/overall, r2))
