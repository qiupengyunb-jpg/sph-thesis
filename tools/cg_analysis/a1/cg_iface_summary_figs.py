"""Summary figures: shape relaxation, radial profile, fluctuation spectrum."""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cg_iface import (bulk_density, contour_metrics, density_field, load_vtp,
                      radial_profile, read_params)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Case lists for the two figure groups.  Override on the command line:
#   python cg_iface_summary_figs.py <root> [relax cases] [profile cases]
DEFAULT_RELAX = ["A0_disc_N600", "A1_ellipse_N600", "A2_rect_N600", "A4_ellipse_dpp4"]
DEFAULT_PROFILE = ["A0_disc_N600", "D1_disc_N200", "D2_disc_N400", "D3_disc_N1200"]
LINE_COLORS = ["#1f4e79", "#2e9e5b", "#c0392b", "#d68910", "#7d3c98", "#117a65"]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("usage: python cg_iface_summary_figs.py <results-root> "
              "[relax-cases(comma separated)] [profile-cases(comma separated)]")
        print("       needs <results-root>/iface_series.csv from cg_iface.py")
        raise SystemExit(2)
    ROOT = Path(sys.argv[1])
    if not ROOT.is_dir():
        raise SystemExit("not a directory: %s" % ROOT)
    series = list(csv.DictReader(open(ROOT / "iface_series.csv", encoding="utf-8")))
    by = {}
    for r in series:
        by.setdefault(r["case"], []).append(r)

    # ---- figure 1: shape relaxation ----
    show = (sys.argv[2].split(",") if len(sys.argv) > 2 else DEFAULT_RELAX)
    show = [c for c in show if c in by] or [c for c in sorted(by)]
    colors = {c: LINE_COLORS[i % len(LINE_COLORS)] for i, c in enumerate(show)}
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.0), dpi=200)
    for key, lab, ylab in (("C", "圆度 C = 4πA/P²", "C"),
                           ("AR", "长短轴比 AR", "AR"),
                           ("P", "周长 P（σ）", "P (σ)"),
                           ("A", "面积 A（σ²）", "A (σ²)")):
        ax = axes[["C", "AR", "P", "A"].index(key)]
        for nm in show:
            t = [float(r["time"]) for r in by[nm]]
            y = [float(r[key]) for r in by[nm]]
            ax.plot(t, y, lw=1.6, color=colors[nm], label=nm.split("_")[0])
        ax.set_xlabel("时间 t"); ax.set_ylabel(ylab); ax.grid(alpha=0.25)
        if key == "C":
            ax.axhline(1.0, ls="--", color="0.4", lw=1.0)
            ax.text(100, 0.99, "理想圆 C=1", fontsize=8, color="0.4")
        if key == "AR":
            ax.axhline(1.0, ls="--", color="0.4", lw=1.0)
        if key == "A":
            ax.legend(fontsize=8)
    fig.suptitle("实验 A：自由凝聚体的形状松弛（无纤维；等密度轮廓定义）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(ROOT / "fig_iface_relax.png"); plt.close(fig)

    # ---- figure 2: radial profile (normalised) ----
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), dpi=200)
    prof_cases = (sys.argv[3].split(",") if len(sys.argv) > 3 else DEFAULT_PROFILE)
    prof_cases = [c for c in prof_cases if c in by] or [c for c in sorted(by)]
    for nm in prof_cases:
        d = ROOT / nm
        p = read_params(d)
        lx, ly = float(p["domain_length"]), float(p["domain_height"])
        out = d / ("output_" + nm)
        fr = []
        with open(d / "selfcheck.csv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                f = out / ("CGParticles_ite_%010d.vtp" % int(r["steps"]))
                if f.exists():
                    fr.append((float(r["time"]), f))
        profs = []
        for t, f in [x for x in fr if x[0] > 0.9 * fr[-1][0]][::max(1, len(fr) // 10)]:
            pos = load_vtp(f, ("Position",))["Position"]
            xs, ys, rho = density_field(pos, lx, ly)
            rb = bulk_density(rho)
            m = contour_metrics(xs, ys, rho, 0.5 * rb)
            if m is None:
                continue
            r, prof = radial_profile(pos, lx, ly, (m["cx"], m["cy"]))
            profs.append(np.interp(r, r, prof) / rb)
            R = m["R_eq"]
        prof = np.mean(np.array(profs), axis=0)
        axes[0].plot(r, prof, lw=1.8, label="%s (R=%.1fσ)" % (nm.split("_")[0][:2], R))
        axes[1].plot(r / R, prof, lw=1.8, label="%s (R=%.1fσ)" % (nm.split("_")[0][:2], R))
    for ax, xl in ((axes[0], "r（σ）"), (axes[1], "r / R_eq")):
        ax.axhline(1.0, ls="--", color="0.4", lw=1.0)
        ax.axhline(0.5, ls=":", color="0.6", lw=1.0)
        ax.set_xlabel(xl); ax.set_ylabel("ρ(r) / ρ_bulk"); ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.6)
    axes[1].set_xlim(0, 1.6); axes[0].set_xlim(0, 30)
    fig.suptitle("实验 B：径向密度剖面（虚线＝体相平台，点线＝界面半高＝等密度轮廓位置）", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(ROOT / "fig_iface_profile.png"); plt.close(fig)

    # ---- figure 3: fluctuation spectrum ----
    sp = list(csv.DictReader(open(ROOT / "iface_spectrum.csv", encoding="utf-8")))
    byq = {}
    for r in sp:
        byq.setdefault(r["case"], []).append((int(r["q"]), float(r["S"])))
    fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=200)
    for nm, v in sorted(byq.items()):
        v.sort()
        q = np.array([a for a, _ in v], dtype=float)
        S = np.array([b for _, b in v])
        m = q >= 2
        ax.loglog(q[m], S[m], "o-", ms=3.5, lw=1.2, label=nm)
    qq = np.array([2, 20.0])
    ax.loglog(qq, 6.0 / qq ** 2, "k--", lw=1.4, label="毛细波 1/q²（参考斜率）")
    ax.set_xlabel("角向模数 q"); ax.set_ylabel("⟨|r$_q$|²⟩（σ²）")
    ax.set_title("实验 A 补充：界面起伏的角向谱（斜率≈−2 即线张力指纹）", fontsize=10)
    ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(ROOT / "fig_iface_spectrum.png"); plt.close(fig)
    print("wrote fig_iface_relax.png, fig_iface_profile.png, fig_iface_spectrum.png")


if __name__ == "__main__":
    main()
