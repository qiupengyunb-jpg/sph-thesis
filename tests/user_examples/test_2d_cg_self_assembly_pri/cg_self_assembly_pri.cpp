/**
 * @file cg_self_assembly_pri.cpp
 * @brief Coarse-grained (mesoscopic) particle self-assembly prototype.
 *
 * ---------------------------------------------------------------------------
 * MODEL PROVENANCE (read before citing anything here)
 *   A-level, taken from the literature interface:
 *     * WCA (LJ truncated at the minimum, energy shifted) as the repulsion core
 *     * single-particle Langevin dynamics with a fluctuation-dissipation
 *       consistent thermostat  (m dv/dt = F - gamma v + sqrt(2 gamma k_B T) R)
 *     * Morse form for the short-range attraction (parameters below)
 *     * axial periodic boundary
 *   C-level, OUR implementation choices (NOT literature-given values):
 *     * Morse r0 = 1.15 sigma and a = 0.25 sigma
 *     * linear force shifting at the Morse cut-off
 *     * the reproduction rules of the discrete thermostat and of the
 *       random-number stream (see below)
 *   The particles are mesoscopic coarse-grained units.  They are NOT molecules.
 *
 *   SALR (medium-range repulsion), fibre adsorption, temperature sweeps,
 *   DPD/SDPD and any real-molecule parameter mapping are deliberately NOT
 *   implemented in this revision.
 * ---------------------------------------------------------------------------
 *
 * Reduced units: m = 1, sigma = 1, k_B T = 1 (all fixed for this revision),
 *   energy unit eps0 = k_B T, so the temperature is NOT a free parameter here.
 *
 * Geometry: 2D rectangle [0,Lx) x [0,Ly), periodic along x.  The rigid
 *   equal-diameter fibre is a horizontal strip of half-width a centred at
 *   y = Ly/2 whose axis is x, so it spans exactly one period.
 *
 * Integrator: BAOAB Langevin splitting (B half-kick, A half-drift, O exact
 *   Ornstein-Uhlenbeck velocity update, A half-drift, B half-kick).  The two
 *   half-kicks use the force before and after the drift, so exactly one force
 *   evaluation is needed per step.  The O step is exact for any dt:
 *       v <- c1 v + c2 xi,  c1 = exp(-gamma dt / m),
 *                           c2 = sqrt(k_B T (1 - c1^2) / m)
 *   which makes the measured kinetic temperature equal k_B T for any dt and
 *   therefore gives a directly verifiable fluctuation-dissipation check.
 *
 * Random numbers: the noise for particle i at step n is a pure function of
 *   (seed, OriginalID_i, n, component).  It uses a counter-based SplitMix64
 *   hash, never a shared stateful generator, so the result cannot depend on
 *   the TBB thread schedule.
 */

#include "sphinxsys.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <tbb/global_control.h>

using namespace SPH;

namespace
{
//=============================================================================
//  Run options (reduced units)
//=============================================================================
struct RunOptions
{
    int case_id = 0;
    int seed = 20260916;
    int threads = 0; /**< 0 keeps the library default */
    Real domain_length = 60.0;
    Real domain_height = 40.0;
    Real fibre_half_width = 5.0;
    // Fibre cross-section / extent.  "strip" is the original geometry: a flat
    // ribbon of half-width fibre_half_width that spans the whole x period, so
    // it has no curvature.  "cylinder" is a capsule: a straight segment of
    // length fibre_length with two half-disc caps of radius fibre_radius, i.e.
    // a finite-radius fibre of finite length whose caps DO have curvature.
    // The physics (both potentials, thermostat, units) is identical; only the
    // function h(x) that measures the distance to the fibre surface changes.
    std::string fibre_shape = "strip"; /**< strip | cylinder */
    Real fibre_radius = 5.0;           /**< cap radius of the cylinder fibre */
    Real fibre_length = 60.0;          /**< straight-segment length of it */
    // Initial condition.  "scatter" is the original random RSA dispersion.
    // "cluster" builds ONE pre-condensed aggregate on a triangular lattice so
    // that the relaxation of a NON-circular free cluster can be measured
    // (interface-tension test).  The lattice spacing is set by the requested
    // packing; exactly --cluster-count sites closest to the shape centre are
    // kept, so two different shapes hold exactly the same particle number.
    std::string init_mode = "scatter";  /**< scatter | cluster | lattice */
    std::string cluster_shape = "disc"; /**< disc | ellipse | rect */
    Real cluster_aspect = 2.5;          /**< long/short axis ratio */
    Real cluster_packing = 0.63;        /**< area fraction of the lattice */
    int cluster_count = 800;            /**< particles in the cluster */
    Real area_fraction = 0.20;
    Real resolution = 2.0; /**< SPH lattice points per unit length */

    Real sigma = 1.0;
    Real wca_epsilon = 1.0;

    Real morse_depth = 0.0;   /**< D / k_B T; 0 disables the attraction */
    Real morse_re = 1.0;      /**< r_e / sigma, the well position */
    Real morse_alpha = 3.0;   /**< alpha * sigma, the INVERSE length parameter */
    Real morse_cutoff = 3.0;  /**< r_c / sigma; the locked default is r_e + 6/alpha */

    Real temperature = 1.0; /**< k_B T in reduced units, fixed at 1 */
    Real friction = 1.0;    /**< gamma / sqrt(m eps0)/sigma, NOT yet calibrated */

    // Particle-fibre wall potential (Case 2).  h is the distance from the
    // particle CENTRE to the fibre SURFACE, so h = 0.5 sigma is geometric
    // contact.  h is never a surface-to-surface gap.
    Real wall_depth_d0 = 0.0;   /**< D0 == eps_pf / k_B T; 0 disables the wall */
    Real wall_alpha = 3.0;      /**< inverse length, 3/sigma */
    Real wall_re = 0.5;         /**< r0 / sigma, the well position = contact */
    Real wall_cutoff = 2.5;     /**< r_c / sigma */
    Real wall_guard_h = 0.10;   /**< numerical backstop when D0 > 0 */
    Real wall_contact_h = 0.50; /**< hard-wall location used when D0 = 0 */
    Real adsorption_cutoff_h = 1.5; /**< h_ads for f_ads */

    // ---- A-1 square-gradient interface free energy -------------------------
    //  F_sg = (lambda/2) Sum_i V0 |grad rho_i|^2,   rho_i = Sum_j W_h(r_ij)
    //  lambda = 0 is a TRUE zero path: both sweeps return before doing any
    //  floating point work, so the trajectory is bit-identical to the
    //  pre-A-1 baseline.  Nothing else in the physics is touched.
    Real interface_lambda = 0.0;   /**< lambda in k_B T sigma^4; 0 disables */
    Real interface_h = 2.0;        /**< kernel support h_rho / sigma */
    Real interface_rho_ref = 0.88; /**< V0 = 1/rho_ref, the volume per particle */

    // ---- M1 axisymmetric (z,r) scaffold ------------------------------------
    //  Stage M1.0 only INTRODUCES the switch.  "off" is the sole accepted
    //  value for now: every other value is rejected in ParseCommandLine, so
    //  no numeric path can change and the A-1/A-2 trajectories stay
    //  bit-identical.  The remaining fields are placeholders reserved for
    //  M1.1 (constant-radius geometry) and M1.2 (film initial state).
    //  Axisymmetric convention: z = fibre axis (periodic), r = radial
    //  coordinate measured from the fibre axis, wall distance h = r - R.
    std::string axisymmetric = "off"; /**< off | filmonly */
    Real axis_radius = 5.0;           /**< R0 / sigma, CONSTANT radius only */
    Real film_thickness = 2.5;        /**< h0 / sigma, uniform annular film */
    Real film_perturb_amp = 0.0;      /**< A / sigma, 0 = no axial modulation */
    int film_perturb_mode = 1;        /**< n in A sin(2 pi n z / Lz) */
    bool film_broadband = false;      /**< Stage 1: broadband axial seed */
    // ---- Stage 9: equilibrate-then-perturb workflow (EXPERIMENTAL, default off)
    //  When --perturb-after-equilibration=1 the M1.4 single-mode mapping is NOT
    //  applied at t=0; instead the film is equilibrated unperturbed until
    //  --equilibrate-until and the same mapping is then applied once to the
    //  CURRENT particle positions.  With the switch off (default) every
    //  existing path is bit-identical.
    bool perturb_after_equilibration = false;
    Real equilibrate_until = 0.0;     /**< t_eq in time units; 0 = no hook */
    // ---- Stage 21: opt-in two-scale Scheme E kernel (EXPERIMENTAL, default off)
    //  --interface-kernel=legacy (default) keeps the historical single-scale
    //  Wendland kernel bit-for-bit; two_scale uses
    //      K_new = [ c_h W_h - beta c_xi W_xi ] / (1 - beta)
    //  with c_h = 144/(5h^2), c_xi = 144/(5 xi^2) so that both terms and the
    //  combination satisfy Int|u|^2 K = 4 (lambda keeps its meaning).
    std::string interface_kernel = "legacy";
    Real interface_xi = 4.0;          /**< xi / sigma (long-range lobe support) */
    Real interface_beta = 0.3;        /**< mixing weight, 0 < beta < 1 */
    bool axisym_jacobian = false;     /**< explicit -kBT ln(2 pi r) measure */
    int sg_fd_check = 0; /**< M1.3/N1: run the energy-force FD check and exit */
    // ---- N3 diagnostic-only switches (all default OFF; none changes the
    //      production trajectory when left at its default) ------------------
    int sg_force_dump = 0;    /**< dump per-particle term(1)/term(2) force split */
    int sg_budget_every = 0;  /**< M1.4: write sg_budget.csv every N steps (0=off) */
    bool noise_off = false;   /**< T = 0 diagnostic: zero the OU noise amplitude */
    bool axisym_j_one = false;/**< force J = 1 in the (z,r) branch (J control) */
    // ---- M1.3R experimental interface-energy schemes (default = production) --
    //  moving_j        : Scheme A, (lambda V0 / 2) Sum_i J_i |G_i|^2   (frozen)
    //  difference_energy: Scheme E, (lambda/2) V0^2 Sum_{i<j} J_ij Kt_ij
    //                     (rho_i - rho_j)^2, Kt = c W, c = 144/(5 h^2)
    //  Scheme E is available only in the axisymmetric branch and is OFF by
    //  default, so every historical trajectory stays bit-identical.
    std::string axisym_interface_scheme = "moving_j";

    Real initial_separation = 1.0;
    Real exclusion_margin = 0.50;
    std::string initial_velocity = "maxwell";

    Real dt = 0.0; /**< 0 means automatic */
    Real dt_cfl = 0.30;
    Real dt_reference_separation = 0.92; /**< reference closest approach in sigma */
    Real end_time = 200.0;
    int frames = 100;
    bool y_reflect = true;
    std::string output_tag = "cg_case0";
    std::string git_commit = CG_GIT_COMMIT;
};

RunOptions run_options;

Real DomainLength() { return run_options.domain_length; }
Real DomainHeight() { return run_options.domain_height; }
Real FibreHalfWidth() { return run_options.fibre_half_width; }
Real FibreCentreY() { return 0.5 * DomainHeight(); }

// ---- M1.1 axisymmetric geometry (scaffold only) ---------------------------
//  Coordinate interpretation in the axisymmetric branch:
//      x -> z : axial coordinate, PERIODIC
//      y -> r : radial coordinate measured from the fibre axis
//  The fibre is R(z) = R0 = const and occupies r < R0, so the wall distance is
//      h_wall = r - R0 = y - R0
//  and the outward surface normal is ALWAYS +r.  Unlike the strip/capsule
//  shapes there is no second side and no end cap.  M1.1 adds NO physics: this
//  branch only supplies h and the normal to the pre-existing Morse wall force
//  and to the pre-existing geometric constraint.
inline bool Axisymmetric() { return run_options.axisymmetric != "off"; }
Real AxisRadius() { return run_options.axis_radius; }

Real FibreRadius()
{
    if (Axisymmetric())
        return AxisRadius();
    return (run_options.fibre_shape == "cylinder") ? run_options.fibre_radius
                                                   : FibreHalfWidth();
}
Real FibreLength()
{
    return (run_options.fibre_shape == "cylinder") ? run_options.fibre_length
                                                   : DomainLength();
}
Real ExclusionRadius() { return FibreRadius() + run_options.exclusion_margin; }
Real Sigma() { return run_options.sigma; }
Real WCACutoff() { return std::pow(2.0, 1.0 / 6.0) * Sigma(); }

Real LargestPairCutoff();

/**
 * Centre-to-surface distance h together with the outward surface normal.
 * h < 0 means the point is inside the fibre.  For the flat strip the normal is
 * along y; for the capsule it is radial on the shaft and points away from the
 * nearest cap centre outside it.
 */
struct FibreContact
{
    Real h;
    Vecd normal;
};

inline Real PeriodicOffsetX(Real x)
{
    const Real lx = DomainLength();
    Real dx = x - 0.5 * lx;
    dx -= lx * std::floor(dx / lx + 0.5);
    return dx;
}

inline FibreContact FibreContactOf(const Vecd &p)
{
    const Real dy = p[1] - FibreCentreY();
    FibreContact c;
    if (run_options.fibre_shape == "none")
    {
        c.h = 1.0e9;          // no fibre at all: no exclusion, no wall force
        c.normal = Vecd(0.0, 1.0);
        return c;
    }
    if (Axisymmetric())
    {
        // Constant-radius cylinder: h is independent of z (no radius gradient,
        // no end cap) and the normal is the radial unit vector, never -r.
        c.h = p[1] - AxisRadius();
        c.normal = Vecd(0.0, 1.0);
        return c;
    }
    if (run_options.fibre_shape == "cylinder")
    {
        const Real dx = PeriodicOffsetX(p[0]);
        const Real half_length = 0.5 * FibreLength();
        const Real s = dx - std::min(std::max(dx, -half_length), half_length);
        const Real d = std::sqrt(s * s + dy * dy);
        c.h = d - FibreRadius();
        if (d > 1.0e-12)
            c.normal = Vecd(s / d, dy / d);
        else
            c.normal = Vecd(0.0, 1.0);
        return c;
    }
    c.h = std::abs(dy) - FibreHalfWidth();
    c.normal = Vecd(0.0, (dy >= 0.0) ? 1.0 : -1.0);
    return c;
}

/** Area occupied by the fibre (used only to convert area fraction to N). */
Real FibreArea()
{
    // Axisymmetric cross-section (z,r): the solid core r < R0 covers R0 * Lz.
    if (Axisymmetric())
        return AxisRadius() * DomainLength();
    if (run_options.fibre_shape == "none")
        return 0.0;
    if (run_options.fibre_shape == "cylinder")
        return FibreLength() * 2.0 * FibreRadius() + Pi * FibreRadius() * FibreRadius();
    return 2.0 * FibreHalfWidth() * DomainLength();
}

/** Perimeter that can be covered: the 2D "circumference" analogue. */
Real FibrePerimeter()
{
    // Physical wetted circumference of the cylinder; informational only.
    if (Axisymmetric())
        return 2.0 * Pi * AxisRadius();
    if (run_options.fibre_shape == "cylinder")
        return 2.0 * FibreLength() + 2.0 * Pi * FibreRadius();
    return 2.0 * DomainLength();
}

Real FreeArea()
{
    return DomainLength() * DomainHeight() - FibreArea();
}

int TargetParticleNumber()
{
    const Real disk_area = 0.25 * Pi * Sigma() * Sigma();
    const Real n = run_options.area_fraction * FreeArea() / disk_area;
    return std::max(1, static_cast<int>(std::llround(n)));
}

/**
 * SPH lattice spacing.  SPHinXsys builds neighbour lists with 2h = 2.6 dx, so
 * dx must satisfy 2.6 dx >= the largest pair cut-off, otherwise the neighbour
 * list would silently truncate the interaction.
 */
Real ParticleSpacing()
{
    const Real requested = 1.0 / run_options.resolution;
    const Real required = LargestPairCutoff() / 2.6;
    return std::max(requested, required);
}

//=============================================================================
//  Pair force: WCA repulsion + truncated, force-shifted Morse attraction
//=============================================================================
Real LargestPairCutoff()
{
    // The Morse cut-off is included even when D = 0 so that the whole batch
    // shares one lattice spacing, one cell size and one neighbour cut-off.
    // Otherwise Case 0 and Case 1 would run on different discretisations and
    // the D scan would no longer be a single-variable comparison.
    return std::max(WCACutoff(), run_options.morse_cutoff * Sigma());
}

/** Repulsive magnitude of the WCA core; positive means push the pair apart. */
Real WCAForce(Real r)
{
    const Real rc = WCACutoff();
    if (r >= rc || r <= TinyReal)
        return 0.0;
    const Real s = Sigma() / r;
    const Real s6 = s * s * s * s * s * s;
    return (24.0 * run_options.wca_epsilon / r) * (2.0 * s6 * s6 - s6);
}

/**
 * Morse potential, written with the INVERSE length parameter alpha (NOT the
 * length parameter a; alpha = 1/a, so the two must never be mixed up):
 *
 *     U_M(r) = D * [ exp(-2 alpha (r - r_e)) - 2 exp(- alpha (r - r_e)) ]
 *
 * Minimum -D at r = r_e; the half-width at U = -D/2 is 1.763 / alpha.
 * Locked first-version convention: alpha = 3/sigma, r_e = sigma,
 * r_c = r_e + 6/alpha = 3 sigma.
 *
 * Truncation: SHIFTED POTENTIAL (not force shifted),
 *     U_M,shifted(r) = U_M(r) - U_M(r_c)   for r < r_c,  else 0.
 * The potential is therefore continuous at r_c while the force steps to zero;
 * the residual force at r_c is recorded in parameters.csv so the size of that
 * step is auditable.
 */
Real MorseAlpha() { return run_options.morse_alpha / Sigma(); }
Real MorseRe() { return run_options.morse_re * Sigma(); }
Real MorseCutoff() { return run_options.morse_cutoff * Sigma(); }

Real MorsePotential(Real r)
{
    const Real d = run_options.morse_depth;
    if (d <= 0.0)
        return 0.0;
    const Real x = MorseAlpha() * (r - MorseRe());
    return d * (std::exp(-2.0 * x) - 2.0 * std::exp(-x));
}

/** Raw Morse force; positive pushes the pair apart. */
Real MorseRawForce(Real r)
{
    const Real d = run_options.morse_depth;
    if (d <= 0.0)
        return 0.0;
    const Real alpha = MorseAlpha();
    const Real x = alpha * (r - MorseRe());
    const Real ex = std::exp(-x);
    return 2.0 * d * alpha * ex * (ex - 1.0);
}

/** With potential shifting the force itself is the raw force, cut at r_c. */
Real MorseForce(Real r)
{
    const Real d = run_options.morse_depth;
    if (d <= 0.0)
        return 0.0;
    const Real rc = MorseCutoff();
    if (r >= rc)
        return 0.0;
    return MorseRawForce(r);
}

Real PairForce(Real r) { return WCAForce(r) + MorseForce(r); }

/** Total conservative pair potential, WCA + shifted Morse. */
Real PairPotential(Real r)
{
    Real u = 0.0;
    const Real eps = run_options.wca_epsilon;
    if (r < WCACutoff() && r > TinyReal)
    {
        const Real s = Sigma() / r;
        const Real s6 = s * s * s * s * s * s;
        u += 4.0 * eps * (s6 * s6 - s6) + eps;
    }
    if (run_options.morse_depth > 0.0 && r < MorseCutoff())
        u += MorsePotential(r) - MorsePotential(MorseCutoff());
    return u;
}

/**
 * eps_eff: the depth of the ACTUAL well of the combined potential, found by a
 * dense scan plus a parabolic refinement.  It differs from D because r_e = sigma
 * lies inside the WCA core, which pushes the combined minimum out to the WCA
 * cut-off.  Reporting both D/k_BT and eps_eff/k_BT keeps the two apart.
 */
Real EffectiveWellDepth()
{
    if (run_options.morse_depth <= 0.0)
        return 0.0;
    const Real lo = 0.6 * Sigma();
    const Real hi = std::max(MorseCutoff(), 1.2 * Sigma());
    const int samples = 20000;
    Real best_r = lo, best_u = PairPotential(lo);
    for (int k = 0; k <= samples; ++k)
    {
        const Real r = lo + (hi - lo) * static_cast<Real>(k) / samples;
        const Real u = PairPotential(r);
        if (u < best_u)
        {
            best_u = u;
            best_r = r;
        }
    }
    const Real h = 1.0e-4 * Sigma();
    for (int it = 0; it != 60; ++it)
    {
        const Real u_l = PairPotential(best_r - h);
        const Real u_c = PairPotential(best_r);
        const Real u_r = PairPotential(best_r + h);
        const Real d1 = (u_r - u_l) / (2.0 * h);
        const Real d2 = (u_r - 2.0 * u_c + u_l) / (h * h);
        if (std::abs(d2) < TinyReal)
            break;
        best_r -= d1 / d2;
    }
    return -PairPotential(best_r);
}

/** -dF/dr of the total pair force, by central difference at a reference r. */
Real PairStiffness(Real r)
{
    const Real h = 1.0e-4 * Sigma();
    return -(PairForce(r + h) - PairForce(r - h)) / (2.0 * h);
}

//=============================================================================
//  Particle-fibre Morse wall (Case 2)
//
//  h is the distance from the particle CENTRE to the fibre SURFACE.
//  With the fibre modelled as a strip of half width a centred at y = Ly/2,
//      h = |y - Ly/2| - a.
//  h = 0.5 sigma is therefore geometric contact for a particle of diameter
//  sigma, and that is where the Morse well sits (r0 = 0.5 sigma).
//
//      E_raw(h) = D0 [ exp(-2 alpha (h - r0)) - 2 exp(-alpha (h - r0)) ]
//      E(h)     = E_raw(h) - E_raw(r_c)                    (h < r_c), else 0
//      F(h)     = -dE/dh = 2 alpha D0 [ exp(-2 alpha (h-r0)) - exp(-alpha (h-r0)) ]
//                 positive = away from the wall (repulsion)
//
//  NOTE ON A SIGN TYPO IN THE LITERATURE INTERFACE: that document writes
//  F = -2 alpha D0 ( exp(-2x) - exp(-x) ), which is the negative of -dE/dh.
//  Its own analytic numbers ("repulsion at h = 0.4 sigma is +2.83 D0/sigma")
//  and the quoted LAMMPS source line both correspond to the expression used
//  here; with the printed minus sign the wall would repel where it must
//  attract.  The physical requirement (a well at h = r0, attraction for
//  h > r0) settles it.
//=============================================================================
Real WallAlpha() { return run_options.wall_alpha / Sigma(); }
Real WallRe() { return run_options.wall_re * Sigma(); }
Real WallCutoff() { return run_options.wall_cutoff * Sigma(); }

// M1.2: innermost film layer = the wall well position (stable contact
// distance), where the wall force is exactly zero.  Same semantics as PC2's
// A-5 helper of the same name, so the two film builders start from an
// identical h_in.  With the default wall_re = 0.5 sigma this is 0.5 sigma.
Real FilmInnerOffset()
{
    return (run_options.wall_depth_d0 > 0.0 ? run_options.wall_re
                                            : run_options.wall_contact_h) *
           Sigma();
}

/** Distance from the particle centre to the nearest fibre surface. */
inline Real WallDistance(const Vecd &p)
{
    return FibreContactOf(p).h;
}

Real WallRawPotential(Real h)
{
    const Real d0 = run_options.wall_depth_d0;
    if (d0 <= 0.0)
        return 0.0;
    const Real x = WallAlpha() * (h - WallRe());
    return d0 * (std::exp(-2.0 * x) - 2.0 * std::exp(-x));
}

/** Positive means push the particle away from the fibre. */
Real WallForce(Real h)
{
    const Real d0 = run_options.wall_depth_d0;
    if (d0 <= 0.0 || h >= WallCutoff())
        return 0.0;
    const Real alpha = WallAlpha();
    const Real x = alpha * (h - WallRe());
    return 2.0 * d0 * alpha * (std::exp(-2.0 * x) - std::exp(-x));
}

/** Depth of the shifted well, measured from the shifted zero at r_c. */
Real WallEffectiveDepth()
{
    if (run_options.wall_depth_d0 <= 0.0)
        return 0.0;
    return run_options.wall_depth_d0 + WallRawPotential(WallCutoff());
}

Real WallStiffness(Real h)
{
    const Real d = 1.0e-4 * Sigma();
    return -(WallForce(h + d) - WallForce(h - d)) / (2.0 * d);
}

//=============================================================================
//  Counter-based deterministic normal deviates
//=============================================================================
inline std::uint64_t SplitMix64(std::uint64_t z)
{
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline Real UniformFromBits(std::uint64_t bits)
{
    // 53 significant bits -> [0,1)
    return static_cast<Real>(bits >> 11) * (1.0 / 9007199254740992.0);
}

/**
 * Standard normal deviate for (seed, particle identity, step, component).
 * Pure function: no shared state, so it cannot depend on execution order.
 */
inline Real DeterministicGaussian(std::uint64_t seed, std::uint64_t id,
                                  std::uint64_t step, std::uint64_t component)
{
    const std::uint64_t base =
        SplitMix64(seed * 0x9E3779B97F4A7C15ULL +
                   id * 0xBF58476D1CE4E5B9ULL +
                   step * 0x94D049BB133111EBULL +
                   component * 0xD1B54A32D192ED03ULL);
    const std::uint64_t second = SplitMix64(base ^ 0xA0761D6478BD642FULL);
    Real u1 = UniformFromBits(base);
    const Real u2 = UniformFromBits(second);
    u1 = std::max(u1, 1.0e-300);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * Pi * u2);
}

//=============================================================================
//  State variables
//=============================================================================
/**
 * A bare SolidBody registers only Density and Mass.  Everything that must
 * travel with a particle has to be an *evolving* variable: the periodic
 * cell-linked-list update sorts the particle array and permutes only the
 * evolving variables, so a plain registered Velocity would be silently
 * scrambled on the first sort.
 */
void RegisterCGStateVariables(BaseParticles &particles)
{
    const Vecd zero = Vecd::Zero().eval();
    particles.registerStateVariableData<Vecd>("Velocity", zero);
    particles.registerStateVariableData<Vecd>("CG_Force", zero);
    // A-1 square-gradient auxiliary fields (recomputed every step; registered
    // as evolving so that a particle sort permutes them consistently).
    particles.registerStateVariableData<Real>("CG_Rho", 1.0);
    particles.registerStateVariableData<Vecd>("CG_RhoGrad", zero);
    particles.registerStateVariableData<Vecd>("CG_SGForce", zero);
    // M1.3R Scheme E auxiliary field: conjugate of the density difference.
    particles.registerStateVariableData<Real>("CG_SGConj", 0.0);

    particles.addEvolvingVariable<Vecd>("Velocity");
    particles.addEvolvingVariable<Vecd>("CG_Force");
    particles.addEvolvingVariable<Real>("CG_Rho");
    particles.addEvolvingVariable<Vecd>("CG_RhoGrad");
    particles.addEvolvingVariable<Vecd>("CG_SGForce");
    particles.addEvolvingVariable<Real>("CG_SGConj");

    particles.addVariableToWrite<Vecd>("Velocity");
    particles.addVariableToWrite<Vecd>("CG_Force");
}

//=============================================================================
//  Command line
//=============================================================================
bool StartsWith(const std::string &s, const char *prefix)
{
    const std::string p(prefix);
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

Real ToReal(const std::string &a, const char *f) { return std::stod(a.substr(std::string(f).size())); }
int ToInt(const std::string &a, const char *f) { return std::stoi(a.substr(std::string(f).size())); }
std::string ToString(const std::string &a, const char *f) { return a.substr(std::string(f).size()); }

void ParseCommandLine(int argc, char *argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string a(argv[i]);
        if (StartsWith(a, "--case="))
            run_options.case_id = ToInt(a, "--case=");
        else if (StartsWith(a, "--seed="))
            run_options.seed = ToInt(a, "--seed=");
        else if (StartsWith(a, "--threads="))
            run_options.threads = ToInt(a, "--threads=");
        else if (StartsWith(a, "--domain-length="))
            run_options.domain_length = ToReal(a, "--domain-length=");
        else if (StartsWith(a, "--domain-height="))
            run_options.domain_height = ToReal(a, "--domain-height=");
        else if (StartsWith(a, "--fibre-half-width="))
            run_options.fibre_half_width = ToReal(a, "--fibre-half-width=");
        else if (StartsWith(a, "--fibre-shape="))
            run_options.fibre_shape = ToString(a, "--fibre-shape=");
        else if (StartsWith(a, "--fibre-radius="))
            run_options.fibre_radius = ToReal(a, "--fibre-radius=");
        else if (StartsWith(a, "--fibre-length="))
            run_options.fibre_length = ToReal(a, "--fibre-length=");
        else if (StartsWith(a, "--init="))
            run_options.init_mode = ToString(a, "--init=");
        else if (StartsWith(a, "--cluster-shape="))
            run_options.cluster_shape = ToString(a, "--cluster-shape=");
        else if (StartsWith(a, "--cluster-aspect="))
            run_options.cluster_aspect = ToReal(a, "--cluster-aspect=");
        else if (StartsWith(a, "--cluster-packing="))
            run_options.cluster_packing = ToReal(a, "--cluster-packing=");
        else if (StartsWith(a, "--cluster-count="))
            run_options.cluster_count = ToInt(a, "--cluster-count=");
        else if (StartsWith(a, "--interface-gradient-lambda="))
            run_options.interface_lambda = ToReal(a, "--interface-gradient-lambda=");
        else if (StartsWith(a, "--interface-gradient-h="))
            run_options.interface_h = ToReal(a, "--interface-gradient-h=");
        else if (StartsWith(a, "--interface-gradient-rho-ref="))
            run_options.interface_rho_ref = ToReal(a, "--interface-gradient-rho-ref=");
        else if (StartsWith(a, "--axisymmetric="))
            run_options.axisymmetric = ToString(a, "--axisymmetric=");
        else if (StartsWith(a, "--fibre-radius-const="))
            run_options.axis_radius = ToReal(a, "--fibre-radius-const=");
        else if (StartsWith(a, "--film-thickness="))
            run_options.film_thickness = ToReal(a, "--film-thickness=");
        else if (StartsWith(a, "--film-perturb-amp="))
            run_options.film_perturb_amp = ToReal(a, "--film-perturb-amp=");
        else if (StartsWith(a, "--film-perturb-mode="))
            run_options.film_perturb_mode = ToInt(a, "--film-perturb-mode=");
        else if (StartsWith(a, "--film-broadband="))
            run_options.film_broadband = ToInt(a, "--film-broadband=") != 0;
        else if (StartsWith(a, "--perturb-after-equilibration="))
            run_options.perturb_after_equilibration =
                ToInt(a, "--perturb-after-equilibration=") != 0;
        else if (StartsWith(a, "--equilibrate-until="))
            run_options.equilibrate_until = ToReal(a, "--equilibrate-until=");
        else if (StartsWith(a, "--interface-kernel="))
            run_options.interface_kernel = ToString(a, "--interface-kernel=");
        else if (StartsWith(a, "--interface-xi="))
            run_options.interface_xi = ToReal(a, "--interface-xi=");
        else if (StartsWith(a, "--interface-beta="))
            run_options.interface_beta = ToReal(a, "--interface-beta=");
        else if (StartsWith(a, "--axisym-jacobian="))
            run_options.axisym_jacobian = ToInt(a, "--axisym-jacobian=") != 0;
        else if (StartsWith(a, "--sg-fd-check="))
            run_options.sg_fd_check = ToInt(a, "--sg-fd-check=");
        else if (StartsWith(a, "--sg-force-dump="))
            run_options.sg_force_dump = ToInt(a, "--sg-force-dump=");
        else if (StartsWith(a, "--sg-budget-every="))
            run_options.sg_budget_every = ToInt(a, "--sg-budget-every=");
        else if (StartsWith(a, "--noise-off="))
            run_options.noise_off = ToInt(a, "--noise-off=") != 0;
        else if (StartsWith(a, "--axisym-j-one="))
            run_options.axisym_j_one = ToInt(a, "--axisym-j-one=") != 0;
        else if (StartsWith(a, "--axisym-interface-scheme="))
            run_options.axisym_interface_scheme =
                ToString(a, "--axisym-interface-scheme=");
        else if (StartsWith(a, "--area-fraction="))
            run_options.area_fraction = ToReal(a, "--area-fraction=");
        else if (StartsWith(a, "--resolution="))
            run_options.resolution = ToReal(a, "--resolution=");
        else if (StartsWith(a, "--sigma="))
            run_options.sigma = ToReal(a, "--sigma=");
        else if (StartsWith(a, "--wca-epsilon="))
            run_options.wca_epsilon = ToReal(a, "--wca-epsilon=");
        else if (StartsWith(a, "--morse-depth-d="))
            run_options.morse_depth = ToReal(a, "--morse-depth-d=");
        else if (StartsWith(a, "--morse-re="))
            run_options.morse_re = ToReal(a, "--morse-re=");
        else if (StartsWith(a, "--morse-alpha="))
            run_options.morse_alpha = ToReal(a, "--morse-alpha=");
        else if (StartsWith(a, "--morse-cutoff="))
            run_options.morse_cutoff = ToReal(a, "--morse-cutoff=");
        else if (StartsWith(a, "--temperature="))
            run_options.temperature = ToReal(a, "--temperature=");
        else if (StartsWith(a, "--friction="))
            run_options.friction = ToReal(a, "--friction=");
        else if (StartsWith(a, "--wall-depth-d0="))
            run_options.wall_depth_d0 = ToReal(a, "--wall-depth-d0=");
        else if (StartsWith(a, "--wall-alpha="))
            run_options.wall_alpha = ToReal(a, "--wall-alpha=");
        else if (StartsWith(a, "--wall-re="))
            run_options.wall_re = ToReal(a, "--wall-re=");
        else if (StartsWith(a, "--wall-cutoff="))
            run_options.wall_cutoff = ToReal(a, "--wall-cutoff=");
        else if (StartsWith(a, "--wall-guard-h="))
            run_options.wall_guard_h = ToReal(a, "--wall-guard-h=");
        else if (StartsWith(a, "--wall-contact-h="))
            run_options.wall_contact_h = ToReal(a, "--wall-contact-h=");
        else if (StartsWith(a, "--adsorption-cutoff-h="))
            run_options.adsorption_cutoff_h = ToReal(a, "--adsorption-cutoff-h=");
        else if (StartsWith(a, "--initial-separation="))
            run_options.initial_separation = ToReal(a, "--initial-separation=");
        else if (StartsWith(a, "--exclusion-margin="))
            run_options.exclusion_margin = ToReal(a, "--exclusion-margin=");
        else if (StartsWith(a, "--initial-velocity="))
            run_options.initial_velocity = ToString(a, "--initial-velocity=");
        else if (StartsWith(a, "--dt="))
            run_options.dt = ToReal(a, "--dt=");
        else if (StartsWith(a, "--dt-cfl="))
            run_options.dt_cfl = ToReal(a, "--dt-cfl=");
        else if (StartsWith(a, "--end-time="))
            run_options.end_time = ToReal(a, "--end-time=");
        else if (StartsWith(a, "--frames="))
            run_options.frames = ToInt(a, "--frames=");
        else if (StartsWith(a, "--y-reflect="))
            run_options.y_reflect = ToInt(a, "--y-reflect=") != 0;
        else if (StartsWith(a, "--output-tag="))
            run_options.output_tag = ToString(a, "--output-tag=");
        else if (StartsWith(a, "--git-commit="))
            run_options.git_commit = ToString(a, "--git-commit=");
    }

    if (run_options.case_id != 0 && run_options.case_id != 1)
        throw std::runtime_error(
            "--case: only 0 (WCA + Langevin) and 1 (WCA + Morse + Langevin) are "
            "implemented; fibre adsorption is deliberately out of scope.");
    if (run_options.case_id == 0 && run_options.morse_depth != 0.0)
        throw std::runtime_error("--case=0 must not carry an attraction; use --case=1.");
    if (run_options.case_id == 1 && run_options.morse_depth < 0.0)
        throw std::runtime_error("--morse-depth-d must be non-negative.");
    if (std::abs(run_options.temperature - 1.0) > 1.0e-12)
        throw std::runtime_error(
            "this revision fixes k_B T = 1 in reduced units; temperature sweeps "
            "are explicitly out of scope.");
    if (run_options.friction <= 0.0)
        throw std::runtime_error("--friction must be positive.");
    if (run_options.initial_velocity != "maxwell" &&
        run_options.initial_velocity != "zero")
        throw std::runtime_error("--initial-velocity must be maxwell or zero.");
    if (run_options.domain_length <= 0.0 || run_options.domain_height <= 0.0)
        throw std::runtime_error("domain size must be positive.");
    if (2.0 * ExclusionRadius() >= run_options.domain_height)
        throw std::runtime_error("fibre does not fit inside the domain.");
    if (run_options.fibre_shape != "strip" && run_options.fibre_shape != "cylinder" &&
        run_options.fibre_shape != "none")
        throw std::runtime_error("--fibre-shape must be strip, cylinder or none.");
    if (run_options.init_mode != "scatter" && run_options.init_mode != "cluster" &&
        run_options.init_mode != "lattice" && run_options.init_mode != "film")
        throw std::runtime_error("--init must be scatter, cluster, lattice or film.");
    if (run_options.axisymmetric != "off" && run_options.axisymmetric != "filmonly")
        throw std::runtime_error("--axisymmetric must be off or filmonly.");
    if (run_options.axisym_interface_scheme != "moving_j" &&
        run_options.axisym_interface_scheme != "difference_energy")
        throw std::runtime_error(
            "--axisym-interface-scheme must be moving_j or difference_energy.");
    if (run_options.axisym_interface_scheme == "difference_energy" && !Axisymmetric())
        throw std::runtime_error(
            "--axisym-interface-scheme=difference_energy requires --axisymmetric="
            "filmonly; the legacy 2D path keeps the frozen A-1 moving-J energy.");
    if (run_options.perturb_after_equilibration)
    {
        if (!Axisymmetric() || run_options.init_mode != "film")
            throw std::runtime_error(
                "--perturb-after-equilibration requires --axisymmetric=filmonly "
                "--init=film.");
        if (run_options.equilibrate_until <= 0.0)
            throw std::runtime_error(
                "--perturb-after-equilibration requires --equilibrate-until > 0.");
    }
    if (run_options.interface_kernel != "legacy" &&
        run_options.interface_kernel != "two_scale")
        throw std::runtime_error("--interface-kernel must be legacy or two_scale.");
    if (run_options.interface_kernel == "two_scale")
    {
        if (!Axisymmetric())
            throw std::runtime_error(
                "--interface-kernel=two_scale requires --axisymmetric=filmonly.");
        if (run_options.axisym_interface_scheme != "difference_energy")
            throw std::runtime_error(
                "--interface-kernel=two_scale requires "
                "--axisym-interface-scheme=difference_energy.");
        if (!(run_options.interface_beta > 0.0 && run_options.interface_beta < 1.0))
            throw std::runtime_error("--interface-beta must satisfy 0 < beta < 1.");
        if (run_options.interface_xi <= run_options.interface_h)
            throw std::runtime_error("--interface-xi must exceed --interface-gradient-h.");
        if (run_options.interface_xi > 0.99 * 2.6 * ParticleSpacing())
            throw std::runtime_error(
                "--interface-xi exceeds the neighbour cut-off (2h = 2.6 dx); the "
                "long-range lobe of the two-scale kernel would be truncated "
                "silently.  Coarsen the lattice or enlarge the cut-off instead.");
    }
    if (run_options.axis_radius <= 0.0)
        throw std::runtime_error("--fibre-radius-const must be positive.");
    if (run_options.film_thickness <= 0.0)
        throw std::runtime_error("--film-thickness must be positive.");
    if (run_options.film_perturb_mode < 1)
        throw std::runtime_error("--film-perturb-mode must be at least 1.");
    if (Axisymmetric())
    {
        // The (z,r) cylinder is R = R0 = const by definition.  A capsule
        // (--fibre-shape=cylinder) or a fibre-less run would silently replace it
        // with a different geometry, so both are refused rather than ignored.
        // "strip" is the RunOptions default and is simply superseded.
        if (run_options.fibre_shape == "cylinder")
            throw std::runtime_error(
                "--fibre-shape=cylinder (capsule, finite end caps) is not "
                "allowed with --axisymmetric; the (z,r) cylinder is R(z) = R0.");
        if (run_options.fibre_shape == "none")
            throw std::runtime_error(
                "--axisymmetric=filmonly requires a fibre; --fibre-shape=none "
                "would remove the wall entirely.");
        if (run_options.axis_radius < 2.0 * Sigma())
            throw std::runtime_error(
                "--fibre-radius-const is below 2 sigma; the work order fixes "
                "R0 = 5 sigma for M1.1 (kernel deficiency is deferred).");
        // M1.4 C1: the film-thickness perturbation is implemented as a
        // continuous radial coordinate mapping (see prepareAxisymmetricFilm).
        // A = 0 is an exact return to the M1.2 state; |A| is limited to a
        // genuinely small-amplitude window.
        if (std::abs(run_options.film_perturb_amp) > 0.2)
            throw std::runtime_error(
                "--film-perturb-amp must satisfy |A| <= 0.2 (small-amplitude "
                "axial perturbation only).");
        if (run_options.film_perturb_mode < 1)
            throw std::runtime_error("--film-perturb-mode must be at least 1.");
        if (run_options.axisym_jacobian)
            throw std::runtime_error(
                "--axisym-jacobian is not implemented yet (M2); the measure term "
                "is deliberately absent in M1.1.");
        if (ExclusionRadius() + 4.0 * Sigma() >= DomainHeight())
            throw std::runtime_error(
                "axisymmetric geometry leaves no radial room: need "
                "R0 + exclusion_margin + 4 sigma < --domain-height.");
    }
    if (run_options.init_mode == "film")
    {
        // M1.2: the film builder exists only for the (z,r) cylinder on this
        // branch.  Refusing everything else keeps the flat-wall strip film
        // (PC2, branch pc2-a5-film) out of this implementation on purpose.
        if (!Axisymmetric())
            throw std::runtime_error(
                "--init=film is implemented only for --axisymmetric=filmonly on "
                "this branch; the flat-wall strip film lives on PC2's "
                "pc2-a5-film branch and is deliberately not copied here.");
        const Real phi = run_options.cluster_packing;
        if (phi <= 0.05 || phi >= 0.72)
            throw std::runtime_error(
                "--cluster-packing must be in (0.05,0.72); 0.72 is the triangular "
                "lattice limit set by the WCA contact distance.");
        const Real h_in = FilmInnerOffset();
        if (run_options.film_thickness <= h_in)
            throw std::runtime_error(
                "--film-thickness must exceed the wall well position (the stable "
                "contact distance), otherwise the film would start inside the wall.");
        // Conservative fit bound: the layer count is derived from the lattice,
        // so allow one extra layer beyond the requested thickness.
        const Real sigma = Sigma();
        const Real d_lat =
            sigma * std::sqrt((0.25 * Pi) / (0.5 * std::sqrt(3.0) * phi));
        const Real dr_lat = 0.5 * std::sqrt(3.0) * d_lat;
        const Real reach_r = AxisRadius() + h_in + run_options.film_thickness + dr_lat;
        if (reach_r + sigma >= run_options.domain_height)
            throw std::runtime_error(
                "--init=film: fibre plus film does not fit inside the domain "
                "(need R0 + h_in + h_film + dr + sigma < --domain-height).");
    }
    if (run_options.init_mode == "cluster")
    {
        if (run_options.cluster_count < 8)
            throw std::runtime_error("--cluster-count must be at least 8.");
        if (run_options.cluster_packing <= 0.05 || run_options.cluster_packing >= 0.72)
            throw std::runtime_error(
                "--cluster-packing must be in (0.05,0.72); 0.72 is the triangular "
                "lattice limit set by the WCA contact distance.");
        if (run_options.cluster_aspect < 1.0)
            throw std::runtime_error("--cluster-aspect must be >= 1.");
        if (run_options.cluster_shape != "disc" && run_options.cluster_shape != "ellipse" &&
            run_options.cluster_shape != "rect")
            throw std::runtime_error("--cluster-shape must be disc, ellipse or rect.");
    }
    if (run_options.fibre_shape == "cylinder")
    {
        if (run_options.fibre_radius <= 0.0 || run_options.fibre_length <= 0.0)
            throw std::runtime_error("--fibre-radius and --fibre-length must be positive.");
        // the fibre plus its periodic image must not overlap themselves
        if (FibreLength() + 2.0 * ExclusionRadius() >= run_options.domain_length)
            throw std::runtime_error(
                "capsule fibre (length + 2*(radius+margin)) must be shorter than Lx.");
    }
    if (run_options.area_fraction <= 0.0 || run_options.area_fraction >= 0.55)
        throw std::runtime_error("--area-fraction must be in (0,0.55).");
    if (run_options.initial_separation <= 0.0 ||
        run_options.initial_separation > LargestPairCutoff())
        throw std::runtime_error("--initial-separation is outside the pair cut-off.");
    if (run_options.frames < 2)
        throw std::runtime_error("--frames must be at least 2.");
    if (run_options.wall_depth_d0 < 0.0)
        throw std::runtime_error("--wall-depth-d0 must be non-negative.");
    if (run_options.interface_lambda < 0.0)
        throw std::runtime_error("--interface-gradient-lambda must be non-negative.");
    if (run_options.interface_lambda > 0.0)
    {
        if (run_options.interface_h <= 0.0)
            throw std::runtime_error("--interface-gradient-h must be positive.");
        if (run_options.interface_rho_ref <= 0.0)
            throw std::runtime_error("--interface-gradient-rho-ref must be positive.");
        // the kernel support must fit inside the neighbour-list cut-off
        if (run_options.interface_h > 0.99 * 2.6 * ParticleSpacing())
            throw std::runtime_error(
                "--interface-gradient-h exceeds the neighbour cut-off (2h = 2.6 dx); "
                "the square-gradient kernel would be truncated silently.");
    }
    if (run_options.wall_depth_d0 > 0.0)
    {
        if (run_options.wall_alpha <= 0.0)
            throw std::runtime_error("--wall-alpha must be positive.");
        if (run_options.wall_re <= 0.0 || run_options.wall_cutoff <= run_options.wall_re)
            throw std::runtime_error("wall well position / cut-off are inconsistent.");
        // The Morse wall already carries the near-wall repulsion; the guard is
        // a numerical backstop only and must sit well inside the well.
        if (run_options.wall_guard_h >= run_options.wall_re)
            throw std::runtime_error(
                "--wall-guard-h must be smaller than --wall-re, otherwise the "
                "guard would double up with the Morse wall repulsion.");
    }
}

//=============================================================================
//  Geometry
//=============================================================================
/** Stage 1 broadband axial seed.
 *
 *  Modes 1,2,3,4,5,6,8,10 with golden-ratio phases and strictly equal weights;
 *  --film-perturb-amp is then the TOTAL amplitude budget, i.e. |f - 1| <= A
 *  pointwise, so the WCA-overlap margin is the same as for the single-mode
 *  seed of the same A.  No wavelength is favoured (equal weights, spread
 *  phases); the realised spectrum is verified from frame 0 by m15_beads.py.
 */
inline Real FilmBroadbandFactor(const Real z, const Real lz, const Real amp)
{
    static const int modes[] = {1, 2, 3, 4, 5, 6, 8, 10};
    static const int nm = 8;
    Real s = 0.0;
    for (int k = 0; k < nm; ++k)
    {
        const Real phi = 2.0 * Pi * std::fmod(modes[k] * 0.6180339887498949, 1.0);
        s += std::sin(2.0 * Pi * static_cast<Real>(modes[k]) * z / lz + phi);
    }
    return 1.0 + (amp / static_cast<Real>(nm)) * s;
}

class CGDomainShape : public ComplexShape
{
  public:
    explicit CGDomainShape(const std::string &shape_name) : ComplexShape(shape_name)
    {
        const Vecd centre(0.5 * DomainLength(), 0.5 * DomainHeight());
        const Vecd half(0.5 * DomainLength(), 0.5 * DomainHeight());
        add<GeometricShapeBox>(Transform(centre), half);
    }
};

class FibreStripShape : public ComplexShape
{
  public:
    explicit FibreStripShape(const std::string &shape_name) : ComplexShape(shape_name)
    {
        if (Axisymmetric())
        {
            // M1.1: the solid core r < R0, i.e. the band y in [0, R0] over the
            // full axial period.  This is the (z,r) image of an infinite
            // constant-radius cylinder.  It is NOT a capsule (no end caps), NOT
            // a disc cross-section and NOT the centred strip.  The fibre body
            // carries no interaction (only particles_inner exists), so this
            // shape affects the written Fibre VTP only, never the trajectory.
            const Vecd centre(0.5 * DomainLength(), 0.5 * AxisRadius());
            const Vecd half(0.5 * DomainLength(), 0.5 * AxisRadius());
            add<GeometricShapeBox>(Transform(centre), half);
            return;
        }
        if (run_options.fibre_shape == "cylinder")
        {
            // capsule = straight segment of length fibre_length plus two caps
            const Vecd axis_centre(0.5 * DomainLength(), FibreCentreY());
            const Vecd half(0.5 * FibreLength(), FibreRadius());
            add<GeometricShapeBox>(Transform(axis_centre), half);
            const Real half_length = 0.5 * FibreLength();
            add<GeometricShapeBall>(
                Vecd(0.5 * DomainLength() - half_length, FibreCentreY()), FibreRadius());
            add<GeometricShapeBall>(
                Vecd(0.5 * DomainLength() + half_length, FibreCentreY()), FibreRadius());
            return;
        }
        const Vecd centre(0.5 * DomainLength(), FibreCentreY());
        const Vecd half(0.5 * DomainLength(), FibreHalfWidth());
        add<GeometricShapeBox>(Transform(centre), half);
    }
};

} // namespace

//=============================================================================
//  Random scattered particle generation
//
//  NOTE: an explicit specialisation of SPH::ParticleGenerator may not be
//  defined inside an anonymous namespace, so this block sits at global scope.
//=============================================================================
struct CGRandomScatter
{
};

template <>
class ParticleGenerator<BaseParticles, CGRandomScatter>
    : public ParticleGenerator<BaseParticles>
{
  public:
    ParticleGenerator(SPHBody &sph_body, BaseParticles &base_particles)
        : ParticleGenerator<BaseParticles>(sph_body, base_particles) {}

    virtual void prepareGeometricData() override
    {
        const Real lx = DomainLength();
        const Real ly = DomainHeight();
        const Real min_separation = run_options.initial_separation;
        if (run_options.init_mode == "cluster")
        {
            prepareCluster();
            return;
        }
        if (run_options.init_mode == "lattice")
        {
            prepareUniformLattice();
            return;
        }
        if (run_options.init_mode == "film")
        {
            prepareFilm();
            return;
        }
        const int target = TargetParticleNumber();

        const Real bucket_size = std::max(min_separation, 1.0e-6);
        const int nx = std::max(1, static_cast<int>(std::floor(lx / bucket_size)));
        const int ny = std::max(1, static_cast<int>(std::floor(ly / bucket_size)));
        const Real dx_bucket = lx / static_cast<Real>(nx);
        const Real dy_bucket = ly / static_cast<Real>(ny);

        std::vector<Vecd> accepted;
        accepted.reserve(target);
        std::vector<std::vector<int>> buckets(
            static_cast<size_t>(nx) * static_cast<size_t>(ny));

        auto bucket_of = [&](const Vecd &p, int &ix, int &iy) {
            ix = static_cast<int>(std::floor(p[0] / dx_bucket));
            iy = static_cast<int>(std::floor(p[1] / dy_bucket));
            ix = ((ix % nx) + nx) % nx;
            iy = std::min(std::max(iy, 0), ny - 1);
        };

        auto accepted_in_ring = [&](const Vecd &candidate) {
            int ix, iy;
            bucket_of(candidate, ix, iy);
            for (int ax = -1; ax <= 1; ++ax)
                for (int ay = -1; ay <= 1; ++ay)
                {
                    const int jx = ((ix + ax) % nx + nx) % nx;
                    const int jy = iy + ay;
                    if (jy < 0 || jy >= ny)
                        continue;
                    const std::vector<int> &bucket =
                        buckets[static_cast<size_t>(jx) * ny + jy];
                    for (int index : bucket)
                    {
                        Vecd delta = candidate - accepted[index];
                        delta[0] -= lx * std::round(delta[0] / lx);
                        if (delta.norm() < min_separation)
                            return true;
                    }
                }
            return false;
        };

        std::mt19937 rng(static_cast<std::mt19937::result_type>(run_options.seed));
        std::uniform_real_distribution<Real> uniform_x(0.0, lx);
        std::uniform_real_distribution<Real> uniform_y(0.0, ly);

        const long long max_attempts = 20000LL * static_cast<long long>(target);
        long long attempts = 0;
        while (static_cast<int>(accepted.size()) < target && attempts < max_attempts)
        {
            ++attempts;
            const Vecd candidate(uniform_x(rng), uniform_y(rng));
            if (FibreContactOf(candidate).h < run_options.exclusion_margin)
                continue;
            if (accepted_in_ring(candidate))
                continue;
            int ix, iy;
            bucket_of(candidate, ix, iy);
            buckets[static_cast<size_t>(ix) * ny + iy].push_back(
                static_cast<int>(accepted.size()));
            accepted.push_back(candidate);
        }

        if (static_cast<int>(accepted.size()) < target)
            throw std::runtime_error(
                "random scattering failed to reach the requested particle "
                "number; lower --area-fraction or --initial-separation.");

        std::cout << "  random scatter: seed=" << run_options.seed
                  << " attempts=" << attempts
                  << " accepted=" << accepted.size()
                  << " min_separation=" << min_separation << "\n";

        for (const Vecd &position : accepted)
            addPositionAndVolumetricMeasure(position, 1.0);
    }

    /**
     * One pre-condensed aggregate on a triangular lattice.  The lattice
     * spacing follows from the requested packing fraction
     *     phi = (pi sigma^2 / 4) / ((sqrt(3)/2) d^2),
     * the shape fixes the target area A = N (pi sigma^2/4) / phi, and the N
     * lattice sites with the smallest normalised shape coordinate are kept, so
     * every shape holds EXACTLY the same particle number.
     */
    void prepareCluster()
    {
        const Real sigma = Sigma();
        const Real phi = run_options.cluster_packing;
        const Real n = static_cast<Real>(run_options.cluster_count);
        const Real area = n * 0.25 * Pi * sigma * sigma / phi;
        const Real d = sigma * std::sqrt((0.25 * Pi) / (0.5 * std::sqrt(3.0) * phi));

        Real a = std::sqrt(area / Pi);           // disc radius
        Real half_l = a, half_w = a;             // ellipse / rect half sizes
        if (run_options.cluster_shape == "ellipse" ||
            run_options.cluster_shape == "rect")
        {
            half_l = std::sqrt(area * run_options.cluster_aspect / Pi);
            half_w = std::sqrt(area / (Pi * run_options.cluster_aspect));
            if (run_options.cluster_shape == "rect")
            {
                half_l = 0.5 * std::sqrt(area * run_options.cluster_aspect);
                half_w = 0.5 * std::sqrt(area / run_options.cluster_aspect);
            }
        }
        if (2.0 * half_l >= DomainLength() - 2.0 * sigma ||
            2.0 * half_w >= DomainHeight() - 2.0 * sigma)
            throw std::runtime_error("cluster does not fit inside the domain.");

        const Real row = 0.5 * std::sqrt(3.0) * d;
        const int nx = static_cast<int>(std::ceil(half_l / d)) + 2;
        const int ny = static_cast<int>(std::ceil(half_w / row)) + 2;
        std::vector<std::pair<Real, Vecd>> sites;
        for (int j = -ny; j <= ny; ++j)
            for (int i = -nx; i <= nx; ++i)
            {
                const Vecd p(static_cast<Real>(i) * d +
                                 ((j % 2) ? 0.5 * d : 0.0),
                             static_cast<Real>(j) * row);
                Real coord;
                if (run_options.cluster_shape == "ellipse")
                    coord = std::sqrt((p[0] / half_l) * (p[0] / half_l) +
                                      (p[1] / half_w) * (p[1] / half_w));
                else if (run_options.cluster_shape == "rect")
                    coord = std::max(std::abs(p[0]) / half_l, std::abs(p[1]) / half_w);
                else
                    coord = p.norm() / half_l;
                sites.emplace_back(coord, p);
            }
        std::sort(sites.begin(), sites.end(),
                  [](const std::pair<Real, Vecd> &l, const std::pair<Real, Vecd> &r) {
                      return l.first < r.first;
                  });
        if (static_cast<int>(sites.size()) < run_options.cluster_count)
            throw std::runtime_error("not enough lattice sites for the cluster.");

        Vecd sum(0.0, 0.0);
        std::vector<Vecd> chosen;
        chosen.reserve(static_cast<size_t>(run_options.cluster_count));
        for (int k = 0; k < run_options.cluster_count; ++k)
        {
            chosen.push_back(sites[static_cast<size_t>(k)].second);
            sum += sites[static_cast<size_t>(k)].second;
        }
        const Vecd shift(0.5 * DomainLength(), FibreCentreY());
        const Vecd centre = sum / static_cast<Real>(run_options.cluster_count);
        for (const Vecd &p : chosen)
            addPositionAndVolumetricMeasure(p - centre + shift, 1.0);

        std::cout << "  pre-condensed cluster: shape=" << run_options.cluster_shape
                  << " aspect=" << run_options.cluster_aspect
                  << " N=" << run_options.cluster_count
                  << " packing=" << phi << " lattice_d=" << d
                  << " area=" << area
                  << " half_sizes=" << half_l << " x " << half_w << "\n";
    }

    /**
     * Uniform triangular lattice filling the whole box, periodic in x.
     * Used by the A-1 gate G1 (homogeneous bulk): with no surface at all the
     * square-gradient term must produce no systematic force.
     */
    void prepareUniformLattice()
    {
        const Real sigma = Sigma();
        const Real phi = run_options.cluster_packing;
        const Real d0 = sigma * std::sqrt((0.25 * Pi) / (0.5 * std::sqrt(3.0) * phi));
        const int nx = std::max(2, static_cast<int>(std::llround(DomainLength() / d0)));
        const Real d = DomainLength() / static_cast<Real>(nx); // exact periodicity
        const Real row = 0.5 * std::sqrt(3.0) * d;
        const int ny = std::max(2, static_cast<int>(std::floor(DomainHeight() / row)));
        const Real y0 = 0.5 * (DomainHeight() - (ny - 1) * row);
        size_t added = 0;
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
            {
                const Real x = (static_cast<Real>(i) + ((j % 2) ? 0.5 : 0.0)) * d;
                const Real y = y0 + static_cast<Real>(j) * row;
                // M1.1: in (z,r) the solid core r < R0 is NOT part of the fluid
                // domain, so lattice sites that fall inside it must be dropped.
                // Without this the run starts with particles inside the solid
                // (h < 0), the geometric constraint projects them onto the
                // surface in one step and the configuration blows up.
                // The filter is deliberately gated on Axisymmetric() so that
                // the legacy 2D lattice runs stay bit-identical.
                if (Axisymmetric() &&
                    FibreContactOf(Vecd(x, y)).h < run_options.exclusion_margin)
                    continue;
                addPositionAndVolumetricMeasure(Vecd(x, y), 1.0);
                ++added;
            }
        const Real packing = static_cast<Real>(added) * 0.25 * Pi * sigma * sigma /
                             (DomainLength() * DomainHeight());
        std::cout << "  uniform lattice: nx=" << nx << " ny=" << ny
                  << " N=" << added << " d=" << d << " row=" << row
                  << " packing=" << packing
                  << " rho=" << packing / (0.25 * Pi * sigma * sigma) << "\n";
    }

    /**
     * M1.2 initial film: pure dispatcher.  Geometry selection lives here and
     * ONLY here, so the two film builders never share a hidden branch.
     */
    void prepareFilm()
    {
        if (Axisymmetric())
        {
            prepareAxisymmetricFilm();
            return;
        }
        throw std::runtime_error(
            "--init=film is implemented only for --axisymmetric=filmonly on this "
            "branch; the flat-wall strip film lives on PC2's pc2-a5-film branch "
            "and is deliberately not copied here.");
    }

    /**
     * M1.2: continuous film outside a CONSTANT-RADIUS (z,r) cylinder.
     *
     * Geometry.  In the axisymmetric branch x -> z (fibre axis, periodic) and
     * y -> r (distance from the axis), the fibre is R(z) = R0 = const and the
     * film occupies the annulus
     *     R0 + h_in  <=  r  <=  R0 + h_in + (rows-1)*dr.
     * Layers are surfaces of constant r, so h(z) is constant along z by
     * construction and the "no holes / continuous film" requirement reduces to
     * "every layer is a complete, evenly spaced z line".
     *
     * Inherited from PC2's A-5 strip film (verified there, reused here as
     * rules only -- the code is NOT copied):
     *     d  = sigma sqrt( (pi/4) / ( (sqrt(3)/2) packing ) )   nearest neighbour
     *     dr = (sqrt(3)/2) d                                     layer spacing
     *     n  = round(Lz/d),  dz = Lz/n                            seam-free period
     *     odd layers shifted by 0.5 dz                            triangular order
     *     innermost layer at h = FilmInnerOffset() = 0.5 sigma    wall force = 0
     *     N = rows*n derived from geometry alone, never hand-set
     *     realised_thickness = rows*dr
     * so the areal density is the project-wide bulk value
     * packing/(pi sigma^2/4) = 0.8913 sigma^-2 at packing 0.70.
     *
     * SCOPE.  This is a GEOMETRY initialiser only.  Particles carry m = 1 and
     * are laid out uniformly in the (z,r) plane, so the implied 3D bead number
     * density scales as 1/r.  That is the known O(h/R) approximation of this
     * stage and is deliberately NOT corrected here (no 2 pi r weighting, no
     * variable mass, no measure term, no density correction).
     */
    void prepareAxisymmetricFilm()
    {
        const Real sigma = Sigma();
        const Real phi = run_options.cluster_packing;
        const Real d = sigma * std::sqrt((0.25 * Pi) / (0.5 * std::sqrt(3.0) * phi));
        const Real dr = 0.5 * std::sqrt(3.0) * d;
        const Real h_in = FilmInnerOffset();
        const Real h_out = run_options.film_thickness;
        const Real lz = DomainLength();
        const Real radius = AxisRadius();
        const int n = std::max(3, static_cast<int>(std::floor(lz / d + 0.5)));
        const Real dz = lz / static_cast<Real>(n);

        std::vector<Vecd> placed;
        std::vector<Real> layer_r;
        std::vector<int> layer_n;
        placed.reserve(256);
        int rows = 0;
        Real h_first = 0.0;
        Real h_last = 0.0;
        // M1.4 C1: the layer count (hence the outer layer height) is known
        // before placement, so the continuous thickness perturbation can be
        // applied to the actual particle positions as they are created.  With
        // A = 0 the factor is exactly 1 and the (h - h_in)*(f - 1) term is
        // exactly zero, so the M1.2 configuration is reproduced bit for bit.
        int rows_planned = 0;
        for (int layer = 0;; ++layer)
        {
            if (h_in + static_cast<Real>(layer) * dr > h_out + 1.0e-9)
                break;
            ++rows_planned;
        }
        const Real h_out_layer =
            h_in + static_cast<Real>(std::max(rows_planned - 1, 0)) * dr;
        // Stage 9: with the equilibrate-then-perturb workflow the mapping is
        // not applied to the lattice at all; it is applied once after Phase E.
        const Real pert_amp = run_options.perturb_after_equilibration
                                  ? 0.0
                                  : run_options.film_perturb_amp;
        const Real pert_kz = (pert_amp != 0.0)
                                 ? 2.0 * Pi *
                                       static_cast<Real>(run_options.film_perturb_mode) / lz
                                 : 0.0;
        for (int layer = 0;; ++layer)
        {
            const Real h = h_in + static_cast<Real>(layer) * dr;
            if (h > h_out + 1.0e-9)
                break;
            const Real shift = ((layer % 2) ? 0.5 : 0.0) * dz;
            for (int k = 0; k < n; ++k)
            {
                Real z = shift + static_cast<Real>(k) * dz;
                z -= lz * std::floor(z / lz);
                const Real f = (pert_amp == 0.0)
                                   ? 1.0
                                   : (run_options.film_broadband
                                          ? FilmBroadbandFactor(z, lz, pert_amp)
                                          : 1.0 + pert_amp * std::sin(pert_kz * z));
                const Real r = radius + h + (h - h_in) * (f - 1.0);
                const Vecd p(z, r);
                addPositionAndVolumetricMeasure(p, 1.0);
                placed.push_back(p);
            }
            if (rows == 0)
                h_first = h;
            h_last = h;
            layer_r.push_back(radius + h);   // unperturbed layer radius
            layer_n.push_back(n);
            ++rows;
        }
        if (placed.empty())
            throw std::runtime_error(
                "--init=film produced no particles; check --film-thickness, the "
                "fibre geometry and the domain size.");

        //-------------------------------------------------------------------------
        //  M1.4 C1: continuous axial film-thickness perturbation
        //
        //      s  = r - R0                        height above the fibre surface
        //      s' = s + (s - s_in) (f - 1),
        //      f  = 1 + A sin(2 pi m z / Lz)
        //
        //  Properties: A = 0 returns the M1.2 state exactly (the loop is skipped
        //  entirely, so the trajectory stays bit-identical); the innermost layer
        //  sits at s = s_in and does not move; the outer interface carries the
        //  full amplitude; f > 0 for |A| <= 0.2 so each particle keeps its layer
        //  index and the layer topology is unchanged; <f> over one axial period
        //  is exactly 1, so the m = 0 thickness is unchanged and the volume
        //  excess is only the O(A^2) term reported below.
        //-------------------------------------------------------------------------
        Real perturb_volume_rel_excess = 0.0;
        if (pert_amp != 0.0)
        {
            const Real mode_m = static_cast<Real>(run_options.film_perturb_mode);
            const Real ds = h_out_layer - h_in;
            const Real r_in = radius + h_in;
            perturb_volume_rel_excess =
                (0.5 * ds * ds * pert_amp * pert_amp) /
                (2.0 * r_in * ds + ds * ds);
            std::cout << "  film perturbation: A=" << pert_amp
                      << " mode m=" << run_options.film_perturb_mode
                      << " kz=" << pert_kz
                      << " lambda_z=" << (lz / mode_m)
                      << " outer_layer_h=" << h_out_layer
                      << " volume_rel_excess_O(A^2)=" << perturb_volume_rel_excess
                      << "\n";
        }

        // Realised geometry.  Every layer stands for one dr-thick band, so the
        // built band is rows*dr thick and holds exactly the requested packing.
        const Real realised_thickness = static_cast<Real>(rows) * dr;
        const size_t kept = placed.size();
        const Real areal_density =
            static_cast<Real>(kept) / (lz * realised_thickness);
        const Real lattice_rho = phi / (0.25 * Pi * sigma * sigma);
        const Real geometric_annulus_volume =
            Pi * ((radius + h_out) * (radius + h_out) - radius * radius) * lz;
        const Real r_in_band = radius + h_in;
        const Real r_out_band = radius + h_in + realised_thickness;
        const Real initialized_band_volume =
            Pi * (r_out_band * r_out_band - r_in_band * r_in_band) * lz;

        // True minimum pair distance of the built configuration, with the
        // minimum image in z (periodic).  O(N^2) is fine: N is a few hundred.
        Real min_pair = std::numeric_limits<Real>::max();
        for (size_t i = 0; i < kept; ++i)
            for (size_t j = i + 1; j < kept; ++j)
            {
                Vecd delta = placed[i] - placed[j];
                delta[0] -= lz * std::round(delta[0] / lz);
                min_pair = std::min(min_pair, delta.norm());
            }
        const Real min_wall_distance = h_first;

        std::cout << "  film: shape=axisymmetric_cylinder R0=" << radius
                  << " d_min=" << h_in << " H_f=" << h_out
                  << " rows=" << rows
                  << " h_first=" << h_first << " h_last=" << h_last
                  << " N=" << kept
                  << " d=" << d << " dr=" << dr << " dz=" << dz << " n=" << n
                  << " min_surface_distance=" << min_wall_distance
                  << " initial_thickness=" << h_out
                  << " realised_thickness=" << realised_thickness
                  << " particle_count=" << kept
                  << " geometric_annulus_volume=" << geometric_annulus_volume
                  << " initialized_band_volume=" << initialized_band_volume
                  << " areal_density=" << areal_density
                  << " lattice_rho=" << lattice_rho
                  << " min_pair_distance=" << min_pair << "\n";

        std::ofstream csv("axisym_film_selfcheck.csv");
        csv << std::setprecision(10);
        csv << "quantity,value,note\n";
        auto row = [&](const std::string &k, const std::string &v,
                       const char *note) { csv << k << "," << v << "," << note << "\n"; };
        row("init_mode", run_options.init_mode, "film initial state");
        row("geometry", "axisymmetric_cylinder",
            "(z,r) constant-radius cylinder; x->z periodic, y->r");
        row("R0", std::to_string(radius), "sigma; R(z) = R0 const");
        row("packing", std::to_string(phi), "areal packing used by the lattice");
        row("d", std::to_string(d), "triangular nearest neighbour");
        row("dr", std::to_string(dr), "(sqrt(3)/2) d; radial layer spacing");
        row("n_per_layer", std::to_string(n), "max(3, round(Lz/d))");
        row("dz", std::to_string(dz), "Lz/n; periodic, seam-free");
        row("Lz", std::to_string(lz), "axial period");
        row("requested_thickness", std::to_string(h_out),
            "H_f as requested on the command line");
        row("realised_thickness", std::to_string(realised_thickness),
            "rows*dr; USE THIS ONE for any downstream geometry comparison");
        row("realised_minus_requested", std::to_string(realised_thickness - h_out),
            "quantisation of the film thickness by dr");
        row("perturb_amp", std::to_string(run_options.film_perturb_amp),
            "M1.4 C1: A in s' = s + (s - s_in)(A sin(2 pi m z / Lz)); 0 = M1.2 state");
        row("perturb_mode", std::to_string(run_options.film_perturb_mode),
            "axial mode index m; lambda_z = Lz / m");
        row("perturb_volume_rel_excess", std::to_string(perturb_volume_rel_excess),
            "analytic O(A^2) relative volume excess of the mapped band");
        row("radial_rows", std::to_string(rows), "number of radial layers");
        row("particle_count", std::to_string(kept), "N = rows*n, never hand-set");
        row("min_wall_distance", std::to_string(min_wall_distance),
            "min(r - R0); must equal d_min and stay > 0");
        row("min_pair_distance", std::to_string(min_pair),
            "true minimum pair distance incl. z minimum image; WCA contact = "
            "1.122462 sigma");
        row("wca_contact_distance", std::to_string(std::pow(2.0, 1.0 / 6.0) * sigma),
            "2^(1/6) sigma; min_pair_distance must exceed this");
        row("geometric_annulus_volume", std::to_string(geometric_annulus_volume),
            "pi[(R0+h_f)^2 - R0^2] Lz; theoretical target annulus, from the "
            "fibre surface");
        row("initialized_band_volume", std::to_string(initialized_band_volume),
            "pi[(R0+h_in+T)^2 - (R0+h_in)^2] Lz; the band actually populated");
        row("particle_number_over_geometric_annulus_volume",
            std::to_string(static_cast<Real>(kept) / geometric_annulus_volume),
            "N / geometric_annulus_volume -- DIAGNOSTIC ONLY: with m=1 and "
            "uniform (z,r) sampling this is NOT a corrected 3D axisymmetric "
            "number density");
        row("particle_number_over_initialized_band_volume",
            std::to_string(static_cast<Real>(kept) / initialized_band_volume),
            "N / initialized_band_volume -- same caveat as above");
        row("areal_density", std::to_string(areal_density),
            "N / (Lz * realised_thickness); should match lattice_rho");
        row("lattice_rho", std::to_string(lattice_rho),
            "packing / (pi sigma^2 / 4); bulk areal density");
        for (int i = 0; i < rows; ++i)
        {
            row("layer_" + std::to_string(i) + "_r", std::to_string(layer_r[i]),
                "radius of this layer; must be R0 + h_in + i*dr");
            row("layer_" + std::to_string(i) + "_count",
                std::to_string(layer_n[i]), "particles in this layer; must equal n");
        }
        csv.close();
        std::cout << "  film selfcheck -> axisym_film_selfcheck.csv\n";
    }
};

namespace
{
//=============================================================================
//  Conservative pair interaction
//=============================================================================
class CGPairInteraction : public LocalDynamics, public DataDelegateInner
{
  public:
    explicit CGPairInteraction(BaseInnerRelation &inner_relation)
        : LocalDynamics(inner_relation.getSPHBody()),
          DataDelegateInner(inner_relation),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          force_(particles_->template getVariableDataByName<Vecd>("CG_Force")),
          min_separation_(std::numeric_limits<Real>::max()),
          periodic_pair_count_(0) {}

    void interaction(size_t index_i, Real dt = 0.0)
    {
        Vecd force = Vecd::Zero();
        Real local_min = std::numeric_limits<Real>::max();

        const Neighborhood &neighborhood = inner_configuration_[index_i];
        for (size_t n = 0; n != neighborhood.current_size_; ++n)
        {
            const size_t index_j = neighborhood.j_[n];
            const Real r = neighborhood.r_ij_[n];
            local_min = std::min(local_min, r);
            if (std::abs(pos_[index_i][0] - pos_[index_j][0]) > 0.5 * DomainLength())
                periodic_pair_count_.fetch_add(1, std::memory_order_relaxed);
            if (r <= TinyReal)
                continue;
            // e_ij points from j to i, so a positive magnitude pushes them apart.
            force += PairForce(r) * neighborhood.e_ij_[n];
        }
        force_[index_i] = force;
        min_separation_ = std::min(min_separation_, local_min);
    }

    void update(size_t index_i, Real dt = 0.0) {}

    Real MinimumSeparation() const
    {
        return min_separation_ == std::numeric_limits<Real>::max() ? -1.0
                                                                   : min_separation_;
    }

    void ResetCounters()
    {
        min_separation_ = std::numeric_limits<Real>::max();
        periodic_pair_count_.store(0, std::memory_order_relaxed);
    }

    size_t PeriodicPairCount() const
    {
        return periodic_pair_count_.load(std::memory_order_relaxed) / 2;
    }

  private:
    Vecd *pos_;
    Vecd *force_;
    Real min_separation_;
    std::atomic<size_t> periodic_pair_count_;
};

//=============================================================================
//  A-1 square-gradient interface free energy
//
//  F_sg = (lambda/2) Sum_i V0 |grad rho_i|^2 ,   rho_i = Sum_j W_h(r_ij)
//
//  This is the direct particle discretisation of  (lambda/2) Int |grad rho|^2
//  dA.  With G_i = grad rho_i = Sum_j W'(r_ij) e_ij and
//  M_ij = W''(r_ij) e_ij (x) e_ij + (W'(r_ij)/r_ij)(I - e_ij (x) e_ij),
//  the exact gradient of the discrete energy is
//      F_a = -lambda V0 Sum_{k!=a} (G_a - G_k) . M_ak      ( = Sum_k f_ak )
//  and f_ak = -f_ka identically, so total momentum is conserved exactly.
//  No empirical "interface normal force" is used anywhere.
//=============================================================================
Real RhoSmoothingH() { return run_options.interface_h * Sigma(); }

/** Wendland C2 kernel with support h_rho (normalised in 2D, C^2 at r = h). */
inline Real RhoKernelW(Real r)
{
    const Real h = RhoSmoothingH();
    const Real q = r / h;
    if (q >= 1.0)
        return 0.0;
    const Real t = 1.0 - q;
    return (7.0 / (Pi * h * h)) * t * t * t * t * (4.0 * q + 1.0);
}

/** dW/dr */
inline Real RhoKernelDW(Real r)
{
    const Real h = RhoSmoothingH();
    const Real q = r / h;
    if (q >= 1.0)
        return 0.0;
    const Real t = 1.0 - q;
    return -(140.0 / (Pi * h * h * h)) * q * t * t * t;
}

/** d2W/dr2 */
inline Real RhoKernelDDW(Real r)
{
    const Real h = RhoSmoothingH();
    const Real q = r / h;
    if (q >= 1.0)
        return 0.0;
    const Real t = 1.0 - q;
    return -(140.0 / (Pi * h * h * h * h)) * t * t * (1.0 - 4.0 * q);
}

//=============================================================================
//  Stage 21: opt-in two-scale Scheme E kernel factory
//
//      K_new(r) = [ c_h W_h(r) - beta c_xi W_xi(r) ] / (1 - beta)
//      c_h = 144/(5 h^2),  c_xi = 144/(5 xi^2)
//
//  Both terms individually satisfy Int|u|^2 (c W) dA = 4, so the combination
//  does as well; the continuum limit of the pair-difference energy (and hence
//  the meaning of lambda) is therefore unchanged and lambda needs no
//  recalibration.  With --interface-kernel=legacy (default) every call falls
//  through to the original single-scale functions, so the historical
//  trajectories stay bit-identical.
//=============================================================================
inline Real TwoScaleKernelWRaw(Real r, Real hw)
{
    const Real q = r / hw;
    if (q >= 1.0)
        return 0.0;
    const Real t = 1.0 - q;
    return (7.0 / (Pi * hw * hw)) * t * t * t * t * (4.0 * q + 1.0);
}

inline Real TwoScaleKernelDWRaw(Real r, Real hw)
{
    const Real q = r / hw;
    if (q >= 1.0)
        return 0.0;
    const Real t = 1.0 - q;
    return -(140.0 / (Pi * hw * hw * hw)) * q * t * t * t;
}

inline bool TwoScaleKernelActive()
{
    return run_options.interface_kernel == "two_scale";
}

/** W of the ACTIVE Scheme E kernel (legacy by default). */
inline Real SchemeKernelW(Real r)
{
    if (!TwoScaleKernelActive())
        return RhoKernelW(r);
    const Real h = RhoSmoothingH();
    const Real xi = run_options.interface_xi * Sigma();
    const Real beta = run_options.interface_beta;
    const Real ch = 144.0 / (5.0 * h * h);
    const Real cxi = 144.0 / (5.0 * xi * xi);
    return (ch * TwoScaleKernelWRaw(r, h) - beta * cxi * TwoScaleKernelWRaw(r, xi)) /
           (1.0 - beta);
}

/** dW/dr of the ACTIVE Scheme E kernel (legacy by default). */
inline Real SchemeKernelDW(Real r)
{
    if (!TwoScaleKernelActive())
        return RhoKernelDW(r);
    const Real h = RhoSmoothingH();
    const Real xi = run_options.interface_xi * Sigma();
    const Real beta = run_options.interface_beta;
    const Real ch = 144.0 / (5.0 * h * h);
    const Real cxi = 144.0 / (5.0 * xi * xi);
    return (ch * TwoScaleKernelDWRaw(r, h) -
            beta * cxi * TwoScaleKernelDWRaw(r, xi)) / (1.0 - beta);
}

/** Pair-loop cutoff for the ACTIVE Scheme E kernel (legacy: h_rho). */
inline Real SchemeKernelSupport()
{
    if (!TwoScaleKernelActive())
        return RhoSmoothingH();
    return std::max(RhoSmoothingH(), run_options.interface_xi * Sigma());
}


/** Sweep 1: local density and its gradient.  True zero path at lambda = 0. */
class CGSquareGradientDensity : public LocalDynamics, public DataDelegateInner
{
  public:
    explicit CGSquareGradientDensity(BaseInnerRelation &inner_relation)
        : LocalDynamics(inner_relation.getSPHBody()),
          DataDelegateInner(inner_relation),
          rho_(particles_->template getVariableDataByName<Real>("CG_Rho")),
          grad_(particles_->template getVariableDataByName<Vecd>("CG_RhoGrad")) {}

    void interaction(size_t index_i, Real dt = 0.0)
    {
        if (run_options.interface_lambda <= 0.0)
            return;
        Real rho = 0.0;
        Vecd grad = Vecd::Zero();
        const Neighborhood &neighborhood = inner_configuration_[index_i];
        for (size_t n = 0; n != neighborhood.current_size_; ++n)
        {
            const Real r = neighborhood.r_ij_[n];
            if (r <= TinyReal)
                continue;
            // density estimator: ALWAYS the legacy kernel (rho_i = Sum W_h);
            // only the energy kernel is allowed to differ (two-scale).
            const Real w = RhoKernelW(r);
            if (w <= 0.0)          // outside the kernel support
                continue;
            rho += w;
            grad += RhoKernelDW(r) * neighborhood.e_ij_[n];
        }
        rho_[index_i] = rho;
        grad_[index_i] = grad;
    }

    void update(size_t index_i, Real dt = 0.0) {}

  private:
    Real *rho_;
    Vecd *grad_;
};

/** Sweep 2: the force that is the exact gradient of F_sg. */
class CGSquareGradientForce : public LocalDynamics, public DataDelegateInner
{
  public:
    explicit CGSquareGradientForce(BaseInnerRelation &inner_relation)
        : LocalDynamics(inner_relation.getSPHBody()),
          DataDelegateInner(inner_relation),
          force_(particles_->template getVariableDataByName<Vecd>("CG_Force")),
          sg_force_(particles_->template getVariableDataByName<Vecd>("CG_SGForce")),
          grad_(particles_->template getVariableDataByName<Vecd>("CG_RhoGrad")),
          pos_(particles_->template getVariableDataByName<Vecd>("Position")) {}

    void interaction(size_t index_i, Real dt = 0.0)
    {
        if (run_options.interface_lambda <= 0.0)
        {
            sg_force_[index_i] = Vecd::Zero();
            return;
        }
        const Vecd g_i = grad_[index_i];
        // f_ak = coeff * [ W'' (dG.e) e + (W'/r) (dG - (dG.e) e) ]
        const Real coeff = -run_options.interface_lambda / run_options.interface_rho_ref;
        // M1.3 / N1: axisymmetric measure factor J(r) = r/R0 applied to the
        // DISCRETE ENERGY   F = (lambda V0 / 2) Sum_i J_i |G_i|^2 .
        // With J == 1 this loop is IDENTICAL to the pre-M1.3 2D A-1 force.
        const bool axisym = Axisymmetric();
        // N3 J control: --axisym-j-one=1 forces J = 1 inside the (z,r) branch,
        // which also kills the dJ/dr branch (2).  Default off = production.
        const bool j_weighted = axisym && !run_options.axisym_j_one;
        const Real r0 = j_weighted ? AxisRadius() : 1.0;
        const Real J_i = j_weighted ? (pos_[index_i][1] / r0) : 1.0;
        Vecd f = Vecd::Zero();
        const Neighborhood &neighborhood = inner_configuration_[index_i];
        for (size_t n = 0; n != neighborhood.current_size_; ++n)
        {
            const size_t index_j = neighborhood.j_[n];
            const Real r = neighborhood.r_ij_[n];
            if (r <= TinyReal)
                continue;
            const Real w1 = RhoKernelDW(r);
            if (w1 == 0.0)         // outside the support
                continue;
            const Vecd e = neighborhood.e_ij_[n];
            // Complete variation of the J-weighted energy: the pair difference
            // carries the weight, i.e. (J_k G_k - J_j G_j), not (G_k - G_j).
            Vecd dG = Vecd::Zero();
            if (j_weighted)
                dG = J_i * g_i - (pos_[index_j][1] / r0) * grad_[index_j];
            else
                dG = g_i - grad_[index_j];
            const Real dGe = dG.dot(e);
            f += coeff * (RhoKernelDDW(r) * dGe * e +
                          (w1 / r) * (dG - dGe * e));
        }
        // M1.3 / N1: the explicit radial branch produced by dJ/dr in the
        // complete variation of that same discrete energy:
        //     F_k += - (lambda V0 / (2 R0)) |G_k|^2 r_hat
        // It is NOT a hand-added curvature or Laplace force: it is the exact
        // gradient of the discrete functional, scales as 1/R0, and vanishes in
        // the flat limit together with J -> 1.  Removing it while keeping the
        // energy would break energy-force consistency (the FD check fails).
        if (j_weighted)
            f += coeff * (0.5 / r0) * g_i.squaredNorm() * Vecd(0.0, 1.0);
        sg_force_[index_i] = f;
        force_[index_i] += f;
    }

    void update(size_t index_i, Real dt = 0.0) {}

    /**
     * M1.3 / N2.2 REF-CACHED.  Recompute the force through a SEPARATE code path
     * from the SAME cached neighbour geometry (r_ij, e_ij) that the production
     * sweep uses, and return the unified error
     *     E_force = max_i |F_recomputed_i - sg_force_i| / max_i |sg_force_i|
     * This verifies that the production force is exactly the formula applied to
     * the cached geometry.  A REF-RECOMPUTED reference (distances taken from the
     * particle positions instead) is a different object and is NOT required to
     * agree with this to 1e-12.
     */
    Real CachedReferenceMaxRelativeError() const
    {
        const size_t total = particles_->TotalRealParticles();
        const bool axisym = Axisymmetric();
        const bool j_weighted = axisym && !run_options.axisym_j_one;
        const Real r0 = j_weighted ? AxisRadius() : 1.0;
        const Real coeff =
            -run_options.interface_lambda / run_options.interface_rho_ref;
        Real dmax = 0.0, fmax = 0.0;
        for (size_t i = 0; i != total; ++i)
        {
            Vecd f = Vecd::Zero();
            if (run_options.interface_lambda > 0.0)
            {
                const Real Ji = j_weighted ? (pos_[i][1] / r0) : 1.0;
                const Neighborhood &nb = inner_configuration_[i];
                for (size_t n = 0; n != nb.current_size_; ++n)
                {
                    const Real r = nb.r_ij_[n];
                    if (r <= TinyReal)
                        continue;
                    const Real w1 = RhoKernelDW(r);
                    if (w1 == 0.0)
                        continue;
                    const Vecd e = nb.e_ij_[n];
                    Vecd dG = Vecd::Zero();
                    if (j_weighted)
                        dG = Ji * grad_[i] -
                             (pos_[nb.j_[n]][1] / r0) * grad_[nb.j_[n]];
                    else
                        dG = grad_[i] - grad_[nb.j_[n]];
                    const Real dGe = dG.dot(e);
                    f += coeff * (RhoKernelDDW(r) * dGe * e +
                                  (w1 / r) * (dG - dGe * e));
                }
                if (j_weighted)
                    f += coeff * (0.5 / r0) * grad_[i].squaredNorm() *
                         Vecd(0.0, 1.0);
            }
            dmax = std::max(dmax, (f - Vecd(sg_force_[i])).norm());
            fmax = std::max(fmax, sg_force_[i].norm());
        }
        return fmax > 0.0 ? dmax / fmax : 0.0;
    }

    /**
     * N3 diagnostic: per-particle split of the square-gradient force into the
     * J-weighted chain-rule part (1) and the explicit dJ/dr part (2).
     * Read-only; only called when --sg-force-dump=1.
     */
    void DumpForceSplit(const std::string &path) const
    {
        std::ofstream csv(path);
        csv << std::setprecision(17);
        csv << "i,z,r,Fr_chain,Fr_explicit_J,Fr_total,Fz_total,Gz,Gr,grad2\n";
        const size_t total = particles_->TotalRealParticles();
        const bool j_weighted = Axisymmetric() && !run_options.axisym_j_one;
        const Real r0 = j_weighted ? AxisRadius() : 1.0;
        const Real coeff =
            -run_options.interface_lambda / run_options.interface_rho_ref;
        for (size_t i = 0; i != total; ++i)
        {
            Vecd fc = Vecd::Zero(), fe = Vecd::Zero();
            if (run_options.interface_lambda > 0.0)
            {
                const Real Ji = j_weighted ? (pos_[i][1] / r0) : 1.0;
                const Neighborhood &nb = inner_configuration_[i];
                for (size_t n = 0; n != nb.current_size_; ++n)
                {
                    const Real r = nb.r_ij_[n];
                    if (r <= TinyReal)
                        continue;
                    const Real w1 = RhoKernelDW(r);
                    if (w1 == 0.0)
                        continue;
                    const Vecd e = nb.e_ij_[n];
                    Vecd dG = Vecd::Zero();
                    if (j_weighted)
                        dG = Ji * grad_[i] -
                             (pos_[nb.j_[n]][1] / r0) * grad_[nb.j_[n]];
                    else
                        dG = grad_[i] - grad_[nb.j_[n]];
                    const Real dGe = dG.dot(e);
                    fc += coeff * (RhoKernelDDW(r) * dGe * e +
                                   (w1 / r) * (dG - dGe * e));
                }
                if (j_weighted)
                    fe += coeff * (0.5 / r0) * grad_[i].squaredNorm() *
                          Vecd(0.0, 1.0);
            }
            const Vecd ft = fc + fe;
            csv << i << "," << pos_[i][0] << "," << pos_[i][1] << "," << fc[1]
                << "," << fe[1] << "," << ft[1] << "," << ft[0] << ","
                << grad_[i][0] << "," << grad_[i][1] << ","
                << grad_[i].squaredNorm() << "\n";
        }
        csv.close();
    }

  private:
    Vecd *force_;
    Vecd *sg_force_;
    Vecd *grad_;
    Vecd *pos_;
};

//  M1.3R: Scheme E -- pairwise-difference interface energy
//
//      F_E = (lambda/2) V0^2 Sum_{i<j} J_ij Kt_ij (rho_i - rho_j)^2
//      J_ij  = (r_i + r_j) / (2 R0)        (pair-midpoint axisymmetric weight)
//      Kt_ij = c W(r_ij),   c = 144 / (5 h^2)   (closed form; see M1.3R report)
//
//  Continuum limit: (lambda/2) Int (r/R0) |grad rho|^2 dr dz, i.e. the planar
//  branch with J = 1 is exactly the A-1 2D convention, so lambda = 12 keeps its
//  meaning and no recalibration is needed.  The full physical measure 2 pi r is
//  obtained by multiplying the energy by 2 pi R0 (a constant, absorbed in
//  lambda_physical = 2 pi R0 * lambda).
//
//  Properties (all verified in tools/cg_analysis/m13r/m13r_scheme_e.py):
//    * constant density  =>  energy and force are EXACTLY zero;
//    * the force is the complete gradient of that same discrete energy
//      (energy-force FD <= 1.4e-8 at delta = 1e-4, <= 7e-10 at 3e-6);
//    * pairs are symmetric => f_ij = -f_ji => momentum conserving;
//    * no reference-lattice labels => valid under large deformation;
//    * cost: 3 neighbour sweeps (vs 2) and one extra Real per particle.
//
//  The measure factor enters as the PAIR midpoint, so there is no single-particle
//  dJ/dr branch (the analogue of A-1's "term 2").  The surviving explicit
//  dJ/dx and dW/dx branches are gated by (rho_i - rho_j)^2, which vanishes
//  identically on a uniform density field -- that is the whole point.
//=============================================================================
inline bool PairDifferenceScheme()
{
    return Axisymmetric() &&
           run_options.axisym_interface_scheme == "difference_energy";
}

/** pref = (lambda/2) V0^2 c, the coefficient multiplying J_ij W(r_ij). */
inline Real SGDifferencePairPref()
{
    const Real V0 = 1.0 / run_options.interface_rho_ref;
    const Real h = RhoSmoothingH();
    const Real ck = 144.0 / (5.0 * h * h);
    return 0.5 * run_options.interface_lambda * V0 * V0 * ck;
}

/** Pair measure factor J_ij.  --axisym-j-one=1 gives the MATHEMATICALLY
 *  CORRESPONDING planar control: the same Scheme E functional with J = 1,
 *  i.e. (lambda/2) V0^2 Sum (rho_i - rho_j)^2 Kt_ij.  Only the interface
 *  measure is changed; nothing else. */
inline Real SGDifferenceJ(const Real ri, const Real rj)
{
    if (run_options.axisym_j_one)
        return 1.0;
    return 0.5 * (ri + rj) / AxisRadius();
}

/** d J_ij / d r_i; zero in the J = 1 control (no radial measure branch). */
inline Real SGDifferenceDJdr()
{
    return run_options.axisym_j_one ? 0.0 : 0.5 / AxisRadius();
}

/** Sweep 2 of Scheme E: conjugate field  S_i = Sum_j 2 A_ij (rho_i - rho_j). */
class CGSquareGradientConjugate : public LocalDynamics, public DataDelegateInner
{
  public:
    explicit CGSquareGradientConjugate(BaseInnerRelation &inner_relation)
        : LocalDynamics(inner_relation.getSPHBody()),
          DataDelegateInner(inner_relation),
          rho_(particles_->template getVariableDataByName<Real>("CG_Rho")),
          conj_(particles_->template getVariableDataByName<Real>("CG_SGConj")),
          pos_(particles_->template getVariableDataByName<Vecd>("Position")) {}

    void interaction(size_t index_i, Real dt = 0.0)
    {
        if (run_options.interface_lambda <= 0.0)
        {
            conj_[index_i] = 0.0;
            return;
        }
        const Real pref = SGDifferencePairPref();
        const Real r0 = AxisRadius();
        const Real ri = pos_[index_i][1];
        Real s = 0.0;
        const Neighborhood &neighborhood = inner_configuration_[index_i];
        for (size_t n = 0; n != neighborhood.current_size_; ++n)
        {
            const size_t index_j = neighborhood.j_[n];
            const Real r = neighborhood.r_ij_[n];
            if (r <= TinyReal)
                continue;
            const Real w = SchemeKernelW(r);
            if (r >= SchemeKernelSupport())   // support gate (keeps negative lobes)
                continue;
            const Real Jij = SGDifferenceJ(ri, pos_[index_j][1]);
            s += 2.0 * pref * Jij * w * (rho_[index_i] - rho_[index_j]);
        }
        conj_[index_i] = s;
    }

    void update(size_t index_i, Real dt = 0.0) {}

  private:
    Real *rho_;
    Real *conj_;
    Vecd *pos_;
};

/** Sweep 3 of Scheme E: F_k = -dF_E/dx_k  (complete gradient). */
class CGSquareGradientDifferenceForce : public LocalDynamics,
                                        public DataDelegateInner
{
  public:
    explicit CGSquareGradientDifferenceForce(BaseInnerRelation &inner_relation)
        : LocalDynamics(inner_relation.getSPHBody()),
          DataDelegateInner(inner_relation),
          force_(particles_->template getVariableDataByName<Vecd>("CG_Force")),
          sg_force_(particles_->template getVariableDataByName<Vecd>("CG_SGForce")),
          rho_(particles_->template getVariableDataByName<Real>("CG_Rho")),
          conj_(particles_->template getVariableDataByName<Real>("CG_SGConj")),
          grad_(particles_->template getVariableDataByName<Vecd>("CG_RhoGrad")),
          pos_(particles_->template getVariableDataByName<Vecd>("Position")) {}

    /** Single-particle force, evaluated from the CACHED neighbour geometry.
     *  Used both by the production sweep and by the REF-CACHED check. */
    Vecd ForceFromCachedGeometry(size_t index_i, size_t *count_terms = nullptr) const
    {
        if (run_options.interface_lambda <= 0.0)
            return Vecd::Zero();
        const Real pref = SGDifferencePairPref();
        const Real r0 = AxisRadius();
        const Real ri = pos_[index_i][1];
        const Real rho_i = rho_[index_i];
        // chain rule through rho: d rho_k / d x_k = +G_k  (verified against FD)
        Vecd f = -conj_[index_i] * Vecd(grad_[index_i]);
        const Vecd radial_hat(0.0, 1.0);
        size_t nterm = 1;
        const Neighborhood &neighborhood = inner_configuration_[index_i];
        for (size_t n = 0; n != neighborhood.current_size_; ++n)
        {
            const size_t index_j = neighborhood.j_[n];
            const Real r = neighborhood.r_ij_[n];
            if (r <= TinyReal)
                continue;
            // w / dw_k : the ENERGY kernel (may be two-scale);
            // dw_rho   : the DENSITY-estimator derivative (always legacy).
            const Real w = SchemeKernelW(r);
            const Real dw_k = SchemeKernelDW(r);
            const Real dw_rho = RhoKernelDW(r);
            const Real dw = dw_rho;
            if (r >= SchemeKernelSupport())   // support gate (keeps negative lobes)
                continue;
            const Vecd e = neighborhood.e_ij_[n];
            const Real Jij = SGDifferenceJ(ri, pos_[index_j][1]);
            // (1) neighbour part of the density chain rule
            f -= conj_[index_j] * dw * e;
            // (2) explicit d(measure)/dx and dW/dx branches, gated by (d rho)^2
            const Real drho = rho_i - rho_[index_j];
            f -= (drho * drho) * pref *
                 (SGDifferenceDJdr() * w * radial_hat + Jij * dw_k * e);
            ++nterm;
        }
        if (count_terms != nullptr)
            *count_terms = nterm;
        return f;
    }

    void interaction(size_t index_i, Real dt = 0.0)
    {
        const Vecd f = ForceFromCachedGeometry(index_i);
        sg_force_[index_i] = f;
        force_[index_i] += f;
    }

    void update(size_t index_i, Real dt = 0.0) {}

    /** REF-CACHED consistency check: recompute through a separate call path. */
    Real CachedReferenceMaxRelativeError() const
    {
        const size_t total = particles_->TotalRealParticles();
        Real dmax = 0.0, fmax = 0.0;
        for (size_t i = 0; i != total; ++i)
        {
            const Vecd f = ForceFromCachedGeometry(i);
            dmax = std::max(dmax, (f - Vecd(sg_force_[i])).norm());
            fmax = std::max(fmax, sg_force_[i].norm());
        }
        return fmax > 0.0 ? dmax / fmax : 0.0;
    }

    /** Per-particle split: density-chain part vs explicit measure/kernel part. */
    void DumpForceSplit(const std::string &path) const
    {
        std::ofstream csv(path);
        csv << std::setprecision(17);
        csv << "i,z,r,Fr_chain,Fr_explicit_J,Fr_total,Fz_total,rho,S,grad2\n";
        const size_t total = particles_->TotalRealParticles();
        const Real pref = SGDifferencePairPref();
        const Real r0 = AxisRadius();
        for (size_t i = 0; i != total; ++i)
        {
            Vecd fc = Vecd::Zero(), fe = Vecd::Zero();
            if (run_options.interface_lambda > 0.0)
            {
                fc = -conj_[i] * Vecd(grad_[i]);
                const Real ri = pos_[i][1];
                const Real rho_i = rho_[i];
                const Neighborhood &nb = inner_configuration_[i];
                for (size_t n = 0; n != nb.current_size_; ++n)
                {
                    const size_t j = nb.j_[n];
                    const Real r = nb.r_ij_[n];
                    if (r <= TinyReal)
                        continue;
                    const Vecd e = nb.e_ij_[n];
                    const Real w = SchemeKernelW(r), dw_k = SchemeKernelDW(r);
                    const Real dw = RhoKernelDW(r);   // density chain rule
                    if (r >= SchemeKernelSupport())   // support gate (keeps negative lobes)
                        continue;
                    const Real Jij = SGDifferenceJ(ri, pos_[j][1]);
                    fc -= conj_[j] * dw * e;
                    const Real drho = rho_i - rho_[j];
                    fe -= (drho * drho) * pref *
                          (SGDifferenceDJdr() * w * Vecd(0.0, 1.0) + Jij * dw_k * e);
                }
            }
            const Vecd ft = fc + fe;
            csv << i << "," << pos_[i][0] << "," << pos_[i][1] << ","
                << fc[1] << "," << fe[1] << "," << ft[1] << "," << ft[0] << ","
                << rho_[i] << "," << conj_[i] << ","
                << grad_[i].squaredNorm() << "\n";
        }
        csv.close();
    }

  private:
    Vecd *force_;
    Vecd *sg_force_;
    Real *rho_;
    Real *conj_;
    Vecd *grad_;
    Vecd *pos_;
};

//=============================================================================
//  Constraints (numerical, reported separately from the physics)
//=============================================================================
class CGConstraints : public LocalDynamics
{
  public:
    explicit CGConstraints(RealBody &real_body)
        : LocalDynamics(real_body),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          vel_(particles_->getVariableDataByName<Vecd>("Velocity")),
          wrap_count_(0), wall_guard_hits_(0), wall_contact_hits_(0),
          escaped_count_(0) {}

    void exec(Real dt = 0.0)
    {
        wrap_count_ = 0;
        wall_guard_hits_ = 0;
        wall_contact_hits_ = 0;
        escaped_count_ = 0;

        const Real lx = DomainLength();
        const Real ly = DomainHeight();
        // With the Morse wall switched on its own h < r0 branch already
        // provides the near-wall repulsion, so the geometric constraint is
        // demoted to a numerical backstop placed far inside the well.
        // With D0 = 0 there is no wall potential at all and the constraint is
        // the physical hard wall, placed at geometric contact r0 = 0.5 sigma
        // so that the excluded volume matches the Morse case.
        const bool morse_wall = run_options.wall_depth_d0 > 0.0;
        const Real wall_limit = (morse_wall ? run_options.wall_guard_h
                                            : run_options.wall_contact_h) * Sigma();
        const size_t total = particles_->TotalRealParticles();

        for (size_t i = 0; i != total; ++i)
        {
            const FibreContact contact = FibreContactOf(pos_[i]);
            if (contact.h < wall_limit)
            {
                if (morse_wall)
                    ++wall_guard_hits_;
                else
                    ++wall_contact_hits_;
                pos_[i] += contact.normal * (wall_limit - contact.h);
                // Elastic (specular) bounce off the fibre core.  Reflecting the
                // normal velocity keeps the constraint dissipation-free, so it
                // does not act as a hidden energy sink that would bias the
                // kinetic-temperature check.
                const Real v_normal = vel_[i].dot(contact.normal);
                if (v_normal < 0.0)
                    vel_[i] -= 2.0 * v_normal * contact.normal;
            }

            if (run_options.y_reflect)
            {
                if (Axisymmetric())
                {
                    // In (z,r) the lower end of the y axis is the fibre CORE,
                    // not a symmetry plane: reflecting there would push
                    // particles back to r > 0, i.e. inside the solid.  The
                    // r >= R0 side is enforced solely by the pre-existing
                    // geometric constraint through h_wall = r - R0.  Only the
                    // outer radial boundary stays reflecting (a confining
                    // scaffold boundary, to be revisited in M1.2).
                    if (pos_[i][1] > ly)
                    {
                        pos_[i][1] = 2.0 * ly - pos_[i][1];
                        vel_[i][1] = -std::abs(vel_[i][1]);
                    }
                }
                else if (pos_[i][1] < 0.0)
                {
                    pos_[i][1] = -pos_[i][1];
                    vel_[i][1] = std::abs(vel_[i][1]);
                }
                else if (pos_[i][1] > ly)
                {
                    pos_[i][1] = 2.0 * ly - pos_[i][1];
                    vel_[i][1] = -std::abs(vel_[i][1]);
                }
            }

            if (pos_[i][0] < 0.0 || pos_[i][0] > lx)
                ++wrap_count_;
            if (pos_[i][1] < -0.01 * ly || pos_[i][1] > 1.01 * ly)
                ++escaped_count_;
        }
    }

    size_t WrapCount() const { return wrap_count_; }
    size_t WallGuardHits() const { return wall_guard_hits_; }
    size_t WallContactHits() const { return wall_contact_hits_; }
    size_t EscapedCount() const { return escaped_count_; }

  private:
    Vecd *pos_;
    Vecd *vel_;
    size_t wrap_count_;
    size_t wall_guard_hits_;
    size_t wall_contact_hits_;
    size_t escaped_count_;
};

//=============================================================================
//  Particle-fibre wall force.  Runs after the pair interaction and ADDS to the
//  stored force, so the pair interaction must always be evaluated first.
//=============================================================================
class CGWallForce : public LocalDynamics
{
  public:
    explicit CGWallForce(RealBody &real_body)
        : LocalDynamics(real_body),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          force_(particles_->template getVariableDataByName<Vecd>("CG_Force")) {}

    void exec(Real dt = 0.0)
    {
        if (run_options.wall_depth_d0 <= 0.0)
            return;
        const size_t total = particles_->TotalRealParticles();
        for (size_t i = 0; i != total; ++i)
        {
            const FibreContact contact = FibreContactOf(pos_[i]);
            if (contact.h >= WallCutoff())
                continue;
            force_[i] += contact.normal * WallForce(contact.h);
        }
    }

  private:
    Vecd *pos_;
    Vecd *force_;
};

//=============================================================================
//  BAOAB Langevin splitting
//=============================================================================
/** B: half kick with the conservative force currently stored in CG_Force. */
class CGLangevinKick : public LocalDynamics
{
  public:
    explicit CGLangevinKick(RealBody &real_body)
        : LocalDynamics(real_body),
          vel_(particles_->getVariableDataByName<Vecd>("Velocity")),
          force_(particles_->template getVariableDataByName<Vecd>("CG_Force")) {}

    void exec(Real dt = 0.0)
    {
        const Real half = 0.5 * dt; /**< unit mass */
        const size_t total = particles_->TotalRealParticles();
        for (size_t i = 0; i != total; ++i)
            vel_[i] += half * force_[i];
    }

  private:
    Vecd *vel_, *force_;
};

/** A: half drift. */
class CGHalfDrift : public LocalDynamics
{
  public:
    explicit CGHalfDrift(RealBody &real_body)
        : LocalDynamics(real_body),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          vel_(particles_->getVariableDataByName<Vecd>("Velocity")) {}

    void exec(Real dt = 0.0)
    {
        const Real half = 0.5 * dt;
        const size_t total = particles_->TotalRealParticles();
        for (size_t i = 0; i != total; ++i)
            pos_[i] += half * vel_[i];
    }

  private:
    Vecd *pos_, *vel_;
};

/**
 * O: exact Ornstein-Uhlenbeck velocity update.
 *   v <- c1 v + c2 xi,  c1 = exp(-gamma dt), c2 = sqrt(k_B T (1 - c1^2))
 * The noise is keyed by (seed, OriginalID, step, component), so it is
 * independent of the thread schedule.
 */
class CGLangevinOU : public LocalDynamics
{
  public:
    explicit CGLangevinOU(RealBody &real_body)
        : LocalDynamics(real_body),
          vel_(particles_->getVariableDataByName<Vecd>("Velocity")),
          id_(particles_->getVariableDataByName<UnsignedInt>("OriginalID")),
          step_(0) {}

    void setStep(std::uint64_t step) { step_ = step; }

    void exec(Real dt = 0.0)
    {
        const Real c1 = std::exp(-run_options.friction * dt);
        // N3 diagnostic: --noise-off=1 removes the stochastic amplitude only
        // (pure damping).  Default 0 keeps the FDT-consistent production value.
        const Real c2 = run_options.noise_off
                            ? 0.0
                            : std::sqrt(run_options.temperature * (1.0 - c1 * c1));
        const size_t total = particles_->TotalRealParticles();
        for (size_t i = 0; i != total; ++i)
        {
            const std::uint64_t id = static_cast<std::uint64_t>(id_[i]);
            Vecd noise;
            noise[0] = DeterministicGaussian(static_cast<std::uint64_t>(run_options.seed),
                                             id, step_, 0);
            noise[1] = DeterministicGaussian(static_cast<std::uint64_t>(run_options.seed),
                                             id, step_, 1);
            vel_[i] = c1 * vel_[i] + c2 * noise;
        }
    }

  private:
    Vecd *vel_;
    UnsignedInt *id_;
    std::uint64_t step_;
};

/** Initial velocity: Maxwell-Boltzmann at k_B T, or exactly zero. */
class CGInitialVelocity : public LocalDynamics
{
  public:
    explicit CGInitialVelocity(RealBody &real_body)
        : LocalDynamics(real_body),
          vel_(particles_->getVariableDataByName<Vecd>("Velocity")),
          id_(particles_->getVariableDataByName<UnsignedInt>("OriginalID")) {}

    void exec(Real dt = 0.0)
    {
        const size_t total = particles_->TotalRealParticles();
        const bool maxwell = run_options.initial_velocity == "maxwell";
        const Real scale = std::sqrt(run_options.temperature);
        for (size_t i = 0; i != total; ++i)
        {
            if (!maxwell)
            {
                vel_[i] = Vecd::Zero();
                continue;
            }
            const std::uint64_t id = static_cast<std::uint64_t>(id_[i]);
            vel_[i][0] = scale * DeterministicGaussian(
                                    static_cast<std::uint64_t>(run_options.seed), id, 0, 100);
            vel_[i][1] = scale * DeterministicGaussian(
                                    static_cast<std::uint64_t>(run_options.seed), id, 0, 101);
        }
    }

  private:
    Vecd *vel_;
    UnsignedInt *id_;
};

//=============================================================================
//  Diagnostics
//=============================================================================
struct FrameDiagnostics
{
    size_t particle_number = 0;
    Real max_speed = 0.0;
    Real kinetic_temperature = 0.0; /**< <v^2> / (d k_B T / m); 1 at equilibrium */
    Real mean_height_above_fibre = 0.0;
    Real f_ads = 0.0;      /**< fraction of particles with h < h_ads */
    Real wall_min_h = 0.0; /**< smallest particle-surface distance, in sigma */
    size_t escaped = 0;
    Real max_abs_vy = 0.0;
    // A-1 square-gradient diagnostics (all exactly zero when lambda = 0)
    Real sg_rho_mean = 0.0;
    Real sg_rho_min = 0.0;
    Real sg_rho_max = 0.0;
    Real sg_grad_max = 0.0;
    Real sg_force_rms = 0.0;
    Real sg_force_max = 0.0;
    Real sg_net_force = 0.0;
    // M1.3 / N0: READ-ONLY interface-energy diagnostics.  They are never used to
    // build any force; they only make the discrete functional evaluable so that
    // the N1 energy-force finite-difference check has something to compare to.
    //   sg_energy                    : (lambda V0 / 2) Sum_i J_i |G_i|^2
    //   sg_energy_planar_reference   : same sum with J == 1 (the legacy 2D A-1)
    //   sg_energy_raw_2pir           : same sum with J = 2 pi r (diagnostic only)
    Real sg_energy = 0.0;
    Real sg_energy_planar_reference = 0.0;
    Real sg_energy_raw_2pir = 0.0;
    Real sg_J_min = 1.0;
    Real sg_J_max = 1.0;
};

FrameDiagnostics CollectDiagnostics(RealBody &real_body)
{
    FrameDiagnostics d;
    auto *pos = real_body.getBaseParticles().getVariableDataByName<Vecd>("Position");
    auto *vel = real_body.getBaseParticles().getVariableDataByName<Vecd>("Velocity");
    d.particle_number = real_body.getBaseParticles().TotalRealParticles();

    const int dimensions = 2;
    Real depth_sum = 0.0;
    Real v2_sum = 0.0;
    Real min_h = std::numeric_limits<Real>::max();
    size_t bound = 0;
    auto *rho = real_body.getBaseParticles().getVariableDataByName<Real>("CG_Rho");
    auto *grad = real_body.getBaseParticles().getVariableDataByName<Vecd>("CG_RhoGrad");
    auto *sgf = real_body.getBaseParticles().getVariableDataByName<Vecd>("CG_SGForce");
    Real rho_sum = 0.0, rho_min = std::numeric_limits<Real>::max(), rho_max = 0.0;
    Real f2_sum = 0.0;
    Vecd f_net = Vecd::Zero();
    Real e_weighted = 0.0, e_planar = 0.0, e_raw = 0.0;
    Real j_min = std::numeric_limits<Real>::max();
    Real j_max = 0.0;
    for (size_t i = 0; i != d.particle_number; ++i)
    {
        const Real v2 = vel[i].squaredNorm();
        v2_sum += v2;
        d.max_speed = std::max(d.max_speed, std::sqrt(v2));
        d.max_abs_vy = std::max(d.max_abs_vy, std::abs(vel[i][1]));
        const Real h = FibreContactOf(pos[i]).h;
        depth_sum += h;
        min_h = std::min(min_h, h);
        if (h < run_options.adsorption_cutoff_h * Sigma())
            ++bound;
        if (pos[i][1] < -0.01 * DomainHeight() || pos[i][1] > 1.01 * DomainHeight())
            ++d.escaped;
        if (run_options.interface_lambda > 0.0)
        {
            rho_sum += rho[i];
            rho_min = std::min(rho_min, rho[i]);
            rho_max = std::max(rho_max, rho[i]);
            d.sg_grad_max = std::max(d.sg_grad_max, grad[i].norm());
            const Real fn = sgf[i].norm();
            f2_sum += fn * fn;
            d.sg_force_max = std::max(d.sg_force_max, fn);
            f_net += sgf[i];
            // M1.3 / N0 read-only energies.  J is the axisymmetric measure
            // factor: J = r/R0 in axisymmetric mode, J = 1 otherwise (so the
            // legacy 2D path degenerates to the planar reference exactly).
            const Real g2 = grad[i].squaredNorm();
            const Real J = Axisymmetric() ? (pos[i][1] / AxisRadius()) : 1.0;
            e_weighted += J * g2;
            e_planar += g2;
            e_raw += (2.0 * Pi * pos[i][1]) * g2;
            j_min = std::min(j_min, J);
            j_max = std::max(j_max, J);
        }
    }
    const Real n = static_cast<Real>(d.particle_number);
    d.mean_height_above_fibre = depth_sum / n;
    d.wall_min_h = (min_h == std::numeric_limits<Real>::max()) ? 0.0 : min_h / Sigma();
    d.f_ads = static_cast<Real>(bound) / n;
    d.kinetic_temperature =
        v2_sum / (n * static_cast<Real>(dimensions) * run_options.temperature);
    if (run_options.interface_lambda > 0.0)
    {
        d.sg_rho_mean = rho_sum / n;
        d.sg_rho_min = (rho_min == std::numeric_limits<Real>::max()) ? 0.0 : rho_min;
        d.sg_rho_max = rho_max;
        d.sg_force_rms = std::sqrt(f2_sum / n);
        d.sg_net_force = f_net.norm();
        const Real energy_scale =
            0.5 * run_options.interface_lambda / run_options.interface_rho_ref;
        d.sg_energy = energy_scale * e_weighted;
        d.sg_energy_planar_reference = energy_scale * e_planar;
        d.sg_energy_raw_2pir = energy_scale * e_raw;
        d.sg_J_min = (j_min == std::numeric_limits<Real>::max()) ? 1.0 : j_min;
        d.sg_J_max = (j_max == 0.0) ? 1.0 : j_max;
    }
    return d;
}

//=============================================================================
//  M1.3 / N1: energy-force finite-difference check (diagnostic only)
//
//  This is an INDEPENDENT re-implementation of the same discrete functional,
//  evaluated from scratch on an arbitrary configuration with the z minimum
//  image.  It exists so that the solver's analytic force can be compared with
//  the central-difference gradient of the energy it claims to be the gradient
//  of.  It never runs unless --sg-fd-check=1, and it changes nothing.
//=============================================================================
inline Real SGMeasureJ(const Vecd &p, int mode)
{
    // mode 0: axisymmetric J = r/R0 (degenerates to 1 outside axisymmetric)
    // mode 1: planar reference J = 1
    // mode 2: raw 2 pi r (scale reference only)
    if (mode == 0)
        return Axisymmetric() ? (p[1] / AxisRadius()) : 1.0;
    if (mode == 2)
        return 2.0 * Pi * p[1];
    return 1.0;
}

void SGConfigRhoGrad(const std::vector<Vecd> &p, std::vector<Real> &rho,
                     std::vector<Vecd> &grad)
{
    const Real h = RhoSmoothingH();
    const Real lz = DomainLength();
    const size_t n = p.size();
    rho.assign(n, 0.0);
    grad.assign(n, Vecd::Zero());
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
        {
            if (i == j)
                continue;
            Vecd d = p[i] - p[j];
            d[0] -= lz * std::round(d[0] / lz);
            const Real r = d.norm();
            if (r <= TinyReal || r >= h)
                continue;
            rho[i] += RhoKernelW(r);
            grad[i] += RhoKernelDW(r) * (d / r);
        }
}

Real SGConfigEnergy(const std::vector<Vecd> &p, int mode)
{
    std::vector<Real> rho;
    std::vector<Vecd> grad;
    SGConfigRhoGrad(p, rho, grad);
    const Real scale =
        0.5 * run_options.interface_lambda / run_options.interface_rho_ref;
    Real e = 0.0;
    for (size_t i = 0; i < p.size(); ++i)
        e += SGMeasureJ(p[i], mode) * grad[i].squaredNorm();
    return scale * e;
}

/** Reference analytic force: the exact gradient of SGConfigEnergy(., mode).
 *  include_geom = false drops only the dJ/dr radial branch (control only). */
void SGConfigForce(const std::vector<Vecd> &p, int mode, bool include_geom,
                   std::vector<Vecd> &F)
{
    const Real h = RhoSmoothingH();
    const Real lz = DomainLength();
    const size_t n = p.size();
    std::vector<Real> rho;
    std::vector<Vecd> grad;
    SGConfigRhoGrad(p, rho, grad);
    const Real coeff =
        -run_options.interface_lambda / run_options.interface_rho_ref;
    F.assign(n, Vecd::Zero());
    for (size_t i = 0; i < n; ++i)
    {
        const Real Ji = SGMeasureJ(p[i], mode);
        for (size_t j = 0; j < n; ++j)
        {
            if (i == j)
                continue;
            Vecd d = p[i] - p[j];
            d[0] -= lz * std::round(d[0] / lz);
            const Real r = d.norm();
            if (r <= TinyReal || r >= h)
                continue;
            const Vecd e = d / r;
            const Vecd dG = Ji * grad[i] - SGMeasureJ(p[j], mode) * grad[j];
            const Real dGe = dG.dot(e);
            F[i] += coeff * (RhoKernelDDW(r) * dGe * e +
                             (RhoKernelDW(r) / r) * (dG - dGe * e));
        }
        // The dJ/dr branch exists only where J actually depends on position,
        // i.e. only in the axisymmetric branch with mode 0.  In the legacy 2D
        // path J == 1, so dJ/dr == 0 and this branch must be absent.
        if (include_geom && mode == 0 && Axisymmetric())
            F[i] += coeff * (0.5 / AxisRadius()) * grad[i].squaredNorm() *
                    Vecd(0.0, 1.0);
    }
}

//=============================================================================
//  M1.3R: independent from-scratch reconstruction of Scheme E (diagnostic only)
//=============================================================================
inline Real SGDifferenceW(const Real r) { return SchemeKernelW(r); }

/** From-scratch Scheme E energy (brute force over all pairs, z minimum image). */
Real SGDifferenceConfigEnergy(const std::vector<Vecd> &p)
{
    const Real h = SchemeKernelSupport();
    const Real lz = DomainLength();
    const Real r0 = AxisRadius();
    const size_t n = p.size();
    std::vector<Real> rho(n, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
        {
            if (i == j)
                continue;
            Vecd d = p[i] - p[j];
            d[0] -= lz * std::round(d[0] / lz);
            const Real r = d.norm();
            if (r <= TinyReal || r >= h)
                continue;
            rho[i] += SGDifferenceW(r);
        }
    const Real pref = SGDifferencePairPref();
    Real acc = 0.0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
        {
            Vecd d = p[i] - p[j];
            d[0] -= lz * std::round(d[0] / lz);
            const Real r = d.norm();
            if (r <= TinyReal || r >= h)
                continue;
            const Real Jij = SGDifferenceJ(p[i][1], p[j][1]);
            const Real drho = rho[i] - rho[j];
            acc += Jij * SGDifferenceW(r) * drho * drho;
        }
    return pref * acc;
}

/** From-scratch Scheme E force.  include_explicit = false drops the
 *  d(measure)/dx + dW/dx branch and is the control that must FAIL the FD. */
void SGDifferenceConfigForce(const std::vector<Vecd> &p, bool include_explicit,
                             std::vector<Vecd> &F)
{
    const Real h = SchemeKernelSupport();
    const Real lz = DomainLength();
    const Real r0 = AxisRadius();
    const size_t n = p.size();
    std::vector<Real> rho(n, 0.0);
    std::vector<Vecd> grad(n, Vecd::Zero());
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
        {
            if (i == j)
                continue;
            Vecd d = p[i] - p[j];
            d[0] -= lz * std::round(d[0] / lz);
            const Real r = d.norm();
            if (r <= TinyReal || r >= h)
                continue;
            rho[i] += SGDifferenceW(r);
            grad[i] += RhoKernelDW(r) * (d / r);
        }
    const Real pref = SGDifferencePairPref();
    std::vector<Real> S(n, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
        {
            if (i == j)
                continue;
            Vecd d = p[i] - p[j];
            d[0] -= lz * std::round(d[0] / lz);
            const Real r = d.norm();
            if (r <= TinyReal || r >= h)
                continue;
            const Real Jij = SGDifferenceJ(p[i][1], p[j][1]);
            S[i] += 2.0 * pref * Jij * SGDifferenceW(r) * (rho[i] - rho[j]);
        }
    F.assign(n, Vecd::Zero());
    for (size_t i = 0; i < n; ++i)
    {
        F[i] = -S[i] * Vecd(grad[i]);   // d rho_i / d x_i = +grad_i
        for (size_t j = 0; j < n; ++j)
        {
            if (i == j)
                continue;
            Vecd d = p[i] - p[j];
            d[0] -= lz * std::round(d[0] / lz);
            const Real r = d.norm();
            if (r <= TinyReal || r >= h)
                continue;
            const Vecd e = d / r;
            const Real drho = rho[i] - rho[j];
            F[i] -= S[j] * RhoKernelDW(r) * e;
            if (include_explicit)
            {
                const Real Jij = SGDifferenceJ(p[i][1], p[j][1]);
                F[i] -= drho * drho * pref *
                        (SGDifferenceDJdr() * SGDifferenceW(r) * Vecd(0.0, 1.0) +
                         Jij * SchemeKernelDW(r) * e);
            }
        }
    }
}

/** Scheme-aware from-scratch energy used by the FD sweep. */
inline Real ConfigInterfaceEnergy(const std::vector<Vecd> &p, int mode)
{
    return PairDifferenceScheme() ? SGDifferenceConfigEnergy(p)
                                  : SGConfigEnergy(p, mode);
}

struct SGSweepResult
{
    // Absolute errors are normalised by the largest force in the configuration,
    // because a uniform lattice has interior particles whose analytic force is
    // ~0 and whose RELATIVE error would be meaningless.  The relative columns
    // are restricted to particles carrying a non-negligible force.
    Real max_abs_z = 0.0, mean_abs_z = 0.0;
    Real max_abs_r = 0.0, mean_abs_r = 0.0;
    Real max_rel_z = 0.0, max_rel_r = 0.0;
};

/** Energy of the functional as the SOLVER computes it: rho and grad rho are
 *  taken from the solver's own arrays, which are refreshed by re-running its
 *  density sweep on the perturbed positions.  The neighbour list is NOT
 *  rebuilt during the sweep -- that is the correct semantics for a force
 *  finite-difference test, because the analytic force is also a single
 *  evaluation on one fixed neighbour list. */
Real SGEnergyFromSolverArrays(int mode, const Vecd *pos, const Real *rho,
                              const Vecd *grad, size_t n)
{
    const Real scale =
        0.5 * run_options.interface_lambda / run_options.interface_rho_ref;
    Real e = 0.0;
    for (size_t i = 0; i < n; ++i)
        e += SGMeasureJ(pos[i], mode) * grad[i].squaredNorm();
    return scale * e;
}

SGSweepResult SGFinitDifferenceSweep(std::vector<Vecd> p, int mode,
                                     const std::vector<Vecd> &target, Real delta,
                                     Real f_scale)
{
    // NOTE: the perturbation must be evaluated with r_ij and e_ij RECOMPUTED
    // from the perturbed positions.  The solver's neighbourhood caches those
    // values, so re-running its density sweep without rebuilding the
    // configuration leaves the energy completely insensitive to the shift and
    // the finite difference collapses to zero.  That is why this sweep uses the
    // from-scratch reconstruction, and why "solver force == formula force" is
    // checked separately below.
    const size_t n = p.size();
    SGSweepResult out;
    Real sum_z = 0.0, sum_r = 0.0;
    for (size_t i = 0; i < n; ++i)
        for (int comp = 0; comp < 2; ++comp)
        {
            const Real saved = p[i][comp];
            p[i][comp] = saved + delta;
            const Real ep = ConfigInterfaceEnergy(p, mode);
            p[i][comp] = saved - delta;
            const Real em = ConfigInterfaceEnergy(p, mode);
            p[i][comp] = saved;
            const Real fd = -(ep - em) / (2.0 * delta);
            const Real an = target[i][comp];
            const Real abs_err = std::abs(fd - an) / f_scale;
            const bool significant = std::abs(an) > 0.01 * f_scale;
            const Real rel_err =
                significant ? std::abs(fd - an) / std::abs(an) : 0.0;
            if (comp == 0)
            {
                out.max_abs_z = std::max(out.max_abs_z, abs_err);
                sum_z += abs_err;
                out.max_rel_z = std::max(out.max_rel_z, rel_err);
            }
            else
            {
                out.max_abs_r = std::max(out.max_abs_r, abs_err);
                sum_r += abs_err;
                out.max_rel_r = std::max(out.max_rel_r, rel_err);
            }
        }
    out.mean_abs_z = sum_z / static_cast<Real>(n);
    out.mean_abs_r = sum_r / static_cast<Real>(n);
    return out;
}

/** Worst |f_ij + f_ji| / max|f_ij| over the chain-rule pair part only. */
Real SGPairAntisymmetryMax(const std::vector<Vecd> &p)
{
    const Real h = RhoSmoothingH();
    const Real lz = DomainLength();
    const size_t n = p.size();
    std::vector<Real> rho;
    std::vector<Vecd> grad;
    SGConfigRhoGrad(p, rho, grad);
    const Real coeff =
        -run_options.interface_lambda / run_options.interface_rho_ref;
    Real worst = 0.0, biggest = 0.0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
        {
            Vecd d = p[i] - p[j];
            d[0] -= lz * std::round(d[0] / lz);
            const Real r = d.norm();
            if (r <= TinyReal || r >= h)
                continue;
            const Vecd e = d / r;
            const Vecd dGij =
                SGMeasureJ(p[i], 0) * grad[i] - SGMeasureJ(p[j], 0) * grad[j];
            const Real dGe = dGij.dot(e);
            const Vecd tij = coeff * (RhoKernelDDW(r) * dGe * e +
                                      (RhoKernelDW(r) / r) * (dGij - dGe * e));
            Vecd e2 = -e;
            const Vecd dGji =
                SGMeasureJ(p[j], 0) * grad[j] - SGMeasureJ(p[i], 0) * grad[i];
            const Real dGe2 = dGji.dot(e2);
            const Vecd tji_check =
                coeff * (RhoKernelDDW(r) * dGe2 * e2 +
                         (RhoKernelDW(r) / r) * (dGji - dGe2 * e2));
            // t_ij is the force on i from j; tji_check the force on j from i.
            // Pair antisymmetry requires their sum to vanish.
            worst = std::max(worst, (tij + tji_check).norm());
            biggest = std::max(biggest, tij.norm());
        }
    return biggest > 0.0 ? worst / biggest : 0.0;
}

void RunSGFinitDifferenceCheck(
    InteractionWithUpdate<CGSquareGradientDensity> &dens, Vecd *pos, Vecd *sgf,
    Real *rho_solver, Vecd *grad_solver, size_t n, Real e_force_cached)
{
    std::vector<Vecd> p(pos, pos + n);
    std::vector<Real> rho0(rho_solver, rho_solver + n);
    std::vector<Vecd> grad0(grad_solver, grad_solver + n);
    std::vector<Vecd> f_solver(sgf, sgf + n);
    // Targets.  "main" is the solver's own force; "no-geom" is the same force
    // with the dJ/dr branch subtracted analytically, which must FAIL the FD
    // test and thereby demonstrate that the branch is required.
    std::vector<Vecd> f_ref, f_nogeo;
    if (PairDifferenceScheme())
    {
        SGDifferenceConfigForce(p, true, f_ref);   // independent Scheme E formula
        SGDifferenceConfigForce(p, false, f_nogeo); // control: explicit branch off
    }
    else
    {
        SGConfigForce(p, 0, true, f_ref);   // independent formula, WITH dJ/dr
        SGConfigForce(p, 0, false, f_nogeo); // control: dJ/dr removed
    }

    // Cross-check the two independent reconstructions of rho and grad rho.
    // If these disagree, nothing downstream can be trusted.
    std::vector<Real> rho_ref;
    std::vector<Vecd> grad_ref;
    SGConfigRhoGrad(p, rho_ref, grad_ref);
    Real rho_num = 0.0, rho_den = 0.0, grd_num = 0.0, grd_den = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        rho_num = std::max(rho_num, std::abs(rho_ref[i] - rho_solver[i]));
        rho_den = std::max(rho_den, std::abs(rho_ref[i]));
        grd_num = std::max(grd_num, (grad_ref[i] - Vecd(grad_solver[i])).norm());
        grd_den = std::max(grd_den, grad_ref[i].norm());
    }
    const Real rho_agree = rho_den > 0.0 ? rho_num / rho_den : 0.0;
    const Real grd_agree = grd_den > 0.0 ? grd_num / grd_den : 0.0;
    // Which side is missing neighbours?  Report the worst particle explicitly.
    size_t worst_i = 0;
    Real worst_d = -1.0;
    int worst_nb = 0;
    for (size_t i = 0; i < n; ++i)
    {
        const Real diff = (grad_ref[i] - Vecd(grad_solver[i])).norm();
        if (diff > worst_d)
        {
            worst_d = diff;
            worst_i = i;
        }
    }
    {
        const Real hh = RhoSmoothingH();
        const Real lzz = DomainLength();
        for (size_t j = 0; j < n; ++j)
        {
            if (j == worst_i)
                continue;
            Vecd d = p[worst_i] - p[j];
            d[0] -= lzz * std::round(d[0] / lzz);
            if (d.norm() < hh)
                ++worst_nb;
        }
    }
    Real rho_ref_max = 0.0, rho_sol_max = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        rho_ref_max = std::max(rho_ref_max, std::abs(rho_ref[i]));
        rho_sol_max = std::max(rho_sol_max, std::abs(rho_solver[i]));
    }

    // Solver force vs this independent reference (must agree to round-off).
    Real dmax = 0.0, fmax = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        dmax = std::max(dmax, (Vecd(sgf[i]) - f_ref[i]).norm());
        fmax = std::max(fmax, sgf[i].norm());
    }
    // Round-off yardstick: the sum of ABSOLUTE pair contributions per particle.
    // The pair sum in grad rho and in the force involves large cancellations, so
    // the meaningful question is whether the residual sits at the floating-point
    // level of the terms being summed, not whether it is small relative to the
    // (possibly near-zero) result.
    Real roundoff_scale = 0.0;
    {
        const Real hh = RhoSmoothingH();
        const Real lzz = DomainLength();
        const Real coeff =
            -run_options.interface_lambda / run_options.interface_rho_ref;
        std::vector<Real> rho_t;
        std::vector<Vecd> grad_t;
        SGConfigRhoGrad(p, rho_t, grad_t);
        for (size_t i = 0; i < n; ++i)
        {
            Real acc = 0.0;
            for (size_t j = 0; j < n; ++j)
            {
                if (i == j)
                    continue;
                Vecd d = p[i] - p[j];
                d[0] -= lzz * std::round(d[0] / lzz);
                const Real r = d.norm();
                if (r <= TinyReal || r >= hh)
                    continue;
                const Vecd e = d / r;
                const Vecd dGj = SGMeasureJ(p[i], 0) * grad_t[i] -
                                 SGMeasureJ(p[j], 0) * grad_t[j];
                const Real dGe = dGj.dot(e);
                acc += std::abs(coeff) * (std::abs(RhoKernelDDW(r) * dGe) +
                                          std::abs(RhoKernelDW(r) / r) *
                                              (dGj - dGe * e).norm());
            }
            roundoff_scale = std::max(roundoff_scale, acc);
        }
    }
    const Real ref_agreement = fmax > 0.0 ? dmax / fmax : 0.0;
    const Real ref_roundoff =
        roundoff_scale > 0.0 ? dmax / roundoff_scale : 0.0;
    Real fmax_ref = 0.0, fmax_planar = 0.0, fmax_nogeo = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        fmax_ref = std::max(fmax_ref, f_ref[i].norm());
        fmax_nogeo = std::max(fmax_nogeo, f_nogeo[i].norm());
    }

    // Closest approach of any pair to the kernel support edge: a pair sitting
    // exactly on r = h_rho would make the energy discontinuous under a
    // perturbation, so it must be reported and kept away from.
    const Real h = RhoSmoothingH();
    const Real lz = DomainLength();
    Real edge = std::numeric_limits<Real>::max();
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
        {
            Vecd d = p[i] - p[j];
            d[0] -= lz * std::round(d[0] / lz);
            edge = std::min(edge, std::abs(d.norm() - h));
        }

    const Real deltas[4] = {1.0e-4, 3.0e-5, 1.0e-5, 3.0e-6};
    const Real asym = SGPairAntisymmetryMax(p);

    std::ofstream csv2("sg_fd_check.csv");
    csv2 << std::setprecision(10);
    csv2 << "quantity,value\n";
    csv2 << "particles," << n << "\n";
    csv2 << "lambda," << run_options.interface_lambda << "\n";
    csv2 << "h_rho," << RhoSmoothingH() << "\n";
    csv2 << "R0," << AxisRadius() << "\n";
    csv2 << "rho_ref," << run_options.interface_rho_ref << "\n";
    csv2 << "rho_solver_vs_reference_max_rel," << rho_agree << "\n";
    csv2 << "grad_solver_vs_reference_max_rel," << grd_agree << "\n";
    csv2 << "force_solver_vs_reference_max_rel," << ref_agreement << "\n";
    csv2 << "E_force_solver_vs_REF_CACHED," << e_force_cached << "\n";
    csv2 << "force_residual_over_sum_of_absolute_terms," << ref_roundoff << "\n";
    csv2 << "force_residual_abs," << dmax << "\n";
    csv2 << "sum_of_absolute_pair_terms_max," << roundoff_scale << "\n";
    csv2 << "max_force_reference," << fmax_ref << "\n";
    csv2 << "max_force_planar," << fmax_planar << "\n";
    csv2 << "max_force_no_geom," << fmax_nogeo << "\n";
    csv2 << "min_pair_distance_to_kernel_edge," << edge << "\n";
    csv2 << "pair_antisymmetry_worst_over_max," << asym << "\n";
    csv2.close();

    {
        // Per-particle dump so that any solver-vs-formula residual can be
        // attributed to specific particles instead of being averaged away.
        std::ofstream pc("sg_fd_particles.csv");
        pc << std::setprecision(17);
        pc << "i,z,r,rho_ref,rho_solver,grad_ref_z,grad_ref_r,grad_solver_z,"
              "grad_solver_r,nbr_ref,rho_diff,grad_diff,f_solver_r,f_ref_r\n";
        const Real hh = RhoSmoothingH();
        const Real lzz = DomainLength();
        for (size_t i = 0; i < n; ++i)
        {
            int nb = 0;
            for (size_t j = 0; j < n; ++j)
            {
                if (i == j)
                    continue;
                Vecd d = p[i] - p[j];
                d[0] -= lzz * std::round(d[0] / lzz);
                if (d.norm() < hh)
                    ++nb;
            }
            pc << i << "," << p[i][0] << "," << p[i][1] << "," << rho_ref[i]
               << "," << rho_solver[i] << "," << grad_ref[i][0] << ","
               << grad_ref[i][1] << "," << grad_solver[i][0] << ","
               << grad_solver[i][1] << "," << nb << ","
               << (rho_ref[i] - rho_solver[i]) << ","
               << (grad_ref[i] - Vecd(grad_solver[i])).norm() << ","
               << f_solver[i][1] << "," << f_ref[i][1] << "\n";
        }
        pc.close();
    }

    std::ofstream csv3("sg_fd_sweeps.csv");
    csv3 << std::setprecision(10);
    csv3 << "sweep,delta,max_abs_err_z,mean_abs_err_z,max_abs_err_r,"
            "mean_abs_err_r,max_rel_err_z_significant,"
            "max_rel_err_r_significant\n";

    std::cout << "\n=== M1.3 / N1 energy-force finite-difference check ===\n"
              << "  particles=" << n << " lambda=" << run_options.interface_lambda
              << " h_rho=" << h << " R0=" << AxisRadius()
              << " axisymmetric=" << (Axisymmetric() ? 1 : 0) << "\n"
              << "  solver vs independent reference (max rel): rho=" << rho_agree
              << "  grad rho=" << grd_agree << "  force=" << ref_agreement
              << "\n"
              << "  force residual: abs=" << dmax
              << "  / sum|terms|=" << roundoff_scale
              << "  => round-off normalised=" << ref_roundoff << "\n"
              << "  E_force (solver vs REF-CACHED, unified definition) = "
              << e_force_cached << "\n"
              << "  max |force|: J-weighted=" << fmax_ref
              << "  planar=" << fmax_planar << "  no-dJ/dr=" << fmax_nogeo << "\n"
              << "  max |rho|: reference=" << rho_ref_max
              << "  solver=" << rho_sol_max << "\n"
              << "  worst grad mismatch at i=" << worst_i
              << "  pos=(" << p[worst_i][0] << "," << p[worst_i][1] << ")"
              << "  reference neighbours within h_rho=" << worst_nb << "\n"
              << "    rho:  reference=" << rho_ref[worst_i]
              << "  solver=" << rho_solver[worst_i] << "\n"
              << "    grad: reference=(" << grad_ref[worst_i][0] << ","
              << grad_ref[worst_i][1] << ")  solver=(" << grad_solver[worst_i][0]
              << "," << grad_solver[worst_i][1] << ")\n"
              << "  min |r_ij - h_rho| over pairs: " << edge
              << (edge < 1.0e-3 ? "   <-- WARNING: pair near the kernel edge\n"
                                : "\n");

    struct Named { const char *name; int mode; const std::vector<Vecd> *tgt; };
    const Named sweeps[2] = {
        {"main          (FD of energy vs SOLVER force)", 0, &f_ref},
        {"no-geom ctrl  (FD of energy vs force WITHOUT dJ/dr)", 0, &f_nogeo}};

    for (const Named &s : sweeps)
    {
        std::cout << "  [" << s.name << "]\n";
        Real f_scale = 0.0;
        for (const Vecd &v : *s.tgt)
            f_scale = std::max(f_scale, v.norm());
        if (f_scale <= 0.0)
            f_scale = 1.0;
        for (Real delta : deltas)
        {
            const SGSweepResult r =
                SGFinitDifferenceSweep(p, s.mode, *s.tgt, delta, f_scale);
            std::cout << "      delta=" << std::setw(8) << delta
                      << "  abs/|F|max: Fz max=" << std::scientific
                      << std::setprecision(3) << r.max_abs_z
                      << " mean=" << r.mean_abs_z << "  Fr max=" << r.max_abs_r
                      << " mean=" << r.mean_abs_r
                      << "   rel(significant): Fz max=" << r.max_rel_z
                      << " Fr max=" << r.max_rel_r << std::defaultfloat << "\n";
            csv3 << s.name << "," << delta << "," << r.max_abs_z << ","
                 << r.mean_abs_z << "," << r.max_abs_r << "," << r.mean_abs_r
                 << "," << r.max_rel_z << "," << r.max_rel_r << "\n";
        }
    }
    std::cout << "  chain-rule pair antisymmetry: max |t_ij + t_ji| / max |t_ij| = "
              << asym << "\n"
              << "  (reference force agreement and antisymmetry are production"
                 " properties; the FD sweeps above are the gate)\n"
              << "=== sg FD check complete; exiting without evolving ===\n";
    csv3.close();
}
} // namespace

//=============================================================================
//  main
//=============================================================================
/**
 * The whole case runs here.  Argument validation and the simulation body both
 * report refusal and failure by throwing std::exception; main() below is a
 * thin wrapper that turns those into a readable message and a defined exit
 * status.  Before this split an escaped exception called std::terminate, which
 * on MSVC aborts through __fastfail (exit 0xC0000409) and prints NOTHING --
 * an invalid command line looked exactly like a crashed process.
 */
static int RunCgCase(int ac, char *av[])
{
    ParseCommandLine(ac, av);

    const Real spacing = ParticleSpacing();
    const Real stiffness = PairStiffness(run_options.dt_reference_separation * Sigma());
    const Real omega = std::sqrt(stiffness / (0.5)); /**< reduced mass m/2 with m = 1 */
    const Real dt_limit = 2.0 / omega;
    const Real dt_auto = run_options.dt_cfl * dt_limit;
    const Real eps_eff_over_kBT = EffectiveWellDepth() / run_options.temperature;
    const Real wall_eps_eff_over_kBT =
        WallEffectiveDepth() / run_options.temperature;
    // The wall adds its own stiffness.  Evaluate it at a reference stand-off
    // that is a little outside the numerical guard, since a particle can never
    // legitimately get closer than that.
    const Real wall_h_ref = 1.5 * run_options.wall_guard_h * Sigma();
    const Real wall_stiffness =
        run_options.wall_depth_d0 > 0.0 ? WallStiffness(wall_h_ref) : 0.0;
    const Real wall_omega = std::sqrt(std::max(wall_stiffness, 0.0));
    const Real wall_dt_limit = wall_omega > TinyReal ? 2.0 / wall_omega : 1.0e9;
    const Real dt_auto_wall =
        run_options.dt_cfl * std::min(dt_limit, wall_dt_limit);
    const Real dt = (run_options.dt > 0.0) ? run_options.dt : dt_auto_wall;

    const Vecd domain_lower(0.0, 0.0);
    const Vecd domain_upper(DomainLength(), DomainHeight());
    BoundingBoxd system_bounds(domain_lower, domain_upper);

    SPHSystem sph_system(system_bounds, spacing);
    sph_system.setReloadParticles(false);
    sph_system.handleCommandlineOptions(1, av); /**< only the program name */
    IO::getEnvironment().appendOutputFolder(run_options.output_tag);

    const int target_number = TargetParticleNumber();
    std::cout << "CG self-assembly prototype -- Langevin revision\n"
              << "  git_commit=" << run_options.git_commit << "\n"
              << "  case=" << run_options.case_id
              << " seed=" << run_options.seed
              << " threads=" << run_options.threads << "\n"
              << "  reduced units: m=1 sigma=" << Sigma()
              << " k_BT=" << run_options.temperature
              << " gamma=" << run_options.friction << "\n"
              << "  axisymmetric mode=" << run_options.axisymmetric
              << (Axisymmetric() ? " -> (z,r) cylinder R(z)=R0 const,"
                                   " h_wall = r - R0, z periodic"
                                 : " -> legacy 2D (x,y) path, bit-identical")
              << "\n"
              << "  domain=" << DomainLength() << " x " << DomainHeight()
              << " (periodic in x) fibre_half_width=" << FibreHalfWidth() << "\n"
              << "  area_fraction=" << run_options.area_fraction
              << " target_particles=" << target_number
              << " spacing=" << spacing << " (2h=" << 2.6 * spacing << ")\n"
              << "  WCA: sigma=" << Sigma()
              << " epsilon=" << run_options.wca_epsilon
              << " cutoff=" << WCACutoff() << "\n"
              << "  Morse: D=" << run_options.morse_depth
              << " r_e=" << MorseRe()
              << " alpha=" << MorseAlpha()
              << " cutoff=" << run_options.morse_cutoff
              << " truncation=shifted_potential\n"
              << "  Morse form: U_M(r) = D[exp(-2 alpha (r - r_e)) - 2 exp(-alpha (r - r_e))]\n"
              << "  eps_eff(k_BT)=" << eps_eff_over_kBT
              << "  (D/k_BT=" << run_options.morse_depth / run_options.temperature
              << "); combined WCA+Morse well depth\n"
              << "  wall: D0=" << run_options.wall_depth_d0
              << " eps_wall_eff(k_BT)=" << wall_eps_eff_over_kBT
              << " alpha=" << WallAlpha() << " r0=" << WallRe()
              << " rc=" << WallCutoff() << " shifted_potential\n"
              << "        fibre shape=" << run_options.fibre_shape
              << " radius=" << FibreRadius() << " length=" << FibreLength()
              << " perimeter=" << FibrePerimeter() << "\n"
              << "        h = centre to fibre SURFACE (h=0.5 sigma is contact;"
                 " strip: h=|y-Ly/2|-a; cylinder: distance to the segment - R)\n"
              << "        guard_h=" << (run_options.wall_depth_d0 > 0.0
                                           ? run_options.wall_guard_h
                                           : run_options.wall_contact_h)
              << (run_options.wall_depth_d0 > 0.0
                      ? " (numerical backstop)" : " (PHYSICAL hard wall, D0=0)")
              << " h_ads=" << run_options.adsorption_cutoff_h << "\n"
              << "  dt=" << dt << " (auto pair limit " << dt_auto
              << ", wall limit " << wall_dt_limit
              << ", chosen " << dt_auto_wall << ")\n";
    if (run_options.interface_lambda > 0.0)
        std::cout << "  A-1 square gradient: lambda=" << run_options.interface_lambda
                  << " h_rho=" << RhoSmoothingH() << " rho_ref=" << run_options.interface_rho_ref
                  << " V0=" << 1.0 / run_options.interface_rho_ref
                  << "  (F_sg = (lambda/2) Sum_i V0 |grad rho_i|^2, Wendland C2 kernel)\n";
    else
        std::cout << "  A-1 square gradient: lambda=0 (disabled, true zero path)\n";

    //-------------------------------------------------------------------------
    //  M1.1 geometry / scaffold validation.  Pure geometry probe: it evaluates
    //  h_wall and the wall-force direction at prescribed radii BEFORE any
    //  dynamics, and writes them to a file.  It touches no particle state.
    //-------------------------------------------------------------------------
    if (Axisymmetric())
    {
        std::ofstream geo("axisym_geometry_selfcheck.csv");
        geo << std::setprecision(10);
        geo << "quantity,value,expected,note\n";
        auto grow = [&](const std::string &k, Real v, const std::string &e,
                        const char *n) { geo << k << "," << v << "," << e << "," << n << "\n"; };

        grow("mode", 0.0, "filmonly", "axisymmetric branch active");
        grow("R0", AxisRadius(), "constant", "R(z) = R0; no gradient, no end cap");
        grow("Lz", DomainLength(), "periodic", "axial period");
        grow("r_outer", DomainHeight(), "reflecting",
             "outer radial boundary is scaffold confinement (M1.2 to revisit)");
        grow("wall_re", WallRe(), "0.5", "Morse-wall well position = contact");
        grow("wall_cutoff", WallCutoff(), "2.5", "wall force is zero beyond this h");

        // Required probes from the work order, plus three extra radii that make
        // the FORCE SIGN test non-trivial (0.5 and 2.5 are both force zeros).
        const Real probes[6] = {-0.5, 0.25, 0.5, 0.75, 1.0, 2.5};
        for (int k = 0; k < 6; ++k)
        {
            const Real dr = probes[k];
            const Vecd p(0.5 * DomainLength(), AxisRadius() + dr);
            const FibreContact c = FibreContactOf(p);
            const Vecd f = c.normal * WallForce(c.h);
            const std::string tag = "r=R0" + std::string(dr < 0.0 ? "-" : "+") +
                                    std::to_string(std::abs(dr));
            grow("h_wall[" + tag + "]", c.h, std::to_string(dr), "h_wall must equal r - R0");
            grow("normal_z[" + tag + "]", c.normal[0], "0", "no axial component of the normal");
            grow("normal_r[" + tag + "]", c.normal[1], "1", "outward +r, never -r");
            grow("force_r[" + tag + "]", f[1],
                 dr < 0.0    ? ">0 repulsive"
                 : dr < 0.5  ? ">0 repulsive (inside the well minimum)"
                 : dr == 0.5 ? "=0 at the well minimum"
                 : dr < 2.5  ? "<0 attractive (toward the cylinder)"
                             : "=0 at the wall cutoff",
                 "WallForce sign convention: + means push away from the fibre");
        }
        geo.close();
        std::cout << "  axisym selfcheck -> axisym_geometry_selfcheck.csv\n";
    }

    //-------------------------------------------------------------------------
    //  Bodies
    //-------------------------------------------------------------------------
    SolidBody fibre(sph_system, makeShared<FibreStripShape>("Fibre"));
    fibre.defineMatterMaterial<Solid>();
    fibre.generateParticles<BaseParticles, Lattice>();

    SolidBody particles_body(sph_system, makeShared<CGDomainShape>("CGParticles"));
    particles_body.defineMatterMaterial<Solid>();
    particles_body.generateParticles<BaseParticles, CGRandomScatter>();
    RegisterCGStateVariables(particles_body.getBaseParticles());

    InnerRelation particles_inner(particles_body);

    //-------------------------------------------------------------------------
    //  Dynamics
    //-------------------------------------------------------------------------
    InteractionWithUpdate<CGPairInteraction> pair_interaction(particles_inner);
    InteractionWithUpdate<CGSquareGradientDensity> sg_density(particles_inner);
    InteractionWithUpdate<CGSquareGradientForce> sg_force(particles_inner);
    InteractionWithUpdate<CGSquareGradientConjugate> sg_conjugate(particles_inner);
    InteractionWithUpdate<CGSquareGradientDifferenceForce> sg_diff_force(particles_inner);

    // M1.3R: the active interface-energy scheme.  moving_j = frozen A-1 path;
    // difference_energy = Scheme E (needs one extra conjugate sweep).
    auto exec_interface_energy = [&](Real dt) {
        sg_density.exec(dt);
        if (PairDifferenceScheme())
        {
            sg_conjugate.exec(dt);
            sg_diff_force.exec(dt);
        }
        else
        {
            sg_force.exec(dt);
        }
    };
    CGWallForce wall_force(particles_body);
    CGInitialVelocity initial_velocity(particles_body);
    CGLangevinKick langevin_kick(particles_body);
    CGHalfDrift half_drift(particles_body);
    CGLangevinOU langevin_ou(particles_body);
    CGConstraints constraints(particles_body);

    PeriodicAlongAxis periodic_along_x(particles_body.getSPHBodyBounds(), xAxis);
    PeriodicConditionUsingCellLinkedList periodic_condition(particles_body, periodic_along_x);

    BodyStatesRecordingToVtp body_states_recording(sph_system);

    std::unique_ptr<tbb::global_control> thread_control;
    if (run_options.threads > 0)
        thread_control = std::make_unique<tbb::global_control>(
            tbb::global_control::max_allowed_parallelism,
            static_cast<size_t>(run_options.threads));

    //-------------------------------------------------------------------------
    //  parameters.csv (the model record, including the Morse truncation)
    //-------------------------------------------------------------------------
    {
        std::ofstream csv("parameters.csv");
        csv << std::setprecision(10);
        auto row = [&](const char *k, const std::string &v, const char *note) {
            csv << k << "," << v << "," << note << "\n";
        };
        csv << "parameter,value,description\n";
        row("schema", "cg-fiber-selfassembly/run-parameters", "written by this case");
        row("git_commit", run_options.git_commit,
            "source revision baked in at build time; --git-commit overrides");
        row("case", std::to_string(run_options.case_id),
            "0 = WCA + Langevin, 1 = WCA + Morse + Langevin");
        row("seed", std::to_string(run_options.seed), "drives the counter-based noise");
        row("threads", std::to_string(run_options.threads), "0 = library default");
        row("reduced_mass", "1", "fixed");
        row("sigma", std::to_string(Sigma()), "fixed length unit");
        row("kBT", std::to_string(run_options.temperature), "fixed at 1 for this revision");
        row("friction_gamma", std::to_string(run_options.friction),
            "NOT calibrated; no literature value yet");
        row("thermostat", "BAOAB with exact OU sub-step",
            "FDT-consistent: c1=exp(-gamma dt), c2=sqrt(kBT(1-c1^2))");
        row("noise_key", "seed + OriginalID + timestep + component",
            "counter-based SplitMix64; independent of thread schedule");
        row("axisymmetric", run_options.axisymmetric,
            "off = legacy 2D (x,y); filmonly = (z,r) cylinder R=R0 const "
            "(geometry scaffold only: no 2 pi r weighting, no measure term)");
        row("axisym_interface_scheme", run_options.axisym_interface_scheme,
            "moving_j = Scheme A (lambda V0/2) Sum_i J_i |G_i|^2 (frozen); "
            "difference_energy = Scheme E (lambda/2) V0^2 Sum_{i<j} J_ij Kt_ij "
            "(rho_i-rho_j)^2, Kt = 144/(5 h^2) W; axisymmetric only");
        row("axis_radius_const", std::to_string(run_options.axis_radius),
            "R0/sigma, constant-radius (z,r) cylinder; active when axisymmetric!=off");
        row("film_thickness", std::to_string(run_options.film_thickness),
            "H_f/sigma, REQUESTED film thickness; use realised_thickness from "
            "axisym_film_selfcheck.csv for geometry comparisons");
        row("film_inner_offset", std::to_string(FilmInnerOffset()),
            "d_min: innermost layer radius offset = wall well position (0.5 sigma)");
        row("film_source", Axisymmetric() ? "axisymmetric_annulus" : "none",
            "which builder produced the film; strip film lives on PC2's "
            "pc2-a5-film branch and is not present here");
        row("film_perturb_amp", std::to_string(run_options.film_perturb_amp),
            "A/sigma, axial modulation amplitude; 0 = none");
        row("film_perturb_mode", std::to_string(run_options.film_perturb_mode),
            "n in A sin(2 pi n z / Lz)");
        row("axisym_jacobian", run_options.axisym_jacobian ? "1" : "0",
            "explicit -kBT ln(2 pi r) measure; off by default in M1");
        row("wca_epsilon", std::to_string(run_options.wca_epsilon), "A-level form");
        row("wca_sigma", std::to_string(Sigma()), "");
        row("wca_cutoff", std::to_string(WCACutoff()), "2^(1/6) sigma, energy shifted");
        row("morse_depth_D", std::to_string(run_options.morse_depth), "0 disables");
        row("morse_D_over_kBT",
            std::to_string(run_options.morse_depth / run_options.temperature),
            "this is the MORSE parameter D, not the depth of the combined well");
        row("eps_eff", std::to_string(eps_eff_over_kBT * run_options.temperature),
            "actual depth of the combined WCA+Morse well");
        row("eps_eff_over_kBT",
            std::to_string(eps_eff_over_kBT),
            "report this alongside D/k_BT");
        row("wall_depth_D0", std::to_string(run_options.wall_depth_d0),
            "nominal eps_pf / k_BT; 0 disables the wall potential");
        row("wall_effective_depth",
            std::to_string(WallEffectiveDepth()),
            "depth of the SHIFTED well; slightly below D0");
        row("wall_effective_depth_over_kBT",
            std::to_string(wall_eps_eff_over_kBT),
            "report this alongside wall_depth_D0");
        row("wall_energy_at_cutoff", std::to_string(WallRawPotential(WallCutoff())),
            "E_raw(rc) subtracted by the shift");
        row("wall_force_step_at_cutoff", std::to_string(WallForce(WallCutoff())),
            "residual force dropped at rc");
        row("wall_alpha", std::to_string(WallAlpha()), "inverse length, 3/sigma");
        row("wall_re", std::to_string(WallRe()),
            "well position = geometric contact = 0.5 sigma");
        row("wall_cutoff", std::to_string(WallCutoff()), "2.5 sigma");
        row("wall_shift_type", "shifted potential (energy shift only)",
            "E = E_raw(h) - E_raw(rc) for h < rc");
        row("wall_h_definition",
            (run_options.fibre_shape == "cylinder")
                ? "h = |r - nearest point of the axis segment| - fibre_radius (centre to SURFACE)"
                : "h = |y - Ly/2| - fibre_half_width (centre to SURFACE)",
            "NOT centre to axis, and NOT a surface-to-surface gap");
        row("wall_guard_h", std::to_string(run_options.wall_guard_h),
            "numerical backstop; only meaningful when D0 > 0");
        row("wall_contact_h", std::to_string(run_options.wall_contact_h),
            "hard-wall stand-off used when D0 = 0 to match the excluded volume");
        row("adsorption_cutoff_h", std::to_string(run_options.adsorption_cutoff_h),
            "h_ads for f_ads");
        row("wall_dt_limit", std::to_string(wall_dt_limit),
            "2/omega from the wall stiffness at 1.5*guard_h");
        row("interface_gradient_lambda", std::to_string(run_options.interface_lambda),
            "A-1 square-gradient strength; 0 = disabled (true zero path)");
        row("interface_gradient_h", std::to_string(run_options.interface_h),
            "kernel support h_rho of the local density, in sigma");
        row("interface_gradient_rho_ref", std::to_string(run_options.interface_rho_ref),
            "V0 = 1/rho_ref, the constant volume per particle in F_sg");
        row("interface_gradient_kernel", "Wendland C2, support h_rho",
            "W = (7/(pi h^2))(1-q)^4(4q+1); dW/dr and d2W/dr2 analytic");
        row("interface_gradient_force", "exact gradient of the discrete F_sg",
            "f_ak = -lambda V0 [W''(dG.e)e + (W'/r)(dG-(dG.e)e)]; f_ak = -f_ka");
        row("morse_formula",
            "U_M(r) = D*[exp(-2*alpha*(r-r_e)) - 2*exp(-alpha*(r-r_e))]",
            "alpha is an INVERSE length; there is no length parameter 'a'");
        row("morse_re", std::to_string(MorseRe()), "well position; locked to sigma");
        row("morse_alpha", std::to_string(MorseAlpha()),
            "inverse length; locked to 3/sigma (half width at -D/2 is 1.763/alpha)");
        row("morse_cutoff", std::to_string(MorseCutoff()),
            "locked default r_e + 6/alpha = 3 sigma");
        row("morse_truncation", "shifted potential (energy shift only)",
            "U_shifted = U_M(r) - U_M(rc) for r < rc; potential continuous, force steps to zero");
        row("morse_U_at_cutoff", std::to_string(MorsePotential(MorseCutoff())),
            "U_M(rc) subtracted by the shift");
        row("morse_force_step_at_cutoff", std::to_string(MorseRawForce(MorseCutoff())),
            "residual force at rc that is dropped (size of the force discontinuity)");
        row("morse_U_at_wca_cutoff", std::to_string(MorsePotential(WCACutoff())),
            "deepest accessible point: r_e < WCA cut-off, so the combined well "
            "minimum sits at the WCA cut-off, not at r_e");
        row("area_fraction", std::to_string(run_options.area_fraction),
            "2D area fraction of discs of diameter sigma");
        row("particle_number",
            std::to_string(run_options.init_mode == "cluster"
                               ? run_options.cluster_count
                               : target_number),
            run_options.init_mode == "cluster"
                ? "actual number of particles in the pre-condensed cluster"
                : "target number from the area fraction");
        row("domain_length", std::to_string(DomainLength()), "periodic in x");
        row("domain_height", std::to_string(DomainHeight()), "reflecting in y");
        row("fibre_half_width", std::to_string(FibreHalfWidth()), "rigid, non-interacting");
        row("fibre_shape", run_options.fibre_shape,
            "strip = flat ribbon spanning the period (no curvature); cylinder = capsule");
        row("fibre_radius", std::to_string(FibreRadius()),
            "cap radius of the cylinder fibre (half-width for the strip)");
        row("fibre_length", std::to_string(FibreLength()),
            "straight-segment length of the cylinder fibre");
        row("fibre_perimeter", std::to_string(FibrePerimeter()),
            "2D circumference available for adsorption");
        row("fibre_area", std::to_string(FibreArea()), "excluded area of the fibre");
        row("free_area", std::to_string(FreeArea()), "domain area minus fibre area");
        row("init_mode", run_options.init_mode,
            "scatter = random RSA dispersion; cluster = one pre-condensed "
            "aggregate; lattice = uniform periodic lattice (G1 bulk test)");
        row("cluster_shape", run_options.cluster_shape,
            "disc | ellipse | rect (only used when init_mode=cluster)");
        row("cluster_aspect", std::to_string(run_options.cluster_aspect),
            "long/short ratio of the initial cluster");
        row("cluster_packing", std::to_string(run_options.cluster_packing),
            "area fraction of the initial triangular lattice");
        row("cluster_count", std::to_string(run_options.cluster_count),
            "particles in the initial cluster");
        row("particle_spacing", std::to_string(spacing), "2h = 2.6 dx >= pair cutoff");
        row("dt", std::to_string(dt), run_options.dt > 0.0 ? "user" : "automatic");
        row("dt_stability_limit", std::to_string(dt_limit),
            "2/omega from the pair stiffness at the reference separation");
        row("end_time", std::to_string(run_options.end_time), "reduced time units");
        row("frames", std::to_string(run_options.frames), "");
        row("initial_separation", std::to_string(run_options.initial_separation), "");
        row("initial_velocity", run_options.initial_velocity, "maxwell or zero");
        row("model_status", "mixed A/C",
            "WCA + Langevin + Morse form are A-level; the locked numbers "
            "(alpha=3/sigma, r_e=sigma, r_c=re+6/alpha, shifted-potential "
            "truncation, gamma=1) are first-version choices");
    }

    //-------------------------------------------------------------------------
    //  Initialise
    //-------------------------------------------------------------------------
    initial_velocity.exec();
    sph_system.initializeSystemCellLinkedLists();
    periodic_condition.update_cell_linked_list_.exec();
    sph_system.initializeSystemConfigurations();

    pair_interaction.ResetCounters();
    pair_interaction.exec();
    wall_force.exec();
    exec_interface_energy(0.0);

    // N3 diagnostic: dump the term(1)/term(2) split once, at t = 0.
    if (run_options.sg_force_dump != 0)
    {
        if (PairDifferenceScheme())
            sg_diff_force.DumpForceSplit("sg_force_split.csv");
        else
            sg_force.DumpForceSplit("sg_force_split.csv");
        std::cout << "  sg force split -> sg_force_split.csv\n";
    }

    // M1.3 / N1 gate: evaluate the analytic force once, then compare it with
    // the finite-difference gradient of the same discrete energy.  Diagnostic
    // path only; --sg-fd-check defaults to 0 and nothing below runs in
    // production.
    if (run_options.sg_fd_check != 0)
    {
        RunSGFinitDifferenceCheck(
            sg_density,
            particles_body.getBaseParticles().getVariableDataByName<Vecd>("Position"),
            particles_body.getBaseParticles().getVariableDataByName<Vecd>("CG_SGForce"),
            particles_body.getBaseParticles().getVariableDataByName<Real>("CG_Rho"),
            particles_body.getBaseParticles().getVariableDataByName<Vecd>("CG_RhoGrad"),
            particles_body.getBaseParticles().TotalRealParticles(),
            PairDifferenceScheme() ? sg_diff_force.CachedReferenceMaxRelativeError()
                                   : sg_force.CachedReferenceMaxRelativeError());
        return 0;
    }

    std::ofstream selfcheck("selfcheck.csv");
    selfcheck << "time,particles,kinetic_temperature,D_over_kBT,eps_eff_over_kBT,"
                 "wall_D0_over_kBT,wall_eps_eff_over_kBT,"
                 "max_speed,max_abs_vy,"
                 "min_neighbour_distance,mean_height_above_fibre,f_ads,wall_min_h,"
                 "wall_guard_hits,wall_contact_hits,"
                 "wrapped_count,escaped_count,periodic_pairs,dt,steps,"
                 "sg_rho_mean,sg_rho_min,sg_rho_max,sg_grad_max,"
                 "sg_force_rms,sg_force_max,sg_net_force,"
                 "sg_energy,sg_energy_planar_reference,sg_energy_raw_2pir,"
                 "sg_J_min,sg_J_max\n";
    selfcheck << std::setprecision(10);

    auto report = [&](Real time, Real dt_now, size_t steps, bool write_frame) {
        const FrameDiagnostics d = CollectDiagnostics(particles_body);
        selfcheck << time << "," << d.particle_number << ","
                  << d.kinetic_temperature << ","
                  << run_options.morse_depth / run_options.temperature << ","
                  << eps_eff_over_kBT << ","
                  << run_options.wall_depth_d0 / run_options.temperature << ","
                  << wall_eps_eff_over_kBT << ","
                  << d.max_speed << ","
                  << d.max_abs_vy << "," << pair_interaction.MinimumSeparation() << ","
                  << d.mean_height_above_fibre << "," << d.f_ads << ","
                  << d.wall_min_h << ","
                  << constraints.WallGuardHits() << ","
                  << constraints.WallContactHits()
                  << "," << constraints.WrapCount() << "," << d.escaped << ","
                  << pair_interaction.PeriodicPairCount() << "," << dt_now << ","
                  << steps << ","
                  << d.sg_rho_mean << "," << d.sg_rho_min << "," << d.sg_rho_max << ","
                  << d.sg_grad_max << "," << d.sg_force_rms << ","
                  << d.sg_force_max << "," << d.sg_net_force << ","
                  << d.sg_energy << "," << d.sg_energy_planar_reference << ","
                  << d.sg_energy_raw_2pir << ","
                  << d.sg_J_min << "," << d.sg_J_max << "\n";
        selfcheck.flush();
        std::cout << std::fixed << std::setprecision(4)
                  << "T=" << time << " N=" << d.particle_number
                  << " T_kin=" << d.kinetic_temperature
                  << " max_speed=" << d.max_speed
                  << " min_sep=" << pair_interaction.MinimumSeparation()
                  << " f_ads=" << d.f_ads
                  << " h_min=" << d.wall_min_h
                  << " guard=" << constraints.WallGuardHits()
                  << " hard=" << constraints.WallContactHits()
                  << " wrapped=" << constraints.WrapCount()
                  << " periodic_pairs=" << pair_interaction.PeriodicPairCount()
                  << " sgE=" << d.sg_energy
                  << " sgE_planar=" << d.sg_energy_planar_reference
                  << " J=[" << d.sg_J_min << "," << d.sg_J_max << "]"
                  << " dt=" << dt_now << "\n";
        if (write_frame)
            body_states_recording.writeToFile(steps);
        pair_interaction.ResetCounters();
    };

    report(0.0, dt, 0, true);

    //-------------------------------------------------------------------------
    //  M1.4 §21: residual-force / error-budget time series.
    //  Written every --sg-budget-every steps.  B_r = |<Fr_measure>| / RMS(F_int)
    //  is the fixed budget metric used to decide whether an observed axial
    //  growth could be a side effect of the radial measure residual.
    //-------------------------------------------------------------------------
    std::ofstream budget_csv;
    const Vecd *budget_sgf = nullptr;

    //-------------------------------------------------------------------------
    //  Stage 9: one-shot post-equilibration single-mode perturbation.
    //  Same radial mapping as M1.4:
    //      h = r - R0,  f = 1 + A sin(2 pi m z / Lz),  r <- r + (h - h_in)(f - 1)
    //  applied to the CURRENT positions once, at the first step whose physical
    //  time reaches --equilibrate-until.  z is untouched and no random
    //  component is added.  Inactive (and bit-identical to the old path) unless
    //  --perturb-after-equilibration=1.
    //-------------------------------------------------------------------------
    bool eq_perturb_done = false;
    auto apply_post_eq_perturbation = [&](Real t_now) {
        if (eq_perturb_done || !run_options.perturb_after_equilibration ||
            run_options.film_perturb_amp == 0.0)
            return;
        const Real amp = run_options.film_perturb_amp;
        const Real mode_m = static_cast<Real>(run_options.film_perturb_mode);
        const Real h_in = FilmInnerOffset();
        const Real r0 = AxisRadius();
        const Real lz = DomainLength();
        const Real kz = 2.0 * Pi * mode_m / lz;
        Vecd *pos = particles_body.getBaseParticles()
                        .getVariableDataByName<Vecd>("Position");
        const size_t n = particles_body.getBaseParticles().TotalRealParticles();
        Real vol_before = 0.0, vol_after = 0.0;
        for (size_t i = 0; i != n; ++i)
        {
            const Real r = pos[i][1];
            const Real h = r - r0;
            const Real f = 1.0 + amp * std::sin(kz * pos[i][0]);
            vol_before += (2.0 * r * h - h * h);
            const Real r_new = r + (h - h_in) * (f - 1.0);
            const Real h_new = r_new - r0;
            vol_after += (2.0 * r_new * h_new - h_new * h_new);
            pos[i][1] = r_new;
        }
        std::cout << "  [Stage9] post-equilibration perturbation applied at t=" << t_now
                  << "  A=" << amp << " m=" << run_options.film_perturb_mode
                  << "  vol_ratio=" << (vol_before > 0.0 ? vol_after / vol_before : 0.0)
                  << "\n";
        std::ofstream fh("eq_perturb_applied.txt");
        fh << std::setprecision(17) << "t_eq=" << t_now << "\nA=" << amp
           << "\nm=" << run_options.film_perturb_mode
           << "\nvol_ratio=" << (vol_before > 0.0 ? vol_after / vol_before : 0.0)
           << "\nvol_metric=sum(2 r h - h^2)\n";
        fh.close();
        eq_perturb_done = true;
    };
    if (run_options.sg_budget_every > 0)
    {
        budget_sgf = particles_body.getBaseParticles()
                         .getVariableDataByName<Vecd>("CG_SGForce");
        budget_csv.open("sg_budget.csv");
        budget_csv << std::setprecision(10);
        budget_csv << "time,steps,N,mean_Fr,rms_Fr,rms_F,rms_Fz,mean_Fz,B_r,"
                      "mean_r,mean_h\n";
    }
    auto write_budget = [&](Real t, size_t st) {
        if (run_options.sg_budget_every <= 0 || !budget_csv.is_open())
            return;
        const Vecd *pos = particles_body.getBaseParticles()
                              .getVariableDataByName<Vecd>("Position");
        const size_t n = particles_body.getBaseParticles().TotalRealParticles();
        Real s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0, s5 = 0.0, sr = 0.0, sh = 0.0;
        for (size_t i = 0; i != n; ++i)
        {
            const Real fr = budget_sgf[i][1];
            const Real fz = budget_sgf[i][0];
            const Real fn = budget_sgf[i].norm();
            s1 += fr;
            s2 += fr * fr;
            s3 += fn * fn;
            s4 += fz * fz;
            s5 += fz;
            sr += pos[i][1];
            sh += pos[i][1] - AxisRadius();
        }
        const Real nn = static_cast<Real>(n);
        const Real mean_fr = s1 / nn;
        const Real rms_f = std::sqrt(s3 / nn);
        budget_csv << t << "," << st << "," << n << "," << mean_fr << ","
                   << std::sqrt(s2 / nn) << "," << rms_f << ","
                   << std::sqrt(s4 / nn) << "," << (s5 / nn) << ","
                   << (rms_f > 0.0 ? std::abs(mean_fr) / rms_f : 0.0) << ","
                   << (sr / nn) << "," << (sh / nn) << "\n";
    };
    write_budget(0.0, 0);

    //-------------------------------------------------------------------------
    //  BAOAB main loop
    //-------------------------------------------------------------------------
    Real physical_time = 0.0;
    const Real output_interval = run_options.end_time / static_cast<Real>(run_options.frames);
    size_t steps = 0;
    size_t frame = 0;
    TickCount wall_start = TickCount::now();

    while (physical_time < run_options.end_time - 1.0e-12)
    {
        Real dt_now = dt;
        if (physical_time + dt_now > run_options.end_time)
            dt_now = run_options.end_time - physical_time;

        langevin_kick.exec(dt_now);   // B/2 with F(x)
        half_drift.exec(dt_now);      // A/2
        langevin_ou.setStep(static_cast<std::uint64_t>(steps) + 1);
        langevin_ou.exec(dt_now);     // O
        half_drift.exec(dt_now);      // A/2

        constraints.exec(dt_now);
        periodic_condition.bounding_.exec();
        particles_body.updateCellLinkedList();
        periodic_condition.update_cell_linked_list_.exec();
        particles_inner.updateConfiguration();

        pair_interaction.exec(dt_now);
        wall_force.exec(dt_now);
        exec_interface_energy(dt_now);
        langevin_kick.exec(dt_now);   // B/2 with F(x_new)

        physical_time += dt_now;
        ++steps;

        // Stage 9 hook: fire once, as soon as Phase E has reached t_eq.
        if (run_options.equilibrate_until > 0.0 &&
            physical_time >= run_options.equilibrate_until)
            apply_post_eq_perturbation(physical_time);

        const FrameDiagnostics now = CollectDiagnostics(particles_body);
        if (!std::isfinite(now.max_speed) || now.max_speed > 1.0e3)
            throw std::runtime_error("blow-up detected: max_speed=" +
                                     std::to_string(now.max_speed));
        if (now.escaped > 0)
            throw std::runtime_error("particle escaped the domain.");
        const Real min_sep = pair_interaction.MinimumSeparation();
        if (min_sep > 0.0 && min_sep < 0.80 * Sigma())
            throw std::runtime_error(
                "pair core violated: min separation " + std::to_string(min_sep) +
                " < 0.80 sigma; reduce dt.");

        if (physical_time >= (frame + 1) * output_interval - 1.0e-9 ||
            physical_time >= run_options.end_time - 1.0e-12)
        {
            ++frame;
            report(physical_time, dt_now, steps, true);
        }
        else if (run_options.sg_budget_every > 0 &&
                 steps % static_cast<size_t>(run_options.sg_budget_every) == 0)
        {
            write_budget(physical_time, steps);
        }
    }

    const Real wall_seconds = (TickCount::now() - wall_start).seconds();
    std::cout << "finished: steps=" << steps
              << " wall_seconds=" << wall_seconds << "\n";
    selfcheck.close();
    return 0;
}

int main(int ac, char *av[])
{
    try
    {
        return RunCgCase(ac, av);
    }
    catch (const std::exception &e)
    {
        // e.g. "--frames must be at least 2." -- say what was wrong instead
        // of dying silently.
        std::cerr << "error: " << e.what() << std::endl;
        return 2;
    }
    catch (...)
    {
        std::cerr << "error: unrecognised exception" << std::endl;
        return 2;
    }
}
//=============================================================================
