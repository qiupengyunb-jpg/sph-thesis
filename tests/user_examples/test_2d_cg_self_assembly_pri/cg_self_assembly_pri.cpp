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
Real ExclusionRadius() { return FibreHalfWidth() + run_options.exclusion_margin; }
Real Sigma() { return run_options.sigma; }
Real WCACutoff() { return std::pow(2.0, 1.0 / 6.0) * Sigma(); }

Real LargestPairCutoff();

Real FreeArea()
{
    return DomainLength() * DomainHeight() -
           2.0 * FibreHalfWidth() * DomainLength();
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

    particles.addEvolvingVariable<Vecd>("Velocity");
    particles.addEvolvingVariable<Vecd>("CG_Force");

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
    if (run_options.area_fraction <= 0.0 || run_options.area_fraction >= 0.55)
        throw std::runtime_error("--area-fraction must be in (0,0.55).");
    if (run_options.initial_separation <= 0.0 ||
        run_options.initial_separation > LargestPairCutoff())
        throw std::runtime_error("--initial-separation is outside the pair cut-off.");
    if (run_options.frames < 2)
        throw std::runtime_error("--frames must be at least 2.");
}

//=============================================================================
//  Geometry
//=============================================================================
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
        const Real centre_y = FibreCentreY();
        const Real exclusion = ExclusionRadius();
        const Real min_separation = run_options.initial_separation;
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
            if (std::abs(candidate[1] - centre_y) < exclusion)
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
//  Constraints (numerical, reported separately from the physics)
//=============================================================================
class CGConstraints : public LocalDynamics
{
  public:
    explicit CGConstraints(RealBody &real_body)
        : LocalDynamics(real_body),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          vel_(particles_->getVariableDataByName<Vecd>("Velocity")),
          wrap_count_(0), inside_fibre_count_(0), escaped_count_(0) {}

    void exec(Real dt = 0.0)
    {
        wrap_count_ = 0;
        inside_fibre_count_ = 0;
        escaped_count_ = 0;

        const Real lx = DomainLength();
        const Real ly = DomainHeight();
        const Real centre_y = FibreCentreY();
        const Real exclusion = ExclusionRadius();
        const size_t total = particles_->TotalRealParticles();

        for (size_t i = 0; i != total; ++i)
        {
            Real offset = pos_[i][1] - centre_y;
            if (std::abs(offset) < exclusion)
            {
                ++inside_fibre_count_;
                const Real sign = (offset >= 0.0) ? 1.0 : -1.0;
                pos_[i][1] = centre_y + sign * exclusion;
                // Elastic (specular) bounce off the fibre core.  Reflecting the
                // normal velocity keeps the constraint dissipation-free, so it
                // does not act as a hidden energy sink that would bias the
                // kinetic-temperature check.
                if (vel_[i][1] * sign < 0.0)
                    vel_[i][1] = -vel_[i][1];
            }

            if (run_options.y_reflect)
            {
                if (pos_[i][1] < 0.0)
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
    size_t InsideFibreCount() const { return inside_fibre_count_; }
    size_t EscapedCount() const { return escaped_count_; }

  private:
    Vecd *pos_;
    Vecd *vel_;
    size_t wrap_count_;
    size_t inside_fibre_count_;
    size_t escaped_count_;
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
        const Real c2 = std::sqrt(run_options.temperature * (1.0 - c1 * c1));
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
    size_t inside_fibre = 0;
    size_t escaped = 0;
    Real max_abs_vy = 0.0;
};

FrameDiagnostics CollectDiagnostics(RealBody &real_body)
{
    FrameDiagnostics d;
    auto *pos = real_body.getBaseParticles().getVariableDataByName<Vecd>("Position");
    auto *vel = real_body.getBaseParticles().getVariableDataByName<Vecd>("Velocity");
    d.particle_number = real_body.getBaseParticles().TotalRealParticles();

    const Real centre_y = FibreCentreY();
    const int dimensions = 2;
    Real depth_sum = 0.0;
    Real v2_sum = 0.0;
    for (size_t i = 0; i != d.particle_number; ++i)
    {
        const Real v2 = vel[i].squaredNorm();
        v2_sum += v2;
        d.max_speed = std::max(d.max_speed, std::sqrt(v2));
        d.max_abs_vy = std::max(d.max_abs_vy, std::abs(vel[i][1]));
        depth_sum += std::abs(pos[i][1] - centre_y) - FibreHalfWidth();
        if (std::abs(pos[i][1] - centre_y) < ExclusionRadius() - 1.0e-9)
            ++d.inside_fibre;
        if (pos[i][1] < -0.01 * DomainHeight() || pos[i][1] > 1.01 * DomainHeight())
            ++d.escaped;
    }
    const Real n = static_cast<Real>(d.particle_number);
    d.mean_height_above_fibre = depth_sum / n;
    d.kinetic_temperature =
        v2_sum / (n * static_cast<Real>(dimensions) * run_options.temperature);
    return d;
}
} // namespace

//=============================================================================
//  main
//=============================================================================
int main(int ac, char *av[])
{
    ParseCommandLine(ac, av);

    const Real spacing = ParticleSpacing();
    const Real stiffness = PairStiffness(run_options.dt_reference_separation * Sigma());
    const Real omega = std::sqrt(stiffness / (0.5)); /**< reduced mass m/2 with m = 1 */
    const Real dt_limit = 2.0 / omega;
    const Real dt_auto = run_options.dt_cfl * dt_limit;
    const Real dt = (run_options.dt > 0.0) ? run_options.dt : dt_auto;
    const Real eps_eff_over_kBT = EffectiveWellDepth() / run_options.temperature;

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
              << "  dt=" << dt << " (auto limit " << dt_auto
              << " from stiffness " << stiffness << " at r="
              << run_options.dt_reference_separation << ")\n";

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
        row("particle_number", std::to_string(target_number), "");
        row("domain_length", std::to_string(DomainLength()), "periodic in x");
        row("domain_height", std::to_string(DomainHeight()), "reflecting in y");
        row("fibre_half_width", std::to_string(FibreHalfWidth()), "rigid, non-interacting");
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

    std::ofstream selfcheck("selfcheck.csv");
    selfcheck << "time,particles,kinetic_temperature,D_over_kBT,eps_eff_over_kBT,"
                 "max_speed,max_abs_vy,"
                 "min_neighbour_distance,mean_height_above_fibre,inside_fibre_count,"
                 "wrapped_count,escaped_count,periodic_pairs,dt,steps\n";
    selfcheck << std::setprecision(10);

    auto report = [&](Real time, Real dt_now, size_t steps, bool write_frame) {
        const FrameDiagnostics d = CollectDiagnostics(particles_body);
        selfcheck << time << "," << d.particle_number << ","
                  << d.kinetic_temperature << ","
                  << run_options.morse_depth / run_options.temperature << ","
                  << eps_eff_over_kBT << ","
                  << d.max_speed << ","
                  << d.max_abs_vy << "," << pair_interaction.MinimumSeparation() << ","
                  << d.mean_height_above_fibre << "," << constraints.InsideFibreCount()
                  << "," << constraints.WrapCount() << "," << d.escaped << ","
                  << pair_interaction.PeriodicPairCount() << "," << dt_now << ","
                  << steps << "\n";
        selfcheck.flush();
        std::cout << std::fixed << std::setprecision(4)
                  << "T=" << time << " N=" << d.particle_number
                  << " T_kin=" << d.kinetic_temperature
                  << " max_speed=" << d.max_speed
                  << " min_sep=" << pair_interaction.MinimumSeparation()
                  << " inside_fibre=" << constraints.InsideFibreCount()
                  << " wrapped=" << constraints.WrapCount()
                  << " periodic_pairs=" << pair_interaction.PeriodicPairCount()
                  << " dt=" << dt_now << "\n";
        if (write_frame)
            body_states_recording.writeToFile(steps);
        pair_interaction.ResetCounters();
    };

    report(0.0, dt, 0, true);

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
        langevin_kick.exec(dt_now);   // B/2 with F(x_new)

        physical_time += dt_now;
        ++steps;

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
    }

    const Real wall_seconds = (TickCount::now() - wall_start).seconds();
    std::cout << "finished: steps=" << steps
              << " wall_seconds=" << wall_seconds << "\n";
    selfcheck.close();
    return 0;
}
