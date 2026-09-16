/**
 * @file cg_self_assembly_pri.cpp
 * @brief Minimal coarse-grained (mesoscopic) particle self-assembly prototype,
 *        stage 1 -- Case 0 "stable dispersion baseline".
 *
 * ---------------------------------------------------------------------------
 * SELF-DEVELOPED TEMPORARY MODEL -- NUMERICAL TEST PURPOSE ONLY.
 * The pair force used here is a simple, bounded, hand-rolled stand-in.  It is
 * NOT taken from the literature and must never be cited as a literature
 * method.  It exists so that geometry, periodic boundary, random
 * initialisation, neighbour search, the integrator and the self-checks can be
 * validated before the literature agent supplies the recommended potentials
 * and parameter ranges.
 * ---------------------------------------------------------------------------
 *
 * Reduced units: particle diameter d0 = 1, particle mass m0 = 1,
 * energy unit e0 = 1, time unit tau0 = sqrt(m0 * d0^2 / e0) = 1.
 *
 * Geometry: 2D rectangle [0,Lx) x [0,Ly), periodic along x.
 *   The rigid equal-diameter fibre is a horizontal strip of half-width a
 *   centred at y = Ly/2.  Its axis is x, i.e. the periodic direction, so the
 *   strip spans exactly one period and the fibre is infinite along x.
 *
 * Case 0 physics: particle-particle short-range repulsion ONLY.
 *   No attraction, no fibre adsorption, no random kick.
 *
 * Mechanisms that are deliberately NOT part of the physics but are needed for
 * a well defined problem, and are therefore reported separately:
 *   * fibre exclusion   -- hard geometric constraint, position only, no force
 *   * transverse bounds -- specular reflection at y = 0 and y = Ly
 *   * drag              -- numerical global damping (default off in Case 0)
 */

#include "sphinxsys.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <limits>
#include <string>
#include <vector>

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
    Real domain_length = 60.0;
    Real domain_height = 40.0;
    Real fibre_half_width = 5.0;
    Real area_fraction = 0.20;
    Real resolution = 2.0; /**< SPH lattice points per unit length */
    Real repulsion_cutoff = 1.0;
    Real repulsion_stiffness = 1.0;
    Real initial_separation = 0.95;
    Real exclusion_margin = 0.50;
    Real drag = 0.0;
    Real dt = 0.0; /**< 0 means automatic */
    Real end_time = 50.0;
    int frames = 50;
    bool y_reflect = true;
    std::string output_tag = "cg_case0";
};

RunOptions run_options;

Real DomainLength() { return run_options.domain_length; }
Real DomainHeight() { return run_options.domain_height; }
Real FibreHalfWidth() { return run_options.fibre_half_width; }
Real FibreCentreY() { return 0.5 * DomainHeight(); }
Real ExclusionRadius() { return FibreHalfWidth() + run_options.exclusion_margin; }

/** Free area = domain minus the fibre strip. */
Real FreeArea()
{
    return DomainLength() * DomainHeight() -
           2.0 * FibreHalfWidth() * DomainLength();
}

/** Target particle number for the requested hard-disk area fraction. */
int TargetParticleNumber()
{
    const Real disk_area = 0.25 * Pi * 1.0 * 1.0; /**< disks of diameter d0 = 1 */
    const Real n = run_options.area_fraction * FreeArea() / disk_area;
    return std::max(1, static_cast<int>(std::llround(n)));
}

/**
 * SPH lattice spacing used for the fibre discretisation AND, more
 * importantly, for the neighbour-search cut-off.
 *
 * SPHinXsys builds the neighbour list with 2h = 2.6 * dx, so the lattice
 * spacing must satisfy 2.6 * dx >= largest pair-interaction cut-off.
 * Otherwise the neighbour list would silently truncate the interaction.
 */
Real ParticleSpacing()
{
    const Real requested = 1.0 / run_options.resolution;
    const Real required = run_options.repulsion_cutoff / 2.6;
    return std::max(requested, required);
}

/**
 * A bare SolidBody only registers Density and Mass on its own; Velocity and
 * Acceleration are created lazily by whichever interaction happens to need
 * them.  This case owns its own dynamics, so the per-particle state is
 * registered explicitly here.
 *
 * Every quantity that must travel with a particle has to be an *evolving*
 * variable: the periodic cell-linked-list update sorts the particle array and
 * permutes only the evolving variables.  Registering Velocity without that
 * flag silently scrambles it as soon as a particle is sorted.
 */
void RegisterCGStateVariables(BaseParticles &particles)
{
    const Vecd zero = Vecd::Zero().eval();
    particles.registerStateVariableData<Vecd>("Velocity", zero);
    particles.registerStateVariableData<Vecd>("Acceleration", zero);
    particles.registerStateVariableData<Vecd>("CG_Force", zero);
    particles.registerStateVariableData<Vecd>("CG_Acceleration", zero);

    particles.addEvolvingVariable<Vecd>("Velocity");
    particles.addEvolvingVariable<Vecd>("Acceleration");
    particles.addEvolvingVariable<Vecd>("CG_Force");
    particles.addEvolvingVariable<Vecd>("CG_Acceleration");

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

Real ToReal(const std::string &argument, const char *flag)
{
    return std::stod(argument.substr(std::string(flag).size()));
}

int ToInt(const std::string &argument, const char *flag)
{
    return std::stoi(argument.substr(std::string(flag).size()));
}

std::string ToString(const std::string &argument, const char *flag)
{
    return argument.substr(std::string(flag).size());
}

void ParseCommandLine(int argc, char *argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument(argv[i]);
        if (StartsWith(argument, "--case="))
            run_options.case_id = ToInt(argument, "--case=");
        else if (StartsWith(argument, "--seed="))
            run_options.seed = ToInt(argument, "--seed=");
        else if (StartsWith(argument, "--domain-length="))
            run_options.domain_length = ToReal(argument, "--domain-length=");
        else if (StartsWith(argument, "--domain-height="))
            run_options.domain_height = ToReal(argument, "--domain-height=");
        else if (StartsWith(argument, "--fibre-half-width="))
            run_options.fibre_half_width = ToReal(argument, "--fibre-half-width=");
        else if (StartsWith(argument, "--area-fraction="))
            run_options.area_fraction = ToReal(argument, "--area-fraction=");
        else if (StartsWith(argument, "--resolution="))
            run_options.resolution = ToReal(argument, "--resolution=");
        else if (StartsWith(argument, "--repulsion-cutoff="))
            run_options.repulsion_cutoff = ToReal(argument, "--repulsion-cutoff=");
        else if (StartsWith(argument, "--repulsion-stiffness="))
            run_options.repulsion_stiffness = ToReal(argument, "--repulsion-stiffness=");
        else if (StartsWith(argument, "--initial-separation="))
            run_options.initial_separation = ToReal(argument, "--initial-separation=");
        else if (StartsWith(argument, "--exclusion-margin="))
            run_options.exclusion_margin = ToReal(argument, "--exclusion-margin=");
        else if (StartsWith(argument, "--drag="))
            run_options.drag = ToReal(argument, "--drag=");
        else if (StartsWith(argument, "--dt="))
            run_options.dt = ToReal(argument, "--dt=");
        else if (StartsWith(argument, "--end-time="))
            run_options.end_time = ToReal(argument, "--end-time=");
        else if (StartsWith(argument, "--frames="))
            run_options.frames = ToInt(argument, "--frames=");
        else if (StartsWith(argument, "--y-reflect="))
            run_options.y_reflect = ToInt(argument, "--y-reflect=") != 0;
        else if (StartsWith(argument, "--output-tag="))
            run_options.output_tag = ToString(argument, "--output-tag=");
    }

    if (run_options.case_id != 0)
        throw std::runtime_error(
            "--case: only case 0 (stable dispersion baseline) is implemented "
            "in this revision; cases 1 and 2 are staged for later rounds.");
    if (run_options.domain_length <= 0.0 || run_options.domain_height <= 0.0)
        throw std::runtime_error("domain size must be positive.");
    if (run_options.fibre_half_width <= 0.0 ||
        2.0 * ExclusionRadius() >= run_options.domain_height)
        throw std::runtime_error("fibre does not fit inside the domain.");
    if (run_options.area_fraction <= 0.0 || run_options.area_fraction >= 0.55)
        throw std::runtime_error("--area-fraction must be in (0,0.55).");
    if (run_options.repulsion_cutoff <= 0.0 ||
        run_options.repulsion_stiffness <= 0.0)
        throw std::runtime_error("repulsion parameters must be positive.");
    if (run_options.initial_separation <= 0.0 ||
        run_options.initial_separation > run_options.repulsion_cutoff)
        throw std::runtime_error(
            "--initial-separation must be in (0, --repulsion-cutoff].");
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

/** Rigid equal-diameter fibre: horizontal strip whose axis is x (periodic). */
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

        // Uniform bucket grid used only to reject too-close candidates.  The
        // x direction is periodic, so the bucket index wraps around.
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
                        delta[0] -= lx * std::round(delta[0] / lx); /**< minimum image in x */
                        if (delta.norm() < min_separation)
                            return true;
                    }
                }
            return false;
        };

        std::mt19937 rng(static_cast<std::mt19937::result_type>(run_options.seed));
        std::uniform_real_distribution<Real> uniform_x(0.0, lx);
        std::uniform_real_distribution<Real> uniform_y(0.0, ly);

        const long long max_attempts = 4000LL * static_cast<long long>(target);
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
                  << " min_separation=" << min_separation
                  << " buckets=" << nx << "x" << ny << "\n";

        // The coarse-grained particle carries unit mass (rho0 = 1, volume = 1).
        for (const Vecd &position : accepted)
            addPositionAndVolumetricMeasure(position, 1.0);
    }
};

namespace
{
//=============================================================================
//  Particle-particle short-range repulsion (Case 0 physics, self-developed)
//=============================================================================
class CGPairRepulsion : public LocalDynamics, public DataDelegateInner
{
  public:
    explicit CGPairRepulsion(BaseInnerRelation &inner_relation)
        : LocalDynamics(inner_relation.getSPHBody()),
          DataDelegateInner(inner_relation),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          force_(particles_->template getVariableDataByName<Vecd>("CG_Force")),
          min_separation_(std::numeric_limits<Real>::max()),
          periodic_pair_count_(0) {}

    void interaction(size_t index_i, Real dt = 0.0)
    {
        const Real cutoff = run_options.repulsion_cutoff;
        const Real stiffness = run_options.repulsion_stiffness;
        Vecd force = Vecd::Zero();
        Real local_min = std::numeric_limits<Real>::max();

        const Neighborhood &neighborhood = inner_configuration_[index_i];
        for (size_t n = 0; n != neighborhood.current_size_; ++n)
        {
            const Real r = neighborhood.r_ij_[n];
            local_min = std::min(local_min, r);
            // The neighbour list is built through the periodic cell lists, so a
            // pair whose particles sit on opposite sides of the domain can
            // still be a neighbour.  Counting those is the decisive check that
            // the axial periodic boundary really is active.
            if (std::abs(pos_[index_i][0] - pos_[neighborhood.j_[n]][0]) >
                0.5 * DomainLength())
                periodic_pair_count_.fetch_add(1, std::memory_order_relaxed);
            if (r >= cutoff || r <= TinyReal)
                continue;
            // e_ij points from j to i, so +e_ij is the repulsive direction.
            const Real magnitude = stiffness * (cutoff - r) / cutoff;
            force += magnitude * neighborhood.e_ij_[n];
        }
        force_[index_i] = force;
        min_separation_ = std::min(min_separation_, local_min);
    }

    void update(size_t index_i, Real dt = 0.0) {}

    /** Smallest neighbour distance seen during the last interaction sweep. */
    Real MinimumSeparation() const
    {
        return min_separation_ == std::numeric_limits<Real>::max()
                   ? -1.0
                   : min_separation_;
    }

    void ResetMinimumSeparation() { min_separation_ = std::numeric_limits<Real>::max(); }

    void ResetCounters()
    {
        ResetMinimumSeparation();
        periodic_pair_count_.store(0, std::memory_order_relaxed);
    }

    /** Number of directed neighbour pairs formed across the periodic boundary. */
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
            // (1) fibre exclusion: hard geometric constraint, no force
            Real offset = pos_[i][1] - centre_y;
            if (std::abs(offset) < exclusion)
            {
                ++inside_fibre_count_;
                const Real sign = (offset >= 0.0) ? 1.0 : -1.0;
                pos_[i][1] = centre_y + sign * exclusion;
                // remove the velocity component pushing into the fibre
                if (vel_[i][1] * sign < 0.0)
                    vel_[i][1] = 0.0;
            }

            // (2) transverse domain bounds: specular reflection
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

            // (3) count how many particles needed an axial wrap this step
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
//  Velocity-Verlet integrator (one force evaluation per step)
//=============================================================================
class CGPositionPrediction : public LocalDynamics
{
  public:
    explicit CGPositionPrediction(RealBody &real_body)
        : LocalDynamics(real_body),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          vel_(particles_->getVariableDataByName<Vecd>("Velocity")),
          acc_(particles_->template getVariableDataByName<Vecd>(
              "CG_Acceleration")) {}

    void exec(Real dt = 0.0)
    {
        const size_t total = particles_->TotalRealParticles();
        for (size_t i = 0; i != total; ++i)
            pos_[i] += vel_[i] * dt + 0.5 * acc_[i] * dt * dt;
    }

  private:
    Vecd *pos_, *vel_, *acc_;
};

class CGVelocityUpdate : public LocalDynamics
{
  public:
    explicit CGVelocityUpdate(RealBody &real_body)
        : LocalDynamics(real_body),
          vel_(particles_->getVariableDataByName<Vecd>("Velocity")),
          acc_(particles_->template getVariableDataByName<Vecd>(
              "CG_Acceleration")),
          force_(particles_->template getVariableDataByName<Vecd>("CG_Force")) {}

    void exec(Real dt = 0.0)
    {
        const size_t total = particles_->TotalRealParticles();
        for (size_t i = 0; i != total; ++i)
        {
            const Vecd acc_new = force_[i]; /**< unit mass: a = F / m, m = 1 */
            vel_[i] += 0.5 * (acc_[i] + acc_new) * dt;
            acc_[i] = acc_new;
        }
    }

  private:
    Vecd *vel_, *acc_, *force_;
};

/** Consistent start-up: seed the stored acceleration with the initial force. */
class CGAccelerationInitialize : public LocalDynamics
{
  public:
    explicit CGAccelerationInitialize(RealBody &real_body)
        : LocalDynamics(real_body),
          acc_(particles_->template getVariableDataByName<Vecd>(
              "CG_Acceleration")),
          force_(particles_->template getVariableDataByName<Vecd>("CG_Force")) {}

    void exec(Real dt = 0.0)
    {
        const size_t total = particles_->TotalRealParticles();
        for (size_t i = 0; i != total; ++i)
            acc_[i] = force_[i];
    }

  private:
    Vecd *acc_, *force_;
};

/** Numerical global damping, NOT part of the physical model. */
class CGNumericalDrag : public LocalDynamics
{
  public:
    explicit CGNumericalDrag(RealBody &real_body)
        : LocalDynamics(real_body),
          vel_(particles_->getVariableDataByName<Vecd>("Velocity")) {}

    void exec(Real dt = 0.0)
    {
        const Real gamma = run_options.drag;
        if (gamma <= 0.0)
            return;
        const Real factor = 1.0 / (1.0 + gamma * dt);
        const size_t total = particles_->TotalRealParticles();
        for (size_t i = 0; i != total; ++i)
            vel_[i] *= factor;
    }

  private:
    Vecd *vel_;
};

//=============================================================================
//  Helpers for the per-frame numerical self-check
//=============================================================================
struct FrameDiagnostics
{
    Real max_speed = 0.0;
    Real max_speed_y = 0.0;
    Real mean_height_from_wall = 0.0;
    size_t inside_fibre = 0;
    size_t escaped = 0;
    size_t wrapped = 0;
    size_t particle_number = 0;
};

FrameDiagnostics CollectDiagnostics(RealBody &real_body)
{
    FrameDiagnostics d;
    auto *pos = real_body.getBaseParticles().getVariableDataByName<Vecd>("Position");
    auto *vel = real_body.getBaseParticles().getVariableDataByName<Vecd>("Velocity");
    d.particle_number = real_body.getBaseParticles().TotalRealParticles();

    const Real centre_y = FibreCentreY();
    Real depth_sum = 0.0;
    for (size_t i = 0; i != d.particle_number; ++i)
    {
        const Real speed = vel[i].norm();
        d.max_speed = std::max(d.max_speed, speed);
        d.max_speed_y = std::max(d.max_speed_y, std::abs(vel[i][1]));
        depth_sum += std::abs(pos[i][1] - centre_y) - FibreHalfWidth();
        if (std::abs(pos[i][1] - centre_y) < ExclusionRadius() - 1.0e-9)
            ++d.inside_fibre;
        if (pos[i][1] < -0.01 * DomainHeight() ||
            pos[i][1] > 1.01 * DomainHeight())
            ++d.escaped;
    }
    d.mean_height_from_wall = depth_sum / static_cast<Real>(d.particle_number);
    return d;
}
} // namespace

//=============================================================================
//  main
//=============================================================================
int main(int ac, char *av[])
{
    ParseCommandLine(ac, av);

    const Vecd domain_lower(0.0, 0.0);
    const Vecd domain_upper(DomainLength(), DomainHeight());
    BoundingBoxd system_bounds(domain_lower, domain_upper);
    const Real spacing = ParticleSpacing();

    SPHSystem sph_system(system_bounds, spacing);
    sph_system.setReloadParticles(false);
    // Only the program name is forwarded; every option of this case is parsed
    // above, and the library rejects unknown boost::program_options entries.
    sph_system.handleCommandlineOptions(1, av);
    IO::getEnvironment().appendOutputFolder(run_options.output_tag);

    const int target_number = TargetParticleNumber();
    std::cout << "CG self-assembly prototype (SELF-DEVELOPED TEMPORARY MODEL)\n"
              << "  case=" << run_options.case_id
              << " seed=" << run_options.seed
              << " domain=" << DomainLength() << " x " << DomainHeight()
              << " (periodic in x)\n"
              << "  fibre_half_width=" << FibreHalfWidth()
              << " exclusion_radius=" << ExclusionRadius()
              << " free_area=" << FreeArea() << "\n"
              << "  area_fraction=" << run_options.area_fraction
              << " target_particles=" << target_number
              << " spacing=" << spacing << " (2h=" << 2.6 * spacing << ")\n"
              << "  repulsion cutoff=" << run_options.repulsion_cutoff
              << " stiffness=" << run_options.repulsion_stiffness
              << " initial_separation=" << run_options.initial_separation << "\n"
              << "  drag=" << run_options.drag
              << " dt=" << run_options.dt
              << " end_time=" << run_options.end_time
              << " frames=" << run_options.frames
              << " y_reflect=" << (run_options.y_reflect ? 1 : 0) << "\n";

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

    //-------------------------------------------------------------------------
    //  Relations
    //-------------------------------------------------------------------------
    InnerRelation particles_inner(particles_body);

    //-------------------------------------------------------------------------
    //  Dynamics
    //-------------------------------------------------------------------------
    InteractionWithUpdate<CGPairRepulsion> pair_repulsion(particles_inner);
    // These four are plain body-wide operators with their own exec() loop, so
    // they are instantiated directly rather than through SimpleDynamics.
    CGAccelerationInitialize acceleration_initialize(particles_body);
    CGPositionPrediction position_prediction(particles_body);
    CGConstraints constraints(particles_body);
    CGVelocityUpdate velocity_update(particles_body);
    CGNumericalDrag numerical_drag(particles_body);

    PeriodicAlongAxis periodic_along_x(particles_body.getSPHBodyBounds(), xAxis);
    PeriodicConditionUsingCellLinkedList periodic_condition(particles_body, periodic_along_x);

    BodyStatesRecordingToVtp body_states_recording(sph_system);

    //-------------------------------------------------------------------------
    //  Initialise
    //-------------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    periodic_condition.update_cell_linked_list_.exec();
    sph_system.initializeSystemConfigurations();

    pair_repulsion.ResetCounters();
    pair_repulsion.exec();
    acceleration_initialize.exec();

    std::ofstream selfcheck("selfcheck.csv");
    selfcheck << "time,particles,max_speed,max_speed_y,min_neighbour_distance,"
                 "mean_height_above_fibre,inside_fibre_count,wrapped_count,"
                 "escaped_count,periodic_pairs,dt,steps\n";
    selfcheck << std::setprecision(10);

    auto report = [&](Real time, Real dt, size_t steps, bool write_frame,
                      bool reset_min) {
        const FrameDiagnostics d = CollectDiagnostics(particles_body);
        selfcheck << time << "," << d.particle_number << ","
                  << d.max_speed << "," << d.max_speed_y << ","
                  << pair_repulsion.MinimumSeparation() << ","
                  << d.mean_height_from_wall << ","
                  << constraints.InsideFibreCount() << ","
                  << constraints.WrapCount() << ","
                  << d.escaped << ","
                  << pair_repulsion.PeriodicPairCount() << ","
                  << dt << "," << steps << "\n";
        selfcheck.flush();
        std::cout << std::fixed << std::setprecision(4)
                  << "T=" << time
                  << " N=" << d.particle_number
                  << " max_speed=" << d.max_speed
                  << " min_sep=" << pair_repulsion.MinimumSeparation()
                  << " h_wall=" << d.mean_height_from_wall
                  << " inside_fibre=" << constraints.InsideFibreCount()
                  << " wrapped=" << constraints.WrapCount()
                  << " periodic_pairs=" << pair_repulsion.PeriodicPairCount()
                  << " dt=" << dt << "\n";
        if (write_frame)
            body_states_recording.writeToFile(steps);
        if (reset_min)
            pair_repulsion.ResetCounters();
    };

    report(0.0, 0.0, 0, true, true);

    //-------------------------------------------------------------------------
    //  Time stepping
    //-------------------------------------------------------------------------
    const Real dt_stiffness =
        0.10 * std::sqrt(1.0 / run_options.repulsion_stiffness);
    Real physical_time = 0.0;
    const Real output_interval = run_options.end_time /
                                 static_cast<Real>(run_options.frames);
    size_t steps = 0;
    size_t frame = 0;
    TickCount wall_start = TickCount::now();

    while (physical_time < run_options.end_time - 1.0e-12)
    {
        const FrameDiagnostics before = CollectDiagnostics(particles_body);
        Real dt = run_options.dt;
        if (dt <= 0.0)
        {
            const Real dt_velocity =
                0.25 * run_options.repulsion_cutoff /
                std::max(before.max_speed, 1.0e-8);
            dt = std::min(dt_stiffness, dt_velocity);
            if (run_options.drag > 0.0)
                dt = std::min(dt, 0.5 / run_options.drag);
        }
        if (physical_time + dt > run_options.end_time)
            dt = run_options.end_time - physical_time;

        position_prediction.exec(dt);
        constraints.exec(dt);
        periodic_condition.bounding_.exec();
        particles_body.updateCellLinkedList();
        periodic_condition.update_cell_linked_list_.exec();
        particles_inner.updateConfiguration();

        pair_repulsion.exec(dt);
        velocity_update.exec(dt);
        numerical_drag.exec(dt);

        physical_time += dt;
        ++steps;

        if (!std::isfinite(dt) || dt <= 0.0)
            throw std::runtime_error("time step became non-finite; aborting.");

        if (physical_time >= (frame + 1) * output_interval - 1.0e-9 ||
            physical_time >= run_options.end_time - 1.0e-12)
        {
            ++frame;
            report(physical_time, dt, steps, true, true);
        }

        const FrameDiagnostics now = CollectDiagnostics(particles_body);
        if (!std::isfinite(now.max_speed) || now.max_speed > 1.0e3)
            throw std::runtime_error("blow-up detected: max_speed=" +
                                     std::to_string(now.max_speed));
        if (now.escaped > 0)
            throw std::runtime_error("particle escaped the domain.");
    }

    const Real wall_seconds = (TickCount::now() - wall_start).seconds();
    std::cout << "finished: steps=" << steps
              << " wall_seconds=" << wall_seconds << "\n";
    selfcheck.close();
    return 0;
}
