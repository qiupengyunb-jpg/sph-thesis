/**
 * @file axisymmetric_free_column_breakup_pri.cpp
 * @brief Diagnostic axisymmetric free-column breakup benchmark.
 *
 * The benchmark deliberately removes the fibre and wetting physics.  It uses
 * the same weakly-compressible SPH integration, local single-phase surface
 * stress and axisymmetric source terms as the fibre-film prototype.  Its only
 * purpose is to answer whether the numerical method can disconnect a resolved
 * free liquid neck before a solid wall is introduced.
 */
#define main retained_film_reference_main
#include "../test_2d_axisymmetric_fiber_film_pri/axisymmetric_fiber_film_pri.cpp"
#undef main

#include <sstream>

namespace
{
Real ColumnRadius() { return fibre_radius; }
Real ColumnLength()
{
    Real fastest_wavelength = 2.0 * Pi * std::sqrt(2.0) * ColumnRadius();
    // The fastest-growing RP wavelength is intentionally unstable and even
    // lattice noise will excite it.  A static Laplace-balance benchmark must
    // use a subcritical periodic length; half lambda_max is below 2*pi*R.
    Real wavelength_factor = run_options.case_name == "static"
                                 ? 0.5
                                 : run_options.WavelengthFactor();
    return wavelength_factor * fastest_wavelength;
}
Real ColumnLowerX() { return -0.5 * ColumnLength(); }
Real ColumnUpperX() { return 0.5 * ColumnLength(); }
Real ColumnSurfaceRadius(Real x)
{
    Real k = 2.0 * Pi / ColumnLength();
    return ColumnRadius() *
           (1.0 - run_options.amplitude_ratio * std::cos(k * x));
}
Real ColumnGenerationRadius(Real x)
{
    // The balanced-curvature path maps a regular cylindrical lattice onto the
    // requested small disturbance after generation. This preserves a 1%
    // disturbance even when its height is smaller than one particle spacing.
    return run_options.surface_formulation == "balanced-curvature"
               ? ColumnRadius()
               : ColumnSurfaceRadius(x);
}

class FreeColumnShape : public MultiPolygonShape
{
  public:
    explicit FreeColumnShape(const std::string &name) : MultiPolygonShape(name)
    {
        std::vector<Vecd> polygon;
        const size_t samples = std::max<size_t>(
            240, static_cast<size_t>(std::ceil(ColumnLength() / ParticleSpacing())) * 3);
        polygon.reserve(samples + 3);
        polygon.emplace_back(ColumnLowerX(), 0.0);
        polygon.emplace_back(ColumnUpperX(), 0.0);
        for (size_t i = 0; i <= samples; ++i)
        {
            Real x = ColumnUpperX() -
                     ColumnLength() * static_cast<Real>(i) /
                         static_cast<Real>(samples);
            polygon.emplace_back(x, ColumnGenerationRadius(x));
        }
        polygon.emplace_back(ColumnLowerX(), 0.0);
        multi_polygon_.addPolygon(polygon, GeometricOps::add);
    }
};

class AxisSymmetryWallShape : public MultiPolygonShape
{
  public:
    explicit AxisSymmetryWallShape(const std::string &name)
        : MultiPolygonShape(name)
    {
        Real width = 4.0 * ParticleSpacing();
        std::vector<Vecd> polygon{
            Vecd(ColumnLowerX(), -width), Vecd(ColumnUpperX(), -width),
            Vecd(ColumnUpperX(), 0.0), Vecd(ColumnLowerX(), 0.0),
            Vecd(ColumnLowerX(), -width)};
        multi_polygon_.addPolygon(polygon, GeometricOps::add);
    }
};

class InitializeColumnState : public LocalDynamics
{
  public:
    explicit InitializeColumnState(SPHBody &body)
        : LocalDynamics(body),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          velocity_(particles_->getVariableDataByName<Vecd>("Velocity")),
          density_(particles_->getVariableDataByName<Real>("Density")),
          pressure_(particles_->registerStateVariableData<Real>("Pressure")) {}

    void update(size_t i, Real dt)
    {
        velocity_[i] = Vecd::Zero();
        if (run_options.surface_formulation == "balanced-curvature" &&
            run_options.case_name != "static")
            pos_[i][1] *= ColumnSurfaceRadius(pos_[i][0]) / ColumnRadius();
        // For the unperturbed axisymmetric column, kappa_axial=0 and
        // kappa_hoop=1/R.  The exact static Young-Laplace state therefore has
        // p_liquid-p_gas=gamma/R.  Initialising p=0 would test capillary
        // collapse, not discrete pressure/capillary balance.
        Real k = 2.0 * Pi / ColumnLength();
        Real radius = ColumnSurfaceRadius(pos_[i][0]);
        Real first_derivative = ColumnRadius() * run_options.amplitude_ratio *
                                k * std::sin(k * pos_[i][0]);
        Real second_derivative = ColumnRadius() * run_options.amplitude_ratio *
                                 k * k * std::cos(k * pos_[i][0]);
        Real metric = std::sqrt(1.0 + first_derivative * first_derivative);
        Real initial_curvature =
            -second_derivative / (metric * metric * metric) +
            1.0 / (radius * metric);
        Real laplace_pressure = surface_tension * initial_curvature;
        density_[i] = reference_density *
                      (1.0 + laplace_pressure /
                                 (reference_density * SoundSpeed() * SoundSpeed()));
        pressure_[i] = laplace_pressure;
    }

  private:
    Vecd *pos_, *velocity_;
    Real *density_, *pressure_;
};

/** Local CSS surface stress with the axisymmetric hoop-divergence term. */
class FreeColumnSurfaceStress : public ForcePrior, public DataDelegateInner
{
  public:
    explicit FreeColumnSurfaceStress(BaseInnerRelation &inner_relation)
        : ForcePrior(inner_relation.getSPHBody(), "ColumnCapillaryForce"),
          DataDelegateInner(inner_relation),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          rho_(particles_->getVariableDataByName<Real>("Density")),
          pressure_(particles_->registerStateVariableData<Real>("Pressure")),
          mass_(particles_->getVariableDataByName<Real>("Mass")),
          Vol_(particles_->getVariableDataByName<Real>("VolumetricMeasure")),
          indicator_(particles_->getVariableDataByName<int>("Indicator")),
          gradient_(particles_->registerStateVariableData<Vecd>("ColumnColorGradient")),
          normal_(particles_->registerStateVariableData<Vecd>("ColumnSurfaceNormal")),
          delta_(particles_->registerStateVariableData<Real>("ColumnSurfaceDelta")),
          axial_curvature_(particles_->registerStateVariableData<Real>(
              "ColumnAxialCurvature")),
          hoop_curvature_(particles_->registerStateVariableData<Real>(
              "ColumnHoopCurvature")),
          total_curvature_(particles_->registerStateVariableData<Real>(
              "ColumnTotalCurvature")),
          stress_(particles_->registerStateVariableData<Matd>("ColumnSurfaceStress")),
          pressure_jump_force_(particles_->registerStateVariableData<Vecd>(
              "ColumnPressureJumpForce")),
          number_of_bins_(std::max<size_t>(
              32, static_cast<size_t>(std::round(ColumnLength() / ParticleSpacing()))))
    {
        particles_->registerSingleVariable<Real>("SurfaceTensionCoef", surface_tension);
        particles_->addEvolvingVariable<Vecd>("ColumnColorGradient");
        particles_->addEvolvingVariable<Vecd>("ColumnSurfaceNormal");
        particles_->addEvolvingVariable<Real>("ColumnSurfaceDelta");
        particles_->addEvolvingVariable<Real>("ColumnAxialCurvature");
        particles_->addEvolvingVariable<Real>("ColumnHoopCurvature");
        particles_->addEvolvingVariable<Real>("ColumnTotalCurvature");
        particles_->addEvolvingVariable<Matd>("ColumnSurfaceStress");
        particles_->addEvolvingVariable<Vecd>("ColumnPressureJumpForce");
        particles_->addVariableToWrite<Vecd>("ColumnColorGradient");
        particles_->addVariableToWrite<Vecd>("ColumnSurfaceNormal");
        particles_->addVariableToWrite<Real>("ColumnSurfaceDelta");
        particles_->addVariableToWrite<Real>("ColumnAxialCurvature");
        particles_->addVariableToWrite<Real>("ColumnHoopCurvature");
        particles_->addVariableToWrite<Real>("ColumnTotalCurvature");
        particles_->addVariableToWrite<Matd>("ColumnSurfaceStress");
        particles_->addVariableToWrite<Vecd>("ColumnPressureJumpForce");
        particles_->addVariableToWrite<Vecd>("ColumnCapillaryForce");
    }

    void exec(Real dt = 0.0)
    {
        const size_t total = particles_->TotalRealParticles();
        const Real gradient_floor = 0.02 / ParticleSpacing();
        std::vector<Vecd> raw_normal(total, Vecd::Zero());
        std::vector<Real> raw_delta(total, 0.0);

        for (size_t i = 0; i < total; ++i)
        {
            gradient_[i] = Vecd::Zero();
            normal_[i] = Vecd::Zero();
            delta_[i] = 0.0;
            axial_curvature_[i] = 0.0;
            hoop_curvature_[i] = 0.0;
            total_curvature_[i] = 0.0;
            stress_[i] = Matd::Zero();
            pressure_jump_force_[i] = Vecd::Zero();
            if (indicator_[i] != 1)
                continue;
            const Neighborhood &neighborhood = inner_configuration_[i];
            for (size_t n = 0; n < neighborhood.current_size_; ++n)
            {
                size_t j = neighborhood.j_[n];
                gradient_[i] -= neighborhood.dW_ij_[n] * Vol_[j] *
                                neighborhood.e_ij_[n];
            }
            raw_delta[i] = gradient_[i].norm();
            if (raw_delta[i] > gradient_floor)
                raw_normal[i] = gradient_[i] / raw_delta[i];
        }

        // One orientation-aware smoothing pass, identical in purpose to the
        // fibre diagnostic but without any wall/contact-angle operation.
        for (size_t i = 0; i < total; ++i)
        {
            if (raw_delta[i] <= gradient_floor)
                continue;
            Vecd average = raw_normal[i];
            Real delta_average = raw_delta[i];
            Real weight_sum = 1.0;
            const Neighborhood &neighborhood = inner_configuration_[i];
            for (size_t n = 0; n < neighborhood.current_size_; ++n)
            {
                size_t j = neighborhood.j_[n];
                Real alignment = raw_normal[i].dot(raw_normal[j]);
                if (raw_delta[j] <= gradient_floor || alignment <= 0.0)
                    continue;
                Real w = std::max(neighborhood.W_ij_[n] * Vol_[j], Real(0.0)) *
                         alignment * alignment;
                average += w * raw_normal[j];
                delta_average += w * raw_delta[j];
                weight_sum += w;
            }
            normal_[i] = average / (average.norm() + TinyReal);
            delta_[i] = delta_average / weight_sum;
            stress_[i] = run_options.capillary_scale * surface_tension * delta_[i] *
                         (Matd::Identity() - normal_[i] * normal_[i].transpose());
        }

        // The balanced formulation uses one periodic interface reconstruction
        // for both curvature traction and the pressure-jump traction.  It is a
        // numerical derivative reconstruction, not a visual smoothing step:
        // the same r(x), n and delta_s enter both sides of the stress jump.
        if (run_options.surface_formulation == "balanced-curvature")
        {
            std::vector<Real> radius(
                number_of_bins_, -std::numeric_limits<Real>::infinity());
            for (size_t i = 0; i < total; ++i)
            {
                if (indicator_[i] != 1)
                    continue;
                Real shifted_x = pos_[i][0] - ColumnLowerX();
                Real periodic_x = shifted_x -
                                  std::floor(shifted_x / ColumnLength()) *
                                      ColumnLength();
                size_t bin = std::min(
                    number_of_bins_ - 1,
                    static_cast<size_t>(periodic_x / ColumnLength() *
                                        number_of_bins_));
                radius[bin] = std::max(
                    radius[bin], pos_[i][1] + 0.5 * ParticleSpacing());
            }
            size_t valid_bins = 0;
            for (Real value : radius)
                valid_bins += std::isfinite(value) ? 1 : 0;
            if (valid_bins > 0)
            {
                for (size_t i = 0; i < number_of_bins_; ++i)
                {
                    if (std::isfinite(radius[i]))
                        continue;
                    size_t left = i, right = i;
                    do
                        left = (left + number_of_bins_ - 1) % number_of_bins_;
                    while (left != i && !std::isfinite(radius[left]));
                    do
                        right = (right + 1) % number_of_bins_;
                    while (right != i && !std::isfinite(radius[right]));
                    radius[i] = 0.5 * (radius[left] + radius[right]);
                }

                Real mean_radius = 0.0;
                for (Real value : radius)
                    mean_radius += value;
                mean_radius /= static_cast<Real>(number_of_bins_);
                size_t retained_modes = std::min<size_t>(
                    static_cast<size_t>(run_options.interface_modes),
                    number_of_bins_ / 8);
                std::vector<Real> cosine(retained_modes + 1, 0.0);
                std::vector<Real> sine(retained_modes + 1, 0.0);
                for (size_t mode = 1; mode <= retained_modes; ++mode)
                {
                    for (size_t bin = 0; bin < number_of_bins_; ++bin)
                    {
                        Real angle = 2.0 * Pi * static_cast<Real>(mode) *
                                     (static_cast<Real>(bin) + 0.5) /
                                     static_cast<Real>(number_of_bins_);
                        Real departure = radius[bin] - mean_radius;
                        cosine[mode] += departure * std::cos(angle);
                        sine[mode] += departure * std::sin(angle);
                    }
                    cosine[mode] *= 2.0 / static_cast<Real>(number_of_bins_);
                    sine[mode] *= 2.0 / static_cast<Real>(number_of_bins_);
                }

                Real basic_wave_number = 2.0 * Pi / ColumnLength();
                Real interface_width = 2.0 * ParticleSpacing();
                for (size_t i = 0; i < total; ++i)
                {
                    Real shifted_x = pos_[i][0] - ColumnLowerX();
                    Real periodic_x = shifted_x -
                                      std::floor(shifted_x / ColumnLength()) *
                                          ColumnLength();
                    Real theta = basic_wave_number * periodic_x;
                    Real reconstructed_radius = mean_radius;
                    Real first_derivative = 0.0;
                    Real second_derivative = 0.0;
                    for (size_t mode = 1; mode <= retained_modes; ++mode)
                    {
                        Real mode_angle = static_cast<Real>(mode) * theta;
                        Real harmonic = cosine[mode] * std::cos(mode_angle) +
                                        sine[mode] * std::sin(mode_angle);
                        Real wave_number =
                            static_cast<Real>(mode) * basic_wave_number;
                        reconstructed_radius += harmonic;
                        first_derivative +=
                            wave_number *
                            (-cosine[mode] * std::sin(mode_angle) +
                             sine[mode] * std::cos(mode_angle));
                        second_derivative -=
                            wave_number * wave_number * harmonic;
                    }
                    Real metric =
                        std::sqrt(1.0 + first_derivative * first_derivative);
                    Vecd reconstructed_normal(-first_derivative / metric,
                                              1.0 / metric);
                    Real axial =
                        -second_derivative / (metric * metric * metric);
                    Real hoop = 1.0 /
                                (std::max(reconstructed_radius,
                                          0.25 * ParticleSpacing()) *
                                 metric);
                    Real depth = reconstructed_radius - pos_[i][1];
                    Real reconstructed_delta = 0.0;
                    if (depth >= -0.5 * ParticleSpacing() &&
                        depth <= interface_width)
                    {
                        Real nonnegative_depth = std::max(depth, Real(0.0));
                        reconstructed_delta =
                            (1.0 + std::cos(Pi * nonnegative_depth /
                                            interface_width)) /
                            interface_width;
                    }
                    normal_[i] = reconstructed_delta > 0.0
                                     ? reconstructed_normal
                                     : Vecd::Zero();
                    delta_[i] = reconstructed_delta;
                    gradient_[i] = reconstructed_delta * normal_[i];
                    axial_curvature_[i] =
                        reconstructed_delta > 0.0 ? axial : 0.0;
                    hoop_curvature_[i] =
                        reconstructed_delta > 0.0 ? hoop : 0.0;
                    total_curvature_[i] =
                        reconstructed_delta > 0.0 ? axial + hoop : 0.0;
                }
            }
        }

        Real pressure_gauge_offset = 0.0;
        if (run_options.surface_formulation == "balanced-curvature" &&
            run_options.free_surface_pressure_jump)
        {
            Real weight_sum = 0.0;
            Real pressure_sum = 0.0;
            Real curvature_sum = 0.0;
            for (size_t i = 0; i < total; ++i)
            {
                if (delta_[i] <= gradient_floor)
                    continue;
                Real weight = delta_[i] * Vol_[i];
                weight_sum += weight;
                pressure_sum += weight * pressure_[i];
                curvature_sum += weight * total_curvature_[i];
            }
            if (weight_sum > TinyReal)
                pressure_gauge_offset =
                    run_options.capillary_scale * surface_tension *
                        curvature_sum / weight_sum -
                    pressure_sum / weight_sum;
        }

        for (size_t i = 0; i < total; ++i)
        {
            Vecd force = Vecd::Zero();
            const Neighborhood &neighborhood = inner_configuration_[i];
            if (run_options.surface_formulation == "css")
            {
                for (size_t n = 0; n < neighborhood.current_size_; ++n)
                {
                    size_t j = neighborhood.j_[n];
                    force += mass_[i] * neighborhood.dW_ij_[n] * Vol_[j] *
                             (stress_[i] + stress_[j]) * neighborhood.e_ij_[n] /
                             (rho_[i] + TinyReal);
                }
            }
            if (run_options.surface_formulation == "css" &&
                delta_[i] > gradient_floor)
            {
                Real r = std::max(pos_[i][1], 0.25 * ParticleSpacing());
                Vecd cylindrical(stress_[i](0, 1),
                                  stress_[i](1, 1) -
                                      run_options.capillary_scale *
                                          surface_tension * delta_[i]);
                force += mass_[i] * cylindrical /
                         ((rho_[i] + TinyReal) * r);
            }
            if (run_options.surface_formulation == "balanced-curvature" &&
                delta_[i] > gradient_floor)
            {
                force -= mass_[i] / (rho_[i] + TinyReal) *
                         run_options.capillary_scale * surface_tension *
                         total_curvature_[i] * delta_[i] * normal_[i];
            }
            // In a one-phase calculation the gas-side pressure neighbours do
            // not exist.  A uniform liquid pressure therefore supplies almost
            // no discrete traction at the truncated support.  Represent the
            // missing normal-stress jump explicitly: +p*n*delta_s.  For an
            // exact static cylinder p=gamma/R and this is the outward partner
            // of the inward capillary traction.
            if (run_options.free_surface_pressure_jump &&
                delta_[i] > gradient_floor)
            {
                Real boundary_pressure =
                    run_options.surface_formulation == "balanced-curvature"
                        ? pressure_[i] + pressure_gauge_offset
                        : (run_options.case_name == "static"
                               ? surface_tension / ColumnRadius()
                               : pressure_[i]);
                pressure_jump_force_[i] =
                    mass_[i] / (rho_[i] + TinyReal) * boundary_pressure *
                    delta_[i] * normal_[i];
                force += pressure_jump_force_[i];
            }
            current_force_[i] = force;
            ForcePrior::update(i, dt);
        }
    }

  private:
    Vecd *pos_, *gradient_, *normal_, *pressure_jump_force_;
    Real *rho_, *pressure_, *mass_, *Vol_, *delta_, *axial_curvature_,
        *hoop_curvature_, *total_curvature_;
    int *indicator_;
    Matd *stress_;
    size_t number_of_bins_;
};

struct ColumnMetrics
{
    Real min_radius = 0.0;
    Real max_radius = 0.0;
    Real min_layers = 0.0;
    Real empty_fraction = 0.0;
    Real largest_empty_width = 0.0;
};

ColumnMetrics MeasureColumn(BaseParticles &particles, Vecd *positions)
{
    const Real dx = ParticleSpacing();
    const size_t bins = std::max<size_t>(8, static_cast<size_t>(std::round(ColumnLength() / dx)));
    const Real bin_width = ColumnLength() / static_cast<Real>(bins);
    std::vector<Real> radius(bins, -std::numeric_limits<Real>::infinity());
    for (size_t i = 0; i < particles.TotalRealParticles(); ++i)
    {
        Real shifted = positions[i][0] - ColumnLowerX();
        shifted -= std::floor(shifted / ColumnLength()) * ColumnLength();
        size_t bin = std::min(bins - 1,
                              static_cast<size_t>(shifted / bin_width));
        radius[bin] = std::max(radius[bin], positions[i][1] + 0.5 * dx);
    }
    ColumnMetrics result;
    result.min_radius = std::numeric_limits<Real>::max();
    result.max_radius = 0.0;
    size_t empty = 0, run = 0, largest_run = 0;
    for (Real value : radius)
    {
        if (!std::isfinite(value))
        {
            ++empty;
            ++run;
            largest_run = std::max(largest_run, run);
        }
        else
        {
            run = 0;
            result.min_radius = std::min(result.min_radius, value);
            result.max_radius = std::max(result.max_radius, value);
        }
    }
    // Account for an empty interval crossing the periodic boundary.
    size_t prefix = 0, suffix = 0;
    while (prefix < bins && !std::isfinite(radius[prefix]))
        ++prefix;
    while (suffix < bins && !std::isfinite(radius[bins - 1 - suffix]))
        ++suffix;
    largest_run = std::max(largest_run, std::min(bins, prefix + suffix));
    if (empty == bins)
        result.min_radius = 0.0;
    result.min_layers = result.min_radius / dx;
    result.empty_fraction = static_cast<Real>(empty) / static_cast<Real>(bins);
    result.largest_empty_width = static_cast<Real>(largest_run) * bin_width;
    return result;
}
} // namespace

int main(int argc, char *argv[])
{
    std::vector<std::string> sph_arguments;
    try
    {
        run_options = ParseRunOptions(argc, argv, sph_arguments);
    }
    catch (const std::exception &error)
    {
        std::cerr << "Argument error: " << error.what() << std::endl;
        return 2;
    }
    std::vector<char *> sph_argument_pointers = MakeArgumentPointers(sph_arguments);
    const Real dx = ParticleSpacing();
    const Real margin = 4.0 * dx;
    BoundingBoxd bounds(Vecd(ColumnLowerX() - margin, -margin),
                        Vecd(ColumnUpperX() + margin, 3.0 * ColumnRadius()));
    SPHSystem sph_system(bounds, dx);
    sph_system.handleCommandlineOptions(static_cast<int>(sph_argument_pointers.size()),
                                        sph_argument_pointers.data());

    std::ostringstream label;
    label << "free_column_r" << run_options.resolution
          << "_a" << static_cast<int>(std::round(10000.0 * run_options.amplitude_ratio))
          << "_cs" << static_cast<int>(std::round(100.0 * run_options.capillary_scale))
          << "_pj" << (run_options.free_surface_pressure_jump ? 1 : 0)
          << "_sf" << (run_options.surface_formulation == "css" ? "css" : "bc");
    IO::getEnvironment().appendOutputFolder(label.str());

    FluidBody liquid(sph_system, makeShared<FreeColumnShape>("FreeColumnHalf"));
    liquid.defineMatterMaterial<WeaklyCompressibleFluid>(reference_density, SoundSpeed());
    liquid.addMaterialProperty<Viscosity>(DynamicViscosity());
    liquid.generateParticles<BaseParticles, Lattice>();

    SolidBody axis_wall(sph_system, makeShared<AxisSymmetryWallShape>("AxisSymmetryWall"));
    axis_wall.defineMatterMaterial<Solid>();
    axis_wall.generateParticles<BaseParticles, Lattice>();

    InnerRelation liquid_inner(liquid);
    ContactRelation liquid_axis_contact(liquid, {&axis_wall});
    ComplexRelation liquid_complex(liquid_inner, liquid_axis_contact);

    SimpleDynamics<NormalDirectionFromBodyShape> axis_wall_normals(axis_wall);
    InteractionWithUpdate<FreeSurfaceIndicationComplex> indicate_free_surface(
        liquid_inner, liquid_axis_contact);
    InteractionWithUpdate<fluid_dynamics::DensitySummationComplexFreeSurface>
        density_by_summation(liquid_inner, liquid_axis_contact);
    FreeColumnSurfaceStress capillary_force(liquid_inner);
    Dynamics1Level<fluid_dynamics::Integration1stHalfWithWallRiemann>
        pressure_relaxation(liquid_inner, liquid_axis_contact);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallRiemann>
        density_relaxation(liquid_inner, liquid_axis_contact);
    InteractionWithUpdate<fluid_dynamics::ViscousForceInner> viscous_force(liquid_inner);
    InteractionWithUpdate<AxisymmetricViscousForce> axisymmetric_viscous_force(liquid_inner);
    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep>
        advection_time_step(liquid, CapillaryVelocity(), 0.10);
    ReduceDynamics<fluid_dynamics::SurfaceTensionTimeStep>
        acoustic_surface_time_step(liquid, 0.08);

    PeriodicAlongAxis periodic_axis(liquid.getSPHBodyBounds(), xAxis);
    PeriodicConditionUsingCellLinkedList periodic_condition(liquid, periodic_axis);
    ParticleSorting particle_sorting(liquid);
    SimpleDynamics<InitializeColumnState> initialize_state(liquid);
    SimpleDynamics<InitializeAxisymmetricWeights> initialize_axisymmetric_weights(liquid);
    SimpleDynamics<UpdateAxisymmetricWeights> update_axisymmetric_weights(liquid);
    SimpleDynamics<AxisymmetricContinuityCorrection> axisymmetric_continuity(liquid);

    BodyStatesRecordingToVtp write_states(sph_system);
    write_states.addToWrite<Real>(liquid, "Density");
    write_states.addToWrite<Real>(liquid, "Pressure");
    write_states.addToWrite<int>(liquid, "Indicator");
    write_states.addToWrite<Vecd>(liquid, "Velocity");
    write_states.addToWrite<Vecd>(liquid, "ColumnCapillaryForce");
    write_states.addToWrite<Vecd>(liquid, "ColumnPressureJumpForce");

    initialize_state.exec();
    initialize_axisymmetric_weights.exec();
    update_axisymmetric_weights.exec();
    sph_system.initializeSystemCellLinkedLists();
    periodic_condition.update_cell_linked_list_.exec();
    sph_system.initializeSystemConfigurations();
    axis_wall_normals.exec();
    indicate_free_surface.exec();
    capillary_force.exec();

    std::string output = IO::getEnvironment().OutputFolder();
    std::ofstream parameters(output + "/parameters.csv");
    parameters << "parameter,value\n"
               << "model,axisymmetric_free_column\n"
               << "radius," << ColumnRadius() << "\n"
               << "length," << ColumnLength() << "\n"
               << "resolution," << run_options.resolution << "\n"
               << "particle_spacing," << dx << "\n"
               << "amplitude_ratio," << run_options.amplitude_ratio << "\n"
               << "initial_laplace_pressure,"
               << (run_options.case_name == "static"
                       ? surface_tension / ColumnRadius()
                       : 0.0)
               << "\n"
               << "density_reinitialization,"
               << ((run_options.case_name == "static" ||
                    run_options.surface_formulation == "balanced-curvature")
                       ? "off"
                       : "on")
               << "\n"
               << "capillary_scale," << run_options.capillary_scale << "\n"
               << "free_surface_pressure_jump,"
               << (run_options.free_surface_pressure_jump ? "on" : "off") << "\n"
               << "surface_formulation," << run_options.surface_formulation << "\n"
               << "Oh," << run_options.oh << "\n"
               << "end_T," << run_options.end_T << "\n";
    parameters.close();

    std::ofstream diagnostics(output + "/column_diagnostics.csv");
    diagnostics << "time,T,min_radius,max_radius,min_radius_layers,empty_fraction,"
                   "largest_empty_width,kernel_support_radius,max_speed,ring_mass_error,"
                   "finite_state,out_of_domain_particles\n"
                << std::scientific << std::setprecision(10);

    BaseParticles &particles = liquid.getBaseParticles();
    Vecd *positions = particles.getVariableDataByName<Vecd>("Position");
    Vecd *velocities = particles.getVariableDataByName<Vecd>("Velocity");
    Real *densities = particles.getVariableDataByName<Real>("Density");
    Real *masses = particles.getVariableDataByName<Real>("Mass");
    Real initial_ring_mass = 0.0;
    for (size_t i = 0; i < particles.TotalRealParticles(); ++i)
        initial_ring_mass += 2.0 * Pi * positions[i][1] * masses[i];

    auto write_diagnostics = [&](Real time)
    {
        ColumnMetrics metrics = MeasureColumn(particles, positions);
        Real ring_mass = 0.0, max_speed = 0.0;
        size_t escaped = 0;
        bool finite = true;
        for (size_t i = 0; i < particles.TotalRealParticles(); ++i)
        {
            ring_mass += 2.0 * Pi * positions[i][1] * masses[i];
            max_speed = std::max(max_speed, velocities[i].norm());
            finite = finite && std::isfinite(densities[i]) &&
                     std::isfinite(positions[i][0]) && std::isfinite(positions[i][1]) &&
                     std::isfinite(velocities[i][0]) && std::isfinite(velocities[i][1]);
            if (positions[i][1] < -dx || positions[i][1] > 3.0 * ColumnRadius())
                ++escaped;
        }
        Real T = surface_tension * time /
                 (DynamicViscosity() * ColumnRadius());
        diagnostics << time << "," << T << "," << metrics.min_radius << ","
                    << metrics.max_radius << "," << metrics.min_layers << ","
                    << metrics.empty_fraction << "," << metrics.largest_empty_width << ","
                    << liquid.getSPHAdaptation().getKernel()->CutOffRadius() << ","
                    << max_speed << ","
                    << (ring_mass - initial_ring_mass) /
                           (initial_ring_mass + TinyReal)
                    << "," << (finite ? 1 : 0) << "," << escaped << "\n";
        diagnostics.flush();
        std::cout << std::fixed << std::setprecision(5)
                  << "T=" << T << " min_radius=" << metrics.min_radius
                  << " layers=" << metrics.min_layers
                  << " empty=" << metrics.empty_fraction
                  << " max_speed=" << max_speed << std::endl;
        return finite && escaped == 0;
    };

    Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
    Real end_time = run_options.end_T * DynamicViscosity() * ColumnRadius() /
                    surface_tension;
    Real output_interval = end_time / static_cast<Real>(run_options.frames);
    size_t iteration = 0;
    write_states.writeToFile(0);
    write_diagnostics(physical_time);
    while (physical_time < end_time)
    {
        Real integration_time = 0.0;
        while (integration_time < output_interval && physical_time < end_time)
        {
            indicate_free_surface.exec();
            // Density summation is useful for the large-deformation breakup
            // case, but it erases the small density increment representing
            // the Laplace pressure in this weakly-compressible static test.
            if (run_options.case_name != "static" &&
                run_options.surface_formulation != "balanced-curvature")
                density_by_summation.exec();
            update_axisymmetric_weights.exec();
            viscous_force.exec();
            axisymmetric_viscous_force.exec();
            Real Dt = advection_time_step.exec();
            Real relaxation_time = 0.0;
            while (relaxation_time < Dt && integration_time < output_interval &&
                   physical_time < end_time)
            {
                capillary_force.exec();
                Real dt = acoustic_surface_time_step.exec();
                dt = std::min(dt, Dt - relaxation_time);
                dt = std::min(dt, output_interval - integration_time);
                dt = std::min(dt, end_time - physical_time);
                pressure_relaxation.exec(dt);
                density_relaxation.exec(dt);
                axisymmetric_continuity.exec(dt);
                update_axisymmetric_weights.exec();
                relaxation_time += dt;
                integration_time += dt;
                physical_time += dt;
            }
            ++iteration;
            periodic_condition.bounding_.exec();
            if (iteration % 100 == 0)
                particle_sorting.exec();
            liquid.updateCellLinkedList();
            periodic_condition.update_cell_linked_list_.exec();
            liquid_complex.updateConfiguration();
        }
        if (!write_diagnostics(physical_time))
            return 3;
        axis_wall.setNewlyUpdated();
        write_states.writeToFile();
    }
    std::cout << "Completed free-column diagnostic. Output: " << output << std::endl;
    return 0;
}
