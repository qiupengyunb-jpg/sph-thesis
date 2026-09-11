/**
 * @file particle_film_impact.cpp
 * @brief A rigid circular particle impacts a liquid film on a rigid substrate.
 *
 * This is a preliminary, dimensionless, two-dimensional model. Surface tension
 * acts at the liquid-air interface. A prescribed contact angle is intentionally
 * not included in this first-stage model.
 */
#include "sphinxsys.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace SPH;

namespace
{
constexpr Real impact_speed = 1.0;

struct RunOptions
{
    std::string case_name = "h020";
    std::string motion = "free";
    Real resolution = 50.0;
    Real end_time = 1.5;
    int frames = 60;

    Real FilmThicknessRatio() const
    {
        if (case_name == "h015")
            return 0.15;
        if (case_name == "h020")
            return 0.20;
        if (case_name == "h025")
            return 0.25;
        throw std::runtime_error("Unknown case '" + case_name + "'. Use h015, h020, or h025.");
    }

    bool FreeMotion() const
    {
        if (motion == "free")
            return true;
        if (motion == "prescribed" || motion == "stationary")
            return false;
        throw std::runtime_error(
            "Unknown motion '" + motion + "'. Use free, prescribed, or stationary.");
    }

    Real PrescribedVerticalSpeed() const
    {
        return motion == "stationary" ? 0.0 : -impact_speed;
    }

    std::string Label() const
    {
        return case_name + "_" + motion + "_r" + std::to_string(static_cast<int>(resolution));
    }
};

bool StartsWith(const std::string &value, const std::string &prefix)
{
    return value.rfind(prefix, 0) == 0;
}

RunOptions ParseRunOptions(int argc, char *argv[], std::vector<std::string> &sph_arguments)
{
    RunOptions options;
    sph_arguments.push_back(argv[0]);
    for (int i = 1; i < argc; ++i)
    {
        std::string argument(argv[i]);
        if (StartsWith(argument, "--case="))
        {
            options.case_name = argument.substr(std::string("--case=").size());
        }
        else if (StartsWith(argument, "--motion="))
        {
            options.motion = argument.substr(std::string("--motion=").size());
        }
        else if (StartsWith(argument, "--resolution="))
        {
            options.resolution = std::stod(argument.substr(std::string("--resolution=").size()));
        }
        else if (StartsWith(argument, "--end-time="))
        {
            options.end_time = std::stod(argument.substr(std::string("--end-time=").size()));
        }
        else if (StartsWith(argument, "--frames="))
        {
            options.frames = std::stoi(argument.substr(std::string("--frames=").size()));
        }
        else
        {
            sph_arguments.push_back(argument);
        }
    }

    if (options.resolution < 20.0)
        throw std::runtime_error("Resolution must be at least 20 particles per diameter.");
    if (options.end_time <= 0.0)
        throw std::runtime_error("End time must be positive.");
    if (options.frames < 2)
        throw std::runtime_error("At least two output frames are required.");

    options.FilmThicknessRatio();
    options.FreeMotion();
    return options;
}

std::vector<char *> MakeArgumentPointers(std::vector<std::string> &arguments)
{
    std::vector<char *> pointers;
    pointers.reserve(arguments.size());
    for (std::string &argument : arguments)
        pointers.push_back(argument.data());
    return pointers;
}

RunOptions run_options;

// Dimensionless geometry.
constexpr Real particle_diameter = 1.0;
constexpr Real particle_radius = 0.5 * particle_diameter;
constexpr Real domain_length = 4.0 * particle_diameter;
constexpr Real domain_height = 2.0 * particle_diameter;
constexpr Real initial_surface_gap = 0.10 * particle_diameter;

// Dimensionless materials: We = 1 and Re = 10.
constexpr Real rho0_liquid = 1.0;
constexpr Real rho0_air = 1.0e-3;
constexpr Real rho0_particle = 2.2;
constexpr Real sound_speed = 10.0 * impact_speed;
constexpr Real liquid_viscosity = 0.1;
constexpr Real air_viscosity = 1.0e-3;
constexpr Real surface_tension = 1.0;
constexpr Real surface_hourglass_coefficient = 0.0;
constexpr Real particle_contact_stiffness = rho0_particle * sound_speed * sound_speed;

Real FilmThickness()
{
    return run_options.FilmThicknessRatio() * particle_diameter;
}

Real ParticleSpacing()
{
    return particle_diameter / run_options.resolution;
}

Real BoundaryWidth()
{
    return 4.0 * ParticleSpacing();
}

Vec2d InitialParticleCenter()
{
    return Vec2d(0.5 * domain_length,
                 FilmThickness() + initial_surface_gap + particle_radius);
}

class LiquidFilmShape : public ComplexShape
{
  public:
    explicit LiquidFilmShape(const std::string &shape_name) : ComplexShape(shape_name)
    {
        // Keep the geometric boundary slightly away from a lattice row. Without
        // this tolerance, H/D=0.25 at D/dx=50 creates one coincident liquid-air
        // particle at the periodic corner and immediately produces NaN values.
        Real film_shape_height = FilmThickness() - 0.05 * ParticleSpacing();
        Vec2d half_size(0.5 * domain_length, 0.5 * film_shape_height);
        Vec2d center(0.5 * domain_length, 0.5 * film_shape_height);
        add<GeometricShapeBox>(Transform(center), half_size);
    }
};

class AirShape : public ComplexShape
{
  public:
    explicit AirShape(const std::string &shape_name) : ComplexShape(shape_name)
    {
        Vec2d domain_half_size(0.5 * domain_length, 0.5 * domain_height);
        Vec2d domain_center(0.5 * domain_length, 0.5 * domain_height);
        Vec2d film_half_size(0.5 * domain_length, 0.5 * FilmThickness());
        Vec2d film_center(0.5 * domain_length, 0.5 * FilmThickness());

        add<GeometricShapeBox>(Transform(domain_center), domain_half_size);
        subtract<GeometricShapeBox>(Transform(film_center), film_half_size);
        subtract<GeometricShapeBall>(InitialParticleCenter(), particle_radius);
    }
};

class WallShape : public ComplexShape
{
  public:
    explicit WallShape(const std::string &shape_name) : ComplexShape(shape_name)
    {
        Real boundary_width = BoundaryWidth();
        Vec2d wall_half_size(0.5 * domain_length + boundary_width,
                             0.5 * boundary_width);
        Vec2d bottom_center(0.5 * domain_length, -0.5 * boundary_width);
        Vec2d top_center(0.5 * domain_length, domain_height + 0.5 * boundary_width);

        add<GeometricShapeBox>(Transform(bottom_center), wall_half_size);
        add<GeometricShapeBox>(Transform(top_center), wall_half_size);
    }
};

class RigidParticleShape : public ComplexShape
{
  public:
    explicit RigidParticleShape(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<GeometricShapeBall>(InitialParticleCenter(), particle_radius);
    }
};

class PrescribedVerticalMotion : public LocalDynamics
{
  public:
    PrescribedVerticalMotion(SPHBody &sph_body, Real vertical_speed)
        : LocalDynamics(sph_body),
          vertical_speed_(vertical_speed),
          physical_time_(0.0),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          pos0_(particles_->registerStateVariableDataFrom<Vecd>("InitialPosition", "Position")),
          vel_(particles_->registerStateVariableData<Vecd>("Velocity")),
          acc_(particles_->registerStateVariableData<Vecd>("Acceleration")) {}

    void setPhysicalTime(Real physical_time)
    {
        physical_time_ = physical_time;
    }

    void update(size_t index_i, Real dt)
    {
        pos_[index_i] = pos0_[index_i] + Vecd(0.0, vertical_speed_ * physical_time_);
        vel_[index_i] = Vecd(0.0, vertical_speed_);
        acc_[index_i] = Vecd::Zero();
    }

  private:
    Real vertical_speed_;
    Real physical_time_;
    Vecd *pos_;
    Vecd *pos0_;
    Vecd *vel_;
    Vecd *acc_;
};

void WriteParameterSummary(const std::string &output_folder)
{
    std::ofstream file(output_folder + "/parameters.csv");
    file << "parameter,value,description\n";
    file << "case," << run_options.case_name << ",film-thickness case\n";
    file << "motion," << run_options.motion << ",particle motion mode\n";
    file << "D," << particle_diameter << ",particle diameter\n";
    file << "H_over_D," << run_options.FilmThicknessRatio() << ",film thickness ratio\n";
    file << "initial_gap_over_D," << initial_surface_gap / particle_diameter << ",particle-film gap\n";
    file << "resolution," << run_options.resolution << ",particles per diameter\n";
    file << "particle_spacing," << ParticleSpacing() << ",reference particle spacing\n";
    file << "domain_length_over_D," << domain_length / particle_diameter << ",periodic horizontal domain length\n";
    file << "domain_height_over_D," << domain_height / particle_diameter << ",domain height\n";
    file << "rho_liquid," << rho0_liquid << ",liquid density\n";
    file << "rho_air," << rho0_air << ",air density\n";
    file << "rho_particle," << rho0_particle << ",particle density\n";
    file << "U0," << impact_speed << ",impact speed\n";
    file << "c0," << sound_speed << ",reference sound speed\n";
    file << "mu_liquid," << liquid_viscosity << ",liquid dynamic viscosity\n";
    file << "mu_air," << air_viscosity << ",air dynamic viscosity\n";
    file << "surface_tension," << surface_tension << ",liquid-air surface tension\n";
    file << "surface_hourglass_coefficient," << surface_hourglass_coefficient << ",surface-stress stabilization coefficient\n";
    file << "gravity,0,gravity disabled\n";
    file << "Re," << rho0_liquid * impact_speed * particle_diameter / liquid_viscosity << ",Reynolds number\n";
    file << "We," << rho0_liquid * impact_speed * impact_speed * particle_diameter / surface_tension << ",Weber number\n";
    file << "end_time," << run_options.end_time << ",dimensionless end time\n";
    file << "frames," << run_options.frames << ",requested output frames\n";
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

    Real particle_spacing_ref = ParticleSpacing();
    Real boundary_width = BoundaryWidth();
    BoundingBoxd system_domain_bounds(
        Vec2d(-boundary_width, -boundary_width),
        Vec2d(domain_length + boundary_width, domain_height + boundary_width));
    SPHSystem sph_system(system_domain_bounds, particle_spacing_ref);
    sph_system.handleCommandlineOptions(static_cast<int>(sph_argument_pointers.size()),
                                        sph_argument_pointers.data());
    IO::getEnvironment().appendOutputFolder(run_options.Label());

    std::cout << "Case: " << run_options.case_name
              << ", motion: " << run_options.motion
              << ", H/D: " << run_options.FilmThicknessRatio()
              << ", D/dx: " << run_options.resolution << std::endl;
    std::cout << "Reynolds number: "
              << rho0_liquid * impact_speed * particle_diameter / liquid_viscosity
              << ", Weber number: "
              << rho0_liquid * impact_speed * impact_speed * particle_diameter / surface_tension
              << std::endl;

    FluidBody liquid_film(sph_system, makeShared<LiquidFilmShape>("LiquidFilm"));
    liquid_film.defineBodyLevelSetShape();
    liquid_film.defineMatterMaterial<WeaklyCompressibleFluid>(rho0_liquid, sound_speed);
    liquid_film.addMaterialProperty<Viscosity>(liquid_viscosity);
    liquid_film.generateParticles<BaseParticles, Lattice>();

    FluidBody air(sph_system, makeShared<AirShape>("Air"));
    air.defineBodyLevelSetShape();
    air.defineMatterMaterial<WeaklyCompressibleFluid>(rho0_air, sound_speed);
    air.addMaterialProperty<Viscosity>(air_viscosity);
    air.generateParticles<BaseParticles, Lattice>();

    SolidBody wall(sph_system, makeShared<WallShape>("Wall"));
    wall.defineMatterMaterial<Solid>();
    wall.generateParticles<BaseParticles, Lattice>();

    SolidBody particle(sph_system, makeShared<RigidParticleShape>("RigidParticle"));
    particle.defineBodyLevelSetShape();
    particle.defineMatterMaterial<Solid>(rho0_particle, particle_contact_stiffness);
    particle.generateParticles<BaseParticles, Lattice>();

    InnerRelation liquid_inner(liquid_film);
    ContactRelation liquid_air_contact(liquid_film, {&air});
    ContactRelation liquid_wall_contact(liquid_film, {&wall, &particle});
    InnerRelation air_inner(air);
    ContactRelation air_liquid_contact(air, {&liquid_film});
    ContactRelation air_wall_contact(air, {&wall, &particle});
    ContactRelation particle_liquid_contact(particle, {&liquid_film});
    SurfaceContactRelation particle_wall_contact(particle, {&wall});

    ComplexRelation liquid_complex(liquid_inner, {&liquid_air_contact, &liquid_wall_contact});
    ComplexRelation air_complex(air_inner, {&air_liquid_contact, &air_wall_contact});

    SimpleDynamics<NormalDirectionFromBodyShape> wall_normals(wall);
    SimpleDynamics<NormalDirectionFromBodyShape> particle_normals(particle);

    Dynamics1Level<fluid_dynamics::MultiPhaseIntegration1stHalfWithWallRiemann>
        liquid_pressure_relaxation(liquid_inner, liquid_air_contact, liquid_wall_contact);
    Dynamics1Level<fluid_dynamics::MultiPhaseIntegration2ndHalfWithWallRiemann>
        liquid_density_relaxation(liquid_inner, liquid_air_contact, liquid_wall_contact);
    Dynamics1Level<fluid_dynamics::MultiPhaseIntegration1stHalfWithWallRiemann>
        air_pressure_relaxation(air_inner, air_liquid_contact, air_wall_contact);
    Dynamics1Level<fluid_dynamics::MultiPhaseIntegration2ndHalfWithWallRiemann>
        air_density_relaxation(air_inner, air_liquid_contact, air_wall_contact);

    InteractionWithUpdate<fluid_dynamics::BaseDensitySummationComplex<Inner<>, Contact<>, Contact<>>>
        update_liquid_density(liquid_inner, liquid_air_contact, liquid_wall_contact);
    InteractionWithUpdate<fluid_dynamics::BaseDensitySummationComplex<Inner<>, Contact<>, Contact<>>>
        update_air_density(air_inner, air_liquid_contact, air_wall_contact);
    InteractionWithUpdate<fluid_dynamics::MultiPhaseTransportVelocityCorrectionComplex<AllParticles>>
        liquid_transport_correction(liquid_inner, liquid_air_contact, liquid_wall_contact);
    InteractionWithUpdate<fluid_dynamics::MultiPhaseTransportVelocityCorrectionComplex<AllParticles>>
        air_transport_correction(air_inner, air_liquid_contact, air_wall_contact);
    InteractionWithUpdate<fluid_dynamics::MultiPhaseViscousForceWithWall>
        liquid_viscous_force(liquid_inner, liquid_air_contact, liquid_wall_contact);
    InteractionWithUpdate<fluid_dynamics::MultiPhaseViscousForceWithWall>
        air_viscous_force(air_inner, air_liquid_contact, air_wall_contact);

    InteractionDynamics<fluid_dynamics::SurfaceTensionStress>
        liquid_surface_tension_stress(liquid_air_contact, surface_tension);
    InteractionDynamics<fluid_dynamics::SurfaceTensionStress>
        air_surface_tension_stress(air_liquid_contact, surface_tension);
    InteractionWithUpdate<fluid_dynamics::SurfaceStressForce<Inner<>>>
        liquid_surface_tension_force_inner(liquid_inner, surface_hourglass_coefficient);
    InteractionWithUpdate<fluid_dynamics::SurfaceStressForce<Contact<>>>
        liquid_surface_tension_force_contact(liquid_air_contact, surface_hourglass_coefficient);
    InteractionWithUpdate<fluid_dynamics::SurfaceStressForce<Inner<>>>
        air_surface_tension_force_inner(air_inner, surface_hourglass_coefficient);
    InteractionWithUpdate<fluid_dynamics::SurfaceStressForce<Contact<>>>
        air_surface_tension_force_contact(air_liquid_contact, surface_hourglass_coefficient);

    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep>
        liquid_advection_time_step(liquid_film, impact_speed, 0.1);
    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep>
        air_advection_time_step(air, impact_speed, 0.1);
    ReduceDynamics<fluid_dynamics::SurfaceTensionTimeStep>
        liquid_time_step(liquid_film, 0.6);
    ReduceDynamics<fluid_dynamics::SurfaceTensionTimeStep>
        air_time_step(air, 0.6);

    PeriodicAlongAxis liquid_periodic_axis(liquid_film.getSPHBodyBounds(), xAxis);
    PeriodicConditionUsingCellLinkedList liquid_periodic_condition(
        liquid_film, liquid_periodic_axis);
    PeriodicAlongAxis air_periodic_axis(air.getSPHBodyBounds(), xAxis);
    PeriodicConditionUsingCellLinkedList air_periodic_condition(
        air, air_periodic_axis);

    InteractionWithUpdate<solid_dynamics::ViscousForceFromFluid>
        viscous_force_from_liquid(particle_liquid_contact);
    InteractionWithUpdate<solid_dynamics::PressureForceFromFluid<decltype(liquid_density_relaxation)>>
        pressure_force_from_liquid(particle_liquid_contact);
    InteractionDynamics<solid_dynamics::ContactFactorSummation>
        update_particle_wall_contact_factor(particle_wall_contact);
    InteractionWithUpdate<solid_dynamics::ContactForceFromWall>
        particle_wall_repulsion(particle_wall_contact);

    SimpleDynamics<PrescribedVerticalMotion>
        prescribed_particle_motion(particle, run_options.PrescribedVerticalSpeed());

    SimTK::MultibodySystem multibody_system;
    SimTK::SimbodyMatterSubsystem matter(multibody_system);
    SimTK::GeneralForceSubsystem forces(multibody_system);
    SolidBodyPartForSimbody particle_body_part(
        particle, makeShared<RigidParticleShape>("ParticleConstraint"));
    SimTK::Body::Rigid particle_body_info(*particle_body_part.body_part_mass_properties_);
    SimTK::Transform slider_frame(
        SimTK::Rotation(Pi / 2.0, SimTK::ZAxis),
        SimTKVec3(InitialParticleCenter()[0], InitialParticleCenter()[1], 0.0));
    SimTK::MobilizedBody::Slider particle_mob(
        matter.Ground(), slider_frame, particle_body_info, SimTK::Transform(SimTKVec3(0.0)));
    SimTK::Force::DiscreteForces body_forces(forces, matter);
    SimTK::State state = multibody_system.realizeTopology();
    particle_mob.setRate(state, -impact_speed);

    SimTK::RungeKuttaMersonIntegrator integrator(multibody_system);
    integrator.setAccuracy(1.0e-4);
    integrator.setAllowInterpolation(false);
    integrator.initialize(state);

    ReduceDynamics<solid_dynamics::TotalForceOnBodyPartForSimBody>
        total_force_for_simbody(particle_body_part, multibody_system, particle_mob, integrator);
    SimpleDynamics<solid_dynamics::ConstraintBodyPartBySimBody>
        constrain_particle_by_simbody(particle_body_part, multibody_system, particle_mob, integrator);

    ReduceDynamics<QuantitySummation<Real>> particle_mass(particle, "Mass");
    ReduceDynamics<QuantityMoment<Vecd, SPHBody>> particle_position_moment(particle, "Position");
    ReduceDynamics<QuantityMoment<Vecd, SPHBody>> particle_velocity_moment(particle, "Velocity");
    ReduceDynamics<QuantitySummation<Vecd>> total_pressure_force(particle, "PressureForceFromFluid");
    ReduceDynamics<QuantitySummation<Vecd>> total_viscous_force(particle, "ViscousForceFromFluid");
    ReduceDynamics<QuantitySummation<Vecd>> total_contact_force(particle, "RepulsionForce");

    BaseParticles &liquid_particles = liquid_film.getBaseParticles();
    BaseParticles &air_particles = air.getBaseParticles();
    Vecd *liquid_positions = liquid_particles.getVariableDataByName<Vecd>("Position");
    Vecd *air_positions = air_particles.getVariableDataByName<Vecd>("Position");
    Real *liquid_densities = liquid_particles.getVariableDataByName<Real>("Density");
    Real *air_densities = air_particles.getVariableDataByName<Real>("Density");
    auto fluid_state_is_finite = [&]()
    {
        auto particles_are_finite = [](BaseParticles &particles, Vecd *positions, Real *densities)
        {
            for (size_t index = 0; index < particles.TotalRealParticles(); ++index)
            {
                if (!std::isfinite(densities[index]))
                    return false;
                for (int axis = 0; axis < Dimensions; ++axis)
                    if (!std::isfinite(positions[index][axis]))
                        return false;
            }
            return true;
        };
        return particles_are_finite(liquid_particles, liquid_positions, liquid_densities) &&
               particles_are_finite(air_particles, air_positions, air_densities);
    };

    BodyStatesRecordingToVtp body_states_recording(sph_system);
    body_states_recording.addToWrite<Real>(liquid_film, "Density");
    body_states_recording.addToWrite<Real>(liquid_film, "Pressure");
    body_states_recording.addToWrite<Matd>(liquid_film, "SurfaceTensionStress");
    body_states_recording.addToWrite<Vecd>(particle, "PressureForceFromFluid");
    body_states_recording.addToWrite<Vecd>(particle, "ViscousForceFromFluid");
    body_states_recording.addToWrite<Vecd>(particle, "RepulsionForce");

    sph_system.initializeSystemCellLinkedLists();
    liquid_periodic_condition.update_cell_linked_list_.exec();
    air_periodic_condition.update_cell_linked_list_.exec();
    sph_system.initializeSystemConfigurations();
    wall_normals.exec();
    particle_normals.exec();
    update_particle_wall_contact_factor.exec();

    if (run_options.FreeMotion())
    {
        constrain_particle_by_simbody.exec();
    }
    else
    {
        prescribed_particle_motion.setPhysicalTime(0.0);
        prescribed_particle_motion.exec();
    }

    std::string output_folder = IO::getEnvironment().OutputFolder();
    WriteParameterSummary(output_folder);
    std::ofstream diagnostics(output_folder + "/particle_dynamics.csv");
    diagnostics << "time,particle_y,particle_vy,clearance,pressure_force_y,"
                   "viscous_force_y,fluid_force_y,contact_force_y,net_force_y\n";
    diagnostics << std::scientific << std::setprecision(10);
    Real last_diagnostic_time = -1.0;

    auto write_diagnostics = [&](Real physical_time)
    {
        if (std::abs(physical_time - last_diagnostic_time) < 1.0e-12)
            return;
        Real mass = particle_mass.exec();
        Vecd center = particle_position_moment.exec() / mass;
        Vecd velocity = particle_velocity_moment.exec() / mass;
        Vecd pressure = total_pressure_force.exec();
        Vecd viscous = total_viscous_force.exec();
        Vecd contact = total_contact_force.exec();
        Vecd fluid = pressure + viscous;
        Vecd net = fluid + contact;
        Real clearance = center[1] - particle_radius;
        diagnostics << physical_time << ","
                    << center[1] / particle_diameter << ","
                    << velocity[1] / impact_speed << ","
                    << clearance / particle_diameter << ","
                    << pressure[1] / surface_tension << ","
                    << viscous[1] / surface_tension << ","
                    << fluid[1] / surface_tension << ","
                    << contact[1] / surface_tension << ","
                    << net[1] / surface_tension << "\n";
        diagnostics.flush();
        last_diagnostic_time = physical_time;
    };

    Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
    Real output_interval = run_options.end_time / static_cast<Real>(run_options.frames);
    Real dt = 0.0;
    size_t iteration = 0;
    int screen_output_interval = 100;
    ParticleSorting liquid_sorting(liquid_film);
    ParticleSorting air_sorting(air);

    body_states_recording.writeToFile(0);
    write_diagnostics(physical_time);

    TickCount computation_start = TickCount::now();
    while (physical_time < run_options.end_time)
    {
        Real integration_time = 0.0;
        while (integration_time < output_interval && physical_time < run_options.end_time)
        {
            Real Dt_liquid = liquid_advection_time_step.exec();
            Real Dt_air = air_advection_time_step.exec();
            Real Dt = SMIN(Dt_liquid, Dt_air);

            update_air_density.exec();
            update_liquid_density.exec();
            air_transport_correction.exec();
            liquid_transport_correction.exec();
            air_viscous_force.exec();
            liquid_viscous_force.exec();
            viscous_force_from_liquid.exec();

            Real relaxation_time = 0.0;
            while (relaxation_time < Dt &&
                   integration_time < output_interval &&
                   physical_time < run_options.end_time)
            {
                liquid_surface_tension_stress.exec();
                air_surface_tension_stress.exec();
                liquid_surface_tension_force_inner.exec();
                liquid_surface_tension_force_contact.exec();
                air_surface_tension_force_inner.exec();
                air_surface_tension_force_contact.exec();

                Real dt_liquid = liquid_time_step.exec();
                Real dt_air = air_time_step.exec();
                dt = SMIN(SMIN(dt_liquid, dt_air),
                          SMIN(Dt - relaxation_time, output_interval - integration_time));
                dt = SMIN(dt, run_options.end_time - physical_time);

                liquid_pressure_relaxation.exec(dt);
                air_pressure_relaxation.exec(dt);
                pressure_force_from_liquid.exec();
                liquid_density_relaxation.exec(dt);
                air_density_relaxation.exec(dt);

                update_particle_wall_contact_factor.exec();
                particle_wall_repulsion.exec();

                Real next_time = physical_time + dt;
                if (run_options.FreeMotion())
                {
                    SimTK::State &state_for_update = integrator.updAdvancedState();
                    body_forces.clearAllBodyForces(state_for_update);
                    body_forces.setOneBodyForce(
                        state_for_update, particle_mob, total_force_for_simbody.exec());
                    integrator.stepBy(dt);
                    constrain_particle_by_simbody.exec();
                }
                else
                {
                    prescribed_particle_motion.setPhysicalTime(next_time);
                    prescribed_particle_motion.exec();
                }

                relaxation_time += dt;
                integration_time += dt;
                physical_time = next_time;
            }

            if (iteration % static_cast<size_t>(screen_output_interval) == 0)
            {
                std::cout << std::fixed << std::setprecision(8)
                          << "N=" << iteration
                          << " time=" << physical_time
                          << " Dt=" << Dt
                          << " dt=" << dt << std::endl;
            }
            ++iteration;

            liquid_periodic_condition.bounding_.exec();
            air_periodic_condition.bounding_.exec();
            if (iteration % 100 == 0)
            {
                liquid_sorting.exec();
                air_sorting.exec();
            }
            liquid_film.updateCellLinkedList();
            air.updateCellLinkedList();
            particle.updateCellLinkedList();
            liquid_periodic_condition.update_cell_linked_list_.exec();
            air_periodic_condition.update_cell_linked_list_.exec();
            liquid_complex.updateConfiguration();
            air_complex.updateConfiguration();
            particle_liquid_contact.updateConfiguration();
            particle_wall_contact.updateConfiguration();
        }

        if (!fluid_state_is_finite())
        {
            std::cerr << "Non-finite fluid state detected at t=" << physical_time
                      << "; stopping before invalid VTP output." << std::endl;
            return 3;
        }
        body_states_recording.writeToFile();
        write_diagnostics(physical_time);
    }

    TimeInterval computation_time = TickCount::now() - computation_start;
    std::cout << "Completed " << run_options.Label()
              << " in " << computation_time.seconds() << " seconds."
              << std::endl;
    return 0;
}
