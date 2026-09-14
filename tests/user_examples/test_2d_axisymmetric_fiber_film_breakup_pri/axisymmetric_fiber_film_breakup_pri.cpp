/**
 * @file axisymmetric_fiber_film_breakup_pri.cpp
 * @brief Topology-capable preliminary model for barrel droplets which remain
 *        attached to a rigid fibre but are separated by dry fibre intervals.
 *
 * This file deliberately reuses the geometry, material and axisymmetric flow
 * corrections of the retained-film example.  Its capillary model is local:
 * it does not reconstruct one global r_s(x), and therefore does not fill dry
 * intervals by interpolation.  Wetting can either use the retained fixed-angle
 * normal correction or a surface-energy prototype.  The latter supplies
 * liquid-gas, solid-liquid and solid-gas energies and lets their imbalance act
 * at the contact line.  No air particles are used.
 */
#define SPHINXSYS_FIBER_FILM_BEAD_DEFAULTS
#define SPHINXSYS_FIBER_FILM_TOPOLOGY_DEFAULTS
#define main retained_film_reference_main
#include "../test_2d_axisymmetric_fiber_film_pri/axisymmetric_fiber_film_pri.cpp"
#undef main

#include <sstream>
#include <memory>
#include <array>
#include <cstdint>

#ifndef SPHINXSYS_BREAKUP_ENTRY_POINT
#define SPHINXSYS_BREAKUP_ENTRY_POINT main
#endif

namespace
{
struct BreakupOptions
{
    // Diagnostic master switch. When disabled, a coated film remains on the
    // reconstructed pre-breakup capillary branch: there is no JFM handover,
    // post-breakup wall adhesion, or topology conversion/recycling.
    bool post_breakup_models =
#if defined(SPHINXSYS_FINITE_TAPERED_FIBER_DEFAULTS) || \
    defined(SPHINXSYS_FINITE_TAPERED_FULL_COATING_DEFAULTS)
        false;
#else
        true;
#endif
    std::string output_tag = "";
    std::string pattern =
#ifdef SPHINXSYS_FINITE_TAPERED_FIBER_DEFAULTS
        "single-center";
#else
        "equal-three";
#endif
    std::string initial_topology =
#ifdef SPHINXSYS_FINITE_TAPERED_FULL_COATING_DEFAULTS
        "full-coating";
#elif defined(SPHINXSYS_FINITE_TAPERED_FIBER_DEFAULTS)
        "finite-film";
#else
        "coated";
#endif
    Real gap_width_over_a = 0.5;
    std::string seeded_pressure = "uniform";
    // Pressure assigned only at t=0 for the closed full-coating diagnostic.
    // local follows the perturbed outer curvature, base omits the imposed
    // perturbation, and zero lets the acoustic solver establish the field.
    std::string full_coating_initial_pressure = "local";
    std::string wetting_model =
#ifdef SPHINXSYS_FINITE_TAPERED_FULL_COATING_DEFAULTS
        "surface-energy";
#elif defined(SPHINXSYS_FINITE_TAPERED_FIBER_DEFAULTS)
        "jfm";
#else
        "surface-energy";
#endif
    Real contact_angle_degrees = 90.0;
    Real gamma_lg = 1.0;
    Real gamma_sl =
#ifdef SPHINXSYS_FINITE_TAPERED_FIBER_DEFAULTS
        0.5;
#else
        0.75;
#endif
    Real gamma_sg =
#ifdef SPHINXSYS_FINITE_TAPERED_FIBER_DEFAULTS
        0.5;
#else
        0.25;
#endif
    bool solid_curvature_traction = false;
    Real wetting_force_scale = 1.0;
    Real contact_core_in_dx = 1.0;
    Real contact_zone_in_dx = 2.25;
    std::string contact_angle_model = "equilibrium";
    Real contact_line_friction = 5.0;
    Real contact_speed_filter_T = 0.25;
    Real contact_speed_capillary_number = 0.10;
    // Coarse-grained solid-liquid adhesion.  In wall-adhesion mode this
    // replaces a prescribed contact angle by an attractive wall traction on
    // liquid particles within the first few particle layers.
    Real wall_adhesion_strength = 0.0;
    Real wall_adhesion_range_in_dx = 1.5;
    // pairwise uses a compact radial interaction profile based on each
    // liquid-solid neighbour distance; traction keeps the earlier
    // distance-window prototype for regression comparisons.
    std::string wall_adhesion_profile = "pairwise";
    std::string wall_adhesion_activation = "always";
    Real wall_adhesion_ramp_T = 0.5;
    bool substep_force_diagnostics = false;
    Real force_sample_dT = 0.05;
    Real force_control_band_in_dx = 4.0;
    Real surface_gradient_floor_in_dx = 0.02;
    int surface_smoothing_passes = 0;
    Real normal_alignment_floor = 0.0;
    Real capillary_cfl = 0.15;
    Real slip_length_in_dx = 0.5;
    // The body-fitted full coating represents sub-spacing interface
    // perturbations directly in particle positions. Replacing density by a
    // kernel sum every advective step interpreted that physical mode as a
    // particle-distribution error and erased about 97% of a 1% disturbance by
    // T=1. Keep the continuity-equation density for this target only.
    bool density_reinitialization =
#ifdef SPHINXSYS_FINITE_TAPERED_FULL_COATING_DEFAULTS
        false;
#else
        true;
#endif
    bool particle_regularization = false;
    Real regularization_coefficient = 0.05;
    bool surface_regularization = false;
    Real surface_regularization_coefficient = 0.05;
    Real surface_max_shift_in_dx = 0.03;
    Real surface_hourglass_coefficient = 4.5;
    std::string normal_reconstruction = "gradient";
    std::string jfm_solid_extension = "hydrophilic-only";
    std::string jfm_neighbor_mode = "mixed-support";
    std::string jfm_endpoint_curvature = "extrapolated";
    bool jfm_curvature_outlier_filter = false;
    Real jfm_curvature_ratio_limit = 3.0;
    int hybrid_min_dry_bins = 3;
    Real hybrid_transition_T = 4.0;
    bool jfm_pressure_balance = false;
    bool css_gradient_correction = false;
    bool separate_contact_endpoint = false;
    bool complete_young_residual = true;
    bool complete_endpoint_vector = false;
    Real endpoint_spread_in_dx = 0.0;
    int support_min_neighbors = 0;
    int support_full_neighbors = 1;
    std::string van_der_waals_model = "off";
    bool van_der_waals_initial_pressure = false;
    std::string van_der_waals_pressure_gauge = "absolute";
    int van_der_waals_gap_smoothing_passes = 0;
    Real van_der_waals_number = 0.02;
    Real van_der_waals_min_gap_in_dx = 1.5;
    Real van_der_waals_resolution_taper_in_dx = 0.0;
    Real van_der_waals_surface_threshold = 0.2;
    Real van_der_waals_ramp_T = 0.5;
    // Xu's attractive H^-3 traction is a pre-rupture thin-film model. Once
    // a genuinely empty wall interval appears it may be faded out smoothly,
    // allowing the post-rupture capillary/adhesion model to take over.
    // Zero preserves the legacy behaviour.
    Real van_der_waals_post_breakup_ramp_T = 0.0;
    Real van_der_waals_cfl = 0.10;
    std::string subgrid_rupture_model = "off";
    Real rupture_cutoff_in_dx = 3.2;
    int rupture_min_bins = 3;
    Real rupture_persistence_T = 0.20;
    int rupture_smoothing_half_width = 1;
    int rupture_max_events = 4;
    Real rupture_max_mass_fraction = 0.10;
    Real recycle_relaxation_T = 4.0;
    Real recycle_max_receiver_fraction = 0.01;
    // Conservative-peel opens an under-resolved neck from its centre rather
    // than converting the complete detected interval in one operation.  The
    // front speed is measured in particle spacings per dimensionless time and
    // each newly reached bin ramps from zero to its full radial cutoff.
    Real rupture_peel_front_speed_dx_per_T = 2.0;
    Real rupture_peel_ramp_T = 0.5;
    bool post_jfm_free_relaxation = false;
    bool post_jfm_continue_recycle = false;
    bool post_jfm_keep_dry_core = false;
    std::string post_jfm_dry_core_profile = "full";
    Real post_breakup_detection_window_T = 1.0;
    std::string random_perturbation = "off";
    Real random_position_in_dx = 0.05;
    Real random_velocity_scale = 0.0;
    int random_seed = 20260829;
    std::string thermal_noise = "off";
    std::string thermal_noise_mode = "all";
    Real thermal_noise_strength = 0.0;
    int thermal_noise_seed = 20260829;
    Real thermal_noise_max_kick_scale = 0.05;
    bool thermal_noise_stop_after_rupture = false;
    // Barker et al. use the thermal scale We*=k_B T/(gamma a^2).  These
    // physical values are diagnostic inputs only: the present SPH model is
    // a one-phase, weakly-compressible free-surface model, not their binary-
    // mixture fluctuating-hydrodynamics solver.
    Real barker_temperature_K = 453.15;
    Real barker_fibre_radius_um = 10.0;
    Real barker_surface_tension_Npm = 0.030;
};

BreakupOptions breakup_options;

// Runtime gate for the wall-adhesion mechanism.  In staged mode it remains
// zero until a resolved dry-fibre interval (or an accepted subgrid rupture
// event) is detected, then rises smoothly to one over the requested ramp.
Real wall_adhesion_activation_factor = 1.0;
Real wall_adhesion_activation_T = 0.0;
bool wall_adhesion_post_breakup_active = true;

/**
 * Thermal-fluctuation scale used by Barker, Bell and Garcia (PNAS, 2023).
 * It permits an honest comparison between a molecular-scale FHD model and
 * this micrometre-scale SPH calculation before any artificial noise is used.
 */
Real BarkerStochasticWeber()
{
    constexpr Real boltzmann_constant = 1.380649e-23; // J/K
    const Real fibre_radius_m = breakup_options.barker_fibre_radius_um * 1.0e-6;
    return boltzmann_constant * breakup_options.barker_temperature_K /
           (breakup_options.barker_surface_tension_Npm *
            fibre_radius_m * fibre_radius_m);
}

Real BarkerThermalLengthMeters()
{
    constexpr Real boltzmann_constant = 1.380649e-23; // J/K
    return std::sqrt(boltzmann_constant * breakup_options.barker_temperature_K /
                     breakup_options.barker_surface_tension_Npm);
}

Real BarkerThermalLengthInParticleSpacings()
{
    const Real fibre_radius_m = breakup_options.barker_fibre_radius_um * 1.0e-6;
    const Real physical_particle_spacing =
        fibre_radius_m * ParticleSpacing() / (fibre_radius + TinyReal);
    return BarkerThermalLengthMeters() /
           (physical_particle_spacing + TinyReal);
}

BreakupOptions ParseBreakupOptions(int argc, char *argv[],
                                   std::vector<std::string> &base_arguments)
{
    BreakupOptions options;
    base_arguments.push_back(argv[0]);
    for (int i = 1; i < argc; ++i)
    {
        std::string argument(argv[i]);
        if (StartsWith(argument, "--post-breakup-models="))
        {
            std::string value = argument.substr(
                std::string("--post-breakup-models=").size());
            if (value != "on" && value != "off")
                throw std::runtime_error(
                    "--post-breakup-models must be on or off.");
            options.post_breakup_models = value == "on";
        }
        else if (StartsWith(argument, "--output-tag="))
            options.output_tag = argument.substr(13);
        else if (StartsWith(argument, "--pattern="))
            options.pattern = argument.substr(10);
        else if (StartsWith(argument, "--initial-topology="))
            options.initial_topology = argument.substr(19);
        else if (StartsWith(argument, "--gap-width="))
            options.gap_width_over_a = std::stod(argument.substr(12));
        else if (StartsWith(argument, "--seeded-pressure="))
            options.seeded_pressure = argument.substr(18);
        else if (StartsWith(argument, "--full-coating-pressure="))
            options.full_coating_initial_pressure = argument.substr(
                std::string("--full-coating-pressure=").size());
        else if (StartsWith(argument, "--wetting="))
            options.wetting_model = argument.substr(10);
        else if (StartsWith(argument, "--contact-angle="))
            options.contact_angle_degrees = std::stod(argument.substr(16));
        else if (StartsWith(argument, "--gamma-lg="))
            options.gamma_lg = std::stod(argument.substr(11));
        else if (StartsWith(argument, "--gamma-sl="))
            options.gamma_sl = std::stod(argument.substr(11));
        else if (StartsWith(argument, "--gamma-sg="))
            options.gamma_sg = std::stod(argument.substr(11));
        else if (StartsWith(argument, "--solid-curvature-traction="))
        {
            std::string value = argument.substr(27);
            if (value != "on" && value != "off")
                throw std::runtime_error("--solid-curvature-traction must be on or off.");
            options.solid_curvature_traction = value == "on";
        }
        else if (StartsWith(argument, "--wetting-force-scale="))
            options.wetting_force_scale = std::stod(argument.substr(22));
        else if (StartsWith(argument, "--contact-core-dx="))
            options.contact_core_in_dx = std::stod(argument.substr(18));
        else if (StartsWith(argument, "--contact-zone-dx="))
            options.contact_zone_in_dx = std::stod(argument.substr(18));
        else if (StartsWith(argument, "--contact-angle-model="))
            options.contact_angle_model = argument.substr(22);
        else if (StartsWith(argument, "--contact-line-friction="))
            options.contact_line_friction = std::stod(argument.substr(24));
        else if (StartsWith(argument, "--contact-speed-filter-T="))
            options.contact_speed_filter_T = std::stod(argument.substr(25));
        else if (StartsWith(argument, "--contact-speed-cap-Ca="))
            options.contact_speed_capillary_number = std::stod(argument.substr(23));
        else if (StartsWith(argument, "--wall-adhesion-strength="))
            options.wall_adhesion_strength = std::stod(
                argument.substr(std::string("--wall-adhesion-strength=").size()));
        else if (StartsWith(argument, "--wall-adhesion-range-dx="))
            options.wall_adhesion_range_in_dx = std::stod(
                argument.substr(std::string("--wall-adhesion-range-dx=").size()));
        else if (StartsWith(argument, "--wall-adhesion-profile="))
            options.wall_adhesion_profile = argument.substr(
                std::string("--wall-adhesion-profile=").size());
        else if (StartsWith(argument, "--wall-adhesion-activation="))
            options.wall_adhesion_activation = argument.substr(
                std::string("--wall-adhesion-activation=").size());
        else if (StartsWith(argument, "--wall-adhesion-ramp-T="))
            options.wall_adhesion_ramp_T = std::stod(
                argument.substr(std::string("--wall-adhesion-ramp-T=").size()));
        else if (StartsWith(argument, "--substep-force-diagnostics="))
        {
            std::string value = argument.substr(28);
            if (value != "on" && value != "off")
                throw std::runtime_error("--substep-force-diagnostics must be on or off.");
            options.substep_force_diagnostics = value == "on";
        }
        else if (StartsWith(argument, "--force-sample-dT="))
            options.force_sample_dT = std::stod(argument.substr(18));
        else if (StartsWith(argument, "--force-control-band-dx="))
            options.force_control_band_in_dx = std::stod(argument.substr(24));
        else if (StartsWith(argument, "--surface-gradient-floor="))
            options.surface_gradient_floor_in_dx = std::stod(argument.substr(25));
        else if (StartsWith(argument, "--surface-smoothing-passes="))
            options.surface_smoothing_passes = std::stoi(argument.substr(27));
        else if (StartsWith(argument, "--normal-alignment-floor="))
            options.normal_alignment_floor = std::stod(argument.substr(25));
        else if (StartsWith(argument, "--capillary-cfl="))
            options.capillary_cfl = std::stod(argument.substr(16));
        else if (StartsWith(argument, "--slip-length-in-dx="))
            options.slip_length_in_dx = std::stod(argument.substr(20));
        else if (StartsWith(argument, "--density-reinit="))
        {
            std::string value = argument.substr(17);
            if (value != "on" && value != "off")
                throw std::runtime_error("--density-reinit must be on or off.");
            options.density_reinitialization = value == "on";
        }
        else if (StartsWith(argument, "--particle-regularization="))
        {
            std::string value = argument.substr(26);
            if (value != "on" && value != "off")
                throw std::runtime_error("--particle-regularization must be on or off.");
            options.particle_regularization = value == "on";
        }
        else if (StartsWith(argument, "--regularization-coefficient="))
            options.regularization_coefficient = std::stod(argument.substr(29));
        else if (StartsWith(argument, "--surface-regularization="))
        {
            std::string value = argument.substr(25);
            if (value != "on" && value != "off")
                throw std::runtime_error("--surface-regularization must be on or off.");
            options.surface_regularization = value == "on";
        }
        else if (StartsWith(argument, "--surface-regularization-coefficient="))
            options.surface_regularization_coefficient = std::stod(argument.substr(37));
        else if (StartsWith(argument, "--surface-max-shift-dx="))
            options.surface_max_shift_in_dx = std::stod(argument.substr(23));
        else if (StartsWith(argument, "--surface-hourglass="))
            options.surface_hourglass_coefficient = std::stod(argument.substr(20));
        else if (StartsWith(argument, "--normal-reconstruction="))
            options.normal_reconstruction = argument.substr(24);
        else if (StartsWith(argument, "--jfm-solid-extension="))
            options.jfm_solid_extension = argument.substr(22);
        else if (StartsWith(argument, "--jfm-neighbor-mode="))
            options.jfm_neighbor_mode = argument.substr(20);
        else if (StartsWith(argument, "--jfm-endpoint-curvature="))
            options.jfm_endpoint_curvature = argument.substr(25);
        else if (StartsWith(argument, "--jfm-curvature-outlier-filter="))
        {
            std::string value = argument.substr(31);
            if (value != "on" && value != "off")
                throw std::runtime_error(
                    "--jfm-curvature-outlier-filter must be on or off.");
            options.jfm_curvature_outlier_filter = value == "on";
        }
        else if (StartsWith(argument, "--jfm-curvature-ratio-limit="))
            options.jfm_curvature_ratio_limit = std::stod(argument.substr(28));
        else if (StartsWith(argument, "--hybrid-min-dry-bins="))
            options.hybrid_min_dry_bins = std::stoi(argument.substr(22));
        else if (StartsWith(argument, "--hybrid-transition-T="))
            options.hybrid_transition_T = std::stod(argument.substr(22));
        else if (StartsWith(argument, "--jfm-pressure-balance="))
        {
            std::string value = argument.substr(23);
            if (value != "on" && value != "off")
                throw std::runtime_error("--jfm-pressure-balance must be on or off.");
            options.jfm_pressure_balance = value == "on";
        }
        else if (StartsWith(argument, "--css-gradient-correction="))
        {
            std::string value = argument.substr(26);
            if (value != "on" && value != "off")
                throw std::runtime_error("--css-gradient-correction must be on or off.");
            options.css_gradient_correction = value == "on";
        }
        else if (StartsWith(argument, "--separate-contact-endpoint="))
        {
            std::string value = argument.substr(28);
            if (value != "on" && value != "off")
                throw std::runtime_error("--separate-contact-endpoint must be on or off.");
            options.separate_contact_endpoint = value == "on";
        }
        else if (StartsWith(argument, "--support-min-neighbors="))
            options.support_min_neighbors = std::stoi(argument.substr(24));
        else if (StartsWith(argument, "--support-full-neighbors="))
            options.support_full_neighbors = std::stoi(argument.substr(25));
        else if (StartsWith(argument, "--van-der-waals="))
            options.van_der_waals_model = argument.substr(16);
        else if (StartsWith(argument, "--vdw-initial-pressure="))
        {
            std::string value = argument.substr(23);
            if (value != "on" && value != "off")
                throw std::runtime_error("--vdw-initial-pressure must be on or off.");
            options.van_der_waals_initial_pressure = value == "on";
        }
        else if (StartsWith(argument, "--vdw-pressure-gauge="))
            options.van_der_waals_pressure_gauge = argument.substr(21);
        else if (StartsWith(argument, "--vdw-gap-smoothing-passes="))
            options.van_der_waals_gap_smoothing_passes =
                std::stoi(argument.substr(27));
        else if (StartsWith(argument, "--vdw-number="))
            options.van_der_waals_number = std::stod(argument.substr(13));
        else if (StartsWith(argument, "--vdw-min-gap-dx="))
            options.van_der_waals_min_gap_in_dx = std::stod(argument.substr(17));
        else if (StartsWith(argument, "--vdw-resolution-taper-dx="))
            options.van_der_waals_resolution_taper_in_dx =
                std::stod(argument.substr(26));
        else if (StartsWith(argument, "--vdw-surface-threshold="))
            options.van_der_waals_surface_threshold = std::stod(argument.substr(24));
        else if (StartsWith(argument, "--vdw-ramp-T="))
            options.van_der_waals_ramp_T = std::stod(argument.substr(13));
        else if (StartsWith(argument, "--vdw-post-breakup-ramp-T="))
            options.van_der_waals_post_breakup_ramp_T = std::stod(
                argument.substr(std::string("--vdw-post-breakup-ramp-T=").size()));
        else if (StartsWith(argument, "--vdw-cfl="))
            options.van_der_waals_cfl = std::stod(argument.substr(10));
        else if (StartsWith(argument, "--subgrid-rupture="))
            options.subgrid_rupture_model = argument.substr(18);
        else if (StartsWith(argument, "--rupture-cutoff-dx="))
            options.rupture_cutoff_in_dx = std::stod(argument.substr(20));
        else if (StartsWith(argument, "--rupture-min-bins="))
            options.rupture_min_bins = std::stoi(argument.substr(19));
        else if (StartsWith(argument, "--rupture-persistence-T="))
            options.rupture_persistence_T = std::stod(argument.substr(24));
        else if (StartsWith(argument, "--rupture-smoothing-half-width="))
            options.rupture_smoothing_half_width = std::stoi(argument.substr(31));
        else if (StartsWith(argument, "--rupture-max-events="))
            options.rupture_max_events = std::stoi(argument.substr(21));
        else if (StartsWith(argument, "--rupture-max-mass-fraction="))
            options.rupture_max_mass_fraction = std::stod(argument.substr(28));
        else if (StartsWith(argument, "--recycle-relaxation-T="))
            options.recycle_relaxation_T = std::stod(argument.substr(23));
        else if (StartsWith(argument, "--recycle-max-receiver-fraction="))
            options.recycle_max_receiver_fraction = std::stod(argument.substr(32));
        else if (StartsWith(argument, "--rupture-peel-front-speed-dx-per-T="))
            options.rupture_peel_front_speed_dx_per_T = std::stod(
                argument.substr(std::string(
                    "--rupture-peel-front-speed-dx-per-T=").size()));
        else if (StartsWith(argument, "--rupture-peel-ramp-T="))
            options.rupture_peel_ramp_T = std::stod(
                argument.substr(std::string("--rupture-peel-ramp-T=").size()));
        else if (StartsWith(argument, "--post-jfm-free-relaxation="))
        {
            std::string value = argument.substr(27);
            if (value != "on" && value != "off")
                throw std::runtime_error(
                    "--post-jfm-free-relaxation must be on or off.");
            options.post_jfm_free_relaxation = value == "on";
        }
        else if (StartsWith(argument, "--post-jfm-continue-recycle="))
        {
            std::string value = argument.substr(28);
            if (value != "on" && value != "off")
                throw std::runtime_error(
                    "--post-jfm-continue-recycle must be on or off.");
            options.post_jfm_continue_recycle = value == "on";
        }
        else if (StartsWith(argument, "--post-jfm-keep-dry-core="))
        {
            std::string value = argument.substr(25);
            if (value != "on" && value != "off")
                throw std::runtime_error(
                    "--post-jfm-keep-dry-core must be on or off.");
            options.post_jfm_keep_dry_core = value == "on";
        }
        else if (StartsWith(argument, "--post-jfm-dry-core-profile="))
            options.post_jfm_dry_core_profile = argument.substr(28);
        else if (StartsWith(argument, "--post-breakup-detection-window-T="))
            options.post_breakup_detection_window_T = std::stod(
                argument.substr(std::string("--post-breakup-detection-window-T=").size()));
        else if (StartsWith(argument, "--random-perturbation="))
            options.random_perturbation = argument.substr(22);
        else if (StartsWith(argument, "--random-position-dx="))
            options.random_position_in_dx = std::stod(argument.substr(21));
        else if (StartsWith(argument, "--random-velocity-scale="))
            options.random_velocity_scale = std::stod(argument.substr(24));
        else if (StartsWith(argument, "--random-seed="))
            options.random_seed = std::stoi(argument.substr(14));
        else if (StartsWith(argument, "--thermal-noise="))
            options.thermal_noise = argument.substr(16);
        else if (StartsWith(argument, "--thermal-noise-mode="))
            options.thermal_noise_mode = argument.substr(21);
        else if (StartsWith(argument, "--thermal-noise-strength="))
            options.thermal_noise_strength = std::stod(argument.substr(25));
        else if (StartsWith(argument, "--thermal-noise-seed="))
            options.thermal_noise_seed = std::stoi(argument.substr(21));
        else if (StartsWith(argument, "--thermal-noise-max-kick="))
            options.thermal_noise_max_kick_scale = std::stod(argument.substr(25));
        else if (StartsWith(argument, "--thermal-noise-stop-after-rupture="))
        {
            std::string value = argument.substr(35);
            if (value != "on" && value != "off")
                throw std::runtime_error(
                    "--thermal-noise-stop-after-rupture must be on or off.");
            options.thermal_noise_stop_after_rupture = value == "on";
        }
        else if (StartsWith(argument, "--barker-temperature-K="))
            options.barker_temperature_K = std::stod(argument.substr(23));
        else if (StartsWith(argument, "--barker-fibre-radius-um="))
            options.barker_fibre_radius_um = std::stod(argument.substr(26));
        else if (StartsWith(argument, "--barker-surface-tension-Npm="))
            options.barker_surface_tension_Npm = std::stod(argument.substr(29));
        else
            base_arguments.push_back(argument);
    }
    if (options.pattern != "mixed" && options.pattern != "equal-three" &&
        options.pattern != "single-center")
        throw std::runtime_error(
            "--pattern must be mixed, equal-three or single-center.");
    if (options.full_coating_initial_pressure != "local" &&
        options.full_coating_initial_pressure != "base" &&
        options.full_coating_initial_pressure != "zero")
        throw std::runtime_error(
            "--full-coating-pressure must be local, base or zero.");
    if (!options.output_tag.empty() &&
        !std::all_of(options.output_tag.begin(), options.output_tag.end(),
                     [](unsigned char character)
                     {
                         return std::isalnum(character) || character == '-' ||
                                character == '_';
                     }))
        throw std::runtime_error(
            "--output-tag may contain only letters, numbers, '-' and '_'.");
    if (options.post_jfm_continue_recycle &&
        !options.post_jfm_free_relaxation)
        throw std::runtime_error(
            "--post-jfm-continue-recycle=on requires "
            "--post-jfm-free-relaxation=on.");
    if (options.post_jfm_keep_dry_core &&
        !options.post_jfm_free_relaxation)
        throw std::runtime_error(
            "--post-jfm-keep-dry-core=on requires "
            "--post-jfm-free-relaxation=on.");
    if (options.post_jfm_dry_core_profile != "full" &&
        options.post_jfm_dry_core_profile != "tapered")
        throw std::runtime_error(
            "--post-jfm-dry-core-profile must be full or tapered.");
    if (options.post_breakup_detection_window_T < 0.0 ||
        options.post_breakup_detection_window_T > 10.0)
        throw std::runtime_error(
            "--post-breakup-detection-window-T must be in [0,10].");
    if (options.random_perturbation != "off" &&
        options.random_perturbation != "position" &&
        options.random_perturbation != "velocity" &&
        options.random_perturbation != "position-velocity")
        throw std::runtime_error(
            "--random-perturbation must be off, position, velocity or position-velocity.");
    if (options.random_position_in_dx < 0.0)
        throw std::runtime_error("--random-position-dx must be non-negative.");
    if (options.random_velocity_scale < 0.0)
        throw std::runtime_error("--random-velocity-scale must be non-negative.");
    if (options.thermal_noise != "off" &&
        options.thermal_noise != "particle")
        throw std::runtime_error("--thermal-noise must be off or particle.");
    if (options.thermal_noise_mode != "all" &&
        options.thermal_noise_mode != "surface-tangent" &&
        options.thermal_noise_mode != "ou-tangent")
        throw std::runtime_error(
            "--thermal-noise-mode must be all, surface-tangent or ou-tangent.");
    if (options.thermal_noise_strength < 0.0)
        throw std::runtime_error("--thermal-noise-strength must be non-negative.");
    if (options.thermal_noise_max_kick_scale <= 0.0 ||
        options.thermal_noise_max_kick_scale > 1.0)
        throw std::runtime_error("--thermal-noise-max-kick must be in (0,1].");
    if (options.barker_temperature_K <= 0.0 ||
        options.barker_fibre_radius_um <= 0.0 ||
        options.barker_surface_tension_Npm <= 0.0)
        throw std::runtime_error(
            "Barker physical-scale inputs must all be positive.");
    if (options.initial_topology != "coated" &&
        options.initial_topology != "seeded-gap" &&
        options.initial_topology != "edge-gap" &&
        options.initial_topology != "finite-film" &&
        options.initial_topology != "full-coating" &&
        options.initial_topology != "compact-crown" &&
        options.initial_topology != "cmc-crown")
        throw std::runtime_error(
            "--initial-topology must be coated, seeded-gap, edge-gap, finite-film, full-coating, compact-crown or cmc-crown.");
    if (options.gap_width_over_a <= 0.0)
        throw std::runtime_error("--gap-width must be positive.");
    if (options.seeded_pressure != "uniform" &&
        options.seeded_pressure != "analytic")
        throw std::runtime_error(
            "--seeded-pressure must be uniform or analytic.");
    if (options.wetting_model != "fixed-angle" &&
        options.wetting_model != "surface-energy" &&
        options.wetting_model != "wall-adhesion" &&
        options.wetting_model != "hybrid-adhesion" &&
        options.wetting_model != "jfm" &&
        options.wetting_model != "hybrid")
        throw std::runtime_error(
            "--wetting must be fixed-angle, surface-energy, wall-adhesion, hybrid-adhesion, jfm or hybrid.");
    if (options.contact_angle_degrees <= 5.0 ||
        options.contact_angle_degrees >= 175.0)
        throw std::runtime_error("--contact-angle must be between 5 and 175 degrees.");
    if (options.gamma_lg <= 0.0 || options.gamma_sl < 0.0 || options.gamma_sg < 0.0)
        throw std::runtime_error("Surface energies require gamma-lg > 0 and gamma-sl,gamma-sg >= 0.");
    if (options.wetting_model != "wall-adhesion" &&
        options.wetting_model != "hybrid-adhesion")
    {
        Real young_cosine = (options.gamma_sg - options.gamma_sl) / options.gamma_lg;
        if (young_cosine < -1.0 || young_cosine > 1.0)
            throw std::runtime_error("Surface energies imply complete wetting/dewetting; this prototype requires |(gamma-sg-gamma-sl)/gamma-lg| <= 1.");
    }
    if (options.wall_adhesion_strength < 0.0 ||
        options.wall_adhesion_strength > 10.0)
        throw std::runtime_error("--wall-adhesion-strength must be in [0,10].");
    if (options.wall_adhesion_range_in_dx < 0.5 ||
        options.wall_adhesion_range_in_dx > 4.0)
        throw std::runtime_error("--wall-adhesion-range-dx must be in [0.5,4].");
    if (options.wall_adhesion_profile != "pairwise" &&
        options.wall_adhesion_profile != "traction")
        throw std::runtime_error(
            "--wall-adhesion-profile must be pairwise or traction.");
    if (options.wall_adhesion_activation != "always" &&
        options.wall_adhesion_activation != "after-rupture")
        throw std::runtime_error(
            "--wall-adhesion-activation must be always or after-rupture.");
    if (options.wall_adhesion_ramp_T < 0.0 ||
        options.wall_adhesion_ramp_T > 20.0)
        throw std::runtime_error(
            "--wall-adhesion-ramp-T must be in [0,20].");
    if (options.wetting_force_scale < 0.0 || options.wetting_force_scale > 2.0)
        throw std::runtime_error("--wetting-force-scale must be in [0,2].");
    if (options.contact_core_in_dx < 0.5 ||
        options.contact_core_in_dx >= options.contact_zone_in_dx)
        throw std::runtime_error("--contact-core-dx must be at least 0.5 and smaller than --contact-zone-dx.");
    if (options.contact_zone_in_dx < 1.0 || options.contact_zone_in_dx > 4.0)
        throw std::runtime_error("--contact-zone-dx must be in [1,4].");
    if (options.contact_angle_model != "equilibrium" &&
        options.contact_angle_model != "linear-friction")
        throw std::runtime_error(
            "--contact-angle-model must be equilibrium or linear-friction.");
    if (options.contact_line_friction < 0.0 ||
        options.contact_line_friction > 50.0)
        throw std::runtime_error("--contact-line-friction must be in [0,50].");
    if (options.contact_speed_filter_T <= 0.0 ||
        options.contact_speed_filter_T > 5.0)
        throw std::runtime_error("--contact-speed-filter-T must be in (0,5].");
    if (options.contact_speed_capillary_number <= 0.0 ||
        options.contact_speed_capillary_number > 1.0)
        throw std::runtime_error("--contact-speed-cap-Ca must be in (0,1].");
    if (options.force_sample_dT <= 0.0 || options.force_sample_dT > 1.0)
        throw std::runtime_error("--force-sample-dT must be in (0,1].");
    if (options.force_control_band_in_dx < 2.0 ||
        options.force_control_band_in_dx > 10.0)
        throw std::runtime_error("--force-control-band-dx must be in [2,10].");
    if (options.surface_gradient_floor_in_dx <= 0.0)
        throw std::runtime_error("--surface-gradient-floor must be positive.");
    if (options.surface_smoothing_passes < 0 ||
        options.surface_smoothing_passes > 4)
        throw std::runtime_error("--surface-smoothing-passes must be in [0,4].");
    if (options.normal_alignment_floor < -0.5 ||
        options.normal_alignment_floor >= 1.0)
        throw std::runtime_error("--normal-alignment-floor must be in [-0.5,1). ");
    if (options.capillary_cfl <= 0.02 || options.capillary_cfl > 0.6)
        throw std::runtime_error("--capillary-cfl must be in (0.02,0.6].");
    if (options.slip_length_in_dx < 0.0 ||
        options.slip_length_in_dx > 10.0)
        throw std::runtime_error("--slip-length-in-dx must be in [0,10].");
    if (options.regularization_coefficient < 0.0 ||
        options.regularization_coefficient > 0.2)
        throw std::runtime_error("--regularization-coefficient must be in [0,0.2].");
    if (options.surface_regularization_coefficient < 0.0 ||
        options.surface_regularization_coefficient > 0.2)
        throw std::runtime_error("--surface-regularization-coefficient must be in [0,0.2].");
    if (options.surface_max_shift_in_dx <= 0.0 ||
        options.surface_max_shift_in_dx > 0.1)
        throw std::runtime_error("--surface-max-shift-dx must be in (0,0.1].");
    if (options.surface_hourglass_coefficient < 0.0 ||
        options.surface_hourglass_coefficient > 6.0)
        throw std::runtime_error("--surface-hourglass must be in [0,6].");
    if (options.normal_reconstruction != "gradient" &&
        options.normal_reconstruction != "pca")
        throw std::runtime_error("--normal-reconstruction must be gradient or pca.");
    if (options.jfm_solid_extension != "all" &&
        options.jfm_solid_extension != "hydrophilic-only")
        throw std::runtime_error(
            "--jfm-solid-extension must be all or hydrophilic-only.");
    if (options.jfm_neighbor_mode != "mixed-support" &&
        options.jfm_neighbor_mode != "paper-split")
        throw std::runtime_error(
            "--jfm-neighbor-mode must be mixed-support or paper-split.");
    if (options.jfm_endpoint_curvature != "raw" &&
        options.jfm_endpoint_curvature != "extrapolated" &&
        options.jfm_endpoint_curvature != "one-sided")
        throw std::runtime_error(
            "--jfm-endpoint-curvature must be raw, extrapolated or one-sided.");
    if (options.jfm_curvature_ratio_limit < 1.5 ||
        options.jfm_curvature_ratio_limit > 20.0)
        throw std::runtime_error(
            "--jfm-curvature-ratio-limit must be in [1.5,20].");
    if (options.hybrid_min_dry_bins < 2 ||
        options.hybrid_min_dry_bins > 12)
        throw std::runtime_error("--hybrid-min-dry-bins must be in [2,12].");
    if (options.hybrid_transition_T <= 0.0 ||
        options.hybrid_transition_T > 20.0)
        throw std::runtime_error("--hybrid-transition-T must be in (0,20].");
    if (options.support_min_neighbors < 0 ||
        options.support_min_neighbors >= options.support_full_neighbors)
        throw std::runtime_error("--support-min-neighbors must be non-negative and smaller than --support-full-neighbors.");
    if (options.support_full_neighbors > 20)
        throw std::runtime_error("--support-full-neighbors must not exceed 20.");
    if (options.van_der_waals_model != "off" &&
        options.van_der_waals_model != "xu")
        throw std::runtime_error("--van-der-waals must be off or xu.");
    if (options.van_der_waals_initial_pressure &&
        options.van_der_waals_model != "xu")
        throw std::runtime_error(
            "--vdw-initial-pressure=on requires --van-der-waals=xu.");
    if (options.van_der_waals_initial_pressure &&
        options.initial_topology != "coated")
        throw std::runtime_error(
            "--vdw-initial-pressure=on is currently validated only for --initial-topology=coated.");
    if (options.van_der_waals_pressure_gauge != "absolute" &&
        options.van_der_waals_pressure_gauge != "mean-film")
        throw std::runtime_error(
            "--vdw-pressure-gauge must be absolute or mean-film.");
    if (options.van_der_waals_gap_smoothing_passes < 0 ||
        options.van_der_waals_gap_smoothing_passes > 8)
        throw std::runtime_error(
            "--vdw-gap-smoothing-passes must be in [0,8].");
    if (options.van_der_waals_number < 0.0 ||
        options.van_der_waals_number > 1.0)
        throw std::runtime_error("--vdw-number must be in [0,1].");
    if (options.van_der_waals_min_gap_in_dx < 0.5 ||
        options.van_der_waals_min_gap_in_dx > 5.0)
        throw std::runtime_error("--vdw-min-gap-dx must be in [0.5,5].");
    if (options.van_der_waals_resolution_taper_in_dx < 0.0 ||
        options.van_der_waals_resolution_taper_in_dx > 12.0 ||
        (options.van_der_waals_resolution_taper_in_dx > 0.0 &&
         options.van_der_waals_resolution_taper_in_dx <=
             options.van_der_waals_min_gap_in_dx))
        throw std::runtime_error(
            "--vdw-resolution-taper-dx must be 0 (off) or larger than --vdw-min-gap-dx and at most 12.");
    if (options.van_der_waals_surface_threshold < 0.05 ||
        options.van_der_waals_surface_threshold > 1.0)
        throw std::runtime_error("--vdw-surface-threshold must be in [0.05,1].");
    if (options.van_der_waals_ramp_T < 0.0 ||
        options.van_der_waals_ramp_T > 10.0)
        throw std::runtime_error("--vdw-ramp-T must be in [0,10].");
    if (options.van_der_waals_post_breakup_ramp_T < 0.0 ||
        options.van_der_waals_post_breakup_ramp_T > 10.0)
        throw std::runtime_error(
            "--vdw-post-breakup-ramp-T must be in [0,10].");
    if (options.van_der_waals_cfl <= 0.01 ||
        options.van_der_waals_cfl > 0.5)
        throw std::runtime_error("--vdw-cfl must be in (0.01,0.5].");
    if (options.subgrid_rupture_model != "off" &&
        options.subgrid_rupture_model != "adsorbed-layer" &&
        options.subgrid_rupture_model != "mass-recycle" &&
        options.subgrid_rupture_model != "conservative-peel")
        throw std::runtime_error(
            "--subgrid-rupture must be off, adsorbed-layer, mass-recycle or conservative-peel.");
    if (options.subgrid_rupture_model != "off" &&
        options.van_der_waals_model != "xu")
        throw std::runtime_error(
            "A subgrid rupture closure currently requires --van-der-waals=xu.");
    if (options.rupture_cutoff_in_dx < 2.0 ||
        options.rupture_cutoff_in_dx > 12.0)
        throw std::runtime_error("--rupture-cutoff-dx must be in [2,12].");
    if (options.rupture_min_bins < 2 || options.rupture_min_bins > 12)
        throw std::runtime_error("--rupture-min-bins must be in [2,12].");
    if (options.rupture_persistence_T < 0.0 ||
        options.rupture_persistence_T > 5.0)
        throw std::runtime_error("--rupture-persistence-T must be in [0,5].");
    if (options.rupture_smoothing_half_width < 0 ||
        options.rupture_smoothing_half_width > 4)
        throw std::runtime_error(
            "--rupture-smoothing-half-width must be in [0,4].");
    if (options.rupture_max_events < 1 || options.rupture_max_events > 20)
        throw std::runtime_error("--rupture-max-events must be in [1,20].");
    if (options.rupture_max_mass_fraction <= 0.0 ||
        options.rupture_max_mass_fraction > 0.25)
        throw std::runtime_error(
            "--rupture-max-mass-fraction must be in (0,0.25].");
    if (options.recycle_relaxation_T <= 0.0 ||
        options.recycle_relaxation_T > 50.0)
        throw std::runtime_error("--recycle-relaxation-T must be in (0,50].");
    if (options.recycle_max_receiver_fraction <= 0.0 ||
        options.recycle_max_receiver_fraction > 0.10)
        throw std::runtime_error(
            "--recycle-max-receiver-fraction must be in (0,0.10].");
    if (options.rupture_peel_front_speed_dx_per_T <= 0.0 ||
        options.rupture_peel_front_speed_dx_per_T > 20.0)
        throw std::runtime_error(
            "--rupture-peel-front-speed-dx-per-T must be in (0,20].");
    if (options.rupture_peel_ramp_T <= 0.0 ||
        options.rupture_peel_ramp_T > 10.0)
        throw std::runtime_error(
            "--rupture-peel-ramp-T must be in (0,10].");
    return options;
}

std::string BreakupLabel()
{
    if (!breakup_options.output_tag.empty())
        return breakup_options.output_tag;
    std::ostringstream label;
    std::string wetting_tag = breakup_options.wetting_model == "surface-energy"
                                  ? "se"
                              : breakup_options.wetting_model == "fixed-angle"
                                  ? "fa"
                              : breakup_options.wetting_model == "wall-adhesion"
                                  ? "wa"
                              : breakup_options.wetting_model == "hybrid-adhesion"
                                  ? "hya"
                              : breakup_options.wetting_model == "hybrid"
                                  ? "hyb"
                                  : "jfm";
    label << "breakup_" << run_options.Label() << "_" << wetting_tag;
    if (breakup_options.wetting_model == "fixed-angle")
        label << "_theta" << static_cast<int>(std::round(breakup_options.contact_angle_degrees));
    else
        label << "_glg" << static_cast<int>(std::round(100.0 * breakup_options.gamma_lg))
              << "_gsl" << static_cast<int>(std::round(100.0 * breakup_options.gamma_sl))
              << "_gsg" << static_cast<int>(std::round(100.0 * breakup_options.gamma_sg));
    label << "_sct" << (breakup_options.solid_curvature_traction ? 1 : 0)
          << "_wfs" << static_cast<int>(std::round(
                             100.0 * breakup_options.wetting_force_scale));
    if (breakup_options.wetting_model == "wall-adhesion" ||
        breakup_options.wetting_model == "hybrid-adhesion")
        label << "_was" << static_cast<int>(std::round(
                             100.0 * breakup_options.wall_adhesion_strength))
              << "_war" << static_cast<int>(std::round(
                             100.0 * breakup_options.wall_adhesion_range_in_dx))
              << "_wap" << (breakup_options.wall_adhesion_profile == "pairwise"
                                  ? "pw"
                                  : "tr")
              << "_waa" << (breakup_options.wall_adhesion_activation ==
                                     "after-rupture"
                                 ? "post"
                                 : "all")
              << "_warT" << static_cast<int>(std::round(
                                  100.0 * breakup_options.wall_adhesion_ramp_T));
    if (breakup_options.post_jfm_keep_dry_core)
        label << "_dcp" << (breakup_options.post_jfm_dry_core_profile == "tapered"
                                   ? "tap"
                                   : "full");
    label << "_" << breakup_options.pattern
          << "_topo" << breakup_options.initial_topology;
    if (breakup_options.initial_topology == "seeded-gap" ||
        breakup_options.initial_topology == "edge-gap" ||
        breakup_options.initial_topology == "compact-crown" ||
        breakup_options.initial_topology == "cmc-crown")
        label << static_cast<int>(std::round(
            100.0 * breakup_options.gap_width_over_a))
              << "_p" << (breakup_options.seeded_pressure == "uniform" ? "u" : "a");
    label
          << "_a" << static_cast<int>(std::round(10000.0 * run_options.amplitude_ratio))
          << "_s" << breakup_options.surface_smoothing_passes
          << "_cfl" << static_cast<int>(std::round(100.0 * breakup_options.capillary_cfl))
          << "_b" << static_cast<int>(std::round(
                            100.0 * breakup_options.slip_length_in_dx))
          << "_dr" << (breakup_options.density_reinitialization ? 1 : 0)
          << "_pr" << static_cast<int>(std::round(
                             100.0 * (breakup_options.particle_regularization
                                          ? breakup_options.regularization_coefficient
                                          : 0.0)))
          << "_sr" << static_cast<int>(std::round(
                             100.0 * (breakup_options.surface_regularization
                                          ? breakup_options.surface_regularization_coefficient
                                          : 0.0)))
          << "_hg" << static_cast<int>(std::round(
                             100.0 * breakup_options.surface_hourglass_coefficient))
          << "_n" << breakup_options.normal_reconstruction
          << "_jse" << (breakup_options.jfm_solid_extension == "all" ? 1 : 0)
          << "_jnm" << (breakup_options.jfm_neighbor_mode == "paper-split" ? 1 : 0)
          << "_jec" << (breakup_options.jfm_endpoint_curvature == "raw"
                              ? 0
                              : (breakup_options.jfm_endpoint_curvature == "extrapolated" ? 1 : 2))
          << "_jcf" << (breakup_options.jfm_curvature_outlier_filter ? 1 : 0)
          << "r" << static_cast<int>(std::round(
                         10.0 * breakup_options.jfm_curvature_ratio_limit))
          << "_hdb" << breakup_options.hybrid_min_dry_bins
          << "t" << static_cast<int>(std::round(
                         10.0 * breakup_options.hybrid_transition_T))
          << "_jpb" << (breakup_options.jfm_pressure_balance ? 1 : 0)
          << "_cam" << (breakup_options.contact_angle_model == "linear-friction" ? 1 : 0)
          << "f" << static_cast<int>(std::round(breakup_options.contact_line_friction))
          << "_cgc" << (breakup_options.css_gradient_correction ? 1 : 0)
          << "_vdw" << (breakup_options.van_der_waals_model == "xu" ? 1 : 0);
    if (breakup_options.van_der_waals_model == "xu")
        label << "n" << static_cast<int>(std::round(
                     1000.0 * breakup_options.van_der_waals_number))
              << "g" << static_cast<int>(std::round(
                     100.0 * breakup_options.van_der_waals_min_gap_in_dx))
              << "z" << static_cast<int>(std::round(
                     100.0 * breakup_options.van_der_waals_resolution_taper_in_dx))
              << "p" << (breakup_options.van_der_waals_initial_pressure ? 1 : 0)
              << "q" << (breakup_options.van_der_waals_pressure_gauge == "mean-film" ? 1 : 0)
              << "s" << breakup_options.van_der_waals_gap_smoothing_passes;
    if (breakup_options.random_perturbation != "off")
        label << "_rnd" << (breakup_options.random_perturbation == "position"
                                ? "p"
                            : breakup_options.random_perturbation == "velocity"
                                ? "v"
                                : "pv")
              << "x" << static_cast<int>(std::round(
                             100.0 * breakup_options.random_position_in_dx))
              << "u" << static_cast<int>(std::round(
                             1000.0 * breakup_options.random_velocity_scale))
              << "s" << breakup_options.random_seed;
    if (breakup_options.thermal_noise != "off")
        label << "_thp" << static_cast<int>(std::round(
                         100.0 * breakup_options.thermal_noise_strength))
              << "s" << breakup_options.thermal_noise_seed
              << "_" << (breakup_options.thermal_noise_mode == "surface-tangent"
                              ? "tan"
                              : breakup_options.thermal_noise_mode == "ou-tangent"
                                    ? "ou"
                                    : "all");
    return label.str();
}

Real YoungCosine()
{
    return std::clamp((breakup_options.gamma_sg - breakup_options.gamma_sl) /
                          breakup_options.gamma_lg,
                      Real(-1.0), Real(1.0));
}

Real YoungAngleDegrees()
{
    return std::acos(YoungCosine()) * 180.0 / Pi;
}

Real BreakupPattern(Real x)
{
    if (breakup_options.pattern == "equal-three")
        return std::cos(3.0 * BasicWaveNumber() * x);
    if (breakup_options.pattern == "single-center")
        return std::cos(BasicWaveNumber() * x);
    return SurfacePattern(x);
}

Real BreakupPatternFirstDerivative(Real x)
{
    if (breakup_options.pattern == "equal-three")
        return -3.0 * BasicWaveNumber() * std::sin(3.0 * BasicWaveNumber() * x);
    if (breakup_options.pattern == "single-center")
        return -BasicWaveNumber() * std::sin(BasicWaveNumber() * x);
    return SurfacePatternFirstDerivative(x);
}

Real BreakupPatternSecondDerivative(Real x)
{
    if (breakup_options.pattern == "equal-three")
    {
        Real k = 3.0 * BasicWaveNumber();
        return -k * k * std::cos(k * x);
    }
    if (breakup_options.pattern == "single-center")
    {
        Real k = BasicWaveNumber();
        return -k * k * std::cos(k * x);
    }
    return SurfacePatternSecondDerivative(x);
}

Real BreakupSurfaceRadius(Real x)
{
    // Measure the film from the local solid surface.  This is identical to the
    // former expression for a uniform fibre and lets the topology-capable
    // solver operate on a conical fibre without placing liquid inside solid.
    return FiberRadius(x) + FilmThickness() +
           InitialAmplitude() * BreakupPattern(x);
}

Real SeedGapWidth()
{
    return breakup_options.gap_width_over_a * fibre_radius;
}

/**
 * Initial liquid thickness for a prescribed bulk thickness.  The two
 * quarter-circle shoulders meet the fibre at 90 degrees and join the flat
 * film with zero slope.  This is an initial-condition test, not a breakup
 * law: subsequent contact-line motion is determined by the SPH equations.
 */
Real SeededGapThickness(Real x, Real bulk_thickness)
{
    Real wet_distance = std::abs(x) - 0.5 * SeedGapWidth();
    if (wet_distance <= 0.0)
        return 0.0;
    if (wet_distance >= bulk_thickness)
        return bulk_thickness;
    return std::sqrt(std::max(
        Real(0.0), 2.0 * bulk_thickness * wet_distance -
                       wet_distance * wet_distance));
}

/**
 * A single connected wet interval centred at x=0, with the prescribed dry
 * fibre interval crossing the periodic boundary. The shoulder is a quarter
 * circle for a 90-degree initial contact and the small cosine perturbation is
 * ramped to zero at the contact edge. Thus the centre is a genuine small
 * disturbance peak, not a preformed bead.
 */
Real EdgeGapThickness(Real x, Real lower, Real upper)
{
    Real bulk_thickness = FilmThickness();
    Real wet_distance = std::min(x - lower, upper - x);
    if (wet_distance <= 0.0)
        return 0.0;
    Real shoulder = wet_distance >= bulk_thickness
                        ? bulk_thickness
                        : std::sqrt(std::max(
                              Real(0.0), 2.0 * bulk_thickness * wet_distance -
                                             wet_distance * wet_distance));
    Real shoulder_weight = std::clamp(
        shoulder / (bulk_thickness + TinyReal), Real(0.0), Real(1.0));
    return std::max(
        Real(0.0), shoulder +
                       InitialAmplitude() * BreakupPattern(x) * shoulder_weight);
}

/** Undisturbed finite coating on the central tapered sidewall.  A circular
 *  shoulder of radius h joins the dry fibre at each end and reaches the
 *  constant-thickness film with zero slope. */
Real FiniteFilmBaseThickness(Real x)
{
    const Real lower = ConeLowerBound();
    const Real upper = ConeUpperBound();
    const Real bulk_thickness = FilmThickness();
    const Real wet_distance = std::min(x - lower, upper - x);
    if (wet_distance <= 0.0)
        return 0.0;
    if (wet_distance >= bulk_thickness)
        return bulk_thickness;
    return std::sqrt(std::max(
        Real(0.0), 2.0 * bulk_thickness * wet_distance -
                       wet_distance * wet_distance));
}

Real FiniteFilmThickness(Real x)
{
    Real base = FiniteFilmBaseThickness(x);
    Real weight = std::clamp(base / (FilmThickness() + TinyReal),
                             Real(0.0), Real(1.0));
    return std::max(Real(0.0),
                    base + InitialAmplitude() * BreakupPattern(x) * weight);
}

Real FiniteFilmSurfaceRadius(Real x)
{
    return FiberRadius(x) + FiniteFilmThickness(x);
}

bool HasCentralWetFootprint()
{
    return breakup_options.initial_topology == "edge-gap" ||
           breakup_options.initial_topology == "finite-film" ||
           breakup_options.initial_topology == "compact-crown" ||
           breakup_options.initial_topology == "cmc-crown";
}

/** Ring-volume measure of the edge-gap film used as the mass reference for
 *  the reverse compact-crown validation. The reference retains the requested
 *  small disturbance, while the compact crown itself is not disturbed. */
Real ReferenceEdgeGapAxisymmetricVolumeMeasure()
{
    constexpr int integration_samples = 4096;
    Real half_gap = 0.5 * SeedGapWidth();
    Real lower = DomainLowerBound() + half_gap;
    Real upper = DomainUpperBound() - half_gap;
    Real dx = (upper - lower) / static_cast<Real>(integration_samples);
    Real integral = 0.0;
    for (int i = 0; i < integration_samples; ++i)
    {
        Real x = lower + (static_cast<Real>(i) + 0.5) * dx;
        Real h = EdgeGapThickness(x, lower, upper);
        integral += ((fibre_radius + h) * (fibre_radius + h) -
                     fibre_radius * fibre_radius) *
                    dx;
    }
    return integral;
}

Real CompactCrownThickness(Real x, Real crown_radius)
{
    Real squared = crown_radius * crown_radius - x * x;
    return squared > 0.0 ? std::sqrt(squared) : 0.0;
}

Real CompactCrownAxisymmetricVolumeMeasure(Real crown_radius)
{
    constexpr int integration_samples = 4096;
    Real lower = -crown_radius;
    Real upper = crown_radius;
    Real dx = (upper - lower) / static_cast<Real>(integration_samples);
    Real integral = 0.0;
    for (int i = 0; i < integration_samples; ++i)
    {
        Real x = lower + (static_cast<Real>(i) + 0.5) * dx;
        Real h = CompactCrownThickness(x, crown_radius);
        integral += ((fibre_radius + h) * (fibre_radius + h) -
                     fibre_radius * fibre_radius) *
                    dx;
    }
    return integral;
}

Real CompactCrownRadius()
{
    static bool initialized = false;
    static Real cached_radius = 0.0;
    if (initialized)
        return cached_radius;

    Real target = ReferenceEdgeGapAxisymmetricVolumeMeasure();
    Real lower = ParticleSpacing();
    Real upper = 0.45 * DomainLength();
    if (CompactCrownAxisymmetricVolumeMeasure(upper) < target)
        throw std::runtime_error(
            "The same-volume compact crown does not fit inside the periodic domain.");
    for (int iteration = 0; iteration < 60; ++iteration)
    {
        Real middle = 0.5 * (lower + upper);
        if (CompactCrownAxisymmetricVolumeMeasure(middle) < target)
            lower = middle;
        else
            upper = middle;
    }
    cached_radius = 0.5 * (lower + upper);
    initialized = true;
    return cached_radius;
}

struct CmcArcState
{
    Real x, r, phi, half_volume;
};

struct CmcHalfArc
{
    CmcArcState endpoint;
    std::vector<Vec2d> points;
};

struct CmcCrownData
{
    Real apex_radius, total_curvature, half_width, outer_half_extent,
        ring_volume_measure;
    std::vector<Vec2d> half_profile;
};

CmcArcState CmcDerivative(const CmcArcState &state, Real curvature)
{
    Real cosine = std::cos(state.phi);
    return CmcArcState{cosine, std::sin(state.phi),
                       cosine / state.r - curvature,
                       (state.r * state.r - fibre_radius * fibre_radius) *
                           cosine};
}

CmcArcState CmcAdvance(const CmcArcState &state,
                       const CmcArcState &increment, Real factor)
{
    return CmcArcState{state.x + factor * increment.x,
                       state.r + factor * increment.r,
                       state.phi + factor * increment.phi,
                       state.half_volume + factor * increment.half_volume};
}

CmcHalfArc IntegrateCmcHalf(Real apex_radius, Real curvature,
                            Real arc_step, bool store_profile)
{
    CmcArcState state{0.0, apex_radius, 0.0, 0.0};
    CmcHalfArc result;
    if (store_profile)
        result.points.push_back(Vec2d(state.x, state.r));
    // Moving from the crown apex to the right contact line, the meridional
    // tangent turns clockwise through the Young angle.  The old implementation
    // hard-coded -pi/2, so every nominal wetting case actually started from a
    // 90-degree crown.  Keeping the profile parametric also permits the
    // overhang that is physical for theta > 90 degrees.
    const Real target_phi = -YoungAngleDegrees() * Pi / 180.0;
    const int maximum_steps = static_cast<int>(100.0 * fibre_radius / arc_step);
    for (int step = 0; step < maximum_steps; ++step)
    {
        CmcArcState previous = state;
        CmcArcState k1 = CmcDerivative(state, curvature);
        CmcArcState k2 = CmcDerivative(
            CmcAdvance(state, k1, 0.5 * arc_step), curvature);
        CmcArcState k3 = CmcDerivative(
            CmcAdvance(state, k2, 0.5 * arc_step), curvature);
        CmcArcState k4 = CmcDerivative(
            CmcAdvance(state, k3, arc_step), curvature);
        state.x += arc_step *
                   (k1.x + 2.0 * k2.x + 2.0 * k3.x + k4.x) / 6.0;
        state.r += arc_step *
                   (k1.r + 2.0 * k2.r + 2.0 * k3.r + k4.r) / 6.0;
        state.phi += arc_step *
                     (k1.phi + 2.0 * k2.phi + 2.0 * k3.phi + k4.phi) / 6.0;
        state.half_volume +=
            arc_step * (k1.half_volume + 2.0 * k2.half_volume +
                        2.0 * k3.half_volume + k4.half_volume) /
            6.0;
        if (!std::isfinite(state.x) || !std::isfinite(state.r) ||
            !std::isfinite(state.phi) || state.r <= 0.0)
            throw std::runtime_error("Invalid CMC integration state.");
        if (state.phi <= target_phi)
        {
            Real fraction = (target_phi - previous.phi) /
                            (state.phi - previous.phi);
            state.x = previous.x + fraction * (state.x - previous.x);
            state.r = previous.r + fraction * (state.r - previous.r);
            state.phi = target_phi;
            state.half_volume = previous.half_volume +
                                fraction * (state.half_volume -
                                            previous.half_volume);
            if (store_profile)
                result.points.push_back(Vec2d(state.x, state.r));
            result.endpoint = state;
            return result;
        }
        if (store_profile)
            result.points.push_back(Vec2d(state.x, state.r));
    }
    throw std::runtime_error("CMC contact-angle event was not reached.");
}

std::array<Real, 2> CmcResidual(Real apex_radius, Real curvature,
                                Real target_volume)
{
    CmcHalfArc arc = IntegrateCmcHalf(
        apex_radius, curvature, 0.004 * fibre_radius, false);
    return { (arc.endpoint.r - fibre_radius) / fibre_radius,
             (2.0 * arc.endpoint.half_volume - target_volume) /
                 target_volume };
}

Real CmcResidualNorm(const std::array<Real, 2> &residual)
{
    return std::sqrt(residual[0] * residual[0] +
                     residual[1] * residual[1]);
}

const CmcCrownData &CmcCrownSolution()
{
    static CmcCrownData solution = []()
    {
        Real target = ReferenceEdgeGapAxisymmetricVolumeMeasure();
        // Angle-aware seeds keep the shooting solve in the physical branch.
        // They interpolate the converged 60/90/120-degree solutions for the
        // fixed reference volume; Newton still enforces r_contact=a and exact
        // volume, so these are only numerical starting values.
        Real angle_offset = YoungAngleDegrees() - 90.0;
        Real apex_slope = angle_offset <= 0.0 ? 0.0067293 : 0.0053763;
        Real curvature_slope = angle_offset <= 0.0 ? 0.0024509 : 0.0020584;
        std::array<Real, 2> parameters{
            (3.0666037 + apex_slope * angle_offset) * fibre_radius,
            (0.7297909 + curvature_slope * angle_offset) / fibre_radius};
        const std::array<Real, 2> lower{1.01 * fibre_radius,
                                        0.01 / fibre_radius};
        const std::array<Real, 2> upper{10.0 * fibre_radius,
                                        10.0 / fibre_radius};
        for (int iteration = 0; iteration < 40; ++iteration)
        {
            std::array<Real, 2> residual =
                CmcResidual(parameters[0], parameters[1], target);
            if (CmcResidualNorm(residual) < 2.0e-8)
                break;
            Real jacobian[2][2];
            for (int column = 0; column < 2; ++column)
            {
                Real finite_step = 2.0e-5 *
                                   std::max(std::abs(parameters[column]),
                                            Real(1.0));
                std::array<Real, 2> shifted = parameters;
                shifted[column] += finite_step;
                std::array<Real, 2> shifted_residual =
                    CmcResidual(shifted[0], shifted[1], target);
                jacobian[0][column] =
                    (shifted_residual[0] - residual[0]) / finite_step;
                jacobian[1][column] =
                    (shifted_residual[1] - residual[1]) / finite_step;
            }
            Real determinant = jacobian[0][0] * jacobian[1][1] -
                               jacobian[0][1] * jacobian[1][0];
            if (std::abs(determinant) < TinyReal)
                throw std::runtime_error("Singular CMC Newton system.");
            std::array<Real, 2> delta{
                (-residual[0] * jacobian[1][1] +
                 jacobian[0][1] * residual[1]) /
                    determinant,
                (-jacobian[0][0] * residual[1] +
                 residual[0] * jacobian[1][0]) /
                    determinant};
            bool accepted = false;
            for (int line_search = 0; line_search < 12; ++line_search)
            {
                Real factor = std::pow(0.5, line_search);
                std::array<Real, 2> trial{
                    std::clamp(parameters[0] + factor * delta[0],
                               lower[0], upper[0]),
                    std::clamp(parameters[1] + factor * delta[1],
                               lower[1], upper[1])};
                if (CmcResidualNorm(CmcResidual(trial[0], trial[1], target)) <
                    CmcResidualNorm(residual))
                {
                    parameters = trial;
                    accepted = true;
                    break;
                }
            }
            if (!accepted)
                throw std::runtime_error("CMC Newton iteration stalled.");
        }
        std::array<Real, 2> final_residual =
            CmcResidual(parameters[0], parameters[1], target);
        if (CmcResidualNorm(final_residual) > 1.0e-6)
            throw std::runtime_error("CMC crown solve did not converge.");
        CmcHalfArc arc = IntegrateCmcHalf(
            parameters[0], parameters[1], 0.0015 * fibre_radius, true);
        Real outer_half_extent = arc.endpoint.x;
        for (const Vec2d &point : arc.points)
            outer_half_extent = std::max(outer_half_extent, point[0]);
        if (outer_half_extent > 0.49 * DomainLength())
            throw std::runtime_error(
                "The Young-angle CMC crown does not fit inside the domain.");
        return CmcCrownData{parameters[0], parameters[1], arc.endpoint.x,
                            outer_half_extent,
                            2.0 * arc.endpoint.half_volume,
                            std::move(arc.points)};
    }();
    return solution;
}

Real CmcCrownSurfaceRadius(Real x)
{
    const CmcCrownData &solution = CmcCrownSolution();
    Real query = std::abs(x);
    if (query >= solution.half_width)
        return fibre_radius;
    auto upper = std::lower_bound(
        solution.half_profile.begin(), solution.half_profile.end(), query,
        [](const Vec2d &point, Real value)
        { return point[0] < value; });
    if (upper == solution.half_profile.begin())
        return upper->operator[](1);
    const Vec2d &right = *upper;
    const Vec2d &left = *(upper - 1);
    Real fraction = (query - left[0]) /
                    (right[0] - left[0] + TinyReal);
    return left[1] + fraction * (right[1] - left[1]);
}

Real SeededAxisymmetricVolumeMeasure(Real bulk_thickness)
{
    constexpr int integration_samples = 4096;
    Real integral = 0.0;
    Real dx = DomainLength() / static_cast<Real>(integration_samples);
    for (int i = 0; i < integration_samples; ++i)
    {
        Real x = DomainLowerBound() +
                 (static_cast<Real>(i) + 0.5) * dx;
        Real h = SeededGapThickness(x, bulk_thickness);
        integral += ((fibre_radius + h) * (fibre_radius + h) -
                     fibre_radius * fibre_radius) *
                    dx;
    }
    return integral;
}

Real SeededBulkThickness()
{
    if (breakup_options.initial_topology != "seeded-gap")
        return FilmThickness();

    static bool initialized = false;
    static Real cached_thickness = 0.0;
    if (initialized)
        return cached_thickness;

    const Real target = DomainLength() *
                        (MeanOuterRadius() * MeanOuterRadius() -
                         fibre_radius * fibre_radius);
    Real lower = FilmThickness();
    Real upper = 2.0 * FilmThickness();
    while (SeededAxisymmetricVolumeMeasure(upper) < target)
        upper *= 1.5;
    for (int iteration = 0; iteration < 60; ++iteration)
    {
        Real middle = 0.5 * (lower + upper);
        if (SeededAxisymmetricVolumeMeasure(middle) < target)
            lower = middle;
        else
            upper = middle;
    }
    cached_thickness = 0.5 * (lower + upper);
    initialized = true;
    return cached_thickness;
}

/** Axial limits and free-surface radius of a closed, uniformly coated fibre.
 *
 * The solid has hemispherical ends.  Adding the film thickness to each end
 * radius is the exact normal offset there.  Along the gently tapered side,
 * h*sqrt(1+r'(x)^2) gives the vertical separation corresponding to a normal
 * coating thickness h.  The resulting outer surface closes on r=0 beyond
 * both solid tips and contains no initial three-phase contact line.
 */
Real FullCoatingLeftCapRadius()
{
#ifdef SPHINXSYS_SYMMETRIC_TAPERED_FIBER_DEFAULTS
    return fibre_radius * run_options.tip_radius_ratio;
#else
    return fibre_radius;
#endif
}
Real FullCoatingRightCapRadius()
{
    return fibre_radius * run_options.tip_radius_ratio;
}
Real FullCoatingLeftCapCentre()
{
    return DomainLowerBound() + FullCoatingLeftCapRadius();
}
Real FullCoatingRightCapCentre()
{
    return DomainUpperBound() - FullCoatingRightCapRadius();
}
Real FullCoatingOuterLowerBound()
{
    return FullCoatingLeftCapCentre() -
           (FullCoatingLeftCapRadius() + FilmThickness());
}
Real FullCoatingOuterUpperBound()
{
    return FullCoatingRightCapCentre() +
           (FullCoatingRightCapRadius() + FilmThickness());
}
Real FullCoatingOuterRadius(Real x)
{
    if (x <= FullCoatingOuterLowerBound() ||
        x >= FullCoatingOuterUpperBound())
        return 0.0;
    if (x < FullCoatingLeftCapCentre())
    {
        const Real radius = FullCoatingLeftCapRadius() + FilmThickness();
        const Real axial = x - FullCoatingLeftCapCentre();
        return std::sqrt(std::max(Real(0.0),
                                  radius * radius - axial * axial));
    }
    if (x > FullCoatingRightCapCentre())
    {
        const Real radius = FullCoatingRightCapRadius() + FilmThickness();
        const Real axial = x - FullCoatingRightCapCentre();
        return std::sqrt(std::max(Real(0.0),
                                  radius * radius - axial * axial));
    }
    const Real slope = FiberRadiusFirstDerivative(x);
    return FiberRadius(x) +
           FilmThickness() * std::sqrt(1.0 + slope * slope);
}

Real FullCoatingInitialCurvature(Real x, bool include_perturbation = true)
{
    if (x < FullCoatingLeftCapCentre())
        return 2.0 / (FullCoatingLeftCapRadius() + FilmThickness());
    if (x > FullCoatingRightCapCentre())
        return 2.0 / (FullCoatingRightCapRadius() + FilmThickness());

    const Real step = std::max(0.5 * ParticleSpacing(), Real(1.0e-4));
    auto perturbed_outer_radius = [include_perturbation](Real position_x)
    {
        Real radius = FullCoatingOuterRadius(position_x);
        if (include_perturbation &&
            position_x >= FullCoatingLeftCapCentre() &&
            position_x <= FullCoatingRightCapCentre())
            radius += InitialAmplitude() * SurfacePattern(position_x);
        return radius;
    };
    const Real rm = perturbed_outer_radius(x - step);
    const Real r0 = perturbed_outer_radius(x);
    const Real rp = perturbed_outer_radius(x + step);
    const Real first = (rp - rm) / (2.0 * step);
    const Real second = (rp - 2.0 * r0 + rm) / (step * step);
    const Real metric = std::sqrt(1.0 + first * first);
    return -second / (metric * metric * metric) +
           1.0 / (std::max(r0, ParticleSpacing()) * metric);
}

class BreakupLiquidFilmShape : public MultiPolygonShape
{
  public:
    explicit BreakupLiquidFilmShape(const std::string &shape_name)
        : MultiPolygonShape(shape_name)
    {
        if (breakup_options.initial_topology == "full-coating")
        {
            AddFullCoating();
        }
        else if (breakup_options.initial_topology == "finite-film")
        {
            AddFiniteFilm();
        }
        else if (breakup_options.initial_topology == "coated")
        {
            AddWetInterval(DomainLowerBound(), DomainUpperBound(), false);
        }
        else if (breakup_options.initial_topology == "compact-crown")
        {
            Real crown_radius = CompactCrownRadius();
            AddWetInterval(-crown_radius, crown_radius, false, false, true);
        }
        else if (breakup_options.initial_topology == "cmc-crown")
        {
            AddCmcCrown();
        }
        else
        {
            Real half_gap = 0.5 * SeedGapWidth();
            if (breakup_options.initial_topology == "seeded-gap")
            {
                AddWetInterval(DomainLowerBound(), -half_gap, true, false);
                AddWetInterval(half_gap, DomainUpperBound(), true, false);
            }
            else
            {
                // The two dry half-gaps touch through the periodic boundary,
                // leaving one wet interval and one small peak at x=0.
                AddWetInterval(DomainLowerBound() + half_gap,
                               DomainUpperBound() - half_gap, false, true);
            }
        }
    }

  private:
    void AddFullCoating()
    {
        const Real outer_lower = FullCoatingOuterLowerBound();
        const Real outer_upper = FullCoatingOuterUpperBound();
        const int outer_samples = std::max(
            240, static_cast<int>(std::ceil((outer_upper - outer_lower) /
                                            (0.4 * ParticleSpacing()))));
        const int solid_samples = std::max(
            220, static_cast<int>(std::ceil(DomainLength() /
                                            (0.4 * ParticleSpacing()))));
        std::vector<Vecd> shell_polygon;
        shell_polygon.reserve(static_cast<size_t>(solid_samples +
                                                   outer_samples + 6));

        // Trace the solid surface from the coarse tip to the fine tip.
        shell_polygon.push_back(Vecd(DomainLowerBound(), 0.0));
        for (int i = 1; i < solid_samples; ++i)
        {
            const Real x = DomainLowerBound() + DomainLength() *
                                                  static_cast<Real>(i) /
                                                  static_cast<Real>(solid_samples);
            shell_polygon.push_back(Vecd(x, FiberRadius(x)));
        }
        shell_polygon.push_back(Vecd(DomainUpperBound(), 0.0));

        // Continue along r=0 to the outer fine tip, then return over the
        // outer free surface.  The final axis segment closes at the solid
        // coarse tip.  This single non-self-intersecting ring is more robust
        // than subtracting a solid polygon which touches the outer polygon on
        // the symmetry axis.
        shell_polygon.push_back(Vecd(outer_upper, 0.0));
        for (int i = outer_samples - 1; i >= 1; --i)
        {
            const Real x = outer_lower + (outer_upper - outer_lower) *
                                             static_cast<Real>(i) /
                                             static_cast<Real>(outer_samples);
            shell_polygon.push_back(Vecd(x, FullCoatingOuterRadius(x)));
        }
        shell_polygon.push_back(Vecd(outer_lower, 0.0));
        shell_polygon.push_back(Vecd(DomainLowerBound(), 0.0));
        multi_polygon_.addPolygon(shell_polygon, GeometricOps::add);
    }

    void AddFiniteFilm()
    {
        const Real lower = ConeLowerBound();
        const Real upper = ConeUpperBound();
        const int samples = std::max(
            80, static_cast<int>(std::ceil((upper - lower) /
                                           (0.5 * ParticleSpacing()))));
        std::vector<Vecd> polygon;
        polygon.reserve(2 * static_cast<size_t>(samples + 1) + 1);
        for (int i = 0; i <= samples; ++i)
        {
            Real x = lower + (upper - lower) * static_cast<Real>(i) /
                                 static_cast<Real>(samples);
            polygon.push_back(Vecd(x, FiberRadius(x)));
        }
        // Generate the lattice from the unperturbed finite film.  The small
        // disturbance is imposed continuously in BreakupInitialLiquidState.
        // Film thickness vanishes at both contact points; omit duplicate
        // vertices and let polygon closure form the two terminal shoulders.
        for (int i = samples - 1; i >= 1; --i)
        {
            Real x = lower + (upper - lower) * static_cast<Real>(i) /
                                 static_cast<Real>(samples);
            polygon.push_back(
                Vecd(x, FiberRadius(x) + FiniteFilmBaseThickness(x)));
        }
        polygon.push_back(Vecd(lower, FiberRadius(lower)));
        multi_polygon_.addPolygon(polygon, GeometricOps::add);
    }

    void AddCmcCrown()
    {
        const CmcCrownData &solution = CmcCrownSolution();
        std::vector<Vecd> polygon;
        polygon.reserve(2 * solution.half_profile.size() + 3);

        // Solid boundary from the left to the right contact point.
        polygon.push_back(Vecd(-solution.half_width, fibre_radius));
        polygon.push_back(Vecd(solution.half_width, fibre_radius));

        // Right free surface: contact point -> apex.
        // The first reverse point is the same contact point already inserted
        // on the solid boundary; skip it to avoid a zero-length polygon edge.
        for (auto point = solution.half_profile.rbegin() + 1;
             point != solution.half_profile.rend(); ++point)
            polygon.push_back(Vecd((*point)[0], (*point)[1]));

        // Left free surface: apex -> contact point.  Skip the duplicated apex.
        for (size_t i = 1; i + 1 < solution.half_profile.size(); ++i)
            polygon.push_back(Vecd(-solution.half_profile[i][0],
                                   solution.half_profile[i][1]));
        polygon.push_back(Vecd(-solution.half_width, fibre_radius));
        multi_polygon_.addPolygon(polygon, GeometricOps::add);
    }

    void AddWetInterval(Real lower, Real upper, bool seeded_gap,
                        bool edge_gap = false, bool compact_crown = false,
                        bool cmc_crown = false)
    {
        std::vector<Vecd> polygon;
        const int samples = std::max(
            40, static_cast<int>(std::ceil((upper - lower) /
                                           (0.5 * ParticleSpacing()))));
        // Trace the actual solid boundary.  Two end points were sufficient for
        // a cylinder, but would cut straight through a tapered fibre.
        polygon.reserve(2 * static_cast<size_t>(samples + 1) + 1);
        for (int i = 0; i <= samples; ++i)
        {
            Real x = lower + (upper - lower) *
                                 static_cast<Real>(i) /
                                 static_cast<Real>(samples);
            polygon.push_back(Vecd(x, FiberRadius(x)));
        }
        Real bulk_thickness = SeededBulkThickness();
        for (int i = samples; i >= 0; --i)
        {
            Real x = lower + (upper - lower) *
                                 static_cast<Real>(i) /
                                 static_cast<Real>(samples);
            Real wall_radius = FiberRadius(x);
            Real radius = cmc_crown
                              ? CmcCrownSurfaceRadius(x)
                              : (compact_crown
                                     ? wall_radius + CompactCrownThickness(
                                                              x, CompactCrownRadius())
                                     : (edge_gap
                                            ? wall_radius +
                                                  EdgeGapThickness(x, lower, upper)
                                            : (seeded_gap
                                                   ? wall_radius + SeededGapThickness(
                                                                        x, bulk_thickness)
                                                   : BreakupSurfaceRadius(x))));
            polygon.push_back(Vecd(x, radius));
        }
        polygon.push_back(Vecd(lower, FiberRadius(lower)));
        multi_polygon_.addPolygon(polygon, GeometricOps::add);
    }
};

class BreakupInitialLiquidState : public LocalDynamics
{
  public:
    explicit BreakupInitialLiquidState(SPHBody &sph_body)
        : LocalDynamics(sph_body),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          rho_(particles_->getVariableDataByName<Real>("Density")),
          pressure_(particles_->registerStateVariableData<Real>("Pressure")),
          velocity_(particles_->registerStateVariableData<Vecd>("Velocity")) {}

    void update(size_t i, Real dt)
    {
        Real x = pos_[i][0];
        if (breakup_options.initial_topology == "full-coating")
        {
            // No position remapping is required: AddFullCoating already
            // generated the constant-normal-thickness reference geometry.
            // Initialise the weakly-compressible liquid with the local
            // Young-Laplace pressure of that closed outer surface.
            Real p0 = 0.0;
            if (breakup_options.full_coating_initial_pressure != "zero")
            {
                const bool include_perturbation =
                    breakup_options.full_coating_initial_pressure == "local";
                p0 = breakup_options.gamma_lg *
                     FullCoatingInitialCurvature(x, include_perturbation);
            }
            rho_[i] = reference_density *
                      (1.0 + p0 /
                                 (reference_density * SoundSpeed() *
                                  SoundSpeed()));
            pressure_[i] = p0;
            velocity_[i] = Vecd::Zero();
            return;
        }
        if (breakup_options.initial_topology == "finite-film")
        {
            Real wall_radius = FiberRadius(x);
            Real base_thickness = FiniteFilmBaseThickness(x);
            Real target_thickness = FiniteFilmThickness(x);
            if (base_thickness > TinyReal)
                pos_[i][1] = wall_radius +
                             (pos_[i][1] - wall_radius) *
                                 target_thickness / base_thickness;

            // Initialise the weakly-compressible pressure with the local
            // Young-Laplace pressure of the prescribed finite profile.  A
            // particle-spacing finite difference is adequate here because
            // it is used only for the initial pressure, not time evolution.
            Real difference_step = ParticleSpacing();
            Real radius_minus = FiniteFilmSurfaceRadius(x - difference_step);
            Real radius_here = FiniteFilmSurfaceRadius(x);
            Real radius_plus = FiniteFilmSurfaceRadius(x + difference_step);
            Real slope = (radius_plus - radius_minus) /
                         (2.0 * difference_step);
            Real second = (radius_plus - 2.0 * radius_here + radius_minus) /
                          (difference_step * difference_step);
            Real metric = std::sqrt(1.0 + slope * slope);
            Real curvature = -second / (metric * metric * metric) +
                             1.0 / (std::max(radius_here, ParticleSpacing()) *
                                    metric);
            Real p0 = breakup_options.gamma_lg * curvature;
            rho_[i] = reference_density *
                      (1.0 + p0 /
                                 (reference_density * SoundSpeed() *
                                  SoundSpeed()));
            pressure_[i] = p0;
            velocity_[i] = Vecd::Zero();
            return;
        }
        if (breakup_options.initial_topology == "cmc-crown")
        {
            Real p0 = breakup_options.gamma_lg *
                      CmcCrownSolution().total_curvature;
            rho_[i] = reference_density *
                      (1.0 + p0 /
                                 (reference_density * SoundSpeed() *
                                  SoundSpeed()));
            pressure_[i] = p0;
            velocity_[i] = Vecd::Zero();
            return;
        }
        if (breakup_options.initial_topology == "compact-crown")
        {
            Real crown_radius = CompactCrownRadius();
            Real h = CompactCrownThickness(x, crown_radius);
            Real surface_radius = fibre_radius + h;
            Real kappa_axial = 1.0 / crown_radius;
            Real kappa_hoop = h /
                               (crown_radius * std::max(surface_radius,
                                                       fibre_radius));
            Real p0 = breakup_options.gamma_lg *
                      (kappa_axial + kappa_hoop);
            rho_[i] = reference_density *
                      (1.0 + p0 /
                                 (reference_density * SoundSpeed() *
                                  SoundSpeed()));
            pressure_[i] = p0;
            velocity_[i] = Vecd::Zero();
            return;
        }
        if (breakup_options.initial_topology == "seeded-gap")
        {
            Real bulk_thickness = SeededBulkThickness();
            Real wet_distance = std::abs(x) - 0.5 * SeedGapWidth();
            Real surface_radius = fibre_radius +
                                  SeededGapThickness(x, bulk_thickness);
            Real kappa_axial =
                (wet_distance > 0.0 && wet_distance < bulk_thickness)
                    ? 1.0 / bulk_thickness
                    : 0.0;
            Real kappa_hoop = 1.0 / std::max(surface_radius, fibre_radius);
            Real p0 = breakup_options.seeded_pressure == "uniform"
                          ? breakup_options.gamma_lg /
                                (fibre_radius + bulk_thickness)
                          : breakup_options.gamma_lg *
                                (kappa_axial + kappa_hoop);
            rho_[i] = reference_density *
                      (1.0 + p0 /
                                 (reference_density * SoundSpeed() *
                                  SoundSpeed()));
            pressure_[i] = p0;
            velocity_[i] = Vecd::Zero();
            return;
        }
        // A disturbance smaller than one lattice spacing does not change which
        // lattice sites are filled.  Map the already generated film smoothly
        // in the radial direction so a 1% disturbance is represented by the
        // particle coordinates while the no-slip fibre surface stays fixed.
        Real wall_radius = FiberRadius(x);
        Real radial_scale = 1.0 + InitialAmplitude() * BreakupPattern(x) /
                                       (FilmThickness() + TinyReal);
        pos_[i][1] = wall_radius +
                     (pos_[i][1] - wall_radius) * radial_scale;
        Real slope = FiberRadiusFirstDerivative(x) +
                     InitialAmplitude() * BreakupPatternFirstDerivative(x);
        Real second = FiberRadiusSecondDerivative(x) +
                      InitialAmplitude() * BreakupPatternSecondDerivative(x);
        Real metric = std::sqrt(1.0 + slope * slope);
        Real kappa_axial = -second / (metric * metric * metric);
        Real kappa_hoop = 1.0 / (BreakupSurfaceRadius(x) * metric);
        Real p0 = breakup_options.gamma_lg * (kappa_axial + kappa_hoop);
        if (breakup_options.van_der_waals_model == "xu" &&
            breakup_options.van_der_waals_initial_pressure)
        {
            Real gap = std::max(
                BreakupSurfaceRadius(x) - wall_radius,
                breakup_options.van_der_waals_min_gap_in_dx *
                    ParticleSpacing());
            Real dimensionless_gap = gap / fibre_radius;
            Real dimensionless_pressure =
                breakup_options.van_der_waals_number /
                (dimensionless_gap * dimensionless_gap *
                     dimensionless_gap + TinyReal);
            if (breakup_options.van_der_waals_pressure_gauge == "mean-film")
            {
                Real reference_gap = FilmThickness() / fibre_radius;
                dimensionless_pressure -=
                    breakup_options.van_der_waals_number /
                    (reference_gap * reference_gap * reference_gap + TinyReal);
            }
            p0 += breakup_options.gamma_lg / fibre_radius *
                  dimensionless_pressure;
        }
        rho_[i] = reference_density *
                  (1.0 + p0 / (reference_density * SoundSpeed() * SoundSpeed()));
        pressure_[i] = p0;
        velocity_[i] = Vecd::Zero();
    }

  private:
    Vecd *pos_, *velocity_;
    Real *rho_, *pressure_;
};

Real DeterministicUnitRandom(size_t particle_index, uint64_t component)
{
    uint64_t state = static_cast<uint64_t>(breakup_options.random_seed) +
                     0x9e3779b97f4a7c15ULL *
                         (static_cast<uint64_t>(particle_index) + 1ULL);
    state ^= component + 0xbf58476d1ce4e5b9ULL;
    state ^= state >> 30;
    state *= 0xbf58476d1ce4e5b9ULL;
    state ^= state >> 27;
    state *= 0x94d049bb133111ebULL;
    state ^= state >> 31;
    return static_cast<Real>(state >> 11) *
           (Real(1.0) / Real(9007199254740992.0));
}

Real DeterministicThermalUnitRandom(size_t particle_index,
                                    uint64_t noise_step,
                                    uint64_t component)
{
    uint64_t state = static_cast<uint64_t>(breakup_options.thermal_noise_seed);
    state += 0x9e3779b97f4a7c15ULL *
             (static_cast<uint64_t>(particle_index) + 1ULL);
    state ^= 0xbf58476d1ce4e5b9ULL * (noise_step + 1ULL);
    state ^= component + 0x94d049bb133111ebULL;
    state ^= state >> 30;
    state *= 0xbf58476d1ce4e5b9ULL;
    state ^= state >> 27;
    state *= 0x94d049bb133111ebULL;
    state ^= state >> 31;
    return static_cast<Real>(state >> 11) *
           (Real(1.0) / Real(9007199254740992.0));
}

Real WrapPeriodicX(Real x)
{
    Real length = DomainLength();
    Real shifted = x - DomainLowerBound();
    shifted -= std::floor(shifted / length) * length;
    return DomainLowerBound() + shifted;
}

class RandomInitialParticlePerturbation : public LocalDynamics
{
  public:
    explicit RandomInitialParticlePerturbation(SPHBody &sph_body)
        : LocalDynamics(sph_body),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          velocity_(particles_->getVariableDataByName<Vecd>("Velocity")),
          displacement_(particles_->registerStateVariableData<Vecd>(
              "InitialRandomDisplacement")),
          velocity_kick_(particles_->registerStateVariableData<Vecd>(
              "InitialRandomVelocity")) {}

    void update(size_t i, Real dt)
    {
        displacement_[i] = Vecd::Zero();
        velocity_kick_[i] = Vecd::Zero();
        if (breakup_options.random_perturbation == "off")
            return;

        bool perturb_position =
            breakup_options.random_perturbation == "position" ||
            breakup_options.random_perturbation == "position-velocity";
        bool perturb_velocity =
            breakup_options.random_perturbation == "velocity" ||
            breakup_options.random_perturbation == "position-velocity";

        if (perturb_position && breakup_options.random_position_in_dx > 0.0)
        {
            Real amplitude =
                breakup_options.random_position_in_dx * ParticleSpacing();
            Vecd displacement(
                amplitude *
                    (2.0 * DeterministicUnitRandom(i, 0x1234ULL) - 1.0),
                amplitude *
                    (2.0 * DeterministicUnitRandom(i, 0x5678ULL) - 1.0));
            pos_[i][0] = WrapPeriodicX(pos_[i][0] + displacement[0]);
            pos_[i][1] += displacement[1];
            const Real wall_floor = FiberRadius(pos_[i][0]) +
                                    0.10 * ParticleSpacing() *
                                        std::sqrt(1.0 + std::pow(
                                            FiberRadiusFirstDerivative(pos_[i][0]), 2));
            pos_[i][1] = std::max(wall_floor, pos_[i][1]);
            displacement_[i] = displacement;
        }

        if (perturb_velocity && breakup_options.random_velocity_scale > 0.0)
        {
            Real amplitude =
                breakup_options.random_velocity_scale * CapillaryVelocity();
            Vecd velocity_kick(
                amplitude *
                    (2.0 * DeterministicUnitRandom(i, 0x9abcULL) - 1.0),
                amplitude *
                    (2.0 * DeterministicUnitRandom(i, 0xdef0ULL) - 1.0));
            velocity_[i] += velocity_kick;
            velocity_kick_[i] = velocity_kick;
        }
    }

  private:
    Vecd *pos_, *velocity_, *displacement_, *velocity_kick_;
};

/**
 * Time-continuous particle-wise random forcing for sensitivity tests.
 *
 * This is not a full fluctuating-hydrodynamics stochastic-stress discretisation:
 * it is a controlled numerical probe for the advisor's suggestion that every
 * particle receives a small persistent disturbance during the evolution.
 */
class ContinuousParticleThermalNoise : public LocalDynamics
{
  public:
    explicit ContinuousParticleThermalNoise(SPHBody &sph_body)
        : LocalDynamics(sph_body),
          velocity_(particles_->getVariableDataByName<Vecd>("Velocity")),
          indicator_(particles_->getVariableDataByName<int>("Indicator")),
          normal_(particles_->getVariableDataByName<Vecd>("TopologyNormal")),
          kick_(particles_->registerStateVariableData<Vecd>(
              "ContinuousThermalNoiseKick")),
          noise_velocity_(particles_->registerStateVariableData<Vecd>(
              "ThermalNoiseVelocity")),
          noise_step_(0), enabled_(true)
    {
        particles_->addVariableToWrite<Vecd>("ContinuousThermalNoiseKick");
        particles_->addEvolvingVariable<Vecd>("ThermalNoiseVelocity");
        particles_->addVariableToWrite<Vecd>("ThermalNoiseVelocity");
        for (size_t i = 0; i < particles_->TotalRealParticles(); ++i)
            noise_velocity_[i] = Vecd::Zero();
    }

    void exec(Real dt = 0.0)
    {
        const size_t total = particles_->TotalRealParticles();
        for (size_t i = 0; i < total; ++i)
            kick_[i] = Vecd::Zero();
        if (!enabled_ ||
            breakup_options.thermal_noise == "off" ||
            breakup_options.thermal_noise_strength <= 0.0 ||
            dt <= TinyReal || total == 0)
        {
            ++noise_step_;
            return;
        }

        const Real capillary_time =
            DynamicViscosity() * fibre_radius /
            (breakup_options.gamma_lg + TinyReal);
        Real amplitude = breakup_options.thermal_noise_strength *
                         CapillaryVelocity() *
                         std::sqrt(dt / (capillary_time + TinyReal));
        const Real maximum_kick =
            breakup_options.thermal_noise_max_kick_scale *
            CapillaryVelocity();
        amplitude = std::min(amplitude, maximum_kick);

        const bool ou_mode =
            breakup_options.thermal_noise_mode == "ou-tangent";
        const Real correlation = ou_mode
                                    ? std::exp(-dt /
                                               (0.5 * capillary_time + TinyReal))
                                    : 0.0;
        const Real ou_scale = ou_mode
                                  ? breakup_options.thermal_noise_strength *
                                        CapillaryVelocity() *
                                        std::sqrt(std::max(
                                            Real(0.0),
                                            1.0 - correlation * correlation))
                                  : 0.0;

        Vecd mean_kick = Vecd::Zero();
        for (size_t i = 0; i < total; ++i)
        {
            Vecd raw_unit(
                    (2.0 * DeterministicThermalUnitRandom(
                               i, noise_step_, 0x13579ULL) -
                     1.0),
                    (2.0 * DeterministicThermalUnitRandom(
                               i, noise_step_, 0x24680ULL) -
                     1.0));
            Vecd projected_unit = raw_unit;
            if (breakup_options.thermal_noise_mode == "surface-tangent" ||
                ou_mode)
            {
                if (indicator_[i] == 1 && normal_[i].squaredNorm() > TinyReal)
                {
                    Vecd normal = normal_[i] /
                                   (normal_[i].norm() + TinyReal);
                    Vecd tangent(-normal[1], normal[0]);
                    projected_unit = tangent * raw_unit.dot(tangent);
                }
                else
                {
                    // Interior particles receive only an axial component;
                    // this preserves the "every particle" probe without
                    // injecting a direct film-thickness shock.
                    projected_unit = Vecd(raw_unit[0], 0.0);
                }
            }

            if (ou_mode)
            {
                Vecd innovation = ou_scale * projected_unit;
                Real innovation_norm = innovation.norm();
                if (innovation_norm > maximum_kick)
                    innovation *= maximum_kick /
                                  (innovation_norm + TinyReal);
                Vecd old_noise = noise_velocity_[i];
                Vecd new_noise = correlation * old_noise + innovation;
                kick_[i] = new_noise - old_noise;
                noise_velocity_[i] = new_noise;
            }
            else
            {
                kick_[i] = amplitude * projected_unit;
            }
            mean_kick += kick_[i];
        }
        mean_kick /= static_cast<Real>(total);
        for (size_t i = 0; i < total; ++i)
        {
            kick_[i] -= mean_kick;
            velocity_[i] += kick_[i];
        }
        ++noise_step_;
    }

    void setEnabled(bool enabled)
    {
        if (!enabled && enabled_ &&
            breakup_options.thermal_noise_mode == "ou-tangent")
        {
            const size_t total = particles_->TotalRealParticles();
            for (size_t i = 0; i < total; ++i)
            {
                velocity_[i] -= noise_velocity_[i];
                noise_velocity_[i] = Vecd::Zero();
            }
        }
        enabled_ = enabled;
    }

  private:
    Vecd *velocity_;
    int *indicator_;
    Vecd *normal_, *kick_, *noise_velocity_;
    uint64_t noise_step_;
    bool enabled_;
};

/**
 * Viscous liquid-fibre coupling with a Navier tangential-slip length b.
 * The pressure Riemann solver still enforces normal impermeability.  b=0
 * exactly recovers the standard no-slip wall viscous interaction.
 */
class FiberNavierSlipWallViscousForce
    : public fluid_dynamics::ViscousForce<Contact<Wall>,
                                             fluid_dynamics::FixedViscosity,
                                             NoKernelCorrection>
{
    using Base = fluid_dynamics::ViscousForce<Contact<Wall>,
                                                fluid_dynamics::FixedViscosity,
                                                NoKernelCorrection>;

  public:
    FiberNavierSlipWallViscousForce(BaseContactRelation &wall_contact_relation,
                                    Real slip_length)
        : Base(wall_contact_relation),
          wall_viscous_force_(particles_->registerStateVariableData<Vecd>(
              "FiberWallViscousForce")),
          slip_length_(slip_length)
    {
        particles_->addEvolvingVariable<Vecd>("FiberWallViscousForce");
        particles_->addVariableToWrite<Vecd>("FiberWallViscousForce");
    }

    void interaction(size_t index_i, Real dt = 0.0)
    {
        Vecd force = Vecd::Zero();
        for (size_t k = 0; k < contact_configuration_.size(); ++k)
        {
            Vecd *wall_velocity = wall_vel_ave_[k];
            Vecd *wall_normal = wall_n_[k];
            Real *wall_volume = wall_Vol_[k];
            const Neighborhood &neighborhood =
                (*contact_configuration_[k])[index_i];
            for (size_t n = 0; n < neighborhood.current_size_; ++n)
            {
                size_t index_j = neighborhood.j_[n];
                Real distance = neighborhood.r_ij_[n];
                const Vecd &direction = neighborhood.e_ij_[n];
                Vecd normal = wall_normal[index_j] /
                              (wall_normal[index_j].norm() + TinyReal);
                Vecd velocity_jump =
                    vel_[index_i] - wall_velocity[index_j];
                Vecd normal_jump = normal * velocity_jump.dot(normal);
                Vecd tangential_jump = velocity_jump - normal_jump;
                Real tangential_factor =
                    distance / (distance + slip_length_ + TinyReal);
                Vecd effective_jump =
                    normal_jump + tangential_factor * tangential_jump;
                Vecd velocity_derivative =
                    2.0 * effective_jump /
                    (distance + 0.01 * smoothing_length_);
                force +=
                    2.0 * direction.dot(kernel_correction_(index_i) * direction) *
                    mu_(index_i, index_i) * velocity_derivative *
                    neighborhood.dW_ij_[n] * wall_volume[index_j];
            }
        }
        wall_viscous_force_[index_i] = force * Vol_[index_i];
        viscous_force_[index_i] += wall_viscous_force_[index_i];
    }

  private:
    Vecd *wall_viscous_force_;
    Real slip_length_;
};

/**
 * Read-only reconstruction of the pressure force used by the first acoustic
 * half-step.  It duplicates the no-correction inner and rigid-wall pressure
 * stencil solely for VTP diagnostics and never updates velocity or density.
 */
class PressureForceDiagnostic : public LocalDynamics,
                                public DataDelegateInner,
                                public DataDelegateContact
{
  public:
    PressureForceDiagnostic(BaseInnerRelation &inner_relation,
                            BaseContactRelation &wall_contact_relation)
        : LocalDynamics(inner_relation.getSPHBody()),
          DataDelegateInner(inner_relation),
          DataDelegateContact(wall_contact_relation),
          rho_(particles_->getVariableDataByName<Real>("Density")),
          pressure_(particles_->getVariableDataByName<Real>("Pressure")),
          mass_(particles_->getVariableDataByName<Real>("Mass")),
          Vol_(particles_->getVariableDataByName<Real>("VolumetricMeasure")),
          force_prior_(particles_->registerStateVariableData<Vecd>("ForcePrior")),
          diagnostic_force_(particles_->registerStateVariableData<Vecd>(
              "DiagnosticPressureForce"))
    {
        for (size_t k = 0; k < contact_particles_.size(); ++k)
        {
            Solid &solid_material = DynamicCast<Solid>(
                this, contact_bodies_[k]->getMatterMaterial());
            wall_acceleration_.push_back(
                solid_material.AverageAcceleration(contact_particles_[k]));
            wall_volume_.push_back(
                contact_particles_[k]->getVariableDataByName<Real>(
                    "VolumetricMeasure"));
        }
        particles_->addEvolvingVariable<Vecd>("DiagnosticPressureForce");
        particles_->addVariableToWrite<Vecd>("DiagnosticPressureForce");
    }

    void exec(Real dt = 0.0)
    {
        for (size_t i = 0; i < particles_->TotalRealParticles(); ++i)
        {
            Vecd force = Vecd::Zero();
            const Neighborhood &inner_neighborhood = inner_configuration_[i];
            for (size_t n = 0; n < inner_neighborhood.current_size_; ++n)
            {
                size_t j = inner_neighborhood.j_[n];
                Real dW_ijV_j = inner_neighborhood.dW_ij_[n] * Vol_[j];
                force -= (pressure_[i] + pressure_[j]) * dW_ijV_j *
                         inner_neighborhood.e_ij_[n];
            }
            for (size_t k = 0; k < contact_configuration_.size(); ++k)
            {
                Vecd *wall_acceleration = wall_acceleration_[k];
                Real *wall_volume = wall_volume_[k];
                const Neighborhood &wall_neighborhood =
                    (*contact_configuration_[k])[i];
                for (size_t n = 0; n < wall_neighborhood.current_size_; ++n)
                {
                    size_t j = wall_neighborhood.j_[n];
                    const Vecd &e_ij = wall_neighborhood.e_ij_[n];
                    Real dW_ijV_j =
                        wall_neighborhood.dW_ij_[n] * wall_volume[j];
                    Real wall_normal_acceleration =
                        (force_prior_[i] / (mass_[i] + TinyReal) -
                         wall_acceleration[j])
                            .dot(-e_ij);
                    Real wall_pressure = pressure_[i] + rho_[i] *
                        wall_neighborhood.r_ij_[n] *
                        std::max(Real(0.0), wall_normal_acceleration);
                    force -= (pressure_[i] + wall_pressure) * dW_ijV_j * e_ij;
                }
            }
            diagnostic_force_[i] = force * Vol_[i];
        }
    }

  private:
    Real *rho_, *pressure_, *mass_, *Vol_;
    Vecd *force_prior_, *diagnostic_force_;
    StdVec<Vecd *> wall_acceleration_;
    StdVec<Real *> wall_volume_;
};

/** Identify contact bodies that may only complete the kernel support.
 *
 * The half-axisymmetric model mirrors the meridional plane about r=0 and
 * fills the r<0 side with a numerical symmetry body.  Those particles must
 * contribute to the Shepard sum and to the colour gradient, otherwise the
 * liquid support near the axis is incomplete and the reconstructed interface
 * normal degenerates there.  They must not, however, act as a wettable solid:
 * the axis is a free-slip mirror plane, so it carries no contact angle, no
 * Young traction and no adhesion.
 */
bool IsSupportOnlyContactBody(SPHBody *body)
{
    return body->Name() == "AxisSymmetryBoundary";
}

/**
 * One-sided continuum-surface-stress (CSS) model with an axisymmetric stress
 * divergence correction.  The liquid support deficiency supplies the local
 * liquid/free-space interface; no second phase is discretized.
 *
 * The three passes are intentionally explicit:
 *   1. local colour gradient and outward normal;
 *   2. optional legacy contact-angle correction and liquid-gas surface stress;
 *   3. divergence of surface stress, cylindrical 1/r term and the surface-
 *      energy wetting forces near the solid contact line.
 */
class LocalAxisymmetricSurfaceStress : public ForcePrior,
                                       public DataDelegateInner,
                                       public DataDelegateContact
{
  public:
    LocalAxisymmetricSurfaceStress(BaseInnerRelation &inner_relation,
                                   BaseContactRelation &fibre_contact_relation)
        : ForcePrior(inner_relation.getSPHBody(), "TopologyCapillaryForce"),
          DataDelegateInner(inner_relation),
          DataDelegateContact(fibre_contact_relation),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          rho_(particles_->getVariableDataByName<Real>("Density")),
          mass_(particles_->getVariableDataByName<Real>("Mass")),
          Vol_(particles_->getVariableDataByName<Real>("VolumetricMeasure")),
          indicator_(particles_->getVariableDataByName<int>("Indicator")),
          gradient_(particles_->registerStateVariableData<Vecd>("TopologyColorGradient")),
          normal_(particles_->registerStateVariableData<Vecd>("TopologyNormal")),
          delta_(particles_->registerStateVariableData<Real>("TopologySurfaceDelta")),
          stress_(particles_->registerStateVariableData<Matd>("TopologySurfaceStress")),
          correction_matrix_(particles_->registerStateVariableData<Matd>("TopologyGradientCorrection")),
          contact_line_(particles_->registerStateVariableData<int>("ContactLineIndicator")),
          contact_angle_weight_(particles_->registerStateVariableData<Real>("ContactAngleWeight")),
          contact_line_partition_(particles_->registerStateVariableData<Real>("ContactLinePartition")),
          dynamic_contact_cosine_(particles_->registerStateVariableData<Real>("DynamicContactCosine")),
          wetting_force_(particles_->registerStateVariableData<Vecd>("SurfaceEnergyWettingForce")),
          wall_adhesion_force_(particles_->getVariableDataByName<Vecd>("WallAdhesionForce")),
          solid_liquid_force_(particles_->registerStateVariableData<Vecd>("SolidLiquidCurvatureForce")),
          support_neighbors_(particles_->registerStateVariableData<int>("SurfaceSupportNeighbors")),
          surface_confidence_(particles_->registerStateVariableData<Real>("SurfaceStressConfidence")),
          pca_normal_used_(particles_->registerStateVariableData<int>("PcaNormalUsed"))
    {
        for (BaseParticles *fibre_particles : contact_particles_)
            fibre_Vol_.push_back(
                fibre_particles->getVariableDataByName<Real>("VolumetricMeasure"));
        for (SPHBody *contact_body : contact_bodies_)
            contact_body_is_wettable_.push_back(
                !IsSupportOnlyContactBody(contact_body));
        particles_->registerSingleVariable<Real>("SurfaceTensionCoef", breakup_options.gamma_lg);
        particles_->registerSingleVariable<Real>(
            "ExternalDynamicContactCosine",
            std::numeric_limits<Real>::quiet_NaN());
        particles_->registerSingleVariable<Real>(
            "ExternalFreeSurfaceCurvature",
            std::numeric_limits<Real>::quiet_NaN());
        particles_->registerSingleVariable<Real>(
            "ExternalContactCenterX",
            std::numeric_limits<Real>::quiet_NaN());
        particles_->addEvolvingVariable<Vecd>("TopologyColorGradient");
        particles_->addEvolvingVariable<Vecd>("TopologyNormal");
        particles_->addEvolvingVariable<Real>("TopologySurfaceDelta");
        particles_->addEvolvingVariable<Matd>("TopologySurfaceStress");
        particles_->addEvolvingVariable<Matd>("TopologyGradientCorrection");
        particles_->addEvolvingVariable<int>("ContactLineIndicator");
        particles_->addEvolvingVariable<Real>("ContactAngleWeight");
        particles_->addEvolvingVariable<Real>("ContactLinePartition");
        particles_->addEvolvingVariable<Real>("DynamicContactCosine");
        particles_->addEvolvingVariable<Vecd>("SurfaceEnergyWettingForce");
        particles_->addEvolvingVariable<Vecd>("SolidLiquidCurvatureForce");
        particles_->addEvolvingVariable<int>("SurfaceSupportNeighbors");
        particles_->addEvolvingVariable<Real>("SurfaceStressConfidence");
        particles_->addEvolvingVariable<int>("PcaNormalUsed");
        particles_->addVariableToWrite<Vecd>("TopologyColorGradient");
        particles_->addVariableToWrite<Vecd>("TopologyNormal");
        particles_->addVariableToWrite<Real>("TopologySurfaceDelta");
        particles_->addVariableToWrite<Matd>("TopologySurfaceStress");
        particles_->addVariableToWrite<Matd>("TopologyGradientCorrection");
        particles_->addVariableToWrite<int>("ContactLineIndicator");
        particles_->addVariableToWrite<Real>("ContactAngleWeight");
        particles_->addVariableToWrite<Real>("ContactLinePartition");
        particles_->addVariableToWrite<Real>("DynamicContactCosine");
        particles_->addVariableToWrite<Vecd>("SurfaceEnergyWettingForce");
        particles_->addVariableToWrite<Vecd>("SolidLiquidCurvatureForce");
        particles_->addVariableToWrite<int>("SurfaceSupportNeighbors");
        particles_->addVariableToWrite<Real>("SurfaceStressConfidence");
        particles_->addVariableToWrite<int>("PcaNormalUsed");
        particles_->addVariableToWrite<Vecd>("TopologyCapillaryForce");
    }

    void exec(Real dt = 0.0)
    {
        const size_t total = particles_->TotalRealParticles();
        const Real gradient_floor =
            breakup_options.surface_gradient_floor_in_dx / ParticleSpacing();

        Real footprint_left = std::numeric_limits<Real>::max();
        Real footprint_right = -std::numeric_limits<Real>::max();
        const size_t footprint_bins = std::max<size_t>(
            24, static_cast<size_t>(std::round(
                    DomainLength() / (2.0 * ParticleSpacing()))));
        std::vector<int> dominant_footprint_bins(footprint_bins, 0);
        auto footprint_bin_index = [&](Real x)
        {
            Real shifted = x - DomainLowerBound();
            Real periodic_x = shifted -
                              std::floor(shifted / DomainLength()) *
                                  DomainLength();
            return std::min(
                footprint_bins - 1, static_cast<size_t>(std::floor(
                                        periodic_x / DomainLength() *
                                        static_cast<Real>(footprint_bins))));
        };
        bool has_resolved_footprint = HasCentralWetFootprint();
        if (has_resolved_footprint)
        {
            std::fill(dominant_footprint_bins.begin(),
                      dominant_footprint_bins.end(), 1);
            for (size_t i = 0; i < total; ++i)
            {
                Real wall_distance = FiberNormalDistance(pos_[i]);
                if (wall_distance < -0.25 * ParticleSpacing() ||
                    wall_distance > 1.5 * ParticleSpacing())
                    continue;
                footprint_left = std::min(footprint_left, pos_[i][0]);
                footprint_right = std::max(footprint_right, pos_[i][0]);
            }
        }
        else
        {
            // A coated film may create a dry interval at runtime.  Recover
            // the principal wall-attached liquid component instead of
            // treating every isolated precursor/satellite particle as a new
            // contact-line endpoint.  One-bin holes are closed first to avoid
            // lattice noise; a footprint is accepted only when the remaining
            // dry run spans the same kernel-resolved width used by the hybrid
            // topology switch.
            std::vector<int> wall_wet(footprint_bins, 0);
            std::vector<size_t> wall_particles(footprint_bins, 0);
            std::vector<Real> outer_radius(
                footprint_bins, -std::numeric_limits<Real>::infinity());
            for (size_t i = 0; i < total; ++i)
            {
                size_t b = footprint_bin_index(pos_[i][0]);
                outer_radius[b] = std::max(
                    outer_radius[b], pos_[i][1] + 0.5 * ParticleSpacing());
                ++wall_particles[b];
                Real geometric_wall_distance = FiberNormalDistance(pos_[i]);
                if (geometric_wall_distance < -0.25 * ParticleSpacing() ||
                    geometric_wall_distance > 1.5 * ParticleSpacing())
                    continue;
            }
            const Real resolved_film_cutoff =
                breakup_options.rupture_cutoff_in_dx * ParticleSpacing();
            for (size_t b = 0; b < footprint_bins; ++b)
            {
                Real bin_x = DomainLowerBound() +
                             (static_cast<Real>(b) + 0.5) * DomainLength() /
                                 static_cast<Real>(footprint_bins);
                if (std::isfinite(outer_radius[b]) &&
                    FiberNormalDistance(Vecd(bin_x, outer_radius[b])) >=
                        resolved_film_cutoff)
                    wall_wet[b] = 1;
            }
            std::vector<int> closed_wet = wall_wet;
            for (size_t b = 0; b < footprint_bins; ++b)
                if (wall_wet[b] == 0 &&
                    wall_wet[(b + footprint_bins - 1) % footprint_bins] == 1 &&
                    wall_wet[(b + 1) % footprint_bins] == 1)
                    closed_wet[b] = 1;

            auto dry_anchor = std::find(closed_wet.begin(),
                                        closed_wet.end(), 0);
            if (dry_anchor != closed_wet.end())
            {
                size_t anchor = static_cast<size_t>(
                    dry_anchor - closed_wet.begin());
                size_t dry_run = 0, maximum_dry_run = 0;
                std::vector<size_t> current_wet_run, dominant_wet_run;
                size_t current_particles = 0, dominant_particles = 0;
                auto finalize_wet_run = [&]()
                {
                    if (current_particles > dominant_particles ||
                        (current_particles == dominant_particles &&
                         current_wet_run.size() > dominant_wet_run.size()))
                    {
                        dominant_wet_run = current_wet_run;
                        dominant_particles = current_particles;
                    }
                    current_wet_run.clear();
                    current_particles = 0;
                };
                for (size_t offset = 1; offset <= footprint_bins; ++offset)
                {
                    size_t b = (anchor + offset) % footprint_bins;
                    if (closed_wet[b] == 1)
                    {
                        dry_run = 0;
                        current_wet_run.push_back(b);
                        current_particles += wall_particles[b];
                    }
                    else
                    {
                        finalize_wet_run();
                        ++dry_run;
                        maximum_dry_run = std::max(maximum_dry_run, dry_run);
                    }
                }
                finalize_wet_run();
                if (maximum_dry_run >= static_cast<size_t>(
                                               breakup_options.hybrid_min_dry_bins) &&
                    !dominant_wet_run.empty())
                {
                    has_resolved_footprint = true;
                    for (size_t b : dominant_wet_run)
                        dominant_footprint_bins[b] = 1;
                    for (size_t i = 0; i < total; ++i)
                    {
                        Real geometric_wall_distance =
                            FiberNormalDistance(pos_[i]);
                        if (geometric_wall_distance <
                                -0.25 * ParticleSpacing() ||
                            geometric_wall_distance >
                                1.5 * ParticleSpacing() ||
                            dominant_footprint_bins[
                                footprint_bin_index(pos_[i][0])] == 0)
                            continue;
                        footprint_left = std::min(footprint_left, pos_[i][0]);
                        footprint_right = std::max(footprint_right, pos_[i][0]);
                    }
                }
            }
        }
        // Pass 1: the negative kernel-gradient sum points from liquid to the
        // missing-support region and is therefore an outward interface normal.
        for (size_t i = 0; i < total; ++i)
        {
            gradient_[i] = Vecd::Zero();
            normal_[i] = Vecd::Zero();
            delta_[i] = 0.0;
            stress_[i] = Matd::Zero();
            correction_matrix_[i] = Matd::Identity();
            contact_line_[i] = 0;
            contact_angle_weight_[i] = 0.0;
            contact_line_partition_[i] = 0.0;
            dynamic_contact_cosine_[i] = 0.0;
            wetting_force_[i] = Vecd::Zero();
            wall_adhesion_force_[i] = Vecd::Zero();
            solid_liquid_force_[i] = Vecd::Zero();
            support_neighbors_[i] = 0;
            surface_confidence_[i] = 0.0;
            pca_normal_used_[i] = 0;
            bool explicit_footprint_endpoint = false;
            if (HasCentralWetFootprint() && footprint_left < footprint_right)
            {
                Real wall_distance = FiberNormalDistance(pos_[i]);
                Real endpoint_band = 1.25 * ParticleSpacing();
                explicit_footprint_endpoint =
                    wall_distance >= -0.25 * ParticleSpacing() &&
                    wall_distance <= 1.5 * ParticleSpacing() &&
                    (pos_[i][0] <= footprint_left + endpoint_band ||
                     pos_[i][0] >= footprint_right - endpoint_band);
                // The remaining wall-adjacent particles belong to the
                // liquid-solid branch, not to the liquid-gas interface.
                if (wall_distance <= 1.5 * ParticleSpacing() &&
                    !explicit_footprint_endpoint)
                    continue;
            }
            if (indicator_[i] != 1 && !explicit_footprint_endpoint)
                continue;

            const Neighborhood &neighborhood = inner_configuration_[i];
            support_neighbors_[i] = static_cast<int>(neighborhood.current_size_);
            Matd local_configuration = Matd::Zero();
            for (size_t n = 0; n < neighborhood.current_size_; ++n)
            {
                const size_t j = neighborhood.j_[n];
                gradient_[i] -= neighborhood.dW_ij_[n] * Vol_[j] *
                                neighborhood.e_ij_[n];
                Vecd gradient_W = neighborhood.dW_ij_[n] *
                                  neighborhood.e_ij_[n];
                Vecd r_ji = neighborhood.r_ij_[n] *
                            neighborhood.e_ij_[n];
                local_configuration -= Vol_[j] * r_ji *
                                       gradient_W.transpose();
            }
            // Solid particles complete the colour-gradient support below the
            // fibre surface. Consequently a fully wetted liquid-solid branch
            // has nearly zero gradient, while the open gas side at a true
            // three-phase endpoint remains visible.
            for (size_t k = 0; k < contact_configuration_.size(); ++k)
            {
                Real *fibre_volume = fibre_Vol_[k];
                const Neighborhood &fibre_neighborhood =
                    (*contact_configuration_[k])[i];
                for (size_t n = 0; n < fibre_neighborhood.current_size_; ++n)
                {
                    size_t j = fibre_neighborhood.j_[n];
                    gradient_[i] -= fibre_neighborhood.dW_ij_[n] *
                                    fibre_volume[j] *
                                    fibre_neighborhood.e_ij_[n];
                    Vecd gradient_W = fibre_neighborhood.dW_ij_[n] *
                                      fibre_neighborhood.e_ij_[n];
                    Vecd r_ji = fibre_neighborhood.r_ij_[n] *
                                fibre_neighborhood.e_ij_[n];
                    local_configuration -= fibre_volume[j] * r_ji *
                                           gradient_W.transpose();
                }
            }
            if (breakup_options.css_gradient_correction)
                correction_matrix_[i] =
                    inverseTikhonov(local_configuration, SqrtEps);
            delta_[i] = gradient_[i].norm();
            if (delta_[i] > gradient_floor)
                normal_[i] = gradient_[i] / delta_[i];
        }

        // Reconstruct the local interface direction from the covariance of
        // nearby free-surface particle positions. The principal direction is
        // the local tangent; its perpendicular is oriented by the raw colour
        // gradient. Oppositely oriented neighbours across a thin neck are
        // excluded so disconnected or nearly disconnected interfaces are not
        // fitted as one surface.
        if (breakup_options.normal_reconstruction == "pca")
        {
            std::vector<Vecd> reconstructed_normal(total, Vecd::Zero());
            for (size_t i = 0; i < total; ++i)
            {
                reconstructed_normal[i] = normal_[i];
                if (delta_[i] <= gradient_floor)
                    continue;
                Real wall_distance_i = FiberNormalDistance(pos_[i]);
                bool reconstructing_contact =
                    wall_distance_i <= breakup_options.contact_zone_in_dx *
                                           ParticleSpacing();
                Matd covariance = Matd::Zero();
                size_t samples = 0;
                const Neighborhood &neighborhood = inner_configuration_[i];
                for (size_t n = 0; n < neighborhood.current_size_; ++n)
                {
                    const size_t j = neighborhood.j_[n];
                    if (delta_[j] <= gradient_floor ||
                        normal_[i].dot(normal_[j]) <= -0.25)
                        continue;
                    if (reconstructing_contact)
                    {
                        // The solid-liquid branch and the liquid-gas branch
                        // meet inside one kernel support.  Only samples above
                        // the wall core belong to the free surface whose
                        // tangent defines the dynamic contact angle.
                        Real wall_distance_j = FiberNormalDistance(pos_[j]);
                        if (wall_distance_j <= breakup_options.contact_core_in_dx *
                                                   ParticleSpacing() ||
                            normal_[i][0] * normal_[j][0] <= 0.0)
                            continue;
                    }
                    Vecd relative = neighborhood.r_ij_[n] * neighborhood.e_ij_[n];
                    Real w = std::max(neighborhood.W_ij_[n] * Vol_[j], Real(0.0));
                    covariance += w * relative * relative.transpose();
                    ++samples;
                }
                Real a = covariance(0, 0);
                Real b = covariance(0, 1);
                Real c = covariance(1, 1);
                Real discriminant = std::sqrt((a - c) * (a - c) + 4.0 * b * b);
                Real lambda_max = 0.5 * (a + c + discriminant);
                Real lambda_min = 0.5 * (a + c - discriminant);
                if (samples >= 2 && lambda_max > 1.2 * lambda_min + TinyReal)
                {
                    Real tangent_angle = 0.5 * std::atan2(2.0 * b, a - c);
                    Vecd tangent(std::cos(tangent_angle), std::sin(tangent_angle));
                    Vecd candidate(-tangent[1], tangent[0]);
                    Real wall_distance = FiberNormalDistance(pos_[i]);
                    bool in_contact_zone =
                        wall_distance <= breakup_options.contact_zone_in_dx *
                                             ParticleSpacing();
                    if (in_contact_zone && std::abs(normal_[i][0]) > 0.10)
                    {
                        // For an obtuse contact angle, missing kernel support
                        // below the free surface biases the raw gradient into
                        // the wall. Its axial component still identifies the
                        // exterior side, and therefore selects the physically
                        // correct sign of the PCA interface normal.
                        if (candidate[0] * normal_[i][0] < 0.0)
                            candidate = -candidate;
                    }
                    else if (candidate.dot(normal_[i]) < 0.0)
                        candidate = -candidate;
                    reconstructed_normal[i] = candidate;
                    pca_normal_used_[i] = 1;
                }
            }
            for (size_t i = 0; i < total; ++i)
                normal_[i] = reconstructed_normal[i];
        }

        // Pass 2: repeatedly smooth particle-scale normal and surface-strength
        // noise before imposing the contact angle. Only similarly oriented
        // neighbours are blended, so opposite sides of a thin neck do not
        // cancel each other and artificially erase a resolved dry interval.
        std::vector<Vecd> smoothed_normal(total, Vecd::Zero());
        std::vector<Real> smoothed_delta(total, 0.0);
        for (size_t i = 0; i < total; ++i)
        {
            smoothed_normal[i] = normal_[i];
            smoothed_delta[i] = delta_[i];
        }
        for (int pass = 0; pass < breakup_options.surface_smoothing_passes; ++pass)
        {
            std::vector<Vecd> next_normal(total, Vecd::Zero());
            std::vector<Real> next_delta(total, 0.0);
            for (size_t i = 0; i < total; ++i)
            {
                if (smoothed_delta[i] <= gradient_floor)
                    continue;
                Vecd average = smoothed_normal[i];
                Real delta_average = smoothed_delta[i];
                Real weight = 1.0;
                const Neighborhood &neighborhood = inner_configuration_[i];
                for (size_t n = 0; n < neighborhood.current_size_; ++n)
                {
                    const size_t j = neighborhood.j_[n];
                    if (smoothed_delta[j] <= gradient_floor)
                        continue;
                    Real alignment = smoothed_normal[i].dot(smoothed_normal[j]);
                    if (alignment <= breakup_options.normal_alignment_floor)
                        continue;
                    Real alignment_weight =
                        (alignment - breakup_options.normal_alignment_floor) /
                        (1.0 - breakup_options.normal_alignment_floor);
                    Real w = std::max(neighborhood.W_ij_[n] * Vol_[j], Real(0.0)) *
                             alignment_weight * alignment_weight;
                    average += w * smoothed_normal[j];
                    delta_average += w * smoothed_delta[j];
                    weight += w;
                }
                average /= weight;
                next_normal[i] = average / (average.norm() + TinyReal);
                next_delta[i] = delta_average / weight;
            }
            smoothed_normal.swap(next_normal);
            smoothed_delta.swap(next_delta);
        }

        const Real theta = breakup_options.contact_angle_degrees * Pi / 180.0;
        const Real contact_core = breakup_options.contact_core_in_dx * ParticleSpacing();
        const Real contact_zone = breakup_options.contact_zone_in_dx * ParticleSpacing();
        for (size_t i = 0; i < total; ++i)
        {
            if (delta_[i] <= gradient_floor)
                continue;
            Vecd n = smoothed_normal[i];
            Real wall_distance = FiberNormalDistance(pos_[i]);
            bool in_central_endpoint_band = true;
            if (HasCentralWetFootprint() && footprint_left < footprint_right)
            {
                Real endpoint_band = 1.25 * ParticleSpacing();
                in_central_endpoint_band =
                    pos_[i][0] <= footprint_left + endpoint_band ||
                    pos_[i][0] >= footprint_right - endpoint_band;
            }
            if (breakup_options.initial_topology != "full-coating" &&
                wall_distance <= contact_zone && in_central_endpoint_band)
            {
                Real transition = std::clamp(
                    (wall_distance - contact_core) / (contact_zone - contact_core),
                    Real(0.0), Real(1.0));
                Real free_surface_weight = transition * transition *
                                           (3.0 - 2.0 * transition);
                Real contact_angle_weight = 1.0 - free_surface_weight;
                if (breakup_options.wetting_model == "fixed-angle")
                {
                    // Legacy comparison model: prescribe the desired normal
                    // in the contact zone rather than derive it from energies.
                    Real axial_sign = n[0] >= 0.0 ? 1.0 : -1.0;
                    Vecd equilibrium_normal(axial_sign * std::sin(theta),
                                             std::cos(theta));
                    Vecd blended = free_surface_weight * n +
                                   contact_angle_weight * equilibrium_normal;
                    n = blended / (blended.norm() + TinyReal);
                }
                contact_line_[i] = 1;
                contact_angle_weight_[i] = contact_angle_weight;
            }
            normal_[i] = n;
            delta_[i] = smoothed_delta[i];
            Real support_transition = std::clamp(
                (static_cast<Real>(support_neighbors_[i]) -
                 static_cast<Real>(breakup_options.support_min_neighbors)) /
                    static_cast<Real>(breakup_options.support_full_neighbors -
                                      breakup_options.support_min_neighbors),
                Real(0.0), Real(1.0));
            surface_confidence_[i] = support_transition * support_transition *
                                     (3.0 - 2.0 * support_transition);
            stress_[i] = breakup_options.gamma_lg * delta_[i] * surface_confidence_[i] *
                         (Matd::Identity() - n * n.transpose());

            // A contact endpoint must have a resolvable axial interface
            // normal. This removes near-wall support-deficiency particles
            // belonging to the solid-liquid branch rather than the endpoint.
            if (contact_line_[i] == 1 && std::abs(n[0]) < 0.25)
            {
                contact_line_[i] = 0;
                contact_angle_weight_[i] = 0.0;
            }
            if (contact_line_[i] == 1 &&
                support_neighbors_[i] <= breakup_options.support_min_neighbors)
            {
                contact_line_[i] = 0;
                contact_angle_weight_[i] = 0.0;
            }
        }

        // Sessile-drop validation fallback. Build the wall-supported liquid
        // footprint from the largest connected near-wall particle component.
        // This prevents an isolated or ejected particle from becoming the new
        // geometric endpoint and repeatedly receiving the complete Young
        // traction. The general multi-bead fibre calculation leaves the
        // external centre as NaN and therefore does not use this construction.
        Real validation_contact_center_x =
            *(particles_->getSingleVariableByName<Real>(
                  "ExternalContactCenterX")
                  ->Data());
        // A prescribed central crown has a known left/right contact-line
        // partition even though it does not use the external flat-wall
        // validation driver.  Leaving the centre as NaN made the geometric
        // endpoint fallback silently inactive after the first particle move.
        if (!std::isfinite(validation_contact_center_x) &&
            HasCentralWetFootprint())
            validation_contact_center_x = 0.0;
        if (std::isfinite(validation_contact_center_x))
        {
            std::vector<int> near_wall(total, 0);
            std::vector<int> near_wall_degree(total, 0);
            std::vector<int> component(total, -1);
            std::vector<size_t> component_size;
            for (size_t i = 0; i < total; ++i)
            {
                Real wall_distance = FiberNormalDistance(pos_[i]);
                near_wall[i] = wall_distance >= -0.25 * ParticleSpacing() &&
                                       wall_distance <= contact_zone
                                   ? 1
                                   : 0;
            }
            for (size_t i = 0; i < total; ++i)
            {
                if (near_wall[i] == 0)
                    continue;
                const Neighborhood &neighborhood = inner_configuration_[i];
                for (size_t n = 0; n < neighborhood.current_size_; ++n)
                    if (near_wall[neighborhood.j_[n]] == 1)
                        ++near_wall_degree[i];
            }

            int component_id = 0;
            for (size_t seed = 0; seed < total; ++seed)
            {
                if (near_wall[seed] == 0 || component[seed] >= 0)
                    continue;
                std::vector<size_t> stack(1, seed);
                component[seed] = component_id;
                size_t count = 0;
                while (!stack.empty())
                {
                    size_t i = stack.back();
                    stack.pop_back();
                    ++count;
                    const Neighborhood &neighborhood = inner_configuration_[i];
                    for (size_t n = 0; n < neighborhood.current_size_; ++n)
                    {
                        size_t j = neighborhood.j_[n];
                        if (near_wall[j] == 0 || component[j] >= 0)
                            continue;
                        component[j] = component_id;
                        stack.push_back(j);
                    }
                }
                component_size.push_back(count);
                ++component_id;
            }

            int largest_component = -1;
            size_t largest_size = 0;
            for (size_t c = 0; c < component_size.size(); ++c)
                if (component_size[c] > largest_size)
                {
                    largest_size = component_size[c];
                    largest_component = static_cast<int>(c);
                }

            // Retain the raw free-surface endpoint representatives only when
            // they belong to the main wetted footprint. This preserves useful
            // interface support while preventing an isolated component from
            // becoming a new Young-force endpoint.
            for (size_t i = 0; i < total; ++i)
                if (contact_line_[i] == 1 &&
                    component[i] != largest_component)
                {
                    contact_line_[i] = 0;
                    contact_angle_weight_[i] = 0.0;
                }

            Real left_edge = std::numeric_limits<Real>::max();
            Real right_edge = -std::numeric_limits<Real>::max();
            for (size_t i = 0; i < total; ++i)
            {
                if (largest_component < 0 ||
                    component[i] != largest_component ||
                    near_wall_degree[i] < 2)
                    continue;
                if (pos_[i][0] < validation_contact_center_x)
                    left_edge = std::min(left_edge, pos_[i][0]);
                else
                    right_edge = std::max(right_edge, pos_[i][0]);
            }
            // Supplement missing raw labels over one kernel-sized axial band.
            const Real endpoint_band = 1.25 * ParticleSpacing();
            for (size_t i = 0; i < total; ++i)
            {
                if (largest_component < 0 ||
                    component[i] != largest_component)
                    continue;
                Real wall_distance = FiberNormalDistance(pos_[i]);
                bool left_endpoint =
                    left_edge < std::numeric_limits<Real>::max() &&
                    pos_[i][0] < validation_contact_center_x &&
                    pos_[i][0] <= left_edge + endpoint_band;
                bool right_endpoint =
                    right_edge > -std::numeric_limits<Real>::max() &&
                    pos_[i][0] >= validation_contact_center_x &&
                    pos_[i][0] >= right_edge - endpoint_band;
                if (!left_endpoint && !right_endpoint)
                    continue;
                Real transition = std::clamp(
                    (wall_distance - contact_core) /
                        (contact_zone - contact_core),
                    Real(0.0), Real(1.0));
                Real free_surface_weight = transition * transition *
                                           (3.0 - 2.0 * transition);
                contact_line_[i] = 1;
                contact_angle_weight_[i] =
                    std::max(contact_angle_weight_[i],
                             Real(1.0) - free_surface_weight);
                surface_confidence_[i] =
                    std::max(surface_confidence_[i], Real(1.0));
            }
        }

        // Partition one physical Young endpoint force over all particle
        // representatives in its local contact-line cluster. Without this
        // normalization, adding another particle layer repeats the same line
        // force and makes the result resolution dependent.
        Real external_contact_center_x =
            *(particles_->getSingleVariableByName<Real>(
                  "ExternalContactCenterX")
                  ->Data());
        if (!std::isfinite(external_contact_center_x) &&
            HasCentralWetFootprint())
            external_contact_center_x = 0.0;
        Real side_normal_y_sum[2] = {0.0, 0.0};
        Real side_normal_weight[2] = {0.0, 0.0};
        if (std::isfinite(external_contact_center_x))
        {
            for (size_t i = 0; i < total; ++i)
            {
                if (contact_line_[i] != 1)
                    continue;
                size_t side = pos_[i][0] >= external_contact_center_x ? 1 : 0;
                Real weight = std::max(contact_angle_weight_[i] *
                                           surface_confidence_[i],
                                       Real(0.0));
                side_normal_y_sum[side] += weight * normal_[i][1];
                side_normal_weight[side] += weight;
            }
        }
        std::vector<Real> raw_line_measure(total, 0.0);
        std::vector<Real> endpoint_dynamic_cosine(total, 0.0);
        for (size_t i = 0; i < total; ++i)
        {
            if (contact_line_[i] != 1)
                continue;
            Real particle_volume = mass_[i] / (rho_[i] + TinyReal);
            Real endpoint_delta = delta_[i] > gradient_floor
                                      ? delta_[i]
                                      : 1.0 / ParticleSpacing();
            raw_line_measure[i] = particle_volume * endpoint_delta *
                                  contact_angle_weight_[i] *
                                  surface_confidence_[i] /
                                  (contact_zone + TinyReal);

            // Reconstruct the contact tangent from free-surface samples above
            // the wall core.  This avoids the acute/obtuse sign ambiguity of
            // a support-deficiency normal at the wall junction.  The tangent
            // points from the endpoint into the resolved free surface.
            Vecd tangent = Vecd::Zero();
            Real tangent_weight = 0.0;
            Real axial_sign = std::isfinite(external_contact_center_x)
                                  ? (pos_[i][0] >= external_contact_center_x ? 1.0 : -1.0)
                                  : (normal_[i][0] >= 0.0 ? 1.0 : -1.0);
            size_t validation_side = axial_sign > 0.0 ? 1 : 0;
            Real side_mean_normal_y =
                side_normal_y_sum[validation_side] /
                (side_normal_weight[validation_side] + TinyReal);
            const Neighborhood &neighborhood = inner_configuration_[i];
            for (size_t n = 0; n < neighborhood.current_size_; ++n)
            {
                const size_t j = neighborhood.j_[n];
                Vecd toward_surface = pos_[j] - pos_[i];
                Real wall_distance_j = FiberNormalDistance(pos_[j]);
                if (indicator_[j] != 1 || delta_[j] <= gradient_floor ||
                    wall_distance_j <= contact_core ||
                    toward_surface[1] <= 0.20 * ParticleSpacing())
                    continue;
                if (std::isfinite(external_contact_center_x) &&
                    side_mean_normal_y < -0.10 &&
                    axial_sign * toward_surface[0] * side_mean_normal_y >= 0.0)
                    continue;
                Real weight = std::max(neighborhood.W_ij_[n] * Vol_[j], Real(0.0));
                tangent += weight * toward_surface /
                           (toward_surface.norm() + TinyReal);
                tangent_weight += weight;
            }
            if (tangent_weight > TinyReal && tangent.norm() > TinyReal)
            {
                tangent /= tangent.norm();
                endpoint_dynamic_cosine[i] = std::clamp(
                    -axial_sign * tangent[0], Real(-1.0), Real(1.0));
            }
            else
                endpoint_dynamic_cosine[i] = normal_[i][1];
        }

        // A sessile-drop validation has exactly one contact endpoint on each
        // side of the externally fitted drop centre. Normalize over the whole
        // side so overlapping particle neighbourhoods cannot apply the same
        // physical Young line force more than once. The general fibre-breakup
        // problem may contain several disconnected wet regions, so it retains
        // the local-cluster normalization below.
        Real global_partition[2] = {0.0, 0.0};
        Real global_cosine_sum[2] = {0.0, 0.0};
        if (std::isfinite(external_contact_center_x))
        {
            for (size_t i = 0; i < total; ++i)
            {
                if (contact_line_[i] != 1)
                    continue;
                size_t side = pos_[i][0] >= external_contact_center_x ? 1 : 0;
                global_partition[side] += raw_line_measure[i];
                global_cosine_sum[side] +=
                    raw_line_measure[i] * endpoint_dynamic_cosine[i];
            }
        }
        for (size_t i = 0; i < total; ++i)
        {
            if (contact_line_[i] != 1)
                continue;
            if (std::isfinite(external_contact_center_x))
            {
                size_t side = pos_[i][0] >= external_contact_center_x ? 1 : 0;
                contact_line_partition_[i] = global_partition[side];
                dynamic_contact_cosine_[i] =
                    global_cosine_sum[side] / (global_partition[side] + TinyReal);
                continue;
            }
            Real partition = raw_line_measure[i];
            Real cosine_sum = raw_line_measure[i] * endpoint_dynamic_cosine[i];
            const Neighborhood &neighborhood = inner_configuration_[i];
            Real side_i = normal_[i][0];
            for (size_t n = 0; n < neighborhood.current_size_; ++n)
            {
                const size_t j = neighborhood.j_[n];
                Real side_j = normal_[j][0];
                if (contact_line_[j] != 1 || side_i * side_j <= 0.0)
                    continue;
                partition += raw_line_measure[j];
                cosine_sum += raw_line_measure[j] * endpoint_dynamic_cosine[j];
            }
            contact_line_partition_[i] = partition;
            dynamic_contact_cosine_[i] = cosine_sum / (partition + TinyReal);
        }

        // Optional conservative regularization for the single sessile-drop
        // validation. The physical Young line force on each side is unchanged,
        // but its discrete delta support is widened from the endpoint labels to
        // a compact liquid neighbourhood. This prevents a few wall particles
        // from reaching the target angle before the drop body can respond.
        std::vector<Vecd> spread_wetting_force(total, Vecd::Zero());
        bool use_endpoint_spread =
            breakup_options.endpoint_spread_in_dx > 0.0 &&
            breakup_options.wetting_model == "surface-energy" &&
            breakup_options.complete_young_residual &&
            std::isfinite(external_contact_center_x);
        if (use_endpoint_spread)
        {
            Real contact_anchor_x[2] = {
                std::numeric_limits<Real>::max(),
                -std::numeric_limits<Real>::max()};
            Real side_dynamic_cosine[2] = {0.0, 0.0};
            size_t side_samples[2] = {0, 0};
            for (size_t i = 0; i < total; ++i)
            {
                if (contact_line_[i] != 1)
                    continue;
                size_t side = pos_[i][0] >= external_contact_center_x ? 1 : 0;
                if (side == 0)
                    contact_anchor_x[0] = std::min(contact_anchor_x[0], pos_[i][0]);
                else
                    contact_anchor_x[1] = std::max(contact_anchor_x[1], pos_[i][0]);
                side_dynamic_cosine[side] += dynamic_contact_cosine_[i];
                ++side_samples[side];
            }

            Real spread_radius =
                breakup_options.endpoint_spread_in_dx * ParticleSpacing();
            Real external_dynamic_cosine =
                *(particles_->getSingleVariableByName<Real>(
                      "ExternalDynamicContactCosine")
                      ->Data());
            for (size_t side = 0; side < 2; ++side)
            {
                if (side_samples[side] == 0 ||
                    !std::isfinite(contact_anchor_x[side]))
                    continue;
                Real local_cosine = side_dynamic_cosine[side] /
                                    static_cast<Real>(side_samples[side]);
                Real selected_cosine = std::isfinite(external_dynamic_cosine)
                                           ? external_dynamic_cosine
                                           : local_cosine;
                selected_cosine = std::clamp(selected_cosine,
                                             Real(-1.0), Real(1.0));
                Real young_residual =
                    breakup_options.gamma_sg - breakup_options.gamma_sl -
                    breakup_options.gamma_lg * selected_cosine;
                Vecd anchor(contact_anchor_x[side], fibre_radius);
                Real weight_sum = 0.0;
                std::vector<Real> weights(total, 0.0);
                for (size_t i = 0; i < total; ++i)
                {
                    bool particle_on_side = side == 0
                                                ? pos_[i][0] < external_contact_center_x
                                                : pos_[i][0] >= external_contact_center_x;
                    if (!particle_on_side)
                        continue;
                    Real distance = (pos_[i] - anchor).norm();
                    if (distance >= spread_radius)
                        continue;
                    Real compact_weight =
                        0.5 * (1.0 + std::cos(Pi * distance / spread_radius));
                    Real particle_volume = mass_[i] / (rho_[i] + TinyReal);
                    weights[i] = compact_weight * particle_volume;
                    weight_sum += weights[i];
                }
                Real axial_sign = side == 0 ? -1.0 : 1.0;
                Vecd side_force = breakup_options.wetting_force_scale *
                                  young_residual * Vecd(axial_sign, 0.0);
                for (size_t i = 0; i < total; ++i)
                    if (weights[i] > 0.0)
                        spread_wetting_force[i] +=
                            weights[i] / (weight_sum + TinyReal) * side_force;
            }
        }

        // Pass 3: meridional stress divergence plus its cylindrical term.
        // For tau=gamma*delta*(I-nn), tau_theta_theta=gamma*delta.
        for (size_t i = 0; i < total; ++i)
        {
            Vecd force = Vecd::Zero();
            Real external_curvature =
                *(particles_->getSingleVariableByName<Real>(
                      "ExternalFreeSurfaceCurvature")
                      ->Data());
            const Neighborhood &neighborhood = inner_configuration_[i];
            if (std::isfinite(external_curvature))
            {
                // Single-phase curvature-force alternative used by the
                // sessile-drop validation. Keep its externally fitted 1/R
                // traction out of the explicit endpoint band: attempts to
                // restore either the full or wall-normal-only contribution
                // double loaded the obtuse spreading case. A future local
                // curvature/CSS discretization must resolve this band
                // consistently instead of adding the global-fit traction.
                if (!(breakup_options.separate_contact_endpoint &&
                      contact_angle_weight_[i] > 0.0) &&
                    delta_[i] > gradient_floor)
                {
                    Real particle_volume = mass_[i] / (rho_[i] + TinyReal);
                    force -= particle_volume * breakup_options.gamma_lg *
                             external_curvature * delta_[i] *
                             surface_confidence_[i] * normal_[i];
                }
            }
            else for (size_t n = 0; n < neighborhood.current_size_; ++n)
            {
                const size_t j = neighborhood.j_[n];
                if (breakup_options.wetting_model == "surface-energy" &&
                    breakup_options.separate_contact_endpoint &&
                    (contact_angle_weight_[i] > 0.0 ||
                     contact_angle_weight_[j] > 0.0))
                {
                    // Do not let the one-sided CSS divergence create a second,
                    // poorly resolved endpoint traction inside the contact
                    // band. The complete Young residual below supplies that
                    // endpoint force explicitly and conservatively by side.
                    continue;
                }
                Matd hourglass_correction = Matd::Zero();
                if (breakup_options.surface_hourglass_coefficient > 0.0 &&
                    (delta_[i] > gradient_floor || delta_[j] > gradient_floor))
                {
                    Matd tangent_i = Matd::Identity() -
                                     normal_[i] * normal_[i].transpose();
                    Matd tangent_j = Matd::Identity() -
                                     normal_[j] * normal_[j].transpose();
                    Vecd gradient_average = 0.5 * (gradient_[i] + gradient_[j]);
                    Matd projected_jump = gradient_average *
                                          neighborhood.e_ij_[n].transpose() *
                                          neighborhood.r_ij_[n];
                    Matd mismatch = -projected_jump * projected_jump /
                                    (projected_jump.norm() + TinyReal);
                    hourglass_correction =
                        breakup_options.surface_hourglass_coefficient *
                        breakup_options.gamma_lg * 0.5 * (tangent_i + tangent_j) *
                        mismatch / (neighborhood.r_ij_[n] + TinyReal);
                }
                Vecd pair_gradient = neighborhood.dW_ij_[n] *
                                     neighborhood.e_ij_[n];
                if (breakup_options.css_gradient_correction)
                    pair_gradient = 0.5 *
                                    (correction_matrix_[i] + correction_matrix_[j]) *
                                    pair_gradient;
                force += mass_[i] * Vol_[j] *
                         (stress_[i] + stress_[j] + hourglass_correction) *
                         pair_gradient /
                         (rho_[i] + TinyReal);
            }
            if (!std::isfinite(external_curvature) &&
                run_options.FullAxisymmetric() && delta_[i] > gradient_floor)
            {
                Real r = std::max(pos_[i][1], 0.25 * ParticleSpacing());
                Vecd cylindrical(stress_[i](0, 1),
                                  stress_[i](1, 1) - breakup_options.gamma_lg * delta_[i] *
                                                               surface_confidence_[i]);
                force += mass_[i] * cylindrical / ((rho_[i] + TinyReal) * r);
            }

            if (breakup_options.wetting_model == "surface-energy")
            {
                Real wall_distance = FiberNormalDistance(pos_[i]);
                Real particle_volume = mass_[i] / (rho_[i] + TinyReal);

                // Apply the complete uncompensated Young traction at the
                // contact line.  In the continuum identity the end point of
                // the liquid-gas surface-stress integral supplies
                // -gamma_lg*cos(theta_d).  That boundary term is not retained
                // reliably by the present one-sided particle CSS divergence,
                // whose support simply terminates at the wall.  Add the full
                // energy residual explicitly instead of prescribing a normal:
                //
                // f_Y = gamma_sg - gamma_sl - gamma_lg*cos(theta_d).
                //
                // For a circular cap above a horizontal wall, the exterior
                // normal at the right contact point is
                // (sin(theta_d), cos(theta_d)); hence cos(theta_d)=n_y.
                // The force vanishes naturally when Young's equation holds.
                if (use_endpoint_spread)
                {
                    wetting_force_[i] = spread_wetting_force[i];
                    force += wetting_force_[i];
                }
                if (!use_endpoint_spread && contact_line_[i] == 1 &&
                    (delta_[i] > gradient_floor ||
                     std::isfinite(external_contact_center_x)))
                {
                    Real external_center_x =
                        *(particles_->getSingleVariableByName<Real>(
                              "ExternalContactCenterX")
                              ->Data());
                    Real axial_sign = std::isfinite(external_center_x)
                                          ? (pos_[i][0] >= external_center_x ? 1.0 : -1.0)
                                          : (normal_[i][0] >= 0.0 ? 1.0 : -1.0);
                    Real solid_energy_difference =
                        breakup_options.gamma_sg - breakup_options.gamma_sl;
                    Real external_dynamic_cosine =
                        *(particles_->getSingleVariableByName<Real>(
                              "ExternalDynamicContactCosine")
                              ->Data());
                    Real selected_dynamic_cosine =
                        std::isfinite(external_dynamic_cosine)
                            ? external_dynamic_cosine
                            : dynamic_contact_cosine_[i];
                    Real dynamic_cosine = std::clamp(selected_dynamic_cosine,
                                                     Real(-1.0), Real(1.0));
                    Real young_residual = breakup_options.complete_young_residual
                                              ? solid_energy_difference -
                                                    breakup_options.gamma_lg * dynamic_cosine
                                              : solid_energy_difference;
                    Real endpoint_partition_weight =
                        raw_line_measure[i] /
                        (contact_line_partition_[i] + TinyReal);
                    Real liquid_gas_vertical =
                        breakup_options.complete_endpoint_vector
                            ? breakup_options.gamma_lg *
                                  std::sqrt(std::max(Real(0.0),
                                                     Real(1.0) - dynamic_cosine * dynamic_cosine))
                            : 0.0;
                    Vecd complete_endpoint_force(axial_sign * young_residual,
                                                 liquid_gas_vertical);
                    wetting_force_[i] =
                        breakup_options.wetting_force_scale *
                        endpoint_partition_weight * complete_endpoint_force;
                    force += wetting_force_[i];
                }

                // Experimental curved solid-liquid traction.  For the rigid
                // cylindrical fibre kappa_sl=1/a.  A compact cosine delta
                // distributes the normal traction through the wall-contact
                // band; the rigid wall supplies the equal and opposite load.
                if (breakup_options.solid_curvature_traction &&
                    wall_distance >= 0.0 && wall_distance <= contact_zone)
                {
                    Real wall_delta =
                        (1.0 + std::cos(Pi * wall_distance / contact_zone)) /
                        (contact_zone + TinyReal);
                    Real kappa_sl = 1.0 / fibre_radius;
                    solid_liquid_force_[i] =
                        -particle_volume * breakup_options.gamma_sl * kappa_sl *
                        wall_delta * Vecd(0.0, 1.0);
                    force += solid_liquid_force_[i];
                }
            }

            // Wall-adhesion mode: represent the unresolved solid-liquid
            // molecular attraction by a smooth, short-range traction on the
            // liquid particles closest to the fibre.  The contact relation
            // supplies both the kernel-weighted wall support and a local
            // direction obtained from the actual solid neighbours.  The
            // latter is important near a contact-line endpoint: the
            // attraction then acquires a physically meaningful axial
            // component instead of being imposed as a globally vertical
            // force.  No target contact angle is prescribed here; the angle
            // is obtained from the balance between this attraction and the
            // liquid-gas surface tension.
            if (breakup_options.wetting_model == "wall-adhesion")
            {
                Real wall_distance = FiberNormalDistance(pos_[i]);
                Real adhesion_range = breakup_options.wall_adhesion_range_in_dx *
                                      ParticleSpacing();
                if (wall_distance >= 0.0 && wall_distance <= adhesion_range &&
                    breakup_options.wall_adhesion_strength > 0.0)
                {
                    Real wall_support = 0.0;
                    Vecd neighbour_direction = Vecd::Zero();
                    for (size_t k = 0; k < contact_configuration_.size(); ++k)
                    {
                        // Support-only bodies (the axis symmetry plane) never
                        // act as an adhesive wall.
                        if (!contact_body_is_wettable_[k])
                            continue;
                        Real *fibre_volume = fibre_Vol_[k];
                        const Neighborhood &fibre_neighborhood =
                            (*contact_configuration_[k])[i];
                        for (size_t n = 0; n < fibre_neighborhood.current_size_; ++n)
                        {
                            if (fibre_neighborhood.r_ij_[n] <= adhesion_range)
                            {
                                Real pair_weight =
                                    fibre_volume[fibre_neighborhood.j_[n]] *
                                    std::max(fibre_neighborhood.W_ij_[n], Real(0.0));
                                Real q_pair = fibre_neighborhood.r_ij_[n] /
                                              (adhesion_range + TinyReal);
                                Real pair_profile =
                                    std::max(Real(0.0), 1.0 - q_pair);
                                // Smooth compact support: phi(0)=1,
                                // phi(1)=0 and dphi/dr=0 at the cut-off.
                                pair_profile *= pair_profile *
                                                (1.0 + 2.0 * q_pair);
                                if (breakup_options.wall_adhesion_profile ==
                                    "pairwise")
                                    pair_weight *= pair_profile;
                                wall_support += pair_weight;

                                // e_ij may be stored with either pair
                                // orientation by a relation.  Orient the
                                // contribution explicitly toward the solid
                                // half-space so the force remains attractive
                                // for both conventions.
                                Vecd toward_solid =
                                    -fibre_neighborhood.e_ij_[n];
                                const Vecd nominal_wall_direction =
                                    -FiberOutwardNormal(pos_[i][0]);
                                if (toward_solid.dot(nominal_wall_direction) < 0.0)
                                    toward_solid = -toward_solid;
                                neighbour_direction += pair_weight * toward_solid;
                            }
                        }
                    }
                    wall_support = std::clamp(wall_support, Real(0.0), Real(1.0));
                    Real q = wall_distance / (adhesion_range + TinyReal);
                    Real distance_weight = 0.5 * (1.0 + std::cos(Pi * q));
                    Real particle_volume = mass_[i] / (rho_[i] + TinyReal);
                    Vecd wall_inward_normal(0.0, -1.0);
                    if (neighbour_direction.norm() > TinyReal)
                        wall_inward_normal =
                            neighbour_direction / neighbour_direction.norm();
                    Real interaction_weight =
                        breakup_options.wall_adhesion_profile == "pairwise"
                            ? 1.0
                            : distance_weight;
                    wall_adhesion_force_[i] =
                        particle_volume * breakup_options.gamma_lg *
                        breakup_options.wall_adhesion_strength * wall_support *
                        interaction_weight * wall_adhesion_activation_factor /
                        ((ParticleSpacing() * ParticleSpacing()) + TinyReal) *
                        wall_inward_normal;
                    force += wall_adhesion_force_[i];
                }
            }
            current_force_[i] = force;
            ForcePrior::update(i, dt);
        }
    }

  private:
    Vecd *pos_, *gradient_, *normal_, *wetting_force_, *wall_adhesion_force_,
        *solid_liquid_force_;
    Real *rho_, *mass_, *Vol_, *delta_, *contact_line_partition_,
        *dynamic_contact_cosine_;
    int *indicator_, *contact_line_, *support_neighbors_, *pca_normal_used_;
    StdVec<Real *> fibre_Vol_;
    StdVec<bool> contact_body_is_wettable_;
    Real *contact_angle_weight_, *surface_confidence_;
    Matd *stress_, *correction_matrix_;
};

/**
 * JFM-2024 single-phase free-surface wetting model in the meridional (x,r)
 * plane of a cylindrical fibre.  The axial curvature is reconstructed from
 * corrected particle normals and the cylindrical hoop contribution n_r/r is
 * added explicitly.  Wetting is imposed by the same near-wall normal rotation
 * used in the flat-wall validation; no separate endpoint Young force is added.
 */
class JfmAxisymmetricFiberWettingForce : public ForcePrior,
                                         public DataDelegateInner,
                                         public DataDelegateContact
{
  public:
    JfmAxisymmetricFiberWettingForce(BaseInnerRelation &inner_relation,
                                     BaseContactRelation &fibre_contact_relation,
                                     bool apply_contact_angle_correction = true,
                                     bool apply_wall_adhesion = false)
        : ForcePrior(inner_relation.getSPHBody(), "TopologyCapillaryForce"),
          DataDelegateInner(inner_relation),
          DataDelegateContact(fibre_contact_relation),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          vel_(particles_->registerStateVariableData<Vecd>("Velocity")),
          rho_(particles_->getVariableDataByName<Real>("Density")),
          pressure_(particles_->registerStateVariableData<Real>("Pressure")),
          mass_(particles_->getVariableDataByName<Real>("Mass")),
          Vol_(particles_->getVariableDataByName<Real>("VolumetricMeasure")),
          shepard_(particles_->registerStateVariableData<Real>("JfmShepardSum")),
          raw_normal_(particles_->registerStateVariableData<Vecd>("JfmRawNormal")),
          smooth_normal_(particles_->registerStateVariableData<Vecd>("JfmSmoothedNormal")),
          corrected_normal_(particles_->registerStateVariableData<Vecd>("TopologyNormal")),
          surface_delta_(particles_->registerStateVariableData<Real>("TopologySurfaceDelta")),
          axial_curvature_(particles_->registerStateVariableData<Real>("JfmAxialCurvature")),
          hoop_curvature_(particles_->registerStateVariableData<Real>("JfmHoopCurvature")),
          total_curvature_(particles_->registerStateVariableData<Real>("JfmTotalCurvature")),
          correction_matrix_(particles_->registerStateVariableData<Matd>("JfmGradientCorrection")),
          library_surface_indicator_(particles_->getVariableDataByName<int>("Indicator")),
          jfm_surface_indicator_(particles_->registerStateVariableData<int>("JfmSurfaceIndicator")),
          contact_line_(particles_->registerStateVariableData<int>("ContactLineIndicator")),
          dynamic_contact_cosine_(particles_->registerStateVariableData<Real>("DynamicContactCosine")),
          contact_line_speed_(particles_->registerStateVariableData<Real>("JfmContactLineSpeed")),
          wetting_diagnostic_(particles_->registerStateVariableData<Vecd>("SurfaceEnergyWettingForce")),
          wall_adhesion_force_(particles_->getVariableDataByName<Vecd>("WallAdhesionForce")),
          solid_liquid_force_(particles_->registerStateVariableData<Vecd>("SolidLiquidCurvatureForce")),
          curvature_force_(particles_->registerStateVariableData<Vecd>("JfmCurvatureForce")),
          pressure_jump_force_(particles_->registerStateVariableData<Vecd>("TopologyPressureJumpForce")),
          pressure_gauge_offset_(particles_->registerStateVariableData<Real>("JfmPressureGaugeOffset")),
          end_buffer_weight_(particles_->registerStateVariableData<Real>("ConicalEndBufferWeight")),
          contact_curvature_extended_(particles_->registerStateVariableData<int>("JfmContactCurvatureExtended")),
          curvature_outlier_filtered_(particles_->registerStateVariableData<int>("JfmCurvatureOutlierFiltered")),
          support_neighbors_(particles_->registerStateVariableData<int>("SurfaceSupportNeighbors")),
          pca_normal_used_(particles_->registerStateVariableData<int>("PcaNormalUsed")),
          physical_time_(sph_system_->svPhysicalTime().Data()),
          W0_(inner_relation.getSPHBody().getSPHAdaptation().getKernel()->W0(ZeroVecd)),
          smoothing_length_(inner_relation.getSPHBody().getSPHAdaptation().ReferenceSmoothingLength()),
          kernel_radius_(inner_relation.getSPHBody().getSPHAdaptation().getKernel()->CutOffRadius()),
          apply_contact_angle_correction_(apply_contact_angle_correction),
          apply_wall_adhesion_(apply_wall_adhesion),
          contact_history_initialized_(false), previous_contact_time_(0.0),
          previous_footprint_{0.0, 0.0}, filtered_contact_speed_{0.0, 0.0},
          force_scale_(1.0)
    {
        particles_->registerSingleVariable<Real>("SurfaceTensionCoef",
                                                  breakup_options.gamma_lg);
        for (BaseParticles *fibre_particles : contact_particles_)
            fibre_Vol_.push_back(
                fibre_particles->getVariableDataByName<Real>("VolumetricMeasure"));
        for (SPHBody *contact_body : contact_bodies_)
            contact_body_is_wettable_.push_back(
                !IsSupportOnlyContactBody(contact_body));

        particles_->addEvolvingVariable<Real>("JfmShepardSum");
        particles_->addEvolvingVariable<Vecd>("JfmRawNormal");
        particles_->addEvolvingVariable<Vecd>("JfmSmoothedNormal");
        particles_->addEvolvingVariable<Vecd>("TopologyNormal");
        particles_->addEvolvingVariable<Real>("TopologySurfaceDelta");
        particles_->addEvolvingVariable<Real>("JfmAxialCurvature");
        particles_->addEvolvingVariable<Real>("JfmHoopCurvature");
        particles_->addEvolvingVariable<Real>("JfmTotalCurvature");
        particles_->addEvolvingVariable<Matd>("JfmGradientCorrection");
        particles_->addEvolvingVariable<int>("JfmSurfaceIndicator");
        particles_->addEvolvingVariable<int>("ContactLineIndicator");
        particles_->addEvolvingVariable<Real>("DynamicContactCosine");
        particles_->addEvolvingVariable<Real>("JfmContactLineSpeed");
        particles_->addEvolvingVariable<Vecd>("SurfaceEnergyWettingForce");
        particles_->addEvolvingVariable<Vecd>("SolidLiquidCurvatureForce");
        particles_->addEvolvingVariable<Vecd>("JfmCurvatureForce");
        particles_->addEvolvingVariable<Vecd>("TopologyPressureJumpForce");
        particles_->addEvolvingVariable<Real>("JfmPressureGaugeOffset");
        particles_->addEvolvingVariable<int>("JfmContactCurvatureExtended");
        particles_->addEvolvingVariable<int>("JfmCurvatureOutlierFiltered");
        particles_->addEvolvingVariable<int>("SurfaceSupportNeighbors");
        particles_->addEvolvingVariable<int>("PcaNormalUsed");

        particles_->addVariableToWrite<Real>("JfmShepardSum");
        particles_->addVariableToWrite<Vecd>("JfmRawNormal");
        particles_->addVariableToWrite<Vecd>("JfmSmoothedNormal");
        particles_->addVariableToWrite<Vecd>("TopologyNormal");
        particles_->addVariableToWrite<Real>("TopologySurfaceDelta");
        particles_->addVariableToWrite<Real>("JfmAxialCurvature");
        particles_->addVariableToWrite<Real>("JfmHoopCurvature");
        particles_->addVariableToWrite<Real>("JfmTotalCurvature");
        particles_->addVariableToWrite<Matd>("JfmGradientCorrection");
        particles_->addVariableToWrite<int>("JfmSurfaceIndicator");
        particles_->addVariableToWrite<int>("ContactLineIndicator");
        particles_->addVariableToWrite<Real>("DynamicContactCosine");
        particles_->addVariableToWrite<Real>("JfmContactLineSpeed");
        particles_->addVariableToWrite<Vecd>("SurfaceEnergyWettingForce");
        particles_->addVariableToWrite<Vecd>("SolidLiquidCurvatureForce");
        particles_->addVariableToWrite<Vecd>("JfmCurvatureForce");
        particles_->addVariableToWrite<Vecd>("TopologyPressureJumpForce");
        particles_->addVariableToWrite<Real>("JfmPressureGaugeOffset");
        particles_->addVariableToWrite<int>("JfmContactCurvatureExtended");
        particles_->addVariableToWrite<int>("JfmCurvatureOutlierFiltered");
        particles_->addVariableToWrite<int>("SurfaceSupportNeighbors");
        particles_->addVariableToWrite<int>("PcaNormalUsed");
        particles_->addVariableToWrite<Vecd>("TopologyCapillaryForce");
    }

    void setForceScale(Real scale)
    {
        force_scale_ = std::clamp(scale, Real(0.0), Real(1.0));
    }

    void exec(Real dt = 0.0)
    {
        const size_t total = particles_->TotalRealParticles();
        const Real normal_threshold =
            breakup_options.surface_gradient_floor_in_dx /
            (ParticleSpacing() + TinyReal);
        const Real target_angle = YoungAngleDegrees() * Pi / 180.0;
        const bool hydrophilic = !apply_contact_angle_correction_ ||
                                 YoungAngleDegrees() <= 90.0;
        const bool include_solid_support =
            breakup_options.jfm_solid_extension == "all" || hydrophilic;
        const bool include_solid_normal_support =
            breakup_options.jfm_neighbor_mode == "mixed-support" ||
            include_solid_support;
        const int total_normal_passes =
            1 + breakup_options.surface_smoothing_passes;
        normal_buffer_a_.resize(total);
        normal_buffer_b_.resize(total);
        liquid_normal_buffer_.resize(total);
        curvature_shepard_buffer_.resize(total, 0.0);
        force_weight_buffer_.resize(total, 0.0);

        // Shepard support, colour-gradient normal and Bonet--Lok matrix.  The
        // fibre particles complete the kernel support below r=a, preventing
        // the entire liquid-solid interface from being mistaken for a free
        // liquid-gas surface.
        for (size_t i = 0; i < total; ++i)
        {
            shepard_[i] = W0_ * Vol_[i];
            raw_normal_[i] = Vecd::Zero();
            surface_delta_[i] = 0.0;
            axial_curvature_[i] = 0.0;
            hoop_curvature_[i] = 0.0;
            total_curvature_[i] = 0.0;
            jfm_surface_indicator_[i] = 0;
            contact_line_[i] = 0;
            curvature_outlier_filtered_[i] = 0;
            dynamic_contact_cosine_[i] = 0.0;
            contact_line_speed_[i] = 0.0;
            wetting_diagnostic_[i] = Vecd::Zero();
            wall_adhesion_force_[i] = Vecd::Zero();
            solid_liquid_force_[i] = Vecd::Zero();
            curvature_force_[i] = Vecd::Zero();
            pressure_jump_force_[i] = Vecd::Zero();
            pressure_gauge_offset_[i] = 0.0;
            contact_curvature_extended_[i] = 0;
            support_neighbors_[i] = 0;
            pca_normal_used_[i] = 0;
            current_force_[i] = Vecd::Zero();

            Matd local_configuration = Matd::Zero();
            const Neighborhood &inner_neighborhood = inner_configuration_[i];
            for (size_t n = 0; n < inner_neighborhood.current_size_; ++n)
            {
                size_t j = inner_neighborhood.j_[n];
                shepard_[i] += Vol_[j] * inner_neighborhood.W_ij_[n];
                Vecd gradient_W = inner_neighborhood.dW_ij_[n] *
                                  inner_neighborhood.e_ij_[n];
                raw_normal_[i] -= Vol_[j] * gradient_W;
                Vecd r_ji = inner_neighborhood.r_ij_[n] *
                            inner_neighborhood.e_ij_[n];
                local_configuration -=
                    Vol_[j] * r_ji * gradient_W.transpose();
            }
            if (include_solid_normal_support)
            {
                // Every contact body contributes here, including the support
                // only symmetry plane.  Near a fully coated fibre tip the film
                // wraps onto the axis, and without the mirrored half of the
                // support the colour gradient collapses onto the radial
                // coordinate and the reconstructed normal degenerates.
                for (size_t k = 0; k < contact_configuration_.size(); ++k)
                {
                    Real *fibre_volume = fibre_Vol_[k];
                    const Neighborhood &fibre_neighborhood =
                        (*contact_configuration_[k])[i];
                    for (size_t n = 0; n < fibre_neighborhood.current_size_; ++n)
                    {
                        size_t j = fibre_neighborhood.j_[n];
                        shepard_[i] += fibre_volume[j] * fibre_neighborhood.W_ij_[n];
                        Vecd gradient_W = fibre_neighborhood.dW_ij_[n] *
                                          fibre_neighborhood.e_ij_[n];
                        raw_normal_[i] -= fibre_volume[j] * gradient_W;
                        Vecd r_ji = fibre_neighborhood.r_ij_[n] *
                                    fibre_neighborhood.e_ij_[n];
                        local_configuration -=
                            fibre_volume[j] * r_ji * gradient_W.transpose();
                    }
                }
            }
            correction_matrix_[i] =
                inverseTikhonov(local_configuration, SqrtEps);
            normal_buffer_a_[i] = raw_normal_[i];
        }

        // Smooth non-unit normals.  One pass is intrinsic to the JFM method;
        // the option requests additional passes, as validated on the flat wall.
        for (int pass = 0; pass < total_normal_passes; ++pass)
        {
            for (size_t i = 0; i < total; ++i)
            {
                Vecd weighted_normal =
                    W0_ * Vol_[i] * normal_buffer_a_[i];
                const Neighborhood &neighborhood = inner_configuration_[i];
                for (size_t n = 0; n < neighborhood.current_size_; ++n)
                {
                    size_t j = neighborhood.j_[n];
                    weighted_normal += Vol_[j] * neighborhood.W_ij_[n] *
                                       normal_buffer_a_[j];
                }
                normal_buffer_b_[i] =
                    weighted_normal / (shepard_[i] + TinyReal);
            }
            normal_buffer_a_.swap(normal_buffer_b_);
        }

        // The edge-gap validation has one connected near-wall footprint.
        // Resolve its two axial endpoints directly from the wall-adjacent
        // particle layer. A 90-degree circular shoulder rises too steeply for
        // the generic outer-surface indicator to label its first lattice row.
        Real footprint_left = std::numeric_limits<Real>::max();
        Real footprint_right = -std::numeric_limits<Real>::max();
        const size_t footprint_bins = std::max<size_t>(
            24, static_cast<size_t>(std::round(
                    DomainLength() / (2.0 * ParticleSpacing()))));
        std::vector<int> dominant_footprint_bins(footprint_bins, 0);
        auto footprint_bin_index = [&](Real x)
        {
            Real shifted = x - DomainLowerBound();
            Real periodic_x = shifted -
                              std::floor(shifted / DomainLength()) *
                                  DomainLength();
            return std::min(
                footprint_bins - 1, static_cast<size_t>(std::floor(
                                        periodic_x / DomainLength() *
                                        static_cast<Real>(footprint_bins))));
        };
        bool has_resolved_footprint = HasCentralWetFootprint();
        // A three-phase contact line needs a wall patch that is exposed to
        // gas.  When the film seals the whole fibre no such patch exists, so
        // the closed end caps must never be reported as contact lines however
        // close their first liquid layer happens to sit to the solid.
        bool wall_exposes_gas = HasCentralWetFootprint();
        if (has_resolved_footprint)
        {
            std::fill(dominant_footprint_bins.begin(),
                      dominant_footprint_bins.end(), 1);
            for (size_t i = 0; i < total; ++i)
            {
                Real geometric_wall_distance = FiberNormalDistance(pos_[i]);
                if (geometric_wall_distance < -0.25 * ParticleSpacing() ||
                    geometric_wall_distance > 1.5 * ParticleSpacing())
                    continue;
                footprint_left = std::min(footprint_left, pos_[i][0]);
                footprint_right = std::max(footprint_right, pos_[i][0]);
            }
        }
        else
        {
            std::vector<int> wall_wet(footprint_bins, 0);
            std::vector<size_t> wall_particles(footprint_bins, 0);
            std::vector<Real> outer_radius(
                footprint_bins, -std::numeric_limits<Real>::infinity());
            for (size_t i = 0; i < total; ++i)
            {
                size_t b = footprint_bin_index(pos_[i][0]);
                outer_radius[b] = std::max(
                    outer_radius[b], pos_[i][1] + 0.5 * ParticleSpacing());
                ++wall_particles[b];
                Real geometric_wall_distance = FiberNormalDistance(pos_[i]);
                if (geometric_wall_distance < -0.25 * ParticleSpacing() ||
                    geometric_wall_distance > 1.5 * ParticleSpacing())
                    continue;
            }
            const Real resolved_film_cutoff =
                breakup_options.rupture_cutoff_in_dx * ParticleSpacing();
            for (size_t b = 0; b < footprint_bins; ++b)
            {
                Real bin_x = DomainLowerBound() +
                             (static_cast<Real>(b) + 0.5) * DomainLength() /
                                 static_cast<Real>(footprint_bins);
                if (std::isfinite(outer_radius[b]) &&
                    FiberNormalDistance(Vecd(bin_x, outer_radius[b])) >=
                        resolved_film_cutoff)
                    wall_wet[b] = 1;
            }
            std::vector<int> closed_wet = wall_wet;
            for (size_t b = 0; b < footprint_bins; ++b)
                if (wall_wet[b] == 0 &&
                    wall_wet[(b + footprint_bins - 1) % footprint_bins] == 1 &&
                    wall_wet[(b + 1) % footprint_bins] == 1)
                    closed_wet[b] = 1;

            auto dry_anchor = std::find(closed_wet.begin(),
                                        closed_wet.end(), 0);
            if (dry_anchor != closed_wet.end())
            {
                wall_exposes_gas = true;
                size_t anchor = static_cast<size_t>(
                    dry_anchor - closed_wet.begin());
                size_t dry_run = 0, maximum_dry_run = 0;
                std::vector<size_t> current_wet_run, dominant_wet_run;
                size_t current_particles = 0, dominant_particles = 0;
                auto finalize_wet_run = [&]()
                {
                    if (current_particles > dominant_particles ||
                        (current_particles == dominant_particles &&
                         current_wet_run.size() > dominant_wet_run.size()))
                    {
                        dominant_wet_run = current_wet_run;
                        dominant_particles = current_particles;
                    }
                    current_wet_run.clear();
                    current_particles = 0;
                };
                for (size_t offset = 1; offset <= footprint_bins; ++offset)
                {
                    size_t b = (anchor + offset) % footprint_bins;
                    if (closed_wet[b] == 1)
                    {
                        dry_run = 0;
                        current_wet_run.push_back(b);
                        current_particles += wall_particles[b];
                    }
                    else
                    {
                        finalize_wet_run();
                        ++dry_run;
                        maximum_dry_run = std::max(maximum_dry_run, dry_run);
                    }
                }
                finalize_wet_run();
                if (maximum_dry_run >= static_cast<size_t>(
                                               breakup_options.hybrid_min_dry_bins) &&
                    !dominant_wet_run.empty())
                {
                    has_resolved_footprint = true;
                    for (size_t b : dominant_wet_run)
                        dominant_footprint_bins[b] = 1;
                    for (size_t i = 0; i < total; ++i)
                    {
                        Real geometric_wall_distance =
                            FiberNormalDistance(pos_[i]);
                        if (geometric_wall_distance <
                                -0.25 * ParticleSpacing() ||
                            geometric_wall_distance >
                                1.5 * ParticleSpacing() ||
                            dominant_footprint_bins[
                                footprint_bin_index(pos_[i][0])] == 0)
                            continue;
                        footprint_left = std::min(footprint_left, pos_[i][0]);
                        footprint_right = std::max(footprint_right, pos_[i][0]);
                    }
                }
            }
        }
        std::array<Real, 2> contact_outward_speed{0.0, 0.0};
        std::array<Real, 2> contact_target_cosine{
            std::cos(target_angle), std::cos(target_angle)};
        if (has_resolved_footprint && footprint_left < footprint_right)
        {
            const Real current_time = *physical_time_;
            if (!contact_history_initialized_ ||
                current_time + TinyReal < previous_contact_time_)
            {
                previous_footprint_ = {footprint_left, footprint_right};
                previous_contact_time_ = current_time;
                filtered_contact_speed_ = {0.0, 0.0};
                contact_history_initialized_ = true;
            }
            else
            {
                const Real elapsed_time = current_time - previous_contact_time_;
                if (elapsed_time > TinyReal)
                {
                    std::array<Real, 2> raw_outward_speed{
                        -(footprint_left - previous_footprint_[0]) / elapsed_time,
                        (footprint_right - previous_footprint_[1]) / elapsed_time};
                    const Real capillary_speed =
                        breakup_options.gamma_lg / (DynamicViscosity() + TinyReal);
                    const Real speed_limit =
                        breakup_options.contact_speed_capillary_number *
                        capillary_speed;
                    const Real filter_time =
                        breakup_options.contact_speed_filter_T *
                        DynamicViscosity() * fibre_radius /
                        (breakup_options.gamma_lg + TinyReal);
                    const Real filter_weight = std::clamp(
                        elapsed_time / (filter_time + elapsed_time),
                        Real(0.0), Real(1.0));
                    for (size_t side = 0; side < 2; ++side)
                    {
                        raw_outward_speed[side] = std::clamp(
                            raw_outward_speed[side], -speed_limit, speed_limit);
                        filtered_contact_speed_[side] += filter_weight *
                            (raw_outward_speed[side] -
                             filtered_contact_speed_[side]);
                    }
                    previous_footprint_ = {footprint_left, footprint_right};
                    previous_contact_time_ = current_time;
                }
            }
            contact_outward_speed = filtered_contact_speed_;
            for (size_t side = 0; side < 2; ++side)
            {
                if (breakup_options.contact_angle_model == "linear-friction")
                {
                    Real capillary_number =
                        DynamicViscosity() * contact_outward_speed[side] /
                        (breakup_options.gamma_lg + TinyReal);
                    contact_target_cosine[side] = std::clamp(
                        YoungCosine() - breakup_options.contact_line_friction *
                                            capillary_number,
                        std::cos(175.0 * Pi / 180.0),
                        std::cos(5.0 * Pi / 180.0));
                }
            }
        }
        for (size_t i = 0; i < total; ++i)
        {
            Vecd filtered = normal_buffer_a_[i];
            if (filtered.norm() >= normal_threshold)
                liquid_normal_buffer_[i] =
                    filtered / (filtered.norm() + TinyReal);
            else
                liquid_normal_buffer_[i] = Vecd::Zero();
            surface_delta_[i] = raw_normal_[i].norm();
            bool local_surface_candidate =
                filtered.norm() >= normal_threshold || shepard_[i] <= 0.95;
            Real wall_distance =
                FiberNormalDistance(pos_[i]) - ParticleSpacing();
            bool explicit_footprint_endpoint = false;
            if (has_resolved_footprint &&
                footprint_left < footprint_right)
            {
                Real geometric_wall_distance = FiberNormalDistance(pos_[i]);
                Real endpoint_band = 1.25 * ParticleSpacing();
                explicit_footprint_endpoint =
                    geometric_wall_distance >= -0.25 * ParticleSpacing() &&
                    geometric_wall_distance <= 1.5 * ParticleSpacing() &&
                    (pos_[i][0] <= footprint_left + endpoint_band ||
                     pos_[i][0] >= footprint_right - endpoint_band);
                if (explicit_footprint_endpoint)
                    local_surface_candidate = true;
            }
            if (has_resolved_footprint &&
                wall_distance <= kernel_radius_ &&
                dominant_footprint_bins[
                    footprint_bin_index(pos_[i][0])] == 0)
                local_surface_candidate = false;
            bool resolved_near_wall_endpoint = false;
            if (local_surface_candidate && wall_distance <= kernel_radius_)
            {
                bool axial_neighbor_negative = false;
                bool axial_neighbor_positive = false;
                Real near_wall_height =
                    fibre_radius +
                    (breakup_options.contact_zone_in_dx + 1.0) *
                        ParticleSpacing();
                const Neighborhood &near_wall_neighborhood =
                    inner_configuration_[i];
                for (size_t n = 0;
                     n < near_wall_neighborhood.current_size_; ++n)
                {
                    size_t j = near_wall_neighborhood.j_[n];
                    if (pos_[j][1] > near_wall_height ||
                        std::abs(near_wall_neighborhood.e_ij_[n][0]) < 0.25)
                        continue;
                    axial_neighbor_negative = axial_neighbor_negative ||
                                                near_wall_neighborhood.e_ij_[n][0] < 0.0;
                    axial_neighbor_positive = axial_neighbor_positive ||
                                                near_wall_neighborhood.e_ij_[n][0] > 0.0;
                }
                // A fully coated fibre has near-wall liquid support on both
                // axial sides. A three-phase contact line has an open gas side.
                if (axial_neighbor_negative && axial_neighbor_positive &&
                    !explicit_footprint_endpoint)
                    local_surface_candidate = false;
                else if (axial_neighbor_negative || axial_neighbor_positive ||
                         explicit_footprint_endpoint)
                    resolved_near_wall_endpoint = true;
            }
            // The library indicator is reliable on a smooth outer interface,
            // but at a 90-degree dry-fibre shoulder its first free-surface
            // particle can sit more than one radial lattice interval above
            // the wall. Retain a locally resolved one-sided footprint edge as
            // a JFM contact endpoint even when the generic indicator omits it.
            if (local_surface_candidate &&
                (library_surface_indicator_[i] == 1 ||
                 resolved_near_wall_endpoint ||
                 explicit_footprint_endpoint))
            {
                if (filtered.norm() > TinyReal)
                    smooth_normal_[i] = filtered / filtered.norm();
                else
                    smooth_normal_[i] = Vecd(
                        pos_[i][0] >= 0.0 ? 1.0 : -1.0, 0.0);
                jfm_surface_indicator_[i] = 1;
            }
            else
                smooth_normal_[i] = Vecd::Zero();
            corrected_normal_[i] = smooth_normal_[i];
        }

        // The fibre surface is r=a in the meridional plane.  It is therefore
        // geometrically horizontal here, while cylindrical geometry enters
        // through the n_r/r curvature and the axisymmetric flow corrections.
        for (size_t i = 0; i < total; ++i)
        {
            if (jfm_surface_indicator_[i] == 0)
                continue;
            Real wall_distance =
                FiberNormalDistance(pos_[i]) - ParticleSpacing();
            if (wall_distance > kernel_radius_)
                continue;

            Vecd fibre_normal = FiberOutwardNormal(pos_[i][0]);
            Vecd fibre_tangent = smooth_normal_[i] -
                                 smooth_normal_[i].dot(fibre_normal) * fibre_normal;
            if (fibre_tangent.norm() <= TinyReal)
            {
                fibre_tangent = Vecd(fibre_normal[1], -fibre_normal[0]);
                if (fibre_tangent.dot(smooth_normal_[i]) < 0.0)
                    fibre_tangent = -fibre_tangent;
            }
            else
                fibre_tangent /= fibre_tangent.norm();
            size_t contact_side =
                pos_[i][0] < 0.5 * (footprint_left + footprint_right) ? 0 : 1;
            // A sealed film has no liquid-gas-solid triple junction, so the
            // near-wall band at its closed caps must not be reported as a
            // contact line.  The blended normal below is still applied: close
            // to the axis the colour-gradient normal loses the mirrored
            // support and this wall-consistent value is the reliable one.
            if (wall_exposes_gas)
            {
                contact_line_[i] = 1;
                contact_line_speed_[i] = contact_outward_speed[contact_side];
            }
            if (apply_contact_angle_correction_)
            {
                Real local_target_angle =
                    std::acos(contact_target_cosine[contact_side]);
                Vecd equilibrium_normal =
                    fibre_tangent * std::sin(local_target_angle) -
                    fibre_normal * std::cos(local_target_angle);
                Real blend_weight = std::clamp(
                    wall_distance / (kernel_radius_ + TinyReal), Real(0.0), Real(1.0));
                Vecd blended = blend_weight * smooth_normal_[i] +
                               (1.0 - blend_weight) * equilibrium_normal;
                corrected_normal_[i] = blended / (blended.norm() + TinyReal);
                dynamic_contact_cosine_[i] = contact_target_cosine[contact_side];
            }
            else
            {
                // Adhesion-based wetting retains the measured liquid-gas
                // normal.  The contact angle is therefore an outcome of the
                // wall attraction rather than a prescribed geometric input.
                corrected_normal_[i] = smooth_normal_[i];
                dynamic_contact_cosine_[i] = std::clamp(
                    -smooth_normal_[i].dot(fibre_normal), Real(-1.0), Real(1.0));
            }
        }

        // Build one common interface weight for the inward curvature traction
        // and the outward one-phase pressure-boundary traction.  Using the
        // same n, kappa, delta_s and resolution fade prevents a numerical
        // mismatch from being introduced by two different interface stencils.
        Real gauge_weight_sum = 0.0;
        Real gauge_pressure_sum = 0.0;
        Real gauge_curvature_sum = 0.0;
        for (size_t i = 0; i < total; ++i)
        {
            if (jfm_surface_indicator_[i] == 0)
                continue;
            Real axial_curvature = 0.0;
            Real normal_shepard = W0_ * Vol_[i];
            const Neighborhood &neighborhood = inner_configuration_[i];
            for (size_t n = 0; n < neighborhood.current_size_; ++n)
            {
                size_t j = neighborhood.j_[n];
                const bool paper_nonwetting =
                    breakup_options.jfm_neighbor_mode == "paper-split" &&
                    !hydrophilic;
                Vecd neighbour_normal = corrected_normal_[j];
                if (paper_nonwetting)
                {
                    // JFM (6.20)-(6.24): the non-wetting curvature uses all
                    // liquid neighbours. Interior liquid particles contribute
                    // a zero interface normal; surface/contact particles carry
                    // the liquid-only or Young-corrected normal. The correction
                    // matrix above is also assembled from liquid neighbours only.
                    neighbour_normal = contact_line_[j] == 1
                                           ? corrected_normal_[j]
                                           : liquid_normal_buffer_[j];
                }
                else if (jfm_surface_indicator_[j] == 0)
                    continue;
                Vecd corrected_gradient = correction_matrix_[i] *
                    (neighborhood.dW_ij_[n] * neighborhood.e_ij_[n]);
                axial_curvature -= 0.5 * Vol_[j] *
                    (corrected_normal_[i] - neighbour_normal)
                        .dot(corrected_gradient);
                if (!paper_nonwetting ||
                    liquid_normal_buffer_[j].norm() > TinyReal ||
                    contact_line_[j] == 1)
                {
                    normal_shepard += Vol_[j] * neighborhood.W_ij_[n];
                    ++support_neighbors_[i];
                }
            }

            bool extend_through_solid = include_solid_support;
            if (extend_through_solid && contact_line_[i] == 1)
            {
                Vecd fibre_normal = FiberOutwardNormal(pos_[i][0]);
                Vecd fibre_tangent(fibre_normal[1], -fibre_normal[0]);
                if (fibre_tangent.dot(corrected_normal_[i]) < 0.0)
                    fibre_tangent = -fibre_tangent;
                Real local_target_angle =
                    std::acos(std::clamp(dynamic_contact_cosine_[i],
                                         Real(-1.0), Real(1.0)));
                Vecd virtual_solid_normal =
                    fibre_tangent * std::sin(local_target_angle) -
                    fibre_normal * std::cos(local_target_angle);
                for (size_t k = 0; k < contact_configuration_.size(); ++k)
                {
                    // Virtual wall normals and the contact-angle extension only
                    // make sense for a physical wettable solid.  The axis
                    // symmetry plane completes the support but is not a wall.
                    if (!contact_body_is_wettable_[k])
                        continue;
                    Real *fibre_volume = fibre_Vol_[k];
                    const Neighborhood &fibre_neighborhood =
                        (*contact_configuration_[k])[i];
                    for (size_t n = 0; n < fibre_neighborhood.current_size_; ++n)
                    {
                        size_t j = fibre_neighborhood.j_[n];
                        Vecd r_ab = fibre_neighborhood.r_ij_[n] *
                                    fibre_neighborhood.e_ij_[n];
                        if (r_ab.dot(corrected_normal_[i]) < 0.0)
                            continue;
                        Vecd corrected_gradient = correction_matrix_[i] *
                            (fibre_neighborhood.dW_ij_[n] *
                             fibre_neighborhood.e_ij_[n]);
                        axial_curvature -= 0.5 * fibre_volume[j] *
                            (corrected_normal_[i] - virtual_solid_normal)
                                .dot(corrected_gradient);
                        normal_shepard +=
                            fibre_volume[j] * fibre_neighborhood.W_ij_[n];
                    }
                }
            }

            axial_curvature_[i] = axial_curvature;
            hoop_curvature_[i] = run_options.FullAxisymmetric()
                                     ? corrected_normal_[i][1] /
                                           (std::max(pos_[i][1], Real(0.25) *
                                                                    ParticleSpacing()) +
                                            TinyReal)
                                     : 0.0;
            total_curvature_[i] = axial_curvature_[i] + hoop_curvature_[i];
            curvature_shepard_buffer_[i] = normal_shepard;
        }

        // The contact endpoint has a one-sided liquid stencil.  Directly
        // differentiating its imposed Young normal makes the Bonet--Lok
        // correction amplify a particle-scale normal jump into a spurious
        // curvature spike.  Extrapolate curvature from the resolved outer
        // interface and blend it back to the raw value over one kernel radius.
        if (breakup_options.jfm_endpoint_curvature == "extrapolated" ||
            breakup_options.jfm_endpoint_curvature == "one-sided")
        {
            for (size_t i = 0; i < total; ++i)
            {
                if (contact_line_[i] == 0)
                    continue;
                Real weighted_curvature = 0.0;
                Real weight_sum = 0.0;
                const Neighborhood &neighborhood = inner_configuration_[i];
                for (size_t n = 0; n < neighborhood.current_size_; ++n)
                {
                    size_t j = neighborhood.j_[n];
                    bool inward_contact_neighbor =
                        contact_line_[j] == 1 &&
                        std::abs(pos_[j][0]) + 0.1 * ParticleSpacing() <
                            std::abs(pos_[i][0]);
                    if (jfm_surface_indicator_[j] == 0 ||
                        (contact_line_[j] == 1 && !inward_contact_neighbor) ||
                        support_neighbors_[j] <= breakup_options.support_min_neighbors)
                        continue;
                    Real weight = Vol_[j] * neighborhood.W_ij_[n];
                    weighted_curvature += weight * axial_curvature_[j];
                    weight_sum += weight;
                }
                if (weight_sum <= TinyReal)
                    continue;
                Real wall_distance = std::max(
                    FiberNormalDistance(pos_[i]) - ParticleSpacing(), Real(0.0));
                Real endpoint_weight = std::clamp(
                    1.0 - wall_distance / (kernel_radius_ + TinyReal),
                    Real(0.0), Real(1.0));
                Real extrapolated_curvature = weighted_curvature / weight_sum;
                axial_curvature_[i] =
                    endpoint_weight * extrapolated_curvature +
                    (1.0 - endpoint_weight) * axial_curvature_[i];
                total_curvature_[i] = axial_curvature_[i] + hoop_curvature_[i];
            }
        }

        // At a 90-degree contact line the first lattice column lies slightly
        // inside the mathematical endpoint and can contain several vertically
        // stacked surface particles.  Their symmetric particle stencil sees a
        // staircase rather than a smooth one-sided interface and may reverse
        // the curvature sign.  Estimate a robust total curvature on each side
        // from the first fully resolved interior band, then extend it toward
        // the endpoint.  The corrected Young normal is retained; only kappa is
        // replaced in the under-resolved band.
        if (breakup_options.jfm_endpoint_curvature == "one-sided" &&
            has_resolved_footprint && footprint_left < footprint_right)
        {
            const Real extension_width = 4.0 * ParticleSpacing();
            const Real reference_width = 6.0 * ParticleSpacing();
            std::array<std::vector<Real>, 2> reference_curvatures;
            for (size_t i = 0; i < total; ++i)
            {
                if (jfm_surface_indicator_[i] == 0 ||
                    support_neighbors_[i] < breakup_options.support_full_neighbors ||
                    !std::isfinite(total_curvature_[i]))
                    continue;
                const size_t side = pos_[i][0] < 0.0 ? 0 : 1;
                Real inward_distance = side == 0
                                           ? pos_[i][0] - footprint_left
                                           : footprint_right - pos_[i][0];
                if (inward_distance >= extension_width &&
                    inward_distance <= extension_width + reference_width)
                    reference_curvatures[side].push_back(total_curvature_[i]);
            }

            std::array<Real, 2> reference_curvature{0.0, 0.0};
            std::array<bool, 2> reference_available{false, false};
            for (size_t side = 0; side < 2; ++side)
            {
                std::vector<Real> &values = reference_curvatures[side];
                if (values.empty())
                    continue;
                std::sort(values.begin(), values.end());
                size_t middle = values.size() / 2;
                reference_curvature[side] = values.size() % 2 == 0
                                                ? 0.5 * (values[middle - 1] + values[middle])
                                                : values[middle];
                reference_available[side] = true;
            }

            for (size_t i = 0; i < total; ++i)
            {
                if (jfm_surface_indicator_[i] == 0)
                    continue;
                const size_t side = pos_[i][0] < 0.0 ? 0 : 1;
                if (!reference_available[side])
                    continue;
                Real inward_distance = side == 0
                                           ? pos_[i][0] - footprint_left
                                           : footprint_right - pos_[i][0];
                if (inward_distance < -0.5 * ParticleSpacing() ||
                    inward_distance >= extension_width)
                    continue;
                Real normalized_distance = std::clamp(
                    inward_distance / extension_width, Real(0.0), Real(1.0));
                Real raw_weight = normalized_distance * normalized_distance *
                                  (3.0 - 2.0 * normalized_distance);
                Real extended_total_curvature =
                    (1.0 - raw_weight) * reference_curvature[side] +
                    raw_weight * total_curvature_[i];
                total_curvature_[i] = extended_total_curvature;
                axial_curvature_[i] =
                    extended_total_curvature - hoop_curvature_[i];
                contact_curvature_extended_[i] = 1;
            }
        }

        // Reject only isolated curvature outliers caused by a singular local
        // normal stencil. A resolved neck or crown changes the curvature of a
        // connected band, so its local signed median changes with it and is
        // not removed by this test.
        if (breakup_options.jfm_curvature_outlier_filter)
        {
            std::vector<Real> filtered_curvature(total, 0.0);
            for (size_t i = 0; i < total; ++i)
                filtered_curvature[i] = total_curvature_[i];

            for (size_t i = 0; i < total; ++i)
            {
                if (jfm_surface_indicator_[i] == 0 ||
                    !std::isfinite(total_curvature_[i]))
                    continue;
                std::vector<Real> local_curvatures;
                local_curvatures.push_back(total_curvature_[i]);
                const Neighborhood &neighborhood = inner_configuration_[i];
                for (size_t n = 0; n < neighborhood.current_size_; ++n)
                {
                    size_t j = neighborhood.j_[n];
                    if (jfm_surface_indicator_[j] == 0 ||
                        !std::isfinite(total_curvature_[j]))
                        continue;
                    local_curvatures.push_back(total_curvature_[j]);
                }
                if (local_curvatures.size() < 5)
                    continue;

                size_t middle = local_curvatures.size() / 2;
                std::nth_element(local_curvatures.begin(),
                                 local_curvatures.begin() + middle,
                                 local_curvatures.end());
                Real local_median = local_curvatures[middle];
                Real reference = std::max(
                    std::abs(local_median),
                    0.5 * std::abs(FiberAxisymmetricCurvature(pos_[i][0])));
                if (std::abs(total_curvature_[i] - local_median) >
                    breakup_options.jfm_curvature_ratio_limit * reference)
                {
                    filtered_curvature[i] = local_median;
                    curvature_outlier_filtered_[i] = 1;
                }
            }

            for (size_t i = 0; i < total; ++i)
            {
                if (curvature_outlier_filtered_[i] == 0)
                    continue;
                total_curvature_[i] = filtered_curvature[i];
                axial_curvature_[i] =
                    total_curvature_[i] - hoop_curvature_[i];
            }
        }

        for (size_t i = 0; i < total; ++i)
        {
            const Real active_weight = IsConicalFiber()
                                           ? 1.0 - std::clamp(end_buffer_weight_[i], Real(0.0), Real(1.0))
                                           : 1.0;
            if (active_weight <= TinyReal)
                continue;
            if (jfm_surface_indicator_[i] == 0)
                continue;
            Real support_correction =
                1.0 + 1.0 / (curvature_shepard_buffer_[i] + TinyReal);
            // Curvature is not resolved once the reconstructed free-surface
            // stencil contains only a few neighbours.  Fade the JFM traction
            // continuously before that limit instead of applying a large,
            // noisy capillary impulse to an isolated particle.
            Real support_transition = std::clamp(
                (static_cast<Real>(support_neighbors_[i]) -
                 static_cast<Real>(breakup_options.support_min_neighbors)) /
                    static_cast<Real>(breakup_options.support_full_neighbors -
                                      breakup_options.support_min_neighbors),
                Real(0.0), Real(1.0));
            Real surface_confidence = support_transition * support_transition *
                                      (3.0 - 2.0 * support_transition);
            // Use the same C1 end-buffer weight as the kinematic reservoir.
            // A hard on/off force switch launches a stationary capillary
            // boundary layer at the inner edge of each buffer on a cone.
            force_weight_buffer_[i] = active_weight * surface_confidence * support_correction;
            Real gauge_weight = force_weight_buffer_[i] *
                                surface_delta_[i] * Vol_[i];
            gauge_weight_sum += gauge_weight;
            gauge_pressure_sum += gauge_weight * pressure_[i];
            gauge_curvature_sum += gauge_weight * total_curvature_[i];
        }

        Real pressure_gauge = 0.0;
        if (breakup_options.jfm_pressure_balance &&
            gauge_weight_sum > TinyReal)
            pressure_gauge = breakup_options.gamma_lg *
                                 gauge_curvature_sum / gauge_weight_sum -
                             gauge_pressure_sum / gauge_weight_sum;

        for (size_t i = 0; i < total; ++i)
        {
            if (jfm_surface_indicator_[i] == 0)
                continue;
            curvature_force_[i] =
                -force_weight_buffer_[i] * mass_[i] *
                breakup_options.gamma_lg /
                (rho_[i] + TinyReal) * total_curvature_[i] *
                corrected_normal_[i] * surface_delta_[i];
            current_force_[i] = curvature_force_[i];
            if (breakup_options.jfm_pressure_balance)
            {
                // Gas particles are absent.  The standard one-phase WCSPH
                // pressure sum therefore misses the gas-side boundary term.
                // Restore it explicitly as +(p-p_g)n delta_s.  The gauge
                // correction fixes p_g=0 without changing resolved pressure
                // differences in the liquid.
                Real boundary_pressure = pressure_[i] + pressure_gauge;
                pressure_jump_force_[i] =
                    force_weight_buffer_[i] * mass_[i] /
                    (rho_[i] + TinyReal) * boundary_pressure *
                    corrected_normal_[i] * surface_delta_[i];
                current_force_[i] += pressure_jump_force_[i];
                pressure_gauge_offset_[i] = pressure_gauge;
            }
            curvature_force_[i] *= force_scale_;
            pressure_jump_force_[i] *= force_scale_;
            current_force_[i] *= force_scale_;
            if (contact_line_[i] == 1)
                wetting_diagnostic_[i] = current_force_[i];
        }

        for (size_t i = 0; i < total; ++i)
        {
            const Real active_weight = IsConicalFiber()
                                           ? 1.0 - std::clamp(end_buffer_weight_[i], Real(0.0), Real(1.0))
                                           : 1.0;
            if (active_weight <= TinyReal)
            {
                current_force_[i] = Vecd::Zero();
                wall_adhesion_force_[i] = Vecd::Zero();
                ForcePrior::update(i, dt);
                continue;
            }
            if (apply_wall_adhesion_ &&
                breakup_options.wall_adhesion_strength > 0.0 &&
                wall_adhesion_activation_factor > 0.0)
            {
                Real wall_distance = FiberNormalDistance(pos_[i]);
                Real adhesion_range =
                    breakup_options.wall_adhesion_range_in_dx * ParticleSpacing();
                if (wall_distance >= 0.0 && wall_distance <= adhesion_range)
                {
                    Real wall_support = 0.0;
                    Vecd neighbour_direction = Vecd::Zero();
                    for (size_t k = 0; k < contact_configuration_.size(); ++k)
                    {
                        // Adhesion is a solid-wall interaction, so the axis
                        // symmetry plane must stay out of it.
                        if (!contact_body_is_wettable_[k])
                            continue;
                        Real *fibre_volume = fibre_Vol_[k];
                        const Neighborhood &fibre_neighborhood =
                            (*contact_configuration_[k])[i];
                        for (size_t n = 0; n < fibre_neighborhood.current_size_; ++n)
                        {
                            if (fibre_neighborhood.r_ij_[n] > adhesion_range)
                                continue;
                            Real pair_weight =
                                fibre_volume[fibre_neighborhood.j_[n]] *
                                std::max(fibre_neighborhood.W_ij_[n], Real(0.0));
                            Real q_pair = fibre_neighborhood.r_ij_[n] /
                                          (adhesion_range + TinyReal);
                            Real pair_profile = std::max(Real(0.0), 1.0 - q_pair);
                            pair_profile *= pair_profile * (1.0 + 2.0 * q_pair);
                            if (breakup_options.wall_adhesion_profile == "pairwise")
                                pair_weight *= pair_profile;
                            wall_support += pair_weight;

                            Vecd toward_solid = -fibre_neighborhood.e_ij_[n];
                            const Vecd nominal_wall_direction =
                                -FiberOutwardNormal(pos_[i][0]);
                            if (toward_solid.dot(nominal_wall_direction) < 0.0)
                                toward_solid = -toward_solid;
                            neighbour_direction += pair_weight * toward_solid;
                        }
                    }
                    wall_support = std::clamp(wall_support, Real(0.0), Real(1.0));
                    Real q = wall_distance / (adhesion_range + TinyReal);
                    Real distance_weight = 0.5 * (1.0 + std::cos(Pi * q));
                    Vecd wall_inward_normal =
                        -FiberOutwardNormal(pos_[i][0]);
                    if (neighbour_direction.norm() > TinyReal)
                        wall_inward_normal =
                            neighbour_direction / neighbour_direction.norm();
                    Real interaction_weight =
                        breakup_options.wall_adhesion_profile == "pairwise"
                            ? 1.0
                            : distance_weight;
                    Real particle_volume = mass_[i] / (rho_[i] + TinyReal);
                    wall_adhesion_force_[i] =
                        active_weight * particle_volume * breakup_options.gamma_lg *
                        breakup_options.wall_adhesion_strength * wall_support *
                        interaction_weight * wall_adhesion_activation_factor /
                        ((ParticleSpacing() * ParticleSpacing()) + TinyReal) *
                        wall_inward_normal;
                    current_force_[i] += wall_adhesion_force_[i];
                }
            }
            ForcePrior::update(i, dt);
        }
    }

  private:
    Vecd *pos_, *vel_, *raw_normal_, *smooth_normal_, *corrected_normal_,
        *wetting_diagnostic_, *wall_adhesion_force_, *solid_liquid_force_, *curvature_force_,
        *pressure_jump_force_;
    Real *rho_, *pressure_, *mass_, *Vol_, *shepard_, *surface_delta_,
        *axial_curvature_, *hoop_curvature_, *total_curvature_,
        *dynamic_contact_cosine_, *contact_line_speed_, *pressure_gauge_offset_,
        *end_buffer_weight_;
    Matd *correction_matrix_;
    int *library_surface_indicator_, *jfm_surface_indicator_, *contact_line_,
        *contact_curvature_extended_, *curvature_outlier_filtered_,
        *support_neighbors_,
        *pca_normal_used_;
    Real *physical_time_;
    StdVec<Real *> fibre_Vol_;
    StdVec<bool> contact_body_is_wettable_;
    StdVec<Vecd> normal_buffer_a_, normal_buffer_b_, liquid_normal_buffer_;
    StdVec<Real> curvature_shepard_buffer_, force_weight_buffer_;
    Real W0_, smoothing_length_, kernel_radius_;
    bool apply_contact_angle_correction_, apply_wall_adhesion_;
    bool contact_history_initialized_;
    Real previous_contact_time_;
    std::array<Real, 2> previous_footprint_, filtered_contact_speed_;
    Real force_scale_;
};

/**
 * Xu et al. (Applied Mathematical Modelling 83, 2020) represent the
 * attractive van der Waals interaction as a surface traction
 *
 *     Pi(H) = A / (6 pi H^3)
 *
 * on the outermost liquid particles and spread it over one particle layer.
 * Here H is the radial distance from the rigid fibre.  The dimensionless
 * input N_A = A/(6 pi gamma_lg a^2) gives Pi/(gamma_lg/a)=N_A/(H/a)^3.
 * The singularity is regularised only below a user-visible resolution cutoff;
 * this is necessary because the continuum expression diverges at rupture.
 */
class XuVanDerWaalsForce : public ForcePrior, public DataDelegateInner
{
  public:
    explicit XuVanDerWaalsForce(BaseInnerRelation &inner_relation)
        : ForcePrior(inner_relation.getSPHBody(), "XuVanDerWaalsForce"),
          DataDelegateInner(inner_relation),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          rho_(particles_->getVariableDataByName<Real>("Density")),
          mass_(particles_->getVariableDataByName<Real>("Mass")),
          Vol_(particles_->getVariableDataByName<Real>("VolumetricMeasure")),
          indicator_(particles_->getVariableDataByName<int>("Indicator")),
          surface_indicator_(particles_->registerStateVariableData<int>("XuSurfaceIndicator")),
          surface_weight_(particles_->registerStateVariableData<Real>("XuSurfaceWeight")),
          film_gap_(particles_->registerStateVariableData<Real>("XuFilmGap")),
          disjoining_pressure_(particles_->registerStateVariableData<Real>("XuDisjoiningPressure")),
          resolution_taper_(particles_->registerStateVariableData<Real>("XuResolutionTaper")),
          end_buffer_weight_(particles_->registerStateVariableData<Real>("ConicalEndBufferWeight")),
          physical_time_(sph_system_->svPhysicalTime().Data()),
          maximum_acceleration_(0.0), enabled_(true),
          post_breakup_ramp_started_(false), post_breakup_ramp_start_T_(0.0)
    {
        particles_->addEvolvingVariable<int>("XuSurfaceIndicator");
        particles_->addEvolvingVariable<Real>("XuSurfaceWeight");
        particles_->addEvolvingVariable<Real>("XuFilmGap");
        particles_->addEvolvingVariable<Real>("XuDisjoiningPressure");
        particles_->addEvolvingVariable<Real>("XuResolutionTaper");
        particles_->addVariableToWrite<int>("XuSurfaceIndicator");
        particles_->addVariableToWrite<Real>("XuSurfaceWeight");
        particles_->addVariableToWrite<Real>("XuFilmGap");
        particles_->addVariableToWrite<Real>("XuDisjoiningPressure");
        particles_->addVariableToWrite<Real>("XuResolutionTaper");
        particles_->addVariableToWrite<Vecd>("XuVanDerWaalsForce");
    }

    void exec(Real dt = 0.0)
    {
        const size_t total = particles_->TotalRealParticles();
        const Real dx = ParticleSpacing();
        if (!enabled_)
        {
            maximum_acceleration_ = 0.0;
            for (size_t i = 0; i < total; ++i)
            {
                current_force_[i] = Vecd::Zero();
                surface_indicator_[i] = 0;
                surface_weight_[i] = 0.0;
                film_gap_[i] = std::max(FiberNormalDistance(pos_[i]), Real(0.0));
                disjoining_pressure_[i] = 0.0;
                resolution_taper_[i] = 0.0;
                ForcePrior::update(i, dt);
            }
            return;
        }
        const Real minimum_gap =
            breakup_options.van_der_waals_min_gap_in_dx * dx;
        const Real threshold =
            breakup_options.van_der_waals_surface_threshold / dx;
        const size_t bins = std::max<size_t>(
            32, static_cast<size_t>(std::round(DomainLength() / dx)));
        std::vector<Real> outer_radius(
            bins, -std::numeric_limits<Real>::infinity());
        std::vector<Real> raw_surface_weight(total, 0.0);

        for (size_t i = 0; i < total; ++i)
        {
            current_force_[i] = Vecd::Zero();
            surface_indicator_[i] = 0;
            surface_weight_[i] = 0.0;
            film_gap_[i] = std::max(FiberNormalDistance(pos_[i]), Real(0.0));
            disjoining_pressure_[i] = 0.0;
            resolution_taper_[i] = 1.0;
            if (breakup_options.van_der_waals_model != "xu" ||
                indicator_[i] != 1 || FiberNormalDistance(pos_[i]) <= 0.25 * dx)
                continue;

            // Xu et al. evaluate the surface weighting function with a
            // dedicated smaller smoothing length h2 = dx, rather than with
            // the bulk-flow kernel.  For the two-dimensional Wendland C2
            // kernel used in their paper,
            //
            //   W = 7/(64*pi*h2^2) (2-q)^4 (1+2q),  0 <= q < 2,
            //
            // and dW/dr = -70/(64*pi*h2^3) q(2-q)^3.  The existing inner
            // neighbour list has a wider support and therefore contains all
            // neighbours required by this dedicated h2=dx evaluation.
            const Neighborhood &neighborhood = inner_configuration_[i];
            Matd local_configuration = Matd::Zero();
            for (size_t n = 0; n < neighborhood.current_size_; ++n)
            {
                const size_t j = neighborhood.j_[n];
                Real q = neighborhood.r_ij_[n] / dx;
                if (q >= 2.0)
                    continue;
                Real two_minus_q = 2.0 - q;
                Real dW_dr = -70.0 / (64.0 * Pi * dx * dx * dx) *
                             q * two_minus_q * two_minus_q * two_minus_q;
                Vecd gradient_W = dW_dr * neighborhood.e_ij_[n];
                Vecd r_ji = neighborhood.r_ij_[n] *
                            neighborhood.e_ij_[n];
                local_configuration -=
                    Vol_[j] * r_ji * gradient_W.transpose();
            }
            Matd correction = inverseTikhonov(local_configuration, SqrtEps);
            Vecd concentration_gradient = Vecd::Zero();
            for (size_t n = 0; n < neighborhood.current_size_; ++n)
            {
                const size_t j = neighborhood.j_[n];
                Real q = neighborhood.r_ij_[n] / dx;
                if (q >= 2.0)
                    continue;
                Real two_minus_q = 2.0 - q;
                Real dW_dr = -70.0 / (64.0 * Pi * dx * dx * dx) *
                             q * two_minus_q * two_minus_q * two_minus_q;
                Vecd corrected_gradient = correction *
                    (dW_dr * neighborhood.e_ij_[n]);
                concentration_gradient -= Vol_[j] * corrected_gradient;
            }
            raw_surface_weight[i] = concentration_gradient.norm();
            if (raw_surface_weight[i] <= threshold ||
                concentration_gradient[1] <= 0.0)
                continue;

            Real shifted = pos_[i][0] - DomainLowerBound();
            Real periodic_x = shifted -
                              std::floor(shifted / DomainLength()) *
                                  DomainLength();
            size_t bin = std::min(
                bins - 1, static_cast<size_t>(std::floor(
                              periodic_x / DomainLength() *
                              static_cast<Real>(bins))));
            outer_radius[bin] = std::max(outer_radius[bin], pos_[i][1]);
        }

        // The outermost lattice row contains particle-scale height steps.
        // Smooth only the height used by the molecular traction; the particle
        // coordinates and the capillary interface remain untouched.  Invalid
        // (dry) bins stay invalid, so this operation cannot bridge a dry gap.
        std::vector<Real> smoothed_outer_radius = outer_radius;
        for (int pass = 0;
             pass < breakup_options.van_der_waals_gap_smoothing_passes;
             ++pass)
        {
            std::vector<Real> next = smoothed_outer_radius;
            for (size_t b = 0; b < bins; ++b)
            {
                if (!std::isfinite(smoothed_outer_radius[b]))
                    continue;
                Real weighted_radius = 2.0 * smoothed_outer_radius[b];
                Real weight = 2.0;
                size_t left = (b + bins - 1) % bins;
                size_t right = (b + 1) % bins;
                if (std::isfinite(smoothed_outer_radius[left]))
                {
                    weighted_radius += smoothed_outer_radius[left];
                    weight += 1.0;
                }
                if (std::isfinite(smoothed_outer_radius[right]))
                {
                    weighted_radius += smoothed_outer_radius[right];
                    weight += 1.0;
                }
                next[b] = weighted_radius / weight;
            }
            smoothed_outer_radius.swap(next);
        }

        Real ramp = 1.0;
        if (!breakup_options.van_der_waals_initial_pressure &&
            breakup_options.van_der_waals_ramp_T > TinyReal)
        {
            Real T = breakup_options.gamma_lg * (*physical_time_) /
                     (DynamicViscosity() * fibre_radius + TinyReal);
            Real phase = std::min(Real(1.0),
                                  T / breakup_options.van_der_waals_ramp_T);
            ramp = 0.5 * (1.0 - std::cos(Pi * phase));
        }
        if (post_breakup_ramp_started_ &&
            breakup_options.van_der_waals_post_breakup_ramp_T > TinyReal)
        {
            Real current_T = breakup_options.gamma_lg * (*physical_time_) /
                             (DynamicViscosity() * fibre_radius + TinyReal);
            Real phase = std::clamp(
                (current_T - post_breakup_ramp_start_T_) /
                    breakup_options.van_der_waals_post_breakup_ramp_T,
                Real(0.0), Real(1.0));
            // C1-continuous decay: one at the naturally detected rupture and
            // zero after the user-visible hand-over interval.
            Real smooth_phase = phase * phase * (3.0 - 2.0 * phase);
            ramp *= 1.0 - smooth_phase;
        }

        maximum_acceleration_ = 0.0;
        for (size_t i = 0; i < total; ++i)
        {
            const Real active_weight = IsConicalFiber()
                                           ? 1.0 - std::clamp(end_buffer_weight_[i], Real(0.0), Real(1.0))
                                           : 1.0;
            if (active_weight <= TinyReal)
            {
                ForcePrior::update(i, dt);
                continue;
            }
            if (raw_surface_weight[i] <= threshold)
            {
                ForcePrior::update(i, dt);
                continue;
            }
            Real shifted = pos_[i][0] - DomainLowerBound();
            Real periodic_x = shifted -
                              std::floor(shifted / DomainLength()) *
                                  DomainLength();
            size_t bin = std::min(
                bins - 1, static_cast<size_t>(std::floor(
                              periodic_x / DomainLength() *
                              static_cast<Real>(bins))));
            if (!std::isfinite(outer_radius[bin]) ||
                pos_[i][1] < outer_radius[bin] - 0.55 * dx)
            {
                ForcePrior::update(i, dt);
                continue;
            }

            surface_indicator_[i] = 1;
            // On a regular flat lattice this tends to 1/dx.  Retaining the
            // measured |grad Gamma| is essential on a deformed surface: it
            // compensates for locally sparse or crowded surface particles so
            // the integrated traction remains approximately independent of
            // the instantaneous particle arrangement.
            surface_weight_[i] = active_weight * raw_surface_weight[i];
            Real bin_x = DomainLowerBound() +
                         (static_cast<Real>(bin) + 0.5) * DomainLength() /
                             static_cast<Real>(bins);
            Real reconstructed_gap =
                std::max(FiberNormalDistance(
                             Vecd(bin_x, smoothed_outer_radius[bin])),
                         Real(0.0)) +
                0.5 * dx;
            // outer_radius stores particle centres; the molecular gap is the
            // reconstructed interface position, half a spacing farther out.
            film_gap_[i] = reconstructed_gap;
            Real effective_gap = std::max(reconstructed_gap, minimum_gap);
            Real dimensionless_gap = effective_gap / fibre_radius;
            Real pressure_scale = breakup_options.gamma_lg / fibre_radius;
            Real dimensionless_pressure =
                breakup_options.van_der_waals_number /
                (dimensionless_gap * dimensionless_gap *
                     dimensionless_gap + TinyReal);
            if (breakup_options.van_der_waals_pressure_gauge == "mean-film")
            {
                Real reference_gap = FilmThickness() / fibre_radius;
                dimensionless_pressure -=
                    breakup_options.van_der_waals_number /
                    (reference_gap * reference_gap * reference_gap + TinyReal);
            }
            Real pressure = ramp * pressure_scale * dimensionless_pressure;
            if (breakup_options.van_der_waals_resolution_taper_in_dx >
                breakup_options.van_der_waals_min_gap_in_dx)
            {
                Real taper_start =
                    breakup_options.van_der_waals_resolution_taper_in_dx * dx;
                Real phase = std::clamp(
                    (reconstructed_gap - minimum_gap) /
                        (taper_start - minimum_gap + TinyReal),
                    Real(0.0), Real(1.0));
                resolution_taper_[i] =
                    phase * phase * (3.0 - 2.0 * phase);
                pressure *= resolution_taper_[i];
            }
            disjoining_pressure_[i] = pressure;
            Vecd inward_fibre_normal = -FiberOutwardNormal(pos_[i][0]);
            Vecd acceleration = inward_fibre_normal * pressure *
                                surface_weight_[i] /
                                (rho_[i] + TinyReal);
            current_force_[i] = mass_[i] * acceleration;
            maximum_acceleration_ =
                std::max(maximum_acceleration_, acceleration.norm());
            ForcePrior::update(i, dt);
        }
    }

    Real stableTimeStep() const
    {
        if (!enabled_ || breakup_options.van_der_waals_model != "xu" ||
            maximum_acceleration_ <= TinyReal)
            return std::numeric_limits<Real>::max();
        return breakup_options.van_der_waals_cfl *
               std::sqrt(ParticleSpacing() /
                         (maximum_acceleration_ + TinyReal));
    }

    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool enabled() const { return enabled_; }

    void startPostBreakupRamp()
    {
        if (post_breakup_ramp_started_ ||
            breakup_options.van_der_waals_post_breakup_ramp_T <= TinyReal)
            return;
        post_breakup_ramp_started_ = true;
        post_breakup_ramp_start_T_ =
            breakup_options.gamma_lg * (*physical_time_) /
            (DynamicViscosity() * fibre_radius + TinyReal);
        std::cout << "Xu post-breakup traction fade started at T="
                  << post_breakup_ramp_start_T_ << " over ramp T="
                  << breakup_options.van_der_waals_post_breakup_ramp_T
                  << std::endl;
    }

    bool postBreakupRampStarted() const { return post_breakup_ramp_started_; }

  private:
    Vecd *pos_;
    Real *rho_, *mass_, *Vol_;
    int *indicator_, *surface_indicator_;
    Real *surface_weight_, *film_gap_, *disjoining_pressure_,
        *resolution_taper_, *end_buffer_weight_;
    Real *physical_time_;
    Real maximum_acceleration_;
    bool enabled_;
    bool post_breakup_ramp_started_;
    Real post_breakup_ramp_start_T_;
};

/** Runtime selector which preserves both legacy capillary implementations. */
class SelectableAxisymmetricCapillaryForce
{
  public:
    SelectableAxisymmetricCapillaryForce(BaseInnerRelation &inner_relation,
                                         BaseContactRelation &fibre_contact_relation,
                                         Real *physical_time)
        : particles_(&inner_relation.getSPHBody().getBaseParticles()),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          mass_(particles_->getVariableDataByName<Real>("Mass")),
          physical_time_(physical_time),
          capillary_force_(nullptr),
          reconstructed_capillary_force_(nullptr),
          jfm_surface_indicator_(nullptr),
          jfm_support_neighbors_(nullptr),
          jfm_contact_line_(nullptr),
          jfm_contact_curvature_extended_(nullptr),
          hybrid_use_jfm_(breakup_options.initial_topology == "seeded-gap"),
          hybrid_transition_started_(hybrid_use_jfm_),
          hybrid_transition_start_time_(*physical_time_),
          hybrid_last_transition_time_(*physical_time_),
          hybrid_resolved_transition_T_(0.0),
          hybrid_support_hold_reported_(false)
    {
        // Keep this diagnostic variable available for every wetting branch;
        // the wall-adhesion branch fills it and the JFM branches leave it
        // zero.
        particles_->registerStateVariableData<Vecd>("WallAdhesionForce");
        particles_->addEvolvingVariable<Vecd>("WallAdhesionForce");
        particles_->addVariableToWrite<Vecd>("WallAdhesionForce");
        if (breakup_options.wetting_model == "jfm")
        {
            jfm_force_ = std::make_unique<JfmAxisymmetricFiberWettingForce>(
                inner_relation, fibre_contact_relation, true, false);
        }
        else if (breakup_options.wetting_model == "hybrid" ||
                 breakup_options.wetting_model == "hybrid-adhesion")
        {
            reconstructed_force_ =
                std::make_unique<ReconstructedAxisymmetricCapillaryForce>(
                    inner_relation.getSPHBody());
            const bool adhesion_wetting =
                breakup_options.wetting_model == "hybrid-adhesion";
            jfm_force_ = std::make_unique<JfmAxisymmetricFiberWettingForce>(
                inner_relation, fibre_contact_relation,
                !adhesion_wetting, adhesion_wetting);
        }
        else
            legacy_force_ =
                std::make_unique<LocalAxisymmetricSurfaceStress>(
                    inner_relation, fibre_contact_relation);
        if (jfm_force_)
        {
            jfm_surface_indicator_ =
                particles_->getVariableDataByName<int>("JfmSurfaceIndicator");
            jfm_support_neighbors_ =
                particles_->getVariableDataByName<int>("SurfaceSupportNeighbors");
            jfm_contact_line_ =
                particles_->getVariableDataByName<int>("ContactLineIndicator");
            jfm_contact_curvature_extended_ = particles_->getVariableDataByName<int>(
                "JfmContactCurvatureExtended");
        }
        capillary_force_ =
            particles_->getVariableDataByName<Vecd>("TopologyCapillaryForce");
        if (reconstructed_force_)
            reconstructed_capillary_force_ =
                particles_->getVariableDataByName<Vecd>(
                    "ReconstructedCapillaryForce");
    }

    void exec(Real dt = 0.0)
    {
        if (reconstructed_force_)
        {
            if (breakup_options.post_breakup_models &&
                !hybrid_use_jfm_ && !hybrid_transition_started_ &&
                (HasHybridTransitionDryInterval() ||
                 ShouldStartConicalLocalTransition()))
            {
                hybrid_transition_started_ = true;
                hybrid_transition_start_time_ = *physical_time_;
                hybrid_last_transition_time_ = *physical_time_;
                hybrid_resolved_transition_T_ = 0.0;
                std::cout << "Hybrid capillary transition started after a kernel-resolved dry interval appeared."
                          << std::endl;
            }
            if (hybrid_use_jfm_)
            {
                jfm_force_->setForceScale(1.0);
                jfm_force_->exec(dt);
            }
            else if (hybrid_transition_started_)
            {
                // A dry interval may disappear while the two capillary
                // discretizations are being blended.  Completing the switch
                // after rewetting would apply the contact-line JFM stencil to
                // a fully coated fibre, even though no contact line exists.
                // Cancel the transition and explicitly clear the JFM force
                // history in that case.
                if (!IsConicalFiber() && !HasHybridTransitionDryInterval())
                {
                    reconstructed_force_->setForceScale(1.0);
                    jfm_force_->setForceScale(0.0);
                    reconstructed_force_->exec(dt);
                    jfm_force_->exec(dt);
                    hybrid_transition_started_ = false;
                    hybrid_resolved_transition_T_ = 0.0;
                    std::cout << "Hybrid capillary transition cancelled because the resolved dry interval disappeared."
                              << std::endl;
                    return;
                }
                // Do not let the contact-line formulation acquire a finite
                // force weight while its free-surface stencil still contains
                // support-deficient particles.  Such particles do not carry
                // a trustworthy normal/curvature and previously produced the
                // terminal force spike.  JFM is still executed at zero scale
                // so its geometric state can become resolved naturally.
                Real incremental_T = breakup_options.gamma_lg *
                                     std::max(Real(0.0),
                                              *physical_time_ - hybrid_last_transition_time_) /
                                     (DynamicViscosity() * fibre_radius + TinyReal);
                hybrid_last_transition_time_ = *physical_time_;
                bool resolved_jfm_surface = HasResolvedJfmSurface();
                if (resolved_jfm_surface)
                    hybrid_resolved_transition_T_ += incremental_T;

                Real phase = std::clamp(
                    hybrid_resolved_transition_T_ /
                        breakup_options.hybrid_transition_T,
                    Real(0.0), Real(1.0));
                Real blend = phase * phase * (3.0 - 2.0 * phase);

                if (!resolved_jfm_surface)
                {
                    // Pause, rather than erase, the resolved transition time.
                    // The current blend is retained to avoid a force jump;
                    // support-deficient JFM particles already carry zero local
                    // force weight inside JfmAxisymmetricFiberWettingForce.
                    reconstructed_force_->setForceScale(1.0 - blend);
                    jfm_force_->setForceScale(blend);
                    reconstructed_force_->exec(dt);
                    jfm_force_->exec(dt);
                    if (!hybrid_support_hold_reported_)
                    {
                        std::cout << "Hybrid capillary transition paused without discarding resolved progress because the JFM surface stencil is under-resolved."
                                  << std::endl;
                        hybrid_support_hold_reported_ = true;
                    }
                    return;
                }
                hybrid_support_hold_reported_ = false;
                reconstructed_force_->setForceScale(1.0 - blend);
                jfm_force_->setForceScale(blend);
                reconstructed_force_->exec(dt);
                jfm_force_->exec(dt);
                if (phase >= 1.0 - SqrtEps)
                {
                    // Execute the reconstructed contribution once at exactly
                    // zero scale so ForcePrior removes its remaining history.
                    reconstructed_force_->setForceScale(0.0);
                    reconstructed_force_->exec(dt);
                    hybrid_use_jfm_ = true;
                    std::cout << "Hybrid capillary transition completed; local JFM is now active."
                              << std::endl;
                }
            }
            else
            {
                reconstructed_force_->setForceScale(1.0);
                reconstructed_force_->exec(dt);
            }
        }
        else if (jfm_force_)
            jfm_force_->exec(dt);
        else
            legacy_force_->exec(dt);
    }

    Real stableTimeStep() const
    {
        Real maximum_acceleration = 0.0;
        for (size_t i = 0; i < particles_->TotalRealParticles(); ++i)
        {
            Vecd combined_force = capillary_force_[i];
            if (reconstructed_capillary_force_ != nullptr)
                combined_force += reconstructed_capillary_force_[i];
            maximum_acceleration = std::max(
                maximum_acceleration,
                combined_force.norm() / (mass_[i] + TinyReal));
        }
        if (maximum_acceleration <= TinyReal)
            return std::numeric_limits<Real>::max();
        return breakup_options.capillary_cfl *
               std::sqrt(ParticleSpacing() /
                         (maximum_acceleration + TinyReal));
    }

    bool isJfmActive() const
    {
        if (breakup_options.wetting_model == "jfm")
            return true;
        return (breakup_options.wetting_model == "hybrid" ||
                breakup_options.wetting_model == "hybrid-adhesion") &&
               hybrid_use_jfm_;
    }

  private:
    bool ShouldStartConicalLocalTransition() const
    {
        if (!IsConicalFiber())
            return false;
        const size_t bins = std::max<size_t>(
            32, static_cast<size_t>(std::round(
                    ConeSectionLength() / ParticleSpacing())));
        std::vector<Real> thickness(
            bins, -std::numeric_limits<Real>::infinity());
        for (size_t i = 0; i < particles_->TotalRealParticles(); ++i)
        {
            const Real x = pos_[i][0];
            if (x <= ConeLowerBound() || x >= ConeUpperBound() ||
                InEndBuffer(x))
                continue;
            Real fraction = (x - ConeLowerBound()) /
                            (ConeSectionLength() + TinyReal);
            size_t bin = std::min(
                bins - 1, static_cast<size_t>(std::floor(
                              fraction * static_cast<Real>(bins))));
            thickness[bin] = std::max(
                thickness[bin], FiberNormalDistance(pos_[i]) +
                                    0.5 * ParticleSpacing());
        }
        Real minimum = std::numeric_limits<Real>::max();
        Real maximum = 0.0;
        std::vector<Real> valid_thickness;
        valid_thickness.reserve(bins);
        size_t valid = 0;
        for (Real value : thickness)
            if (std::isfinite(value))
            {
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
                valid_thickness.push_back(value);
                ++valid;
            }
        if (valid < bins / 2)
            return false;
        // The Fourier graph is accurate and quiet during linear growth.  On
        // a cone, however, both the physical axial drainage and isolated
        // acoustic particle-column excursions make the raw global max-min
        // range an unreliable hand-over signal.  Use a spatially robust
        // 5--95 percentile span, plus a separate finite-width neck test.
        std::sort(valid_thickness.begin(), valid_thickness.end());
        const size_t p05_index = static_cast<size_t>(std::floor(
            0.05 * static_cast<Real>(valid_thickness.size() - 1)));
        const size_t p95_index = static_cast<size_t>(std::floor(
            0.95 * static_cast<Real>(valid_thickness.size() - 1)));
        const Real robust_span =
            valid_thickness[p95_index] - valid_thickness[p05_index];
        (void)minimum;
        (void)maximum;
        const Real neck_threshold = 0.70 * FilmThickness();
        const size_t required_run = static_cast<size_t>(
            std::max(3, breakup_options.hybrid_min_dry_bins));
        size_t current_run = 0;
        size_t longest_run = 0;
        for (Real value : thickness)
        {
            if (std::isfinite(value) && value < neck_threshold)
            {
                ++current_run;
                longest_run = std::max(longest_run, current_run);
            }
            else
                current_run = 0;
        }
        return longest_run >= required_run ||
               robust_span > 0.45 * FilmThickness();
    }

    bool HasHybridTransitionDryInterval() const
    {
        if (breakup_options.wetting_model == "hybrid-adhesion" &&
            breakup_options.wall_adhesion_activation == "after-rupture" &&
            !wall_adhesion_post_breakup_active)
            return false;
        // Once the topology closure has accepted breakup and the caller has
        // explicitly requested dry-core protection, the dry interval is a
        // persistent topology state.  Do not let a one-step contact-stencil
        // miss repeatedly cancel and restart the capillary transition.
        const bool latched_protected_breakup =
            breakup_options.wetting_model == "hybrid-adhesion" &&
            breakup_options.wall_adhesion_activation == "after-rupture" &&
            breakup_options.post_jfm_keep_dry_core &&
            wall_adhesion_post_breakup_active;
        return latched_protected_breakup || HasResolvedDryInterval();
    }

    bool HasResolvedDryInterval() const
    {
        size_t bins = std::max<size_t>(
            24, static_cast<size_t>(std::round(
                    DomainLength() / (2.0 * ParticleSpacing()))));
        std::vector<int> wet(bins, 0);
        for (size_t i = 0; i < particles_->TotalRealParticles(); ++i)
        {
            if (FiberNormalDistance(pos_[i]) >
                (breakup_options.contact_zone_in_dx + 0.75) *
                    ParticleSpacing())
                continue;
            Real shifted = pos_[i][0] - DomainLowerBound();
            Real periodic_x = shifted -
                              std::floor(shifted / DomainLength()) *
                                  DomainLength();
            size_t bin = std::min(
                bins - 1, static_cast<size_t>(std::floor(
                              periodic_x / DomainLength() *
                              static_cast<Real>(bins))));
            wet[bin] = 1;
        }
        auto wet_anchor = std::find(wet.begin(), wet.end(), 1);
        if (wet_anchor == wet.end())
            return true;
        size_t anchor = static_cast<size_t>(wet_anchor - wet.begin());
        size_t dry_run = 0;
        size_t maximum_dry_run = 0;
        for (size_t offset = 1; offset <= bins; ++offset)
        {
            size_t bin = (anchor + offset) % bins;
            if (wet[bin] == 0)
            {
                ++dry_run;
                maximum_dry_run = std::max(maximum_dry_run, dry_run);
            }
            else
                dry_run = 0;
        }
        return maximum_dry_run >=
               static_cast<size_t>(breakup_options.hybrid_min_dry_bins);
    }

    bool HasResolvedJfmSurface() const
    {
        if (jfm_surface_indicator_ == nullptr ||
            jfm_support_neighbors_ == nullptr ||
            jfm_contact_line_ == nullptr ||
            jfm_contact_curvature_extended_ == nullptr)
            return false;
        size_t left_contact_particles = 0;
        size_t right_contact_particles = 0;
        size_t left_resolved_particles = 0;
        size_t right_resolved_particles = 0;
        for (size_t i = 0; i < particles_->TotalRealParticles(); ++i)
            if (jfm_surface_indicator_[i] == 1 &&
                jfm_contact_line_[i] == 1)
            {
                const bool resolved_endpoint =
                    jfm_support_neighbors_[i] >
                        breakup_options.support_full_neighbors ||
                    jfm_contact_curvature_extended_[i] == 1;
                if (pos_[i][0] < 0.0)
                {
                    ++left_contact_particles;
                    if (resolved_endpoint)
                        ++left_resolved_particles;
                }
                else
                {
                    ++right_contact_particles;
                    if (resolved_endpoint)
                        ++right_resolved_particles;
                }
            }
        return left_contact_particles > 0 && right_contact_particles > 0 &&
               left_resolved_particles > 0 && right_resolved_particles > 0;
    }

    BaseParticles *particles_;
    Vecd *pos_;
    Real *mass_;
    Real *physical_time_;
    Vecd *capillary_force_;
    Vecd *reconstructed_capillary_force_;
    int *jfm_surface_indicator_;
    int *jfm_support_neighbors_;
    int *jfm_contact_line_;
    int *jfm_contact_curvature_extended_;
    bool hybrid_use_jfm_;
    bool hybrid_transition_started_;
    Real hybrid_transition_start_time_;
    Real hybrid_last_transition_time_;
    Real hybrid_resolved_transition_T_;
    bool hybrid_support_hold_reported_;
    std::unique_ptr<JfmAxisymmetricFiberWettingForce> jfm_force_;
    std::unique_ptr<LocalAxisymmetricSurfaceStress> legacy_force_;
    std::unique_ptr<ReconstructedAxisymmetricCapillaryForce> reconstructed_force_;
};

/**
 * Re-distribute free-surface particles only along the local interface tangent.
 * The normal component of the standard kernel-gradient position correction is
 * removed, so this improves particle spacing without deliberately shrinking or
 * expanding the represented liquid/free-space interface. At the numerical
 * contact line the correction is restricted to the fibre-axis direction.
 */
class TangentialSurfaceRegularization : public LocalDynamics, public DataDelegateInner
{
  public:
    explicit TangentialSurfaceRegularization(BaseInnerRelation &inner_relation)
        : LocalDynamics(inner_relation.getSPHBody()), DataDelegateInner(inner_relation),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          Vol_(particles_->getVariableDataByName<Real>("VolumetricMeasure")),
          indicator_(particles_->getVariableDataByName<int>("Indicator")),
          normal_(particles_->getVariableDataByName<Vecd>("TopologyNormal")),
          contact_line_(particles_->getVariableDataByName<int>("ContactLineIndicator")),
          shift_(particles_->registerStateVariableData<Vecd>("SurfaceRegularizationShift"))
    {
        particles_->addEvolvingVariable<Vecd>("SurfaceRegularizationShift");
        particles_->addVariableToWrite<Vecd>("SurfaceRegularizationShift");
    }

    void interaction(size_t i, Real dt = 0.0)
    {
        shift_[i] = Vecd::Zero();
        if (indicator_[i] != 1 || normal_[i].squaredNorm() < TinyReal)
            return;
        const Neighborhood &neighborhood = inner_configuration_[i];
        if (neighborhood.current_size_ < 4)
            return;

        Vecd inconsistency = Vecd::Zero();
        for (size_t n = 0; n < neighborhood.current_size_; ++n)
        {
            const size_t j = neighborhood.j_[n];
            inconsistency -= 2.0 * neighborhood.dW_ij_[n] * Vol_[j] *
                             neighborhood.e_ij_[n];
        }

        Vecd tangent_correction = inconsistency -
                                  normal_[i] * inconsistency.dot(normal_[i]);
        if (contact_line_[i] == 1)
            tangent_correction[1] = 0.0;
        Vecd proposed = breakup_options.surface_regularization_coefficient *
                        ParticleSpacing() * ParticleSpacing() * tangent_correction;
        Real maximum_shift = breakup_options.surface_max_shift_in_dx * ParticleSpacing();
        Real magnitude = proposed.norm();
        if (magnitude > maximum_shift)
            proposed *= maximum_shift / (magnitude + TinyReal);
        shift_[i] = proposed;
    }

    void update(size_t i, Real dt = 0.0)
    {
        pos_[i] += shift_[i];
    }

  private:
    Vecd *pos_, *normal_, *shift_;
    Real *Vol_;
    int *indicator_, *contact_line_;
};

struct TopologyMetrics
{
    Real dry_fraction = 0.0;
    Real central_gap_width = 0.0;
    Real left_contact_x = 0.0;
    Real right_contact_x = 0.0;
    size_t wet_regions = 0;
    size_t dry_bins = 0;
    size_t total_bins = 0;
    Real minimum_film = 0.0;
    Real maximum_film = 0.0;
    Real minimum_film_layers = 0.0;
};

TopologyMetrics MeasureTopology(BaseParticles &particles, Vecd *positions)
{
    // Two particle spacings per bin suppress empty-bin noise caused solely by
    // irregular particle positions.  A bin is wet when a liquid particle lies
    // within the numerical contact zone above the fibre.
    size_t bins = std::max<size_t>(24, static_cast<size_t>(
        std::round(DomainLength() / (2.0 * ParticleSpacing()))));
    std::vector<int> wet(bins, 0);
    std::vector<Real> outer(bins, -std::numeric_limits<Real>::infinity());
    Real left_contact_x = -std::numeric_limits<Real>::infinity();
    Real right_contact_x = std::numeric_limits<Real>::infinity();
    for (size_t i = 0; i < particles.TotalRealParticles(); ++i)
    {
        Real shifted = positions[i][0] - DomainLowerBound();
        Real periodic_x = shifted - std::floor(shifted / DomainLength()) * DomainLength();
        size_t b = std::min(bins - 1, static_cast<size_t>(
            std::floor(periodic_x / DomainLength() * static_cast<Real>(bins))));
        if (FiberNormalDistance(positions[i]) <=
            (breakup_options.contact_zone_in_dx + 0.75) * ParticleSpacing())
        {
            wet[b] = 1;
            if (positions[i][0] < 0.0)
                left_contact_x = std::max(left_contact_x, positions[i][0]);
            else
                right_contact_x = std::min(right_contact_x, positions[i][0]);
        }
        outer[b] = std::max(outer[b], positions[i][1] + 0.5 * ParticleSpacing());
    }

    TopologyMetrics metrics;
    metrics.total_bins = bins;
    metrics.minimum_film = std::numeric_limits<Real>::max();
    metrics.maximum_film = 0.0;
    for (size_t b = 0; b < bins; ++b)
    {
        if (!wet[b])
            ++metrics.dry_bins;
        Real bin_x = DomainLowerBound() +
                     (static_cast<Real>(b) + 0.5) * DomainLength() /
                         static_cast<Real>(bins);
        Real thickness = std::isfinite(outer[b])
                             ? FiberNormalDistance(Vecd(bin_x, outer[b]))
                             : 0.0;
        metrics.minimum_film = std::min(metrics.minimum_film, thickness);
        metrics.maximum_film = std::max(metrics.maximum_film, thickness);
    }
    metrics.dry_fraction = static_cast<Real>(metrics.dry_bins) /
                           static_cast<Real>(bins);
    metrics.minimum_film_layers = metrics.minimum_film / ParticleSpacing();
    if (std::isfinite(left_contact_x) && std::isfinite(right_contact_x))
    {
        metrics.left_contact_x = left_contact_x;
        metrics.right_contact_x = right_contact_x;
        // Particle centres bracket the dry interval. Subtract one spacing to
        // estimate the distance between the two finite-sized particle edges.
        metrics.central_gap_width = std::max(
            Real(0.0), right_contact_x - left_contact_x - ParticleSpacing());
    }

    if (metrics.dry_bins == 0)
        metrics.wet_regions = 1;
    else if (metrics.dry_bins < bins)
    {
        for (size_t b = 0; b < bins; ++b)
            if (wet[b] && !wet[(b + bins - 1) % bins])
                ++metrics.wet_regions;
    }
    return metrics;
}

/**
 * Resolution-aware topology closure for a wall film thinner than the SPH
 * stencil can resolve.  It is deliberately not presented as a first-principles
 * rupture law.  A neck must be below a user-visible thickness threshold over
 * several consecutive axial bins, must first be observed while thinning, and
 * must remain there for a prescribed dimensionless time.  The corresponding
 * particles are then switched to SPHinXsys buffer particles, which removes
 * them from all hydrodynamic neighbour lists.  Their axisymmetric mass is not
 * discarded: it is accumulated in an adsorbed-layer ledger and written to a
 * separate event file and to the main mass diagnostics.
 */
class ResolutionAwareFilmRupture
{
  public:
    ResolutionAwareFilmRupture(BaseParticles &particles, Real *physical_time,
                               Real initial_ring_mass,
                               const std::string &output_folder)
        : particles_(particles), physical_time_(physical_time),
          initial_ring_mass_(initial_ring_mass),
          pos_(particles.getVariableDataByName<Vecd>("Position")),
          velocity_(particles.getVariableDataByName<Vecd>("Velocity")),
          mass_(particles.getVariableDataByName<Real>("Mass")),
          density_(particles.getVariableDataByName<Real>("Density")),
          Vol_(particles.getVariableDataByName<Real>("VolumetricMeasure")),
          ring_mass_(particles.getVariableDataByName<Real>("AxisymmetricRingMass")),
          surface_indicator_(particles.getVariableDataByName<int>("Indicator")),
          bins_(std::max<size_t>(
              24, static_cast<size_t>(std::round(
                      DomainLength() /
                      ((breakup_options.subgrid_rupture_model ==
                        "conservative-peel")
                           ? ParticleSpacing()
                           : 2.0 * ParticleSpacing()))))),
          qualifying_since_(bins_, -1.0),
          previous_smoothed_thickness_(
              bins_, std::numeric_limits<Real>::infinity()),
          protected_dry_bins_(bins_, 0),
          protected_dry_core_profile_(
              bins_, breakup_options.subgrid_rupture_model ==
                             "conservative-peel"
                         ? 0.0
                         : 1.0),
          claimed_peel_bins_(bins_, 0),
          peel_bin_activation_T_(bins_, -1.0),
          initialized_(false), cumulative_converted_ring_mass_(0.0),
          cumulative_adsorbed_ring_mass_(0.0),
          cumulative_recycled_ring_mass_(0.0), pending_recycle_ring_mass_(0.0),
          pending_recycle_ring_momentum_(Vecd::Zero()), last_recycle_T_(0.0),
          recycle_clock_initialized_(false), removed_particles_(0),
          accepted_events_(0), ledger_records_(0), first_accepted_T_(-1.0),
          mass_guard_reported_(false)
    {
        event_file_.open(output_folder + "/rupture_events.csv");
        event_file_ << "record_id,event_id,time,T,status,start_bin,end_bin,bin_count,"
                       "x_start,x_end_periodic,removed_particles,"
                       "converted_ring_mass,cumulative_converted_ring_mass,"
                       "cumulative_adsorbed_ring_mass,cumulative_recycled_ring_mass,"
                       "active_particles\n";
        event_file_ << std::scientific << std::setprecision(10);
    }

    bool exec()
    {
        if (breakup_options.subgrid_rupture_model == "off")
            return false;
        const Real current_T = breakup_options.gamma_lg * (*physical_time_) /
                               (DynamicViscosity() * fibre_radius + TinyReal);
        if (breakup_options.subgrid_rupture_model == "conservative-peel")
            AdvanceConservativePeel(current_T);
        if (free_relaxation_frozen_)
        {
            bool maintained_dry_core = false;
            if (breakup_options.post_jfm_keep_dry_core)
                maintained_dry_core = MaintainProtectedDryCore(current_T);
            if (free_relaxation_continue_recycle_)
            {
                ReleaseRecycleReservoir(current_T);
            }
            return maintained_dry_core;
        }

        ReleaseRecycleReservoir(current_T);
        // Once a kernel-resolved dry nucleus has been created, prevent the
        // same under-resolved particles from numerically healing it in the
        // next neighbour-list update. Contact lines remain free outside this
        // minimum core; this is an irreversible nucleation closure, not a
        // general no-wetting boundary condition.
        // Maintaining an already accepted dry core must not stop the scan for
        // other necks.  In a nominally symmetric film one neck can cross the
        // discrete threshold a fraction of a time step earlier than its
        // partner; returning here previously made that tiny ordering error
        // permanent.
        MaintainProtectedDryCore(current_T);
        if (accepted_events_ >=
            static_cast<size_t>(breakup_options.rupture_max_events))
            return false;

        const Real dx = ParticleSpacing();
        const Real bin_width = DomainLength() / static_cast<Real>(bins_);
        std::vector<Real> outer(bins_,
                                -std::numeric_limits<Real>::infinity());
        for (size_t i = 0; i < particles_.TotalRealParticles(); ++i)
        {
            size_t b = BinIndex(pos_[i][0]);
            outer[b] = std::max(outer[b], pos_[i][1] + 0.5 * dx);
        }

        std::vector<Real> smoothed(bins_,
                                   std::numeric_limits<Real>::infinity());
        const int half_width = breakup_options.rupture_smoothing_half_width;
        for (size_t b = 0; b < bins_; ++b)
        {
            // A dry centre bin stays dry; averaging is only used to reject
            // jagged outer-particle noise in bins that still contain liquid.
            if (!std::isfinite(outer[b]))
                continue;
            Real sum = 0.0;
            size_t count = 0;
            for (int offset = -half_width; offset <= half_width; ++offset)
            {
                size_t q = static_cast<size_t>(
                    (static_cast<long long>(b) + offset +
                     static_cast<long long>(bins_)) %
                    static_cast<long long>(bins_));
                if (std::isfinite(outer[q]))
                {
                    Real q_x = DomainLowerBound() +
                               (static_cast<Real>(q) + 0.5) * DomainLength() /
                                   static_cast<Real>(bins_);
                    sum += FiberNormalDistance(Vecd(q_x, outer[q]));
                    ++count;
                }
            }
            if (count > 0)
                smoothed[b] = sum / static_cast<Real>(count);
        }

        if (!initialized_)
        {
            previous_smoothed_thickness_ = smoothed;
            initialized_ = true;
            return false;
        }

        const Real cutoff = breakup_options.rupture_cutoff_in_dx * dx;
        std::vector<int> eligible(bins_, 0);
        for (size_t b = 0; b < bins_; ++b)
        {
            const bool below = std::isfinite(smoothed[b]) &&
                               smoothed[b] <= cutoff &&
                               claimed_peel_bins_[b] == 0;
            const bool first_observed_while_thinning =
                below && std::isfinite(previous_smoothed_thickness_[b]) &&
                smoothed[b] < previous_smoothed_thickness_[b] - 1.0e-4 * dx;
            if (!below)
                qualifying_since_[b] = -1.0;
            else if (qualifying_since_[b] < 0.0 &&
                     first_observed_while_thinning)
                qualifying_since_[b] = current_T;

            if (qualifying_since_[b] >= 0.0 &&
                current_T - qualifying_since_[b] + 1.0e-12 >=
                    breakup_options.rupture_persistence_T)
                eligible[b] = 1;
        }
        previous_smoothed_thickness_ = smoothed;

        std::vector<std::vector<size_t>> accepted_runs;
        auto dry_anchor = std::find(eligible.begin(), eligible.end(), 0);
        if (dry_anchor == eligible.end())
            return false; // Safety: never convert the entire periodic film.
        size_t anchor = static_cast<size_t>(dry_anchor - eligible.begin());
        std::vector<size_t> run;
        auto finalize_run = [&]()
        {
            if (run.size() >=
                    static_cast<size_t>(breakup_options.rupture_min_bins) &&
                accepted_events_ + accepted_runs.size() <
                    static_cast<size_t>(breakup_options.rupture_max_events))
                accepted_runs.push_back(run);
            run.clear();
        };
        for (size_t offset = 1; offset <= bins_; ++offset)
        {
            size_t b = (anchor + offset) % bins_;
            if (eligible[b] == 1)
                run.push_back(b);
            else
                finalize_run();
        }
        finalize_run();
        if (accepted_runs.empty())
            return false;

        if (breakup_options.subgrid_rupture_model ==
            "conservative-peel")
        {
            RegisterConservativePeelEvents(accepted_runs, current_T,
                                           bin_width);
            return false;
        }

        std::vector<int> remove_bin(bins_, 0);
        if (accepted_events_ == 0 && !accepted_runs.empty())
            first_accepted_T_ = current_T;
        for (const auto &accepted_run : accepted_runs)
            for (size_t b : accepted_run)
                remove_bin[b] = 1;

        for (const auto &accepted_run : accepted_runs)
            for (size_t k = 0; k < accepted_run.size(); ++k)
            {
                const size_t b = accepted_run[k];
                protected_dry_bins_[b] = 1;
                if (breakup_options.post_jfm_dry_core_profile == "tapered" &&
                    accepted_run.size() > 1)
                {
                    const Real phase = Pi * static_cast<Real>(k + 1) /
                                       static_cast<Real>(accepted_run.size() + 1);
                    protected_dry_core_profile_[b] =
                        std::max(protected_dry_core_profile_[b],
                                 std::sin(phase));
                }
            }

        Real proposed_mass = 0.0;
        for (size_t i = 0; i < particles_.TotalRealParticles(); ++i)
            if (remove_bin[BinIndex(pos_[i][0])] == 1 &&
                IsInsideProtectedDryCore(i))
                proposed_mass += CurrentRingMass(i);
        if (cumulative_converted_ring_mass_ + proposed_mass >
            breakup_options.rupture_max_mass_fraction * initial_ring_mass_)
        {
            event_file_ << ledger_records_++ << "," << accepted_events_ << ","
                        << *physical_time_ << "," << current_T
                        << ",rejected_mass_guard,-1,-1,0,0,0,0,"
                        << proposed_mass << ","
                        << cumulative_converted_ring_mass_ << ","
                        << cumulative_adsorbed_ring_mass_ << ","
                        << cumulative_recycled_ring_mass_ << ","
                        << particles_.TotalRealParticles() << "\n";
            event_file_.flush();
            for (const auto &accepted_run : accepted_runs)
                for (size_t b : accepted_run)
                {
                    qualifying_since_[b] = -1.0;
                    protected_dry_bins_[b] = 0;
                    protected_dry_core_profile_[b] = 1.0;
                }
            return false;
        }

        std::vector<Real> mass_by_bin(bins_, 0.0);
        std::vector<Vecd> momentum_by_bin(bins_, Vecd::Zero());
        std::vector<size_t> particles_by_bin(bins_, 0);
        size_t i = 0;
        while (i < particles_.TotalRealParticles())
        {
            const size_t b = BinIndex(pos_[i][0]);
            if (remove_bin[b] == 0 || !IsInsideProtectedDryCore(i))
            {
                ++i;
                continue;
            }
            const Real converted_mass = CurrentRingMass(i);
            mass_by_bin[b] += converted_mass;
            momentum_by_bin[b] += converted_mass * velocity_[i];
            ++particles_by_bin[b];
            ++removed_particles_;
            // This copies the last active particle into i and shortens the
            // active prefix, so i must be tested again rather than incremented.
            particles_.switchToBufferParticle(static_cast<UnsignedInt>(i));
        }
        for (const auto &accepted_run : accepted_runs)
        {
            Real event_mass = 0.0;
            Vecd event_ring_momentum = Vecd::Zero();
            Real source_cosine = 0.0;
            Real source_sine = 0.0;
            size_t event_particles = 0;
            for (size_t b : accepted_run)
            {
                event_mass += mass_by_bin[b];
                event_ring_momentum += momentum_by_bin[b];
                const Real bin_x = DomainLowerBound() +
                                   (static_cast<Real>(b) + 0.5) * bin_width;
                const Real angle = 2.0 * Pi *
                                   (bin_x - DomainLowerBound()) / DomainLength();
                source_cosine += mass_by_bin[b] * std::cos(angle);
                source_sine += mass_by_bin[b] * std::sin(angle);
                event_particles += particles_by_bin[b];
                qualifying_since_[b] = -1.0;
                previous_smoothed_thickness_[b] =
                    std::numeric_limits<Real>::infinity();
            }
            const size_t first_bin = accepted_run.front();
            const size_t last_bin = accepted_run.back();
            Real x_start = DomainLowerBound() +
                           static_cast<Real>(first_bin) * bin_width;
            Real x_end = x_start +
                         static_cast<Real>(accepted_run.size()) * bin_width;
            Real source_angle = std::atan2(source_sine, source_cosine);
            if (source_angle < 0.0)
                source_angle += 2.0 * Pi;
            const Real source_x = DomainLowerBound() +
                                  source_angle * DomainLength() / (2.0 * Pi);
            StoreOrRecycleConvertedMass(event_mass, event_ring_momentum,
                                        source_x);
            event_file_ << ledger_records_++ << "," << accepted_events_ << ","
                        << *physical_time_ << "," << current_T
                        << ",accepted," << first_bin << ","
                        << last_bin << "," << accepted_run.size() << ","
                        << x_start << "," << x_end << ","
                        << event_particles << "," << event_mass << ","
                        << cumulative_converted_ring_mass_ << ","
                        << cumulative_adsorbed_ring_mass_ << ","
                        << cumulative_recycled_ring_mass_ << ","
                        << particles_.TotalRealParticles() << "\n";
            ++accepted_events_;
        }
        event_file_.flush();
        std::cout << "Subgrid rupture converted " << proposed_mass
                  << " ring mass in " << accepted_runs.size()
                  << " resolved neck interval(s) at T=" << current_T
                  << "; active particles=" << particles_.TotalRealParticles()
                  << std::endl;
        return true;
    }

    Real cumulativeAdsorbedRingMass() const
    {
        return cumulative_adsorbed_ring_mass_;
    }
    Real cumulativeConvertedRingMass() const
    {
        return cumulative_converted_ring_mass_;
    }
    Real cumulativeRecycledRingMass() const
    {
        return cumulative_recycled_ring_mass_;
    }
    Real pendingRecycleRingMass() const
    {
        return pending_recycle_ring_mass_;
    }
    size_t removedParticles() const { return removed_particles_; }
    size_t acceptedEvents() const { return accepted_events_; }
    bool hasAcceptedEvents() const { return accepted_events_ > 0; }
    Real firstAcceptedT() const { return first_accepted_T_; }
    bool postBreakupDetectionWindowElapsed(Real current_T) const
    {
        return hasAcceptedEvents() && first_accepted_T_ >= 0.0 &&
               current_T - first_accepted_T_ + 1.0e-12 >=
                   breakup_options.post_breakup_detection_window_T;
    }
    void freezeForFreeRelaxation(bool continue_recycle)
    {
        free_relaxation_frozen_ = true;
        free_relaxation_continue_recycle_ = continue_recycle;
    }
    bool freeRelaxationFrozen() const { return free_relaxation_frozen_; }
    bool freeRelaxationContinuesRecycle() const
    {
        return free_relaxation_continue_recycle_;
    }

  private:
    struct RecycleParcel
    {
        Real ring_mass;
        Vecd ring_momentum;
        Real source_x;
    };

    struct PeelInterval
    {
        std::vector<size_t> bins;
        size_t centre_index;
        Real start_T;
        size_t left_front_bin;
        size_t right_front_bin;
        Real last_left_activation_T;
        Real last_right_activation_T;
    };

    void ActivatePeelBin(size_t bin, Real current_T)
    {
        if (protected_dry_bins_[bin] == 1)
            return;
        protected_dry_bins_[bin] = 1;
        protected_dry_core_profile_[bin] = 0.0;
        peel_bin_activation_T_[bin] = current_T;
        claimed_peel_bins_[bin] = 1;
    }

    void RegisterConservativePeelEvents(
        const std::vector<std::vector<size_t>> &accepted_runs,
        Real current_T, Real bin_width)
    {
        size_t registered_now = 0;
        const size_t guard_bins = static_cast<size_t>(std::ceil(
            std::max(Real(breakup_options.rupture_min_bins) * bin_width,
                     2.5 * FilmThickness()) /
            (bin_width + TinyReal)));
        for (const std::vector<size_t> &accepted_run : accepted_runs)
        {
            if (accepted_events_ >=
                static_cast<size_t>(breakup_options.rupture_max_events))
                break;
            const size_t centre_index = accepted_run.size() / 2;
            const size_t centre_bin = accepted_run[centre_index];
            bool belongs_to_existing_neck = false;
            for (const PeelInterval &interval : peel_intervals_)
            {
                const size_t existing_bin =
                    interval.bins[interval.centre_index];
                const size_t direct_distance = centre_bin > existing_bin
                                                   ? centre_bin - existing_bin
                                                   : existing_bin - centre_bin;
                const size_t periodic_distance =
                    std::min(direct_distance, bins_ - direct_distance);
                if (periodic_distance <= guard_bins)
                {
                    belongs_to_existing_neck = true;
                    break;
                }
            }
            if (belongs_to_existing_neck)
                continue;

            if (accepted_events_ == 0)
                first_accepted_T_ = current_T;
            for (size_t bin : accepted_run)
            {
                qualifying_since_[bin] = -1.0;
                previous_smoothed_thickness_[bin] =
                    std::numeric_limits<Real>::infinity();
            }
            for (long long offset = -static_cast<long long>(guard_bins);
                 offset <= static_cast<long long>(guard_bins); ++offset)
            {
                const size_t bin = static_cast<size_t>(
                    (static_cast<long long>(centre_bin) + offset +
                     static_cast<long long>(bins_)) %
                    static_cast<long long>(bins_));
                qualifying_since_[bin] = -1.0;
            }
            ActivatePeelBin(centre_bin, current_T);
            peel_intervals_.push_back(
                PeelInterval{accepted_run, centre_index, current_T,
                             centre_bin, centre_bin, current_T, current_T});

            const size_t first_bin = accepted_run.front();
            const size_t last_bin = accepted_run.back();
            const Real x_start = DomainLowerBound() +
                                 static_cast<Real>(first_bin) * bin_width;
            const Real x_end = x_start +
                               static_cast<Real>(accepted_run.size()) *
                                   bin_width;
            event_file_ << ledger_records_++ << "," << accepted_events_
                        << "," << *physical_time_ << "," << current_T
                        << ",peel_registered," << first_bin << ","
                        << last_bin << "," << accepted_run.size() << ","
                        << x_start << "," << x_end
                        << ",0,0," << cumulative_converted_ring_mass_ << ","
                        << cumulative_adsorbed_ring_mass_ << ","
                        << cumulative_recycled_ring_mass_ << ","
                        << particles_.TotalRealParticles() << "\n";
            ++accepted_events_;
            ++registered_now;
        }
        event_file_.flush();
        if (registered_now > 0)
            std::cout << "Conservative peel registered "
                      << registered_now
                      << " under-resolved neck interval(s) at T=" << current_T
                      << "; dry opening starts from each neck centre."
                      << std::endl;
    }

    void AdvanceConservativePeel(Real current_T)
    {
        const Real dx = ParticleSpacing();
        const Real bin_width = DomainLength() /
                               static_cast<Real>(bins_);
        const Real propagation_delay_T = bin_width /
            (breakup_options.rupture_peel_front_speed_dx_per_T * dx + TinyReal);

        // Determine whether the complete liquid column in an axial bin is
        // already below the resolvable-film threshold.  Expansion is allowed
        // only through such bins, so the front cannot eat into a resolved
        // crown merely because its bottom particles touch the fibre.
        std::vector<Real> outer_thickness(
            bins_, -std::numeric_limits<Real>::infinity());
        for (size_t i = 0; i < particles_.TotalRealParticles(); ++i)
        {
            const size_t b = BinIndex(pos_[i][0]);
            outer_thickness[b] = std::max(
                outer_thickness[b], FiberNormalDistance(pos_[i]) + 0.5 * dx);
        }
        const Real cutoff = breakup_options.rupture_cutoff_in_dx * dx;
        auto can_advance_into = [&](size_t bin)
        {
            return claimed_peel_bins_[bin] == 0 &&
                   std::isfinite(outer_thickness[bin]) &&
                   outer_thickness[bin] <= cutoff;
        };

        // Each neck owns exactly two connected fronts.  A front may move by
        // one axial bin only after the prescribed propagation time has
        // elapsed.  This replaces independent bin deletion, which could leave
        // salt-and-pepper one-particle remnants and non-physical contact-line
        // jumps.
        for (PeelInterval &interval : peel_intervals_)
        {
            if (current_T - interval.last_left_activation_T + TinyReal >=
                propagation_delay_T)
            {
                const size_t candidate =
                    (interval.left_front_bin + bins_ - 1) % bins_;
                const bool was_registered_neck_bin =
                    std::find(interval.bins.begin(), interval.bins.end(),
                              candidate) != interval.bins.end();
                if (was_registered_neck_bin || can_advance_into(candidate))
                {
                    ActivatePeelBin(candidate, current_T);
                    interval.left_front_bin = candidate;
                    interval.last_left_activation_T = current_T;
                }
            }
            if (current_T - interval.last_right_activation_T + TinyReal >=
                propagation_delay_T)
            {
                const size_t candidate =
                    (interval.right_front_bin + 1) % bins_;
                const bool was_registered_neck_bin =
                    std::find(interval.bins.begin(), interval.bins.end(),
                              candidate) != interval.bins.end();
                if (was_registered_neck_bin || can_advance_into(candidate))
                {
                    ActivatePeelBin(candidate, current_T);
                    interval.right_front_bin = candidate;
                    interval.last_right_activation_T = current_T;
                }
            }
        }

        for (size_t b = 0; b < bins_; ++b)
        {
            if (protected_dry_bins_[b] == 0 ||
                peel_bin_activation_T_[b] < 0.0)
                continue;
            const Real phase = std::clamp(
                (current_T - peel_bin_activation_T_[b]) /
                    (breakup_options.rupture_peel_ramp_T + TinyReal),
                Real(0.0), Real(1.0));
            protected_dry_core_profile_[b] =
                phase * phase * (3.0 - 2.0 * phase);
        }
    }

    Real WrapPeriodicX(Real x) const
    {
        Real shifted = x - DomainLowerBound();
        shifted -= std::floor(shifted / DomainLength()) * DomainLength();
        return DomainLowerBound() + shifted;
    }

    Real SignedPeriodicDistance(Real x, Real centre) const
    {
        Real distance = x - centre;
        distance -= std::round(distance / DomainLength()) * DomainLength();
        return distance;
    }

    void AddRecycleParcel(Real converted_mass,
                          const Vecd &converted_ring_momentum,
                          Real source_x)
    {
        source_x = WrapPeriodicX(source_x);
        const Real merge_distance = 2.5 * ParticleSpacing();
        for (RecycleParcel &parcel : recycle_parcels_)
        {
            const Real displacement =
                SignedPeriodicDistance(source_x, parcel.source_x);
            if (std::abs(displacement) > merge_distance)
                continue;
            const Real combined_mass = parcel.ring_mass + converted_mass;
            parcel.source_x = WrapPeriodicX(
                parcel.source_x + converted_mass /
                                      (combined_mass + TinyReal) * displacement);
            parcel.ring_mass = combined_mass;
            parcel.ring_momentum += converted_ring_momentum;
            return;
        }
        recycle_parcels_.push_back(
            RecycleParcel{converted_mass, converted_ring_momentum, source_x});
    }

    void StoreOrRecycleConvertedMass(Real converted_mass,
                                     const Vecd &converted_ring_momentum,
                                     Real source_x)
    {
        if (converted_mass <= TinyReal)
            return;
        cumulative_converted_ring_mass_ += converted_mass;
        if (breakup_options.subgrid_rupture_model != "mass-recycle" &&
            breakup_options.subgrid_rupture_model != "conservative-peel")
        {
            cumulative_adsorbed_ring_mass_ += converted_mass;
            return;
        }

        // Do not inject a whole under-resolved neck into the crown in one
        // operation. Store its conserved ring mass and momentum, then release
        // both gradually on a user-controlled capillary time scale.
        pending_recycle_ring_mass_ += converted_mass;
        pending_recycle_ring_momentum_ += converted_ring_momentum;
        AddRecycleParcel(converted_mass, converted_ring_momentum, source_x);
    }

    void ReleaseRecycleReservoir(Real current_T)
    {
        if (!recycle_clock_initialized_)
        {
            last_recycle_T_ = current_T;
            recycle_clock_initialized_ = true;
            return;
        }
        const Real delta_T = std::max(Real(0.0), current_T - last_recycle_T_);
        last_recycle_T_ = current_T;
        if ((breakup_options.subgrid_rupture_model != "mass-recycle" &&
             breakup_options.subgrid_rupture_model != "conservative-peel") ||
            pending_recycle_ring_mass_ <= TinyReal || delta_T <= TinyReal)
            return;

        std::vector<size_t> receivers;
        const Real receiver_distance = std::max(
            Real(0.5) * FilmThickness(),
            (breakup_options.contact_zone_in_dx + 1.0) * ParticleSpacing());
        for (size_t i = 0; i < particles_.TotalRealParticles(); ++i)
            if (protected_dry_bins_[BinIndex(pos_[i][0])] == 0 &&
                FiberNormalDistance(pos_[i]) >= receiver_distance &&
                surface_indicator_[i] == 0)
                receivers.push_back(i);
        if (receivers.empty())
            for (size_t i = 0; i < particles_.TotalRealParticles(); ++i)
                if (protected_dry_bins_[BinIndex(pos_[i][0])] == 0 &&
                    FiberNormalDistance(pos_[i]) >= receiver_distance)
                    receivers.push_back(i);
        if (receivers.empty())
            return;

        const Real exponential_fraction = 1.0 - std::exp(
            -delta_T / (breakup_options.recycle_relaxation_T + TinyReal));
        std::vector<Real> remaining_capacity(
            particles_.TotalRealParticles(), 0.0);
        for (size_t i : receivers)
            remaining_capacity[i] =
                breakup_options.recycle_max_receiver_fraction *
                CurrentRingMass(i);

        const Real local_band = std::max(
            FilmThickness(), 6.0 * ParticleSpacing());
        const Real local_sigma = std::max(
            0.5 * FilmThickness(), 4.0 * ParticleSpacing());
        Real total_released_mass = 0.0;
        for (RecycleParcel &parcel : recycle_parcels_)
        {
            if (parcel.ring_mass <= TinyReal)
                continue;

            Real nearest_left = std::numeric_limits<Real>::infinity();
            Real nearest_right = std::numeric_limits<Real>::infinity();
            for (size_t i : receivers)
            {
                if (remaining_capacity[i] <= TinyReal)
                    continue;
                const Real distance =
                    SignedPeriodicDistance(pos_[i][0], parcel.source_x);
                if (distance < 0.0)
                    nearest_left = std::min(nearest_left, -distance);
                else
                    nearest_right = std::min(nearest_right, distance);
            }

            std::vector<size_t> local_receivers;
            std::vector<Real> raw_weights;
            Real left_weight = 0.0;
            Real right_weight = 0.0;
            for (size_t i : receivers)
            {
                if (remaining_capacity[i] <= TinyReal)
                    continue;
                const Real distance =
                    SignedPeriodicDistance(pos_[i][0], parcel.source_x);
                const Real side_nearest = distance < 0.0
                                              ? nearest_left
                                              : nearest_right;
                if (!std::isfinite(side_nearest) ||
                    std::abs(distance) > side_nearest + local_band)
                    continue;
                const Real offset = std::max(
                    Real(0.0), std::abs(distance) - side_nearest);
                const Real weight = CurrentRingMass(i) * std::exp(
                    -0.5 * std::pow(offset / (local_sigma + TinyReal), 2));
                local_receivers.push_back(i);
                raw_weights.push_back(weight);
                if (distance < 0.0)
                    left_weight += weight;
                else
                    right_weight += weight;
            }
            if (local_receivers.empty())
                continue;

            const bool has_both_sides = left_weight > TinyReal &&
                                        right_weight > TinyReal;
            std::vector<Real> normalized_weights(raw_weights.size(), 0.0);
            Real available_capacity = 0.0;
            for (size_t q = 0; q < local_receivers.size(); ++q)
            {
                const size_t i = local_receivers[q];
                const Real distance =
                    SignedPeriodicDistance(pos_[i][0], parcel.source_x);
                const Real side_weight = distance < 0.0
                                             ? left_weight
                                             : right_weight;
                normalized_weights[q] = raw_weights[q] /
                                        (side_weight + TinyReal) *
                                        (has_both_sides ? 0.5 : 1.0);
                available_capacity += remaining_capacity[i];
            }

            const Real old_parcel_mass = parcel.ring_mass;
            const Real requested_mass = old_parcel_mass *
                                        exponential_fraction;
            const Real released_mass = std::min(requested_mass,
                                                available_capacity);
            if (released_mass <= TinyReal)
                continue;

            std::vector<Real> shares(local_receivers.size(), 0.0);
            Real undistributed = released_mass;
            for (size_t iteration = 0;
                 iteration <= local_receivers.size() &&
                 undistributed > TinyReal;
                 ++iteration)
            {
                Real active_weight = 0.0;
                for (size_t q = 0; q < local_receivers.size(); ++q)
                    if (remaining_capacity[local_receivers[q]] > TinyReal)
                        active_weight += normalized_weights[q];
                if (active_weight <= TinyReal)
                    break;
                Real distributed_now = 0.0;
                for (size_t q = 0; q < local_receivers.size(); ++q)
                {
                    const size_t i = local_receivers[q];
                    if (remaining_capacity[i] <= TinyReal)
                        continue;
                    const Real proposed_share = undistributed *
                                                normalized_weights[q] /
                                                active_weight;
                    const Real accepted_share = std::min(
                        proposed_share, remaining_capacity[i]);
                    shares[q] += accepted_share;
                    remaining_capacity[i] -= accepted_share;
                    distributed_now += accepted_share;
                }
                if (distributed_now <= TinyReal)
                    break;
                undistributed -= distributed_now;
            }
            const Real actually_released = released_mass - undistributed;
            if (actually_released <= TinyReal)
                continue;

            const Vecd parcel_mean_velocity =
                parcel.ring_momentum / (old_parcel_mass + TinyReal);
            for (size_t q = 0; q < local_receivers.size(); ++q)
            {
                const Real ring_share = shares[q];
                if (ring_share <= TinyReal)
                    continue;
                const size_t i = local_receivers[q];
                const Real old_ring_mass = CurrentRingMass(i);
                velocity_[i] = (old_ring_mass * velocity_[i] +
                                ring_share * parcel_mean_velocity) /
                               (old_ring_mass + ring_share + TinyReal);
                const Real mass_increment = run_options.FullAxisymmetric()
                                                ? ring_share /
                                                      (2.0 * Pi * pos_[i][1] + TinyReal)
                                                : ring_share / (2.0 * Pi);
                mass_[i] += mass_increment;
                Vol_[i] = mass_[i] / (density_[i] + TinyReal);
                ring_mass_[i] = pos_[i][1] * mass_[i];
            }
            const Real released_fraction = actually_released /
                                           (old_parcel_mass + TinyReal);
            parcel.ring_momentum *=
                std::max(Real(0.0), Real(1.0) - released_fraction);
            parcel.ring_mass -= actually_released;
            total_released_mass += actually_released;
        }

        recycle_parcels_.erase(
            std::remove_if(recycle_parcels_.begin(), recycle_parcels_.end(),
                           [&](const RecycleParcel &parcel)
                           {
                               return parcel.ring_mass <=
                                      1.0e-12 * initial_ring_mass_;
                           }),
            recycle_parcels_.end());
        pending_recycle_ring_mass_ = 0.0;
        pending_recycle_ring_momentum_ = Vecd::Zero();
        for (const RecycleParcel &parcel : recycle_parcels_)
        {
            pending_recycle_ring_mass_ += parcel.ring_mass;
            pending_recycle_ring_momentum_ += parcel.ring_momentum;
        }
        if (pending_recycle_ring_mass_ <=
            1.0e-12 * initial_ring_mass_)
        {
            pending_recycle_ring_mass_ = 0.0;
            pending_recycle_ring_momentum_ = Vecd::Zero();
        }
        cumulative_recycled_ring_mass_ += total_released_mass;
    }

    bool MaintainProtectedDryCore(Real current_T)
    {
        if (std::find(protected_dry_bins_.begin(),
                      protected_dry_bins_.end(), 1) ==
            protected_dry_bins_.end())
            return false;

        Real proposed_mass = 0.0;
        size_t proposed_particles = 0;
        for (size_t i = 0; i < particles_.TotalRealParticles(); ++i)
            if (IsInsideProtectedDryCore(i))
            {
                const Real converted_mass = CurrentRingMass(i);
                proposed_mass += converted_mass;
                ++proposed_particles;
            }
        if (proposed_particles == 0)
            return false;

        const Real mass_limit = breakup_options.rupture_max_mass_fraction *
                                initial_ring_mass_;
        if (cumulative_converted_ring_mass_ + proposed_mass > mass_limit)
        {
            if (!mass_guard_reported_)
            {
                event_file_ << ledger_records_++ << "," << accepted_events_
                            << "," << *physical_time_ << "," << current_T
                            << ",maintenance_rejected_mass_guard,-1,-1,0,0,0,0,"
                            << proposed_mass << ","
                            << cumulative_converted_ring_mass_ << ","
                            << cumulative_adsorbed_ring_mass_ << ","
                            << cumulative_recycled_ring_mass_ << ","
                            << particles_.TotalRealParticles() << "\n";
                event_file_.flush();
                mass_guard_reported_ = true;
            }
            return false;
        }

        size_t removed_now = 0;
        size_t i = 0;
        while (i < particles_.TotalRealParticles())
        {
            if (!IsInsideProtectedDryCore(i))
            {
                ++i;
                continue;
            }
            const Real converted_mass = CurrentRingMass(i);
            const Vecd converted_momentum = converted_mass * velocity_[i];
            const Real source_x = pos_[i][0];
            ++removed_particles_;
            ++removed_now;
            particles_.switchToBufferParticle(static_cast<UnsignedInt>(i));
            StoreOrRecycleConvertedMass(converted_mass, converted_momentum,
                                        source_x);
        }
        event_file_ << ledger_records_++ << "," << accepted_events_ << ","
                    << *physical_time_ << "," << current_T
                    << ",maintained_dry_core,-1,-1,"
                    << std::count(protected_dry_bins_.begin(),
                                  protected_dry_bins_.end(), 1)
                    << ",0,0," << removed_now << "," << proposed_mass << ","
                    << cumulative_converted_ring_mass_ << ","
                    << cumulative_adsorbed_ring_mass_ << ","
                    << cumulative_recycled_ring_mass_ << ","
                    << particles_.TotalRealParticles() << "\n";
        event_file_.flush();
        return true;
    }

    size_t BinIndex(Real x) const
    {
        Real shifted = x - DomainLowerBound();
        Real periodic_x = shifted -
                          std::floor(shifted / DomainLength()) *
                              DomainLength();
        return std::min(
            bins_ - 1, static_cast<size_t>(std::floor(
                           periodic_x / DomainLength() *
                           static_cast<Real>(bins_))));
    }

    bool IsInsideProtectedDryCore(size_t i) const
    {
        const size_t b = BinIndex(pos_[i][0]);
        if (protected_dry_bins_[b] == 0)
            return false;
        const Real local_cutoff =
            protected_dry_core_profile_[b] *
            breakup_options.rupture_cutoff_in_dx * ParticleSpacing();
        return FiberNormalDistance(pos_[i]) <= local_cutoff;
    }

    Real CurrentRingMass(size_t i) const
    {
        return run_options.FullAxisymmetric()
                   ? 2.0 * Pi * pos_[i][1] * mass_[i]
                   : 2.0 * Pi * ring_mass_[i];
    }

    BaseParticles &particles_;
    Real *physical_time_;
    Real initial_ring_mass_;
    Vecd *pos_, *velocity_;
    Real *mass_, *density_, *Vol_, *ring_mass_;
    int *surface_indicator_;
    size_t bins_;
    std::vector<Real> qualifying_since_;
    std::vector<Real> previous_smoothed_thickness_;
    std::vector<int> protected_dry_bins_;
    std::vector<Real> protected_dry_core_profile_;
    std::vector<int> claimed_peel_bins_;
    std::vector<Real> peel_bin_activation_T_;
    std::vector<PeelInterval> peel_intervals_;
    bool initialized_;
    Real cumulative_converted_ring_mass_;
    Real cumulative_adsorbed_ring_mass_;
    Real cumulative_recycled_ring_mass_;
    Real pending_recycle_ring_mass_;
    Vecd pending_recycle_ring_momentum_;
    std::vector<RecycleParcel> recycle_parcels_;
    Real last_recycle_T_;
    bool recycle_clock_initialized_;
    bool free_relaxation_frozen_ = false;
    bool free_relaxation_continue_recycle_ = false;
    size_t removed_particles_, accepted_events_, ledger_records_;
    Real first_accepted_T_;
    bool mass_guard_reported_;
    std::ofstream event_file_;
};

void WriteBreakupParameters(const std::string &output_folder)
{
    std::ofstream file(output_folder + "/parameters.csv");
    file << "parameter,value,description\n" << std::scientific << std::setprecision(10);
    file << "post_breakup_models,"
         << (breakup_options.post_breakup_models ? 1 : 0)
         << ",master switch for JFM handover, wall adhesion and topology conversion/recycling\n";
    file << "output_tag," << (breakup_options.output_tag.empty() ? "automatic" : breakup_options.output_tag)
         << ",short user label used to keep Windows VTP paths below the legacy length limit\n";
    file << "case," << run_options.case_name << ",initial wavelength pattern\n";
    file << "pattern," << breakup_options.pattern << ",mixed modes or three equal necks\n";
    file << "initial_topology," << breakup_options.initial_topology
         << ",coated film, dry-fibre seed, edge gap or same-volume compact crown\n";
    file << "initial_gap_width_over_a," << breakup_options.gap_width_over_a
         << ",requested central dry interval in fibre-radius units\n";
    file << "seeded_bulk_thickness,"
         << (breakup_options.initial_topology == "seeded-gap"
                 ? SeededBulkThickness()
                 : FilmThickness())
         << ",volume-compensated plateau thickness; mean film value when inactive\n";
    file << "compact_crown_radius,"
         << (breakup_options.initial_topology == "compact-crown"
                 ? CompactCrownRadius()
                 : Real(0.0))
         << ",semicircular meridional radius; zero when inactive\n";
    if (breakup_options.initial_topology == "cmc-crown")
    {
        const CmcCrownData &cmc = CmcCrownSolution();
        file << "cmc_crown_apex_radius," << cmc.apex_radius
             << ",apex radius of the constant-mean-curvature crown\n";
        file << "cmc_crown_total_curvature," << cmc.total_curvature
             << ",uniform Young-Laplace curvature of the theoretical crown\n";
        file << "cmc_crown_contact_width," << 2.0 * cmc.half_width
             << ",theoretical liquid-solid footprint width\n";
        file << "cmc_crown_outer_width," << 2.0 * cmc.outer_half_extent
             << ",maximum meridional width including nonwetting overhang\n";
    }
    else
    {
        file << "cmc_crown_apex_radius,0,inactive for this initial topology\n";
        file << "cmc_crown_total_curvature,0,inactive for this initial topology\n";
        file << "cmc_crown_contact_width,0,inactive for this initial topology\n";
        file << "cmc_crown_outer_width,0,inactive for this initial topology\n";
    }
    file << "reference_edge_gap_ring_volume," << ReferenceEdgeGapAxisymmetricVolumeMeasure()
         << ",axisymmetric volume measure matched by compact-crown geometry\n";
    file << "seeded_pressure," << breakup_options.seeded_pressure
         << ",uniform contact-line test pressure or analytic local-curvature pressure\n";
    file << "full_coating_initial_pressure,"
         << breakup_options.full_coating_initial_pressure
         << ",local perturbed curvature, unperturbed base curvature, or zero initial gauge pressure\n";
    file << "axisym," << run_options.axisym << ",axisymmetric correction level\n";
    file << "a," << fibre_radius << ",fibre radius\n";
    file << "h_over_a," << run_options.film_ratio << ",mean film thickness ratio\n";
    file << "rho," << reference_density << ",liquid density\n";
    file << "gamma_lg," << breakup_options.gamma_lg << ",liquid-gas surface energy\n";
    file << "gamma_sl," << breakup_options.gamma_sl << ",solid-liquid surface energy\n";
    file << "gamma_sg," << breakup_options.gamma_sg << ",solid-gas surface energy\n";
    file << "wetting_model," << breakup_options.wetting_model
         << ",fixed normal, surface-energy, wall-adhesion or JFM contact-line model\n";
    file << "wall_adhesion_strength," << breakup_options.wall_adhesion_strength
         << ",dimensionless solid-liquid attraction scale; active in wall-adhesion and hybrid-adhesion modes\n";
    file << "wall_adhesion_range_dx," << breakup_options.wall_adhesion_range_in_dx
         << ",short-range wall attraction range divided by particle spacing\n";
    file << "wall_adhesion_profile," << breakup_options.wall_adhesion_profile
         << ",pairwise compact radial interaction or legacy distance-window traction\n";
    file << "wall_adhesion_activation," << breakup_options.wall_adhesion_activation
         << ",always applies from T=0; after-rupture waits for a dry interval or accepted rupture event\n";
    file << "wall_adhesion_ramp_T," << breakup_options.wall_adhesion_ramp_T
         << ",smoothstep duration for post-rupture adhesion activation\n";
    file << "young_cosine," << YoungCosine()
         << ",(gamma_sg-gamma_sl)/gamma_lg\n";
    file << "young_angle_deg," << YoungAngleDegrees()
         << ",equilibrium angle predicted by Young equation\n";
    file << "solid_liquid_laplace_pressure," << breakup_options.gamma_sl / fibre_radius
         << ",gamma_sl times cylindrical solid curvature\n";
    file << "solid_curvature_traction," << (breakup_options.solid_curvature_traction ? 1 : 0)
         << ",distributed curved solid-liquid normal traction\n";
    file << "wetting_force_scale," << breakup_options.wetting_force_scale
         << ",prototype contact-line regularization scale\n";
    file << "Oh," << run_options.oh << ",Ohnesorge number\n";
    file << "mu," << DynamicViscosity() << ",dynamic viscosity\n";
    file << "epsilon_over_h," << InitialAmplitude() / FilmThickness() << ",initial disturbance ratio\n";
    file << "resolution," << run_options.resolution << ",particles through mean film\n";
    file << "particle_spacing," << ParticleSpacing() << ",reference particle spacing\n";
    file << "nominal_domain_length," << NominalDomainLength()
         << ",unrounded requested periodic length\n";
    file << "periodic_domain_length," << DomainLength()
         << ",length aligned to an integer number of particle spacings\n";
    file << "periodic_spacing_count," << DomainLength() / ParticleSpacing()
         << ",number of lattice intervals around the periodic direction\n";
    file << "end_buffer_fraction," << run_options.end_buffer_fraction
         << ",fraction of total domain constrained at each end; zero disables both buffers\n";
    file << "conical_end_buffer," << (UseConicalEndBuffer() ? 1 : 0)
         << ",one means end-state relaxation and capillary suppression are active\n";
    file << "end_T," << run_options.end_T << ",gamma*t/(mu*a)\n";
    file << "frames," << run_options.frames << ",output intervals\n";
    file << "contact_angle_deg," << breakup_options.contact_angle_degrees
         << ",used only by fixed-angle comparison mode\n";
    file << "contact_core_dx," << breakup_options.contact_core_in_dx << ",full contact-angle weight below this distance\n";
    file << "contact_zone_dx," << breakup_options.contact_zone_in_dx << ",numerical contact-zone width\n";
    file << "contact_angle_model," << breakup_options.contact_angle_model
         << ",equilibrium angle or velocity-dependent linear contact-line friction\n";
    file << "contact_line_friction," << breakup_options.contact_line_friction
         << ",Lambda in cos(theta_d)=cos(theta_e)-Lambda*mu*U_cl/gamma\n";
    file << "contact_speed_filter_T," << breakup_options.contact_speed_filter_T
         << ",dimensionless low-pass time for geometric contact-line speed\n";
    file << "contact_speed_capillary_number," << breakup_options.contact_speed_capillary_number
         << ",absolute cap on mu*U_cl/gamma before temporal filtering\n";
    file << "substep_force_diagnostics," << (breakup_options.substep_force_diagnostics ? 1 : 0)
         << ",sample force budget inside the acoustic loop\n";
    file << "force_sample_dT," << breakup_options.force_sample_dT
         << ",dimensionless sampling interval for synchronized force budget\n";
    file << "force_control_band_dx," << breakup_options.force_control_band_in_dx
         << ",axial width of each moving geometric endpoint control band\n";
    file << "contact_angle_transition,smoothstep,continuous normal blending from contact core to zone edge\n";
    file << "surface_smoothing_passes," << breakup_options.surface_smoothing_passes
         << ",orientation-aware local normal and surface-strength filtering\n";
    file << "normal_alignment_floor," << breakup_options.normal_alignment_floor
         << ",neighbours below this normal dot product are excluded from smoothing\n";
    file << "capillary_cfl," << breakup_options.capillary_cfl
         << ",surface-tension/acoustic time-step safety coefficient\n";
    file << "slip_length_in_dx," << breakup_options.slip_length_in_dx
         << ",Navier tangential-slip length divided by particle spacing\n";
    file << "density_reinitialization," << (breakup_options.density_reinitialization ? 1 : 0)
         << ",free-surface density summation once per advection step\n";
    file << "particle_regularization," << (breakup_options.particle_regularization ? 1 : 0)
         << ",bulk-only transport-velocity position correction\n";
    file << "regularization_coefficient," << breakup_options.regularization_coefficient
         << ",bulk particle position-correction strength\n";
    file << "surface_regularization," << (breakup_options.surface_regularization ? 1 : 0)
         << ",tangential-only free-surface particle correction\n";
    file << "surface_regularization_coefficient," << breakup_options.surface_regularization_coefficient
         << ",free-surface tangential correction strength\n";
    file << "surface_max_shift_dx," << breakup_options.surface_max_shift_in_dx
         << ",maximum tangential position correction per advection step\n";
    file << "surface_hourglass_coefficient," << breakup_options.surface_hourglass_coefficient
         << ",pairwise surface-stress consistency stabilization\n";
    file << "normal_reconstruction," << breakup_options.normal_reconstruction
         << ",local free-surface normal estimator\n";
    file << "jfm_solid_extension," << breakup_options.jfm_solid_extension
         << ",virtual interface-normal extension through the solid support\n";
    file << "jfm_neighbor_mode," << breakup_options.jfm_neighbor_mode
         << ",mixed support baseline or experimental liquid-only non-wetting stencil\n";
    file << "jfm_endpoint_curvature," << breakup_options.jfm_endpoint_curvature
         << ",raw endpoint, local endpoint extrapolation, or one-sided contact-band extension\n";
    file << "jfm_curvature_outlier_filter,"
         << (breakup_options.jfm_curvature_outlier_filter ? 1 : 0)
         << ",local signed-median replacement of isolated curvature spikes\n";
    file << "jfm_curvature_ratio_limit,"
         << breakup_options.jfm_curvature_ratio_limit
         << ",maximum isolated deviation relative to local median scale\n";
    file << "hybrid_min_dry_bins," << breakup_options.hybrid_min_dry_bins
         << ",consecutive dry bins required before hybrid transition; each bin is about 2dx\n";
    file << "hybrid_transition_T," << breakup_options.hybrid_transition_T
         << ",dimensionless smoothstep duration for reconstructed-to-JFM force blending\n";
    file << "jfm_pressure_balance," << (breakup_options.jfm_pressure_balance ? 1 : 0)
         << ",paired liquid-gas curvature traction and missing one-phase pressure-boundary traction\n";
    file << "css_gradient_correction," << (breakup_options.css_gradient_correction ? 1 : 0)
         << ",symmetric Bonet-Lok correction for conservative surface-stress divergence\n";
    file << "support_min_neighbors," << breakup_options.support_min_neighbors
         << ",capillary stress is zero at or below this inner-neighbour count\n";
    file << "support_full_neighbors," << breakup_options.support_full_neighbors
         << ",capillary stress reaches full strength at this inner-neighbour count\n";
    file << "support_transition,smoothstep,resolution guard for under-resolved fragments\n";
    file << "van_der_waals_model," << breakup_options.van_der_waals_model
         << ",Xu et al. outermost-layer disjoining-pressure model\n";
    file << "van_der_waals_surface_weighting,corrected_wendland_h1dx_gradient"
         << ",lambda=abs(M^-1 grad Gamma) from Xu equations 19-20 and 30-34 with h2=dx\n";
    file << "van_der_waals_initial_pressure,"
         << (breakup_options.van_der_waals_initial_pressure ? 1 : 0)
         << ",initialize p=gamma*kappa+Pi(H) to remove the uniform acoustic startup transient\n";
    file << "van_der_waals_pressure_gauge,"
         << breakup_options.van_der_waals_pressure_gauge
         << ",absolute traction or mean-film gauge with the constant base pressure removed\n";
    file << "van_der_waals_gap_smoothing_passes,"
         << breakup_options.van_der_waals_gap_smoothing_passes
         << ",periodic 1-2-1 smoothing passes applied only to the gap used by Xu traction\n";
    file << "van_der_waals_number," << breakup_options.van_der_waals_number
         << ",A/(6*pi*gamma_lg*a^2)\n";
    file << "van_der_waals_min_gap_dx," << breakup_options.van_der_waals_min_gap_in_dx
         << ",regularization cutoff for the H^-3 singularity\n";
    file << "van_der_waals_resolution_taper_dx,"
         << breakup_options.van_der_waals_resolution_taper_in_dx
         << ",smoothly removes unresolved Xu traction between this gap and the minimum-gap cutoff\n";
    file << "van_der_waals_surface_threshold," << breakup_options.van_der_waals_surface_threshold
         << ",Xu concentration-gradient threshold Lambda\n";
    file << "van_der_waals_ramp_T," << breakup_options.van_der_waals_ramp_T
         << ",smooth force buildup in dimensionless time\n";
    file << "van_der_waals_post_breakup_ramp_T,"
         << breakup_options.van_der_waals_post_breakup_ramp_T
         << ",C1 fade duration after the first naturally empty wall bin; zero disables\n";
    file << "van_der_waals_cfl," << breakup_options.van_der_waals_cfl
         << ",acceleration-based time-step coefficient\n";
    file << "subgrid_rupture_model," << breakup_options.subgrid_rupture_model
         << ",off, adsorbed-layer ledger, instantaneous mass recycle, or progressive conservative peel\n";
    file << "rupture_cutoff_dx," << breakup_options.rupture_cutoff_in_dx
         << ",smoothed film-thickness threshold in particle spacings; values above about 6 are sensitivity tests for unresolved necks\n";
    file << "rupture_min_bins," << breakup_options.rupture_min_bins
         << ",minimum consecutive axial bins required for conversion\n";
    file << "rupture_persistence_T," << breakup_options.rupture_persistence_T
         << ",time the thinning criterion must persist before conversion\n";
    file << "rupture_smoothing_half_width," << breakup_options.rupture_smoothing_half_width
         << ",half-width of the periodic bin average used only by the rupture detector\n";
    file << "rupture_max_events," << breakup_options.rupture_max_events
         << ",safety cap on accepted rupture intervals\n";
    file << "rupture_max_mass_fraction," << breakup_options.rupture_max_mass_fraction
         << ",maximum cumulative initial liquid mass processed by the topology closure\n";
    file << "recycle_relaxation_T," << breakup_options.recycle_relaxation_T
         << ",dimensionless exponential time scale for gradual mass-momentum return\n";
    file << "recycle_max_receiver_fraction," << breakup_options.recycle_max_receiver_fraction
         << ",maximum fractional ring-mass increment accepted by each receiver per release\n";
    file << "rupture_peel_front_speed_dx_per_T,"
         << breakup_options.rupture_peel_front_speed_dx_per_T
         << ",axial conservative-peel front speed in particle spacings per dimensionless time\n";
    file << "rupture_peel_ramp_T," << breakup_options.rupture_peel_ramp_T
         << ",smooth radial opening time assigned independently to each newly reached peel bin\n";
    file << "recycle_localization,source_centred_two_sided_gaussian"
         << ",each rupture parcel returns mass to the nearest resolved crown on both axial sides\n";
    file << "post_jfm_free_relaxation," << (breakup_options.post_jfm_free_relaxation ? 1 : 0)
         << ",after breakup disables Xu traction; dry-core maintenance and recycling are controlled separately\n";
    file << "post_jfm_continue_recycle," << (breakup_options.post_jfm_continue_recycle ? 1 : 0)
         << ",continue only gradual local mass recycling during post-JFM relaxation\n";
    file << "post_jfm_keep_dry_core," << (breakup_options.post_jfm_keep_dry_core ? 1 : 0)
         << ",after breakup keep the accepted dry bins protected while Xu and new rupture conversions remain off\n";
    file << "post_jfm_dry_core_profile," << breakup_options.post_jfm_dry_core_profile
         << ",full removes each protected neck column; tapered removes the centre completely and leaves a smooth edge transition\n";
    file << "post_breakup_detection_window_T,"
         << breakup_options.post_breakup_detection_window_T
         << ",time after the first accepted rupture during which additional necks may still satisfy the same rupture criterion\n";
    file << "random_perturbation," << breakup_options.random_perturbation
         << ",off, initial position jitter, initial velocity jitter, or both\n";
    file << "random_position_dx," << breakup_options.random_position_in_dx
         << ",maximum initial position jitter divided by particle spacing\n";
    file << "random_velocity_scale," << breakup_options.random_velocity_scale
         << ",maximum initial velocity jitter divided by capillary velocity\n";
    file << "random_seed," << breakup_options.random_seed
         << ",deterministic seed for reproducible particle-wise random values\n";
    file << "thermal_noise," << breakup_options.thermal_noise
         << ",off or particle-wise persistent random velocity kick during every acoustic step\n";
    file << "thermal_noise_mode," << breakup_options.thermal_noise_mode
         << ",all-particle Cartesian kick or free-surface tangential kick\n";
    file << "thermal_noise_strength," << breakup_options.thermal_noise_strength
         << ",dimensionless amplitude of persistent random kick, scaled by capillary velocity and sqrt(dt/t_cap)\n";
    file << "thermal_noise_seed," << breakup_options.thermal_noise_seed
         << ",deterministic seed for reproducible time-continuous particle-wise random values\n";
    file << "thermal_noise_max_kick," << breakup_options.thermal_noise_max_kick_scale
         << ",maximum single-step random kick divided by capillary velocity\n";
    file << "thermal_noise_stop_after_rupture," << (breakup_options.thermal_noise_stop_after_rupture ? 1 : 0)
         << ",optional switch for separating noise-driven rupture triggering from post-rupture crown relaxation\n";
    file << "thermal_noise_scope,sensitivity_test,not a full fluctuating-hydrodynamics stochastic-stress discretisation\n";
    file << "barker_temperature_K," << breakup_options.barker_temperature_K
         << ",physical temperature used only for thermal-fluctuation scale diagnostic\n";
    file << "barker_fibre_radius_um," << breakup_options.barker_fibre_radius_um
         << ",physical fibre radius used only for thermal-fluctuation scale diagnostic\n";
    file << "barker_surface_tension_Npm," << breakup_options.barker_surface_tension_Npm
         << ",physical liquid-gas surface tension used only for thermal-fluctuation scale diagnostic\n";
    file << "barker_stochastic_weber," << BarkerStochasticWeber()
         << ",We_star=k_B*T/(gamma*a^2) following Barker et al.\n";
    file << "barker_thermal_length_m," << BarkerThermalLengthMeters()
         << ",sqrt(k_B*T/gamma) following Barker et al.\n";
    file << "barker_thermal_length_dx," << BarkerThermalLengthInParticleSpacings()
         << ",thermal length divided by current physical SPH particle spacing\n";
    file << "barker_model_scope,diagnostic_only,full FHD stochastic stress is not claimed in this one-phase SPH prototype\n";
    file << "surface_delta_filter,orientation_aware_local_kernel_average,reduces particle-scale stress variation\n";
    file << "capillary_method,"
         << ((breakup_options.wetting_model == "jfm" ||
              breakup_options.wetting_model == "hybrid" ||
              breakup_options.wetting_model == "hybrid-adhesion")
                 ? ((breakup_options.wetting_model == "hybrid" ||
                     breakup_options.wetting_model == "hybrid-adhesion")
                        ? (breakup_options.wetting_model == "hybrid-adhesion"
                               ? "reconstructed_RP_then_JFM_geometry_plus_wall_adhesion"
                               : "reconstructed_RP_then_local_JFM")
                        : "JFM2024_single_phase_axisymmetric_CSF")
                 : "local_one_sided_axisymmetric_CSS")
         << ",topology-capable preliminary model\n";
    file << "jfm_hoop_curvature,"
         << ((breakup_options.wetting_model == "jfm" ||
              breakup_options.wetting_model == "hybrid" ||
              breakup_options.wetting_model == "hybrid-adhesion")
                 ? 1
                 : 0)
         << ",adds n_r/r to meridional curvature\n";
    file << "explicit_endpoint_young_force,"
         << ((breakup_options.wetting_model == "jfm" ||
              breakup_options.wetting_model == "hybrid" ||
              breakup_options.wetting_model == "hybrid-adhesion")
                 ? 0
                 : (breakup_options.separate_contact_endpoint ? 1 : 0))
         << ",JFM embeds wetting in corrected normal and curvature\n";
    file << "topology_model,dry_fibre_intervals_allowed,no global interface interpolation\n";
    file << "air_particles,0,air phase intentionally omitted\n";
    file << "annealing_mapping,0,surface energies are direct inputs\n";
    file << "model_status,"
         << ((breakup_options.wetting_model == "wall-adhesion" ||
              breakup_options.wetting_model == "hybrid-adhesion")
                 ? "wall_adhesion_mechanism_prototype"
                 : "mechanism_prototype")
         << ",requires flat-wall contact-angle calibration and resolution validation\n";
}

/** Numerical symmetry boundary below r=0.
 *
 * Liquid exists close to the axis only when it wraps around a finite fibre
 * tip.  These wall particles provide the missing kernel support and enforce
 * zero normal velocity.  They are deliberately excluded from the viscous and
 * wetting relations, so the axis is free-slip and cannot act as a physical
 * solid surface or contact line.
 */
class FullCoatingAxisSymmetryShape : public MultiPolygonShape
{
  public:
    explicit FullCoatingAxisSymmetryShape(const std::string &shape_name)
        : MultiPolygonShape(shape_name)
    {
        const Real lower = FullCoatingOuterLowerBound() - 4.0 * BoundaryWidth();
        const Real upper = FullCoatingOuterUpperBound() + 4.0 * BoundaryWidth();
        const Real depth = 4.0 * ParticleSpacing();
        std::vector<Vecd> polygon{
            Vecd(lower, -depth), Vecd(upper, -depth),
            Vecd(upper, 0.0), Vecd(lower, 0.0),
            Vecd(lower, -depth)};
        multi_polygon_.addPolygon(polygon, GeometricOps::add);
    }
};
} // namespace

namespace SPH
{
/** Tag and direct generator for a body-fitted full coating.
 *
 * Lattice filling a shallow cone changes the number of Cartesian rows each
 * time the wall crosses one dx.  Those repeated steps are then amplified by
 * curvature differentiation.  This generator instead constructs parallel
 * coating curves, distributes particles at nearly constant arc-length dx,
 * and advances one normal layer at a time.
 */
class FullCoatingNormalLayers
{
};

class FullCoatingFiberLayers
{
};

template <>
class ParticleGenerator<BaseParticles, FullCoatingNormalLayers>
    : public ParticleGenerator<BaseParticles>
{
  public:
    ParticleGenerator(SPHBody &sph_body, BaseParticles &base_particles)
        : ParticleGenerator<BaseParticles>(sph_body, base_particles)
    {
    }

    virtual void prepareGeometricData() override
    {
        const Real dx = ParticleSpacing();
        const int layers = std::max(
            1, static_cast<int>(std::round(FilmThickness() / dx)));
        const Real layer_width = FilmThickness() /
                                 static_cast<Real>(layers);
        const Real left_centre = FullCoatingLeftCapCentre();
        const Real right_centre = FullCoatingRightCapCentre();

        for (int layer = 0; layer < layers; ++layer)
        {
            const Real reference_distance =
                (static_cast<Real>(layer) + 0.5) * layer_width;
            const Real layer_fraction = reference_distance /
                                        (FilmThickness() + TinyReal);
            const Real left_radius = FullCoatingLeftCapRadius() + reference_distance;
            const Real right_radius = FullCoatingRightCapRadius() + reference_distance;
            std::vector<Vecd> curve;

            const int coarse_samples = std::max(
                32, static_cast<int>(std::ceil(
                        0.5 * Pi * left_radius / (0.20 * dx))));
            curve.reserve(static_cast<size_t>(coarse_samples) + 1024);
            for (int i = 0; i <= coarse_samples; ++i)
            {
                const Real angle = Pi - 0.5 * Pi *
                                            static_cast<Real>(i) /
                                            static_cast<Real>(coarse_samples);
                curve.push_back(Vecd(left_centre +
                                         left_radius * std::cos(angle),
                                     left_radius * std::sin(angle)));
            }

            const int side_samples = std::max(
                160, static_cast<int>(std::ceil(
                         (right_centre - left_centre) / (0.20 * dx))));
            for (int i = 1; i <= side_samples; ++i)
            {
                const Real x = left_centre +
                               (right_centre - left_centre) *
                                   static_cast<Real>(i) /
                                   static_cast<Real>(side_samples);
                const Real wall_radius = FiberRadius(x);
                const Real slope = FiberRadiusFirstDerivative(x);
                const Real metric = std::sqrt(1.0 + slope * slope);
                // Stretch each normal column in proportion to its distance
                // from the wall.  The continuum wall position therefore
                // stays fixed while the outer layer receives the full small
                // perturbation.  SurfacePattern is smoothly zero in both end
                // reservoirs, so the two round caps are not pre-deformed.
                const Real distance = reference_distance +
                                      layer_fraction * InitialAmplitude() *
                                          SurfacePattern(x);
                curve.push_back(Vecd(x - distance * slope / metric,
                                     wall_radius + distance / metric));
            }

            const int fine_samples = std::max(
                24, static_cast<int>(std::ceil(
                        0.5 * Pi * right_radius / (0.20 * dx))));
            for (int i = 1; i <= fine_samples; ++i)
            {
                const Real angle = 0.5 * Pi *
                                   (1.0 - static_cast<Real>(i) /
                                              static_cast<Real>(fine_samples));
                curve.push_back(Vecd(right_centre +
                                         right_radius * std::cos(angle),
                                     right_radius * std::sin(angle)));
            }

            std::vector<Real> arc_length(curve.size(), 0.0);
            for (size_t i = 1; i < curve.size(); ++i)
                arc_length[i] = arc_length[i - 1] +
                                (curve[i] - curve[i - 1]).norm();
            const Real total_length = arc_length.back();
            const int particles_on_layer = std::max(
                2, static_cast<int>(std::round(total_length / dx)));
            const Real tangential_spacing = total_length /
                                             static_cast<Real>(particles_on_layer);
            size_t segment = 1;
            for (int k = 0; k < particles_on_layer; ++k)
            {
                const Real target =
                    (static_cast<Real>(k) + 0.5) * tangential_spacing;
                while (segment + 1 < arc_length.size() &&
                       arc_length[segment] < target)
                    ++segment;
                const Real s0 = arc_length[segment - 1];
                const Real s1 = arc_length[segment];
                const Real fraction = (target - s0) /
                                      (s1 - s0 + TinyReal);
                const Vecd position = curve[segment - 1] +
                                      fraction *
                                          (curve[segment] - curve[segment - 1]);
                addPositionAndVolumetricMeasure(
                    position, layer_width * tangential_spacing);
            }
        }
    }
};

/** Body-fitted boundary particles for the finite rigid fibre.
 *
 * Four solid layers adjacent to the wall follow the same tangential curves as
 * the liquid.  Deeper solid particles retain an inexpensive Cartesian fill.
 * This removes the alternating liquid/solid neighbour stencil produced when
 * a curvilinear liquid layer contacts a staircase solid lattice.
 */
template <>
class ParticleGenerator<BaseParticles, FullCoatingFiberLayers>
    : public ParticleGenerator<BaseParticles>
{
  public:
    ParticleGenerator(SPHBody &sph_body, BaseParticles &base_particles)
        : ParticleGenerator<BaseParticles>(sph_body, base_particles)
    {
    }

    virtual void prepareGeometricData() override
    {
        const Real dx = ParticleSpacing();
        const int boundary_layers = 4;
        const Real left_centre = FullCoatingLeftCapCentre();
        const Real right_centre = FullCoatingRightCapCentre();

        // Cartesian core, retained only beyond the body-fitted contact band.
        for (Real x = DomainLowerBound() + 0.5 * dx;
             x < DomainUpperBound(); x += dx)
        {
            const Real wall_radius = FiberRadius(x);
            for (Real r = 0.5 * dx;
                 r < wall_radius - static_cast<Real>(boundary_layers) * dx;
                 r += dx)
                addPositionAndVolumetricMeasure(Vecd(x, r), dx * dx);
        }

        for (int layer = 0; layer < boundary_layers; ++layer)
        {
            const Real distance =
                (static_cast<Real>(layer) + 0.5) * dx;
            const Real left_radius = FullCoatingLeftCapRadius() - distance;
            const Real right_radius = FullCoatingRightCapRadius() - distance;
            if (left_radius <= 0.5 * dx || right_radius <= 0.5 * dx)
                continue;
            std::vector<Vecd> curve;

            const int coarse_samples = std::max(
                24, static_cast<int>(std::ceil(
                        0.5 * Pi * left_radius / (0.20 * dx))));
            curve.reserve(static_cast<size_t>(coarse_samples) + 1024);
            for (int i = 0; i <= coarse_samples; ++i)
            {
                const Real angle = Pi - 0.5 * Pi *
                                            static_cast<Real>(i) /
                                            static_cast<Real>(coarse_samples);
                curve.push_back(Vecd(left_centre +
                                         left_radius * std::cos(angle),
                                     left_radius * std::sin(angle)));
            }

            const int side_samples = std::max(
                160, static_cast<int>(std::ceil(
                         (right_centre - left_centre) / (0.20 * dx))));
            for (int i = 1; i <= side_samples; ++i)
            {
                const Real x = left_centre +
                               (right_centre - left_centre) *
                                   static_cast<Real>(i) /
                                   static_cast<Real>(side_samples);
                const Real wall_radius = FiberRadius(x);
                const Real slope = FiberRadiusFirstDerivative(x);
                const Real metric = std::sqrt(1.0 + slope * slope);
                curve.push_back(Vecd(x + distance * slope / metric,
                                     wall_radius - distance / metric));
            }

            const int fine_samples = std::max(
                20, static_cast<int>(std::ceil(
                        0.5 * Pi * right_radius / (0.20 * dx))));
            for (int i = 1; i <= fine_samples; ++i)
            {
                const Real angle = 0.5 * Pi *
                                   (1.0 - static_cast<Real>(i) /
                                              static_cast<Real>(fine_samples));
                curve.push_back(Vecd(right_centre +
                                         right_radius * std::cos(angle),
                                     right_radius * std::sin(angle)));
            }

            std::vector<Real> arc_length(curve.size(), 0.0);
            for (size_t i = 1; i < curve.size(); ++i)
                arc_length[i] = arc_length[i - 1] +
                                (curve[i] - curve[i - 1]).norm();
            const Real total_length = arc_length.back();
            const int particles_on_layer = std::max(
                2, static_cast<int>(std::round(total_length / dx)));
            const Real tangential_spacing = total_length /
                                             static_cast<Real>(particles_on_layer);
            size_t segment = 1;
            for (int k = 0; k < particles_on_layer; ++k)
            {
                const Real target =
                    (static_cast<Real>(k) + 0.5) * tangential_spacing;
                while (segment + 1 < arc_length.size() &&
                       arc_length[segment] < target)
                    ++segment;
                const Real s0 = arc_length[segment - 1];
                const Real s1 = arc_length[segment];
                const Real fraction = (target - s0) /
                                      (s1 - s0 + TinyReal);
                const Vecd position = curve[segment - 1] +
                                      fraction *
                                          (curve[segment] - curve[segment - 1]);
                addPositionAndVolumetricMeasure(
                    position, dx * tangential_spacing);
            }
        }
    }
};
} // namespace SPH

int SPHINXSYS_BREAKUP_ENTRY_POINT(int argc, char *argv[])
{
    std::vector<std::string> base_arguments;
    std::vector<std::string> sph_arguments;
    try
    {
        breakup_options = ParseBreakupOptions(argc, argv, base_arguments);
        std::vector<char *> base_pointers = MakeArgumentPointers(base_arguments);
        run_options = ParseRunOptions(static_cast<int>(base_pointers.size()),
                                      base_pointers.data(), sph_arguments);
        if (IsConicalFiber() && !IsFiniteFiber() &&
            breakup_options.initial_topology != "coated")
            throw std::runtime_error(
                "The conical topology solver currently starts from --initial-topology=coated; seeded crowns/gaps still use cylindrical reference geometry.");
        if (IsFiniteFiber() &&
            breakup_options.initial_topology != "finite-film" &&
            breakup_options.initial_topology != "full-coating")
            throw std::runtime_error(
                "The finite-conical solver requires --initial-topology=finite-film or full-coating.");
        if (!IsFiniteFiber() &&
            (breakup_options.initial_topology == "finite-film" ||
             breakup_options.initial_topology == "full-coating"))
            throw std::runtime_error(
                "finite-film and full-coating topologies require --fiber-profile=finite-conical.");
        bool amplitude_was_given = false;
        for (const std::string &argument : base_arguments)
            amplitude_was_given = amplitude_was_given ||
                                  StartsWith(argument, "--amplitude-ratio=");
        if (!amplitude_was_given)
            run_options.amplitude_ratio = 0.01;
        if (breakup_options.initial_topology == "seeded-gap")
        {
            if (std::abs(run_options.amplitude_ratio) > 1.0e-12)
                throw std::runtime_error(
                    "seeded-gap validation requires --amplitude-ratio=0 so contact-line dynamics are isolated.");
        }
        if (breakup_options.initial_topology == "edge-gap")
        {
            if (run_options.case_name == "static" ||
                breakup_options.pattern != "single-center")
                throw std::runtime_error(
                    "edge-gap validation requires a non-static case and --pattern=single-center.");
            if (run_options.amplitude_ratio <= 0.0 ||
                run_options.amplitude_ratio > 0.05)
                throw std::runtime_error(
                    "edge-gap validation requires --amplitude-ratio in (0,0.05].");
        }
        if (breakup_options.initial_topology == "compact-crown")
        {
            if (run_options.case_name == "static" ||
                breakup_options.pattern != "single-center")
                throw std::runtime_error(
                    "compact-crown validation requires a non-static reference case and --pattern=single-center.");
            if (run_options.amplitude_ratio <= 0.0 ||
                run_options.amplitude_ratio > 0.05)
                throw std::runtime_error(
                    "compact-crown uses --amplitude-ratio in (0,0.05] only to match the edge-gap reference volume.");
        }
        if (breakup_options.initial_topology == "cmc-crown")
        {
            if (run_options.case_name == "static" ||
                breakup_options.pattern != "single-center")
                throw std::runtime_error(
                    "cmc-crown validation requires a non-static reference case and --pattern=single-center.");
            if (run_options.amplitude_ratio <= 0.0 ||
                run_options.amplitude_ratio > 0.05)
                throw std::runtime_error(
                    "cmc-crown uses --amplitude-ratio in (0,0.05] only to match the edge-gap reference volume.");
        }
        if (breakup_options.initial_topology == "seeded-gap" ||
            breakup_options.initial_topology == "edge-gap" ||
            breakup_options.initial_topology == "compact-crown" ||
            breakup_options.initial_topology == "cmc-crown")
        {
            Real gap_width = SeedGapWidth();
            if (gap_width < 2.0 * ParticleSpacing())
                throw std::runtime_error(
                    "--gap-width is under-resolved; use at least two particle spacings.");
            if (gap_width > DomainLength() - 4.0 * ParticleSpacing())
                throw std::runtime_error(
                    "--gap-width leaves too little liquid inside the periodic domain.");
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "Argument error: " << error.what() << std::endl;
        return 2;
    }
    // A pre-seeded dry interval is already a post-breakup continuation.  For
    // a continuous coated film, after-rupture mode starts with zero adhesion
    // and is opened only after the topology detector accepts a rupture.
    wall_adhesion_post_breakup_active =
        breakup_options.post_breakup_models &&
        (breakup_options.wall_adhesion_activation == "always" ||
         breakup_options.initial_topology != "coated");
    wall_adhesion_activation_T =
        wall_adhesion_post_breakup_active ? 0.0 : -1.0;
    wall_adhesion_activation_factor =
        wall_adhesion_post_breakup_active ? 1.0 : 0.0;
    std::vector<char *> sph_argument_pointers = MakeArgumentPointers(sph_arguments);

    Real spacing = ParticleSpacing();
    Real boundary_width = BoundaryWidth();
    Real maximum_initial_radius = IsConicalFiber()
                                      ? RadialDomainUpperBound() - 2.0 * boundary_width
                                      : MeanOuterRadius() + InitialAmplitude();
    if (breakup_options.initial_topology == "seeded-gap")
        maximum_initial_radius = fibre_radius + SeededBulkThickness();
    if (breakup_options.initial_topology == "compact-crown")
        maximum_initial_radius = fibre_radius + CompactCrownRadius();
    if (breakup_options.initial_topology == "cmc-crown")
        maximum_initial_radius = CmcCrownSolution().apex_radius;
    const bool fully_coated_finite =
        breakup_options.initial_topology == "full-coating";
    const Real axial_lower = fully_coated_finite
                                 ? FullCoatingOuterLowerBound()
                                 : DomainLowerBound();
    const Real axial_upper = fully_coated_finite
                                 ? FullCoatingOuterUpperBound()
                                 : DomainUpperBound();
    const Real axial_margin = fully_coated_finite
                                  ? 4.0 * boundary_width
                                  : boundary_width;
    BoundingBoxd bounds(Vec2d(axial_lower - axial_margin, -boundary_width),
                        Vec2d(axial_upper + axial_margin,
                              maximum_initial_radius + 2.0 * boundary_width));
    SPHSystem sph_system(bounds, spacing);
    sph_system.handleCommandlineOptions(static_cast<int>(sph_argument_pointers.size()),
                                        sph_argument_pointers.data());
    IO::getEnvironment().appendOutputFolder(BreakupLabel());

    std::cout << "Topology-capable attached-barrel prototype: case=" << run_options.case_name
              << " axisym=" << run_options.axisym
              << " h/a=" << run_options.film_ratio
              << " resolution=" << run_options.resolution
              << " topology=" << breakup_options.initial_topology
              << " wetting=" << breakup_options.wetting_model;
    if (breakup_options.wetting_model == "fixed-angle")
        std::cout << " theta=" << breakup_options.contact_angle_degrees << " deg\n";
    else if (breakup_options.wetting_model == "wall-adhesion" ||
             breakup_options.wetting_model == "hybrid-adhesion")
        std::cout << " wall_adhesion_strength="
                  << breakup_options.wall_adhesion_strength
                  << " range_dx=" << breakup_options.wall_adhesion_range_in_dx
                  << " profile=" << breakup_options.wall_adhesion_profile
                  << " activation=" << breakup_options.wall_adhesion_activation
                  << " ramp_T=" << breakup_options.wall_adhesion_ramp_T
                  << " rupture_scan_window_T="
                  << breakup_options.post_breakup_detection_window_T
                  << " (gamma_lg=" << breakup_options.gamma_lg << ")\n";
    else
        std::cout << " gamma(lg,sl,sg)=(" << breakup_options.gamma_lg << ","
                  << breakup_options.gamma_sl << "," << breakup_options.gamma_sg
                  << ") Young-angle=" << YoungAngleDegrees() << " deg\n";
    std::cout << "full_coating_initial_pressure="
              << breakup_options.full_coating_initial_pressure << "\n"
              << "Oh=" << run_options.oh << " end_T=" << run_options.end_T
              << " solver_end_time=" << PhysicalEndTime()
              << " post_breakup_models="
              << (breakup_options.post_breakup_models ? "on" : "off")
              << " smoothing_passes=" << breakup_options.surface_smoothing_passes
              << " capillary_cfl=" << breakup_options.capillary_cfl
              << " density_reinit=" << (breakup_options.density_reinitialization ? "on" : "off")
              << " particle_regularization=" << (breakup_options.particle_regularization ? "on" : "off")
              << " surface_regularization=" << (breakup_options.surface_regularization ? "on" : "off")
              << " surface_hourglass=" << breakup_options.surface_hourglass_coefficient
              << " normal_reconstruction=" << breakup_options.normal_reconstruction
              << " jfm_neighbor_mode=" << breakup_options.jfm_neighbor_mode
              << " jfm_pressure_balance=" << (breakup_options.jfm_pressure_balance ? "on" : "off")
              << " contact_angle_model=" << breakup_options.contact_angle_model
              << " contact_line_friction=" << breakup_options.contact_line_friction
              << " contact_speed_filter_T=" << breakup_options.contact_speed_filter_T
              << " contact_speed_cap_Ca=" << breakup_options.contact_speed_capillary_number
              << " substep_force_diagnostics="
              << (breakup_options.substep_force_diagnostics ? "on" : "off")
              << " van_der_waals=" << breakup_options.van_der_waals_model
              << " vdw_number=" << breakup_options.van_der_waals_number
              << " vdw_pressure_gauge="
              << breakup_options.van_der_waals_pressure_gauge
               << " vdw_gap_smoothing_passes="
               << breakup_options.van_der_waals_gap_smoothing_passes
               << " vdw_post_breakup_ramp_T="
               << breakup_options.van_der_waals_post_breakup_ramp_T
               << " subgrid_rupture="
              << breakup_options.subgrid_rupture_model
              << " rupture_cutoff_dx="
              << breakup_options.rupture_cutoff_in_dx
              << " rupture_min_bins="
              << breakup_options.rupture_min_bins
              << " random_perturbation="
              << breakup_options.random_perturbation
              << " random_position_dx="
              << breakup_options.random_position_in_dx
              << " random_velocity_scale="
              << breakup_options.random_velocity_scale
              << " random_seed="
              << breakup_options.random_seed
              << " thermal_noise="
              << breakup_options.thermal_noise
              << " thermal_noise_mode="
              << breakup_options.thermal_noise_mode
              << " thermal_noise_strength="
              << breakup_options.thermal_noise_strength
              << " thermal_noise_seed="
              << breakup_options.thermal_noise_seed
              << " thermal_noise_max_kick="
              << breakup_options.thermal_noise_max_kick_scale
              << " thermal_noise_stop_after_rupture="
              << (breakup_options.thermal_noise_stop_after_rupture ? "on" : "off")
              << " Barker_We*="
              << BarkerStochasticWeber()
              << " Barker_l*/dx="
              << BarkerThermalLengthInParticleSpacings()
              << std::endl;

    FluidBody liquid(sph_system, makeShared<BreakupLiquidFilmShape>("LiquidFilmHalf"));
    liquid.defineMatterMaterial<WeaklyCompressibleFluid>(reference_density, SoundSpeed());
    liquid.addMaterialProperty<Viscosity>(DynamicViscosity());
    if (fully_coated_finite)
        liquid.generateParticles<BaseParticles, FullCoatingNormalLayers>();
    else
        liquid.generateParticles<BaseParticles, Lattice>();

    SolidBody fibre(sph_system, makeShared<RigidFiberShape>("RigidFiberHalf"));
    fibre.defineMatterMaterial<Solid>();
    if (fully_coated_finite)
        fibre.generateParticles<BaseParticles, FullCoatingFiberLayers>();
    else
        fibre.generateParticles<BaseParticles, Lattice>();

    SolidBody axis_symmetry(
        sph_system,
        makeShared<FullCoatingAxisSymmetryShape>("AxisSymmetryBoundary"));
    axis_symmetry.defineMatterMaterial<Solid>();
    axis_symmetry.generateParticles<BaseParticles, Lattice>();

    InnerRelation liquid_inner(liquid);
    ContactRelation liquid_fibre_contact(liquid, {&fibre});
    ContactRelation liquid_wall_contact(liquid, {&fibre, &axis_symmetry});
    ComplexRelation liquid_complex(liquid_inner, liquid_wall_contact);

    SimpleDynamics<NormalDirectionFromBodyShape> fibre_normals(fibre);
    SimpleDynamics<NormalDirectionFromBodyShape> axis_symmetry_normals(axis_symmetry);
    InteractionWithUpdate<FreeSurfaceIndicationComplex> indicate_free_surface(
        liquid_inner, liquid_wall_contact);
    InteractionWithUpdate<fluid_dynamics::DensitySummationComplexFreeSurface>
        density_by_summation(liquid_inner, liquid_wall_contact);
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>>
        regularize_bulk_particles(
            DynamicsArgs(liquid_inner, breakup_options.regularization_coefficient),
            liquid_fibre_contact);
    // The capillary model receives the combined wall relation so that the
    // symmetry-plane particles complete its kernel support: without them the
    // reconstructed free-surface normal degenerates where the film wraps
    // around a fibre tip.  Those particles are support only, and the
    // wettability mask inside the force classes keeps the axis free-slip.
    SelectableAxisymmetricCapillaryForce local_capillary_force(
        liquid_inner, liquid_wall_contact,
        sph_system.svPhysicalTime().Data());
    XuVanDerWaalsForce van_der_waals_force(liquid_inner);
    InteractionWithUpdate<TangentialSurfaceRegularization>
        regularize_surface_particles(liquid_inner);

    Dynamics1Level<fluid_dynamics::Integration1stHalfWithWallRiemann>
        pressure_relaxation(liquid_inner, liquid_wall_contact);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallRiemann>
        density_relaxation(liquid_inner, liquid_wall_contact);
    InteractionWithUpdate<fluid_dynamics::ViscousForceInner>
        viscous_force_inner(liquid_inner);
    InteractionWithUpdate<FiberNavierSlipWallViscousForce>
        viscous_force_wall(
            liquid_fibre_contact,
            breakup_options.slip_length_in_dx * ParticleSpacing());
    PressureForceDiagnostic pressure_force_diagnostic(
        liquid_inner, liquid_wall_contact);
    InteractionWithUpdate<AxisymmetricViscousForce> axisymmetric_viscous_force(liquid_inner);
    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep>
        advection_time_step(liquid, CapillaryVelocity(), 0.10);
    ReduceDynamics<fluid_dynamics::SurfaceTensionTimeStep>
        acoustic_surface_time_step(liquid, breakup_options.capillary_cfl);

    PeriodicAlongAxis periodic_axis(liquid.getSPHBodyBounds(), xAxis);
    PeriodicConditionUsingCellLinkedList periodic_condition(liquid, periodic_axis);
    ParticleSorting particle_sorting(liquid);

    SimpleDynamics<BreakupInitialLiquidState> initialize_liquid(liquid);
    SimpleDynamics<RandomInitialParticlePerturbation> random_initial_perturbation(liquid);
    ContinuousParticleThermalNoise continuous_thermal_noise(liquid);
    ConicalEndBuffer conical_end_buffer(liquid.getBaseParticles());
    SimpleDynamics<InitializeAxisymmetricWeights> initialize_axisymmetric_weights(liquid);
    SimpleDynamics<UpdateAxisymmetricWeights> update_axisymmetric_weights(liquid);
    SimpleDynamics<AxisymmetricContinuityCorrection> axisymmetric_continuity(liquid);

    BodyStatesRecordingToVtp write_states(sph_system);
    write_states.addToWrite<Real>(liquid, "Density");
    write_states.addToWrite<Real>(liquid, "Pressure");
    write_states.addToWrite<int>(liquid, "Indicator");
    write_states.addToWrite<Vecd>(liquid, "Velocity");
    write_states.addToWrite<Vecd>(liquid, "TopologyCapillaryForce");
    write_states.addToWrite<Vecd>(liquid, "FiberWallViscousForce");
    write_states.addToWrite<Vecd>(liquid, "DiagnosticPressureForce");
    write_states.addToWrite<Vecd>(liquid, "SurfaceEnergyWettingForce");
    write_states.addToWrite<Vecd>(liquid, "WallAdhesionForce");
    write_states.addToWrite<Vecd>(liquid, "SolidLiquidCurvatureForce");
    if (breakup_options.wetting_model == "jfm" ||
        breakup_options.wetting_model == "hybrid" ||
        breakup_options.wetting_model == "hybrid-adhesion")
    {
        write_states.addToWrite<Vecd>(liquid, "JfmCurvatureForce");
        write_states.addToWrite<Vecd>(liquid, "TopologyPressureJumpForce");
        write_states.addToWrite<Real>(liquid, "JfmPressureGaugeOffset");
        write_states.addToWrite<int>(liquid, "JfmContactCurvatureExtended");
        write_states.addToWrite<Real>(liquid, "JfmContactLineSpeed");
    }
    write_states.addToWrite<Vecd>(liquid, "ViscousForce");
    write_states.addToWrite<Vecd>(liquid, "AxisymmetricViscousForce");
    write_states.addToWrite<Vecd>(liquid, "XuVanDerWaalsForce");
    write_states.addToWrite<int>(liquid, "XuSurfaceIndicator");
    write_states.addToWrite<Real>(liquid, "XuSurfaceWeight");
    write_states.addToWrite<Real>(liquid, "XuFilmGap");
    write_states.addToWrite<Real>(liquid, "XuDisjoiningPressure");
    if (breakup_options.random_perturbation != "off")
    {
        write_states.addToWrite<Vecd>(liquid, "InitialRandomDisplacement");
        write_states.addToWrite<Vecd>(liquid, "InitialRandomVelocity");
    }
    if (breakup_options.thermal_noise != "off")
        write_states.addToWrite<Vecd>(liquid, "ThermalNoiseVelocity");
    if (breakup_options.thermal_noise != "off")
        write_states.addToWrite<Vecd>(liquid, "ContinuousThermalNoiseKick");

    initialize_liquid.exec();
    random_initial_perturbation.exec();
    conical_end_buffer.capture();
    initialize_axisymmetric_weights.exec();
    if (run_options.FullAxisymmetric())
        update_axisymmetric_weights.exec();
    sph_system.initializeSystemCellLinkedLists();
    if (!IsConicalFiber())
        periodic_condition.update_cell_linked_list_.exec();
    sph_system.initializeSystemConfigurations();
    fibre_normals.exec();
    axis_symmetry_normals.exec();
    indicate_free_surface.exec();
    local_capillary_force.exec();
    van_der_waals_force.exec();
    pressure_force_diagnostic.exec();

    std::string output_folder = IO::getEnvironment().OutputFolder();
    WriteBreakupParameters(output_folder);
    std::ofstream topology_file(output_folder + "/topology_diagnostics.csv");
    topology_file << "time,T,dry_fraction,dry_bins,total_bins,attached_wet_regions,"
                     "central_gap_width,left_contact_x,right_contact_x,contact_line_particles,"
                     "surface_contact_gap_width,surface_left_contact_x,surface_right_contact_x,"
                     "surface_left_edge_vx,surface_right_edge_vx,"
                     "surface_left_edge_fx,surface_right_edge_fx,"
                     "surface_left_edge_nx,surface_left_edge_nr,"
                     "surface_left_edge_kappa_axial,surface_left_edge_kappa_hoop,"
                     "surface_left_edge_support,"
                     "left_contact_particles,right_contact_particles,"
                     "left_capillary_fx,left_capillary_fr,right_capillary_fx,right_capillary_fr,"
                     "left_mean_nx,left_mean_nr,right_mean_nx,right_mean_nr,"
                     "left_geometric_contact_angle_deg,right_geometric_contact_angle_deg,"
                     "left_corrected_contact_angle_deg,right_corrected_contact_angle_deg,"
                     "left_target_contact_angle_deg,right_target_contact_angle_deg,"
                     "left_mean_vx,left_mean_vr,right_mean_vx,right_mean_vr,"
                     "left_mean_pressure,right_mean_pressure,"
                     "left_mean_kappa_axial,left_mean_kappa_hoop,"
                     "right_mean_kappa_axial,right_mean_kappa_hoop,"
                     "minimum_film,max_film,minimum_film_layers,ring_mass,ring_mass_error,"
                     "converted_ring_mass,adsorbed_ring_mass,recycled_ring_mass,pending_recycle_ring_mass,"
                     "conserved_ring_mass,conserved_ring_mass_error,"
                     "rupture_removed_particles,rupture_events,max_speed,"
                     "max_capillary_force,max_wetting_force,max_solid_liquid_force,"
                     "max_wall_adhesion_force,wall_adhesion_activation_factor,max_vdw_force,"
                     "max_vdw_pressure,vdw_surface_particles,"
                     "min_density,max_density,finite_state,out_of_domain_particles,"
                     "under_resolved_surface_particles,max_radial_coordinate,pca_normal_fraction,"
                     "post_jfm_free_relaxation_active\n";
    topology_file << std::scientific << std::setprecision(10);

    BaseParticles &particles = liquid.getBaseParticles();
    Vecd *positions = particles.getVariableDataByName<Vecd>("Position");
    Vecd *velocities = particles.getVariableDataByName<Vecd>("Velocity");
    Real *densities = particles.getVariableDataByName<Real>("Density");
    Real *pressures = particles.getVariableDataByName<Real>("Pressure");
    Real *masses = particles.getVariableDataByName<Real>("Mass");
    Real *ring_masses = particles.getVariableDataByName<Real>("AxisymmetricRingMass");
    Vecd *capillary_forces = particles.getVariableDataByName<Vecd>("TopologyCapillaryForce");
    Vecd *jfm_curvature_forces = nullptr;
    Vecd *pressure_jump_forces = nullptr;
    if (breakup_options.wetting_model == "jfm" ||
        breakup_options.wetting_model == "hybrid" ||
        breakup_options.wetting_model == "hybrid-adhesion")
    {
        jfm_curvature_forces =
            particles.getVariableDataByName<Vecd>("JfmCurvatureForce");
        pressure_jump_forces =
            particles.getVariableDataByName<Vecd>("TopologyPressureJumpForce");
    }
    Vecd *diagnostic_pressure_forces = particles.getVariableDataByName<Vecd>("DiagnosticPressureForce");
    Vecd *viscous_forces = particles.getVariableDataByName<Vecd>("ViscousForce");
    Vecd *axisymmetric_viscous_forces = particles.getVariableDataByName<Vecd>("AxisymmetricViscousForce");
    Vecd *wall_viscous_forces = particles.getVariableDataByName<Vecd>("FiberWallViscousForce");
    Vecd *wetting_forces = particles.getVariableDataByName<Vecd>("SurfaceEnergyWettingForce");
    Vecd *wall_adhesion_forces = particles.getVariableDataByName<Vecd>("WallAdhesionForce");
    Vecd *solid_liquid_forces = particles.getVariableDataByName<Vecd>("SolidLiquidCurvatureForce");
    Vecd *van_der_waals_forces = particles.getVariableDataByName<Vecd>("XuVanDerWaalsForce");
    Real *van_der_waals_pressures = particles.getVariableDataByName<Real>("XuDisjoiningPressure");
    int *van_der_waals_surface = particles.getVariableDataByName<int>("XuSurfaceIndicator");
    int *surface_indicators = particles.getVariableDataByName<int>(
        (breakup_options.wetting_model == "jfm" ||
         breakup_options.wetting_model == "hybrid" ||
         breakup_options.wetting_model == "hybrid-adhesion")
            ? "JfmSurfaceIndicator"
            : "Indicator");
    int *contact_line_indicators =
        particles.getVariableDataByName<int>("ContactLineIndicator");
    Vecd *topology_normals =
        particles.getVariableDataByName<Vecd>("TopologyNormal");
    Vecd *jfm_smoothed_normals = nullptr;
    Real *dynamic_contact_cosines =
        particles.getVariableDataByName<Real>("DynamicContactCosine");
    if (breakup_options.wetting_model == "jfm" ||
        breakup_options.wetting_model == "hybrid" ||
        breakup_options.wetting_model == "hybrid-adhesion")
        jfm_smoothed_normals =
            particles.getVariableDataByName<Vecd>("JfmSmoothedNormal");
    else
        // The legacy CSS branch has no separate JFM-smoothed normal; its
        // topology normal is the closest equivalent for diagnostics.
        jfm_smoothed_normals = topology_normals;
    Real *jfm_axial_curvatures = nullptr;
    Real *jfm_hoop_curvatures = nullptr;
    if (breakup_options.wetting_model == "jfm" ||
        breakup_options.wetting_model == "hybrid" ||
        breakup_options.wetting_model == "hybrid-adhesion")
    {
        jfm_axial_curvatures =
            particles.getVariableDataByName<Real>("JfmAxialCurvature");
        jfm_hoop_curvatures =
            particles.getVariableDataByName<Real>("JfmHoopCurvature");
    }
    int *surface_support_neighbors = particles.getVariableDataByName<int>("SurfaceSupportNeighbors");
    int *pca_normal_used = particles.getVariableDataByName<int>("PcaNormalUsed");
    Real initial_ring_mass = 0.0;
    for (size_t i = 0; i < particles.TotalRealParticles(); ++i)
        initial_ring_mass += 2.0 * Pi * ring_masses[i];
    Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
    ResolutionAwareFilmRupture film_rupture(
        particles, &physical_time, initial_ring_mass, output_folder);
    bool post_jfm_free_relaxation_active = false;
    int dry_topology_persistence_checks = 0;
    auto update_wall_adhesion_activation = [&]()
    {
        if (!breakup_options.post_breakup_models)
        {
            wall_adhesion_post_breakup_active = false;
            wall_adhesion_activation_factor = 0.0;
            return;
        }
        const bool adhesion_mode =
            breakup_options.wetting_model == "wall-adhesion" ||
            breakup_options.wetting_model == "hybrid-adhesion";
        const bool xu_post_breakup_handover =
            breakup_options.van_der_waals_model == "xu" &&
            breakup_options.van_der_waals_post_breakup_ramp_T > TinyReal;
        if (!adhesion_mode && !xu_post_breakup_handover)
            return;
        const TopologyMetrics topology = MeasureTopology(particles, positions);
        // This does not create or enlarge a hole: the Xu fade can begin only
        // after the ordinary particle topology already contains a truly empty
        // wall bin.  The molecular traction has then completed its pre-rupture
        // role and its attractive-only H^-3 continuation is no longer valid.
        if (xu_post_breakup_handover && topology.dry_bins > 0)
            van_der_waals_force.startPostBreakupRamp();
        if (!adhesion_mode)
            return;
        const Real current_T = breakup_options.gamma_lg * physical_time /
                               (DynamicViscosity() * fibre_radius + TinyReal);
        if (breakup_options.wall_adhesion_activation == "always")
        {
            wall_adhesion_post_breakup_active = true;
            wall_adhesion_activation_T = 0.0;
            wall_adhesion_activation_factor = 1.0;
            return;
        }

        // Accepted subgrid conversion is the strongest signal.  When the
        // user leaves that closure off, require two consecutive diagnostics
        // with a kernel-resolved dry interval before opening adhesion.
        if (topology.dry_bins >= static_cast<size_t>(
                                  std::max(1, breakup_options.rupture_min_bins)))
            ++dry_topology_persistence_checks;
        else
            dry_topology_persistence_checks = 0;

        const bool breakup_detection_complete =
            film_rupture.hasAcceptedEvents()
                ? film_rupture.postBreakupDetectionWindowElapsed(current_T)
                : dry_topology_persistence_checks >= 2;
        if (!wall_adhesion_post_breakup_active &&
            breakup_detection_complete)
        {
            wall_adhesion_post_breakup_active = true;
            wall_adhesion_activation_T = current_T;
            std::cout << "Post-breakup wall adhesion activated at T="
                      << current_T << " with smooth ramp T="
                      << breakup_options.wall_adhesion_ramp_T << std::endl;
        }

        if (!wall_adhesion_post_breakup_active)
        {
            wall_adhesion_activation_factor = 0.0;
            return;
        }
        const Real ramp = breakup_options.wall_adhesion_ramp_T;
        const Real s = ramp <= TinyReal
                           ? 1.0
                           : std::clamp((current_T - wall_adhesion_activation_T) /
                                            ramp,
                                        Real(0.0), Real(1.0));
        // C1 smoothstep avoids an impulsive change in the force budget.
        wall_adhesion_activation_factor = s * s * (3.0 - 2.0 * s);
    };
    update_wall_adhesion_activation();

    std::ofstream substep_force_file;
    if (breakup_options.substep_force_diagnostics)
    {
        substep_force_file.open(output_folder + "/substep_force_budget.csv");
        substep_force_file
            << "time,T,side,endpoint_x,control_particles,mean_vx,"
               "curvature_fx,boundary_pressure_fx,sph_pressure_fx,"
               "total_viscous_fx,wall_viscous_fx,wall_adhesion_fx,"
               "wetting_fx,solid_liquid_fx,vdw_fx,resolved_total_fx\n";
        substep_force_file << std::scientific << std::setprecision(10);
    }
    Real next_force_sample_T = 0.0;
    auto write_substep_force_budget = [&](Real time)
    {
        if (!breakup_options.substep_force_diagnostics)
            return;
        const Real T = breakup_options.gamma_lg * time /
                       (DynamicViscosity() * fibre_radius);
        if (T + 1.0e-10 < next_force_sample_T)
            return;

        Real footprint_left = std::numeric_limits<Real>::max();
        Real footprint_right = -std::numeric_limits<Real>::max();
        for (size_t i = 0; i < particles.TotalRealParticles(); ++i)
        {
            Real wall_distance = FiberNormalDistance(positions[i]);
            if (wall_distance < -0.25 * ParticleSpacing() ||
                wall_distance > 1.5 * ParticleSpacing())
                continue;
            footprint_left = std::min(footprint_left, positions[i][0]);
            footprint_right = std::max(footprint_right, positions[i][0]);
        }
        if (!(footprint_left < footprint_right))
            return;

        const Real band_width =
            breakup_options.force_control_band_in_dx * ParticleSpacing();
        const Real radial_limit = fibre_radius +
            (breakup_options.contact_zone_in_dx + 2.0) * ParticleSpacing();
        auto write_side = [&](int side, Real endpoint)
        {
            size_t count = 0;
            Real velocity_x = 0.0;
            Real curvature_fx = 0.0;
            Real boundary_pressure_fx = 0.0;
            Real sph_pressure_fx = 0.0;
            Real total_viscous_fx = 0.0;
            Real wall_viscous_fx = 0.0;
            Real wall_adhesion_fx = 0.0;
            Real wetting_fx = 0.0;
            Real solid_liquid_fx = 0.0;
            Real vdw_fx = 0.0;
            for (size_t i = 0; i < particles.TotalRealParticles(); ++i)
            {
                Real wall_distance = FiberNormalDistance(positions[i]);
                if (wall_distance < -0.25 * ParticleSpacing() ||
                    wall_distance >
                        (breakup_options.contact_zone_in_dx + 2.0) *
                            ParticleSpacing())
                    continue;
                Real inward_distance = side < 0
                                           ? positions[i][0] - endpoint
                                           : endpoint - positions[i][0];
                if (inward_distance < -0.5 * ParticleSpacing() ||
                    inward_distance > band_width)
                    continue;
                ++count;
                velocity_x += velocities[i][0];
                if (jfm_curvature_forces != nullptr)
                    curvature_fx += jfm_curvature_forces[i][0];
                if (pressure_jump_forces != nullptr)
                    boundary_pressure_fx += pressure_jump_forces[i][0];
                sph_pressure_fx += diagnostic_pressure_forces[i][0];
                total_viscous_fx += viscous_forces[i][0] +
                                    axisymmetric_viscous_forces[i][0];
                wall_viscous_fx += wall_viscous_forces[i][0];
                wall_adhesion_fx += wall_adhesion_forces[i][0];
                wetting_fx += wetting_forces[i][0];
                solid_liquid_fx += solid_liquid_forces[i][0];
                vdw_fx += van_der_waals_forces[i][0];
            }
            Real resolved_total_fx = curvature_fx + boundary_pressure_fx +
                                     sph_pressure_fx + total_viscous_fx +
                                     wall_viscous_fx + wall_adhesion_fx +
                                     wetting_fx + solid_liquid_fx + vdw_fx;
            substep_force_file << time << "," << T << ","
                               << (side < 0 ? "left" : "right") << ","
                               << endpoint << "," << count << ","
                               << velocity_x / (static_cast<Real>(count) + TinyReal) << ","
                               << curvature_fx << "," << boundary_pressure_fx << ","
                               << sph_pressure_fx << "," << total_viscous_fx << ","
                               << wall_viscous_fx << "," << wall_adhesion_fx << ","
                               << wetting_fx << "," << solid_liquid_fx << ","
                               << vdw_fx << ","
                               << resolved_total_fx << "\n";
        };
        write_side(-1, footprint_left);
        write_side(1, footprint_right);
        substep_force_file.flush();
        do
            next_force_sample_T += breakup_options.force_sample_dT;
        while (next_force_sample_T <= T + 1.0e-10);
    };
    Real last_diagnostic_time = -1.0;

    auto write_diagnostics = [&](Real time)
    {
        if (last_diagnostic_time >= 0.0 &&
            std::abs(time - last_diagnostic_time) <
                1.0e-10 * std::max(Real(1.0), PhysicalEndTime()))
            return true;
        indicate_free_surface.exec();
        local_capillary_force.exec();
        pressure_force_diagnostic.exec();
        TopologyMetrics topology = MeasureTopology(particles, positions);
        bool finite = true;
        size_t escaped = 0;
        Real max_speed = 0.0;
        Real max_capillary_force = 0.0;
        Real max_wetting_force = 0.0;
        Real max_solid_liquid_force = 0.0;
        Real max_wall_adhesion_force = 0.0;
        Real max_van_der_waals_force = 0.0;
        Real max_van_der_waals_pressure = 0.0;
        size_t van_der_waals_surface_particles = 0;
        Real min_density = std::numeric_limits<Real>::max();
        Real max_density = -std::numeric_limits<Real>::max();
        Real max_radial_coordinate = -std::numeric_limits<Real>::max();
        Real ring_mass = 0.0;
        size_t surface_particles = 0;
        size_t under_resolved_surface_particles = 0;
        size_t pca_surface_particles = 0;
        size_t contact_line_particles = 0;
        size_t left_contact_particles = 0;
        size_t right_contact_particles = 0;
        Vecd left_capillary_force = Vecd::Zero();
        Vecd right_capillary_force = Vecd::Zero();
        Vecd left_normal_sum = Vecd::Zero();
        Vecd right_normal_sum = Vecd::Zero();
        Vecd left_geometric_normal_sum = Vecd::Zero();
        Vecd right_geometric_normal_sum = Vecd::Zero();
        Real left_geometric_cosine_sum = 0.0;
        Real right_geometric_cosine_sum = 0.0;
        Real left_corrected_cosine_sum = 0.0;
        Real right_corrected_cosine_sum = 0.0;
        Real left_target_cosine_sum = 0.0;
        Real right_target_cosine_sum = 0.0;
        Vecd left_velocity_sum = Vecd::Zero();
        Vecd right_velocity_sum = Vecd::Zero();
        Real left_pressure_sum = 0.0;
        Real right_pressure_sum = 0.0;
        Real left_axial_curvature_sum = 0.0;
        Real left_hoop_curvature_sum = 0.0;
        Real right_axial_curvature_sum = 0.0;
        Real right_hoop_curvature_sum = 0.0;
        Real surface_left_contact_x = -std::numeric_limits<Real>::infinity();
        Real surface_right_contact_x = std::numeric_limits<Real>::infinity();
        Real surface_left_edge_vx = 0.0;
        Real surface_right_edge_vx = 0.0;
        Real surface_left_edge_fx = 0.0;
        Real surface_right_edge_fx = 0.0;
        Vecd surface_left_edge_normal = Vecd::Zero();
        Real surface_left_edge_kappa_axial = 0.0;
        Real surface_left_edge_kappa_hoop = 0.0;
        int surface_left_edge_support = 0;
        for (size_t i = 0; i < particles.TotalRealParticles(); ++i)
        {
            finite = finite && std::isfinite(densities[i]) &&
                     std::isfinite(positions[i][0]) && std::isfinite(positions[i][1]) &&
                     std::isfinite(velocities[i][0]) && std::isfinite(velocities[i][1]);
            if (FiberNormalDistance(positions[i]) < -spacing ||
                positions[i][1] > RadialDomainUpperBound() - boundary_width ||
                positions[i][0] < axial_lower - axial_margin ||
                positions[i][0] > axial_upper + axial_margin)
                ++escaped;
            max_speed = std::max(max_speed, velocities[i].norm());
            max_capillary_force = std::max(max_capillary_force, capillary_forces[i].norm());
            max_wetting_force = std::max(max_wetting_force, wetting_forces[i].norm());
            max_solid_liquid_force = std::max(max_solid_liquid_force,
                                              solid_liquid_forces[i].norm());
            max_wall_adhesion_force = std::max(max_wall_adhesion_force,
                                               wall_adhesion_forces[i].norm());
            max_van_der_waals_force = std::max(max_van_der_waals_force,
                                               van_der_waals_forces[i].norm());
            max_van_der_waals_pressure = std::max(max_van_der_waals_pressure,
                                                  van_der_waals_pressures[i]);
            van_der_waals_surface_particles +=
                van_der_waals_surface[i] == 1 ? 1 : 0;
            min_density = std::min(min_density, densities[i]);
            max_density = std::max(max_density, densities[i]);
            max_radial_coordinate = std::max(max_radial_coordinate, positions[i][1]);
            if (surface_indicators[i] == 1)
            {
                ++surface_particles;
                if (surface_support_neighbors[i] <= 1)
                    ++under_resolved_surface_particles;
                if (pca_normal_used[i] == 1)
                    ++pca_surface_particles;
            }
            if (contact_line_indicators[i] == 1)
            {
                ++contact_line_particles;
                if (positions[i][0] < 0.0)
                {
                    if (positions[i][0] > surface_left_contact_x)
                    {
                        surface_left_contact_x = positions[i][0];
                        surface_left_edge_vx = velocities[i][0];
                        surface_left_edge_fx = capillary_forces[i][0];
                        surface_left_edge_normal = topology_normals[i];
                        surface_left_edge_support = surface_support_neighbors[i];
                        if (jfm_axial_curvatures != nullptr)
                        {
                            surface_left_edge_kappa_axial =
                                jfm_axial_curvatures[i];
                            surface_left_edge_kappa_hoop =
                                jfm_hoop_curvatures[i];
                        }
                    }
                    ++left_contact_particles;
                    left_capillary_force += capillary_forces[i];
                    left_normal_sum += topology_normals[i];
                    left_geometric_normal_sum += jfm_smoothed_normals[i];
                    if (jfm_smoothed_normals[i].norm() > TinyReal)
                        left_geometric_cosine_sum +=
                            -jfm_smoothed_normals[i][1] /
                            jfm_smoothed_normals[i].norm();
                    if (topology_normals[i].norm() > TinyReal)
                        left_corrected_cosine_sum +=
                            -topology_normals[i][1] /
                            topology_normals[i].norm();
                    left_target_cosine_sum += dynamic_contact_cosines[i];
                    left_velocity_sum += velocities[i];
                    left_pressure_sum += pressures[i];
                    if (jfm_axial_curvatures != nullptr)
                    {
                        left_axial_curvature_sum += jfm_axial_curvatures[i];
                        left_hoop_curvature_sum += jfm_hoop_curvatures[i];
                    }
                }
                else
                {
                    if (positions[i][0] < surface_right_contact_x)
                    {
                        surface_right_contact_x = positions[i][0];
                        surface_right_edge_vx = velocities[i][0];
                        surface_right_edge_fx = capillary_forces[i][0];
                    }
                    ++right_contact_particles;
                    right_capillary_force += capillary_forces[i];
                    right_normal_sum += topology_normals[i];
                    right_geometric_normal_sum += jfm_smoothed_normals[i];
                    if (jfm_smoothed_normals[i].norm() > TinyReal)
                        right_geometric_cosine_sum +=
                            -jfm_smoothed_normals[i][1] /
                            jfm_smoothed_normals[i].norm();
                    if (topology_normals[i].norm() > TinyReal)
                        right_corrected_cosine_sum +=
                            -topology_normals[i][1] /
                            topology_normals[i].norm();
                    right_target_cosine_sum += dynamic_contact_cosines[i];
                    right_velocity_sum += velocities[i];
                    right_pressure_sum += pressures[i];
                    if (jfm_axial_curvatures != nullptr)
                    {
                        right_axial_curvature_sum += jfm_axial_curvatures[i];
                        right_hoop_curvature_sum += jfm_hoop_curvatures[i];
                    }
                }
            }
            ring_mass += run_options.FullAxisymmetric()
                             ? 2.0 * Pi * positions[i][1] * masses[i]
                             : 2.0 * Pi * ring_masses[i];
        }
        Real T = surface_tension * time / (DynamicViscosity() * fibre_radius);
        Real ring_error = (ring_mass - initial_ring_mass) / (initial_ring_mass + TinyReal);
        Real adsorbed_ring_mass = film_rupture.cumulativeAdsorbedRingMass();
        Real converted_ring_mass = film_rupture.cumulativeConvertedRingMass();
        Real recycled_ring_mass = film_rupture.cumulativeRecycledRingMass();
        Real pending_recycle_ring_mass = film_rupture.pendingRecycleRingMass();
        Real conserved_ring_mass = ring_mass + adsorbed_ring_mass +
                                   pending_recycle_ring_mass;
        Real conserved_ring_error =
            (conserved_ring_mass - initial_ring_mass) /
            (initial_ring_mass + TinyReal);
        Real pca_fraction = static_cast<Real>(pca_surface_particles) /
                            (static_cast<Real>(surface_particles) + TinyReal);
        Real left_count = static_cast<Real>(left_contact_particles) + TinyReal;
        Real right_count = static_cast<Real>(right_contact_particles) + TinyReal;
        auto contact_angle_from_cosine = [](Real cosine_sum,
                                            size_t particle_count)
        {
            if (particle_count == 0)
                return Real(-1.0);
            Real cosine = std::clamp(
                cosine_sum / static_cast<Real>(particle_count),
                Real(-1.0), Real(1.0));
            return std::acos(cosine) * 180.0 / Pi;
        };
        Real left_geometric_contact_angle = contact_angle_from_cosine(
            left_geometric_cosine_sum, left_contact_particles);
        Real right_geometric_contact_angle = contact_angle_from_cosine(
            right_geometric_cosine_sum, right_contact_particles);
        Real left_corrected_contact_angle = contact_angle_from_cosine(
            left_corrected_cosine_sum, left_contact_particles);
        Real right_corrected_contact_angle = contact_angle_from_cosine(
            right_corrected_cosine_sum, right_contact_particles);
        Real left_target_contact_angle = contact_angle_from_cosine(
            left_target_cosine_sum, left_contact_particles);
        Real right_target_contact_angle = contact_angle_from_cosine(
            right_target_cosine_sum, right_contact_particles);
        Real surface_contact_gap_width = 0.0;
        if (std::isfinite(surface_left_contact_x) &&
            std::isfinite(surface_right_contact_x))
            surface_contact_gap_width = std::max(
                Real(0.0), surface_right_contact_x - surface_left_contact_x -
                               ParticleSpacing());
        topology_file << time << "," << T << "," << topology.dry_fraction << ","
                      << topology.dry_bins << "," << topology.total_bins << ","
                      << topology.wet_regions << "," << topology.central_gap_width << ","
                      << topology.left_contact_x << "," << topology.right_contact_x << ","
                      << contact_line_particles << "," << surface_contact_gap_width << ","
                      << surface_left_contact_x << "," << surface_right_contact_x << ","
                      << surface_left_edge_vx << "," << surface_right_edge_vx << ","
                      << surface_left_edge_fx << "," << surface_right_edge_fx << ","
                      << surface_left_edge_normal[0] << ","
                      << surface_left_edge_normal[1] << ","
                      << surface_left_edge_kappa_axial << ","
                      << surface_left_edge_kappa_hoop << ","
                      << surface_left_edge_support << ","
                      << left_contact_particles << ","
                      << right_contact_particles << ","
                      << left_capillary_force[0] << "," << left_capillary_force[1] << ","
                      << right_capillary_force[0] << "," << right_capillary_force[1] << ","
                      << left_normal_sum[0] / left_count << ","
                      << left_normal_sum[1] / left_count << ","
                      << right_normal_sum[0] / right_count << ","
                      << right_normal_sum[1] / right_count << ","
                      << left_geometric_contact_angle << ","
                      << right_geometric_contact_angle << ","
                      << left_corrected_contact_angle << ","
                      << right_corrected_contact_angle << ","
                      << left_target_contact_angle << ","
                      << right_target_contact_angle << ","
                      << left_velocity_sum[0] / left_count << ","
                      << left_velocity_sum[1] / left_count << ","
                      << right_velocity_sum[0] / right_count << ","
                      << right_velocity_sum[1] / right_count << ","
                      << left_pressure_sum / left_count << ","
                      << right_pressure_sum / right_count << ","
                      << left_axial_curvature_sum / left_count << ","
                      << left_hoop_curvature_sum / left_count << ","
                      << right_axial_curvature_sum / right_count << ","
                      << right_hoop_curvature_sum / right_count << ","
                      << topology.minimum_film << ","
                      << topology.maximum_film << "," << topology.minimum_film_layers << ","
                      << ring_mass << "," << ring_error << ","
                      << converted_ring_mass << "," << adsorbed_ring_mass << ","
                      << recycled_ring_mass << "," << pending_recycle_ring_mass << ","
                      << conserved_ring_mass << ","
                      << conserved_ring_error << ","
                      << film_rupture.removedParticles() << ","
                      << film_rupture.acceptedEvents() << ","
                      << max_speed << "," << max_capillary_force << ","
                      << max_wetting_force << "," << max_solid_liquid_force << ","
                      << max_wall_adhesion_force << ","
                      << wall_adhesion_activation_factor << ","
                      << max_van_der_waals_force << ","
                      << max_van_der_waals_pressure << ","
                      << van_der_waals_surface_particles << ","
                      << min_density << "," << max_density << ","
                      << (finite ? 1 : 0) << "," << escaped << ","
                      << under_resolved_surface_particles << ","
                      << max_radial_coordinate << "," << pca_fraction << ","
                      << (post_jfm_free_relaxation_active ? 1 : 0) << "\n";
        topology_file.flush();
        std::cout << std::fixed << std::setprecision(5)
                  << "T=" << T << " dry=" << topology.dry_fraction
                  << " gap=" << topology.central_gap_width
                  << " surface_gap=" << surface_contact_gap_width
                  << " contact_particles=" << contact_line_particles
                  << " attached_regions=" << topology.wet_regions
                  << " max_speed=" << max_speed
                  << " under_resolved_surface=" << under_resolved_surface_particles
                  << std::endl;
        last_diagnostic_time = time;
        return finite && escaped == 0;
    };

    Real output_interval = PhysicalEndTime() / static_cast<Real>(run_options.frames);
    Real dt = 0.0;
    size_t iteration = 0;
    write_substep_force_budget(physical_time);
    write_states.writeToFile(0);
    write_diagnostics(physical_time);

    TickCount start = TickCount::now();
    while (physical_time < PhysicalEndTime())
    {
        Real integration_time = 0.0;
        while (integration_time < output_interval && physical_time < PhysicalEndTime())
        {
            if (run_options.FullAxisymmetric())
                update_axisymmetric_weights.exec();
            indicate_free_surface.exec();
            if (breakup_options.density_reinitialization)
            {
                density_by_summation.exec();
                if (run_options.FullAxisymmetric())
                    update_axisymmetric_weights.exec();
            }
            viscous_force_inner.exec();
            viscous_force_wall.exec();
            if (run_options.FullAxisymmetric())
                axisymmetric_viscous_force.exec();
            Real Dt = advection_time_step.exec();
            Real relaxation_time = 0.0;
            while (relaxation_time < Dt && integration_time < output_interval &&
                   physical_time < PhysicalEndTime())
            {
                local_capillary_force.exec();
                if (breakup_options.post_breakup_models &&
                    breakup_options.post_jfm_free_relaxation &&
                    !post_jfm_free_relaxation_active &&
                    film_rupture.postBreakupDetectionWindowElapsed(
                        breakup_options.gamma_lg * physical_time /
                        (DynamicViscosity() * fibre_radius + TinyReal)))
                {
                    post_jfm_free_relaxation_active = true;
                    van_der_waals_force.setEnabled(false);
                    if (breakup_options.thermal_noise_stop_after_rupture)
                        continuous_thermal_noise.setEnabled(false);
                    film_rupture.freezeForFreeRelaxation(
                        breakup_options.post_jfm_continue_recycle);
                    Real activation_T = breakup_options.gamma_lg * physical_time /
                                        (DynamicViscosity() * fibre_radius + TinyReal);
                    std::cout << "Post-breakup free relaxation activated at T="
                              << activation_T
                              << ": Xu traction and rupture maintenance are frozen; mass recycling is "
                              << (breakup_options.post_jfm_continue_recycle
                                      ? "continued."
                                      : "frozen.")
                              << " Dry-core protection is "
                              << (breakup_options.post_jfm_keep_dry_core
                                      ? "continued."
                                      : "off.")
                              << std::endl;
                }
                van_der_waals_force.exec();
                if (breakup_options.substep_force_diagnostics)
                {
                    pressure_force_diagnostic.exec();
                    write_substep_force_budget(physical_time);
                }
                dt = acoustic_surface_time_step.exec();
                dt = std::min(dt, local_capillary_force.stableTimeStep());
                dt = std::min(dt, van_der_waals_force.stableTimeStep());
                dt = std::min(dt, Dt - relaxation_time);
                dt = std::min(dt, output_interval - integration_time);
                dt = std::min(dt, PhysicalEndTime() - physical_time);
                pressure_relaxation.exec(dt);
                density_relaxation.exec(dt);
                continuous_thermal_noise.exec(dt);
                if (run_options.FullAxisymmetric())
                {
                    axisymmetric_continuity.exec(dt);
                    update_axisymmetric_weights.exec();
                }
                conical_end_buffer.exec(dt);
                relaxation_time += dt;
                integration_time += dt;
                physical_time += dt;
            }

            if (breakup_options.particle_regularization &&
                breakup_options.regularization_coefficient > 0.0)
                regularize_bulk_particles.exec();
            if (breakup_options.surface_regularization &&
                breakup_options.surface_regularization_coefficient > 0.0)
                regularize_surface_particles.exec();
            if (run_options.FullAxisymmetric() &&
                (breakup_options.particle_regularization ||
                 breakup_options.surface_regularization))
                update_axisymmetric_weights.exec();
            if (breakup_options.post_breakup_models)
                film_rupture.exec();
            update_wall_adhesion_activation();
            ++iteration;
            if (!IsConicalFiber())
                periodic_condition.bounding_.exec();
            if (!IsConicalFiber() && iteration % 100 == 0)
                particle_sorting.exec();
            liquid.updateCellLinkedList();
            if (!IsConicalFiber())
                periodic_condition.update_cell_linked_list_.exec();
            liquid_complex.updateConfiguration();
            // The capillary and fibre-viscous models intentionally see only
            // the physical fibre, not the numerical symmetry boundary.
            liquid_fibre_contact.updateConfiguration();
        }

        if (!write_diagnostics(physical_time))
        {
            std::cerr << "Non-finite or escaped-particle state at time="
                      << physical_time << std::endl;
            // Preserve the first rejected state for particle-level diagnosis.
            // Without this emergency frame the CSV reports the failure, but
            // the particles which caused it are lost before they can be
            // inspected in ParaView.
            fibre.setNewlyUpdated();
            write_states.writeToFile();
            return 3;
        }
        fibre.setNewlyUpdated();
        write_states.writeToFile();
    }

    TimeInterval elapsed = TickCount::now() - start;
    std::cout << "Completed " << BreakupLabel() << " in " << elapsed.seconds()
              << " seconds. Output: " << output_folder << std::endl;
    return 0;
}
