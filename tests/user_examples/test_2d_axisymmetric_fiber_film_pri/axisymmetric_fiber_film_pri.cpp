/**
 * @file axisymmetric_fiber_film_pri.cpp
 * @brief Preliminary half-meridional SPH model of the Plateau-Rayleigh
 *        instability of a viscous film coating a rigid circular fibre.
 *
 * The computational coordinates are (x,r). Only r >= 0 is solved. The VTP
 * output is mirrored during post-processing. This user example deliberately
 * does not modify the SPHinXsys public API.
 */
#include "sphinxsys.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace SPH;

namespace
{
struct RunOptions
{
#ifdef SPHINXSYS_FINITE_TAPERED_FIBER_DEFAULTS
    std::string case_name = "dominant";
    std::string axisym = "full";
    std::string regime = "early";
    Real film_ratio = 1.0;
    Real amplitude_ratio = 0.01;
    Real resolution = 16.0;
    Real oh = 0.5;
    Real end_T = 8.0;
    int frames = 16;
    int interface_modes = 12;
#elif defined(SPHINXSYS_CONICAL_FIBER_DEFAULTS)
    std::string case_name = "static";
    std::string axisym = "full";
    std::string regime = "early";
    Real film_ratio = 0.50;
    Real amplitude_ratio = 0.0;
    Real resolution = 12.0;
    Real oh = 0.5;
    Real end_T = 4.0;
    int frames = 8;
    int interface_modes = 12;
#else
    std::string case_name = "dominant";
    std::string axisym = "full";
#ifdef SPHINXSYS_FIBER_FILM_BEAD_DEFAULTS
    std::string regime = "bead";
    Real film_ratio = 1.0;
    Real amplitude_ratio = 0.40;
    Real resolution = 12.0;
    Real oh = 0.20;
    Real end_T = 250.0;
    int frames = 80;
    int interface_modes = 24;
#else
    std::string regime = "early";
    Real film_ratio = 1.0;
    Real amplitude_ratio = 0.10;
    Real resolution = 16.0;
    Real oh = 0.5;
    Real end_T = 40.0;
    int frames = 80;
    int interface_modes = 12;
#endif
#endif
    // Optional fixed axial domain used for data-driven wavelength selection.
    // A positive value is expressed in fibre radii and deliberately removes
    // the theoretical fastest wavelength from the initial-condition setup.
    Real domain_length_over_a =
#ifdef SPHINXSYS_SYMMETRIC_TAPERED_FIBER_DEFAULTS
        60.0;
#elif defined(SPHINXSYS_CONICAL_FIBER_DEFAULTS) || \
    defined(SPHINXSYS_FINITE_TAPERED_FIBER_DEFAULTS)
        30.0;
#else
        0.0;
#endif
    // Optional single-mode wavelength for a direct dispersion scan. This is
    // separate from the broadband domain length so that a candidate physical
    // wavelength can be tested without the integer relation lambda=L/m.
    Real wavelength_over_a = 0.0;
    int broadband_mode_min = 3;
    int broadband_mode_max = 12;
    int disturbance_seed = 0;
    Real capillary_scale = 1.0;
    bool free_surface_pressure_jump = false;
    std::string surface_formulation = "balanced-curvature";
    // The particle staircase shifts the sampled zero Fourier mode by up to
    // half a spacing.  "nominal" anchors only that zero mode to the prescribed
    // mean outer radius; all resolved perturbation modes remain data-driven.
    std::string surface_mean = "nominal";
    // A one-sided regularized delta must integrate to unity through the
    // interface band. Column normalization removes the dependence of its
    // discrete integral on the vertical lattice phase.
    std::string surface_delta = "column-normalized";
    // The explicit SPHinXsys viscous-force operator is the physical default.
    // Pairwise implicit damping remains available only as a numerical
    // diagnostic; the independent decay benchmark shows excessive damping.
    std::string viscosity_integration = "explicit";
    std::string fiber_profile =
#ifdef SPHINXSYS_FINITE_TAPERED_FIBER_DEFAULTS
        "finite-conical";
#elif defined(SPHINXSYS_CONICAL_FIBER_DEFAULTS)
        "conical";
#else
        "uniform";
#endif
    // Radius at the thin end divided by the nominal thick-end radius a.
    Real tip_radius_ratio = 0.60;
    // Straight coated extensions placed outside the tapered observation
    // section. The far-end relaxation buffer is moved into these reservoirs.
    Real reservoir_length_over_a =
#ifdef SPHINXSYS_FINITE_TAPERED_FIBER_DEFAULTS
        5.0;
#else
        0.0;
#endif
    // Length used to ramp the wall slope continuously between each straight
    // reservoir and the linear central cone.
    Real cone_transition_over_a = 2.0;
    // Length of the smooth fine-end cap in the finite-conical model.  The
    // coarse head is a true half circle with axial length a. The remainder of
    // each reservoir extension is a dry straight shoulder.
    Real finite_cap_length_over_a = 2.0;
    // Fluid particles in these end bands are held at the initial coated-film
    // state. This supplies finite-domain reservoirs without pretending that
    // unequal-radius ends are periodic copies of one another.
    Real end_buffer_fraction = 0.12;

    bool FullAxisymmetric() const
    {
        if (axisym == "full")
            return true;
        if (axisym == "curvature-only")
            return false;
        throw std::runtime_error("Unknown --axisym value '" + axisym +
                                 "'. Use full or curvature-only.");
    }

    Real WavelengthFactor() const
    {
        if (case_name == "static" || case_name == "dominant" ||
            case_name == "viscous-decay")
            return 1.0;
        if (case_name == "short")
            return 0.5;
        if (case_name == "long")
            return 1.5;
        if (case_name == "multi")
            return 3.0;
        if (case_name == "broadband")
            return 1.0;
        throw std::runtime_error("Unknown --case value '" + case_name +
                                 "'. Use static, viscous-decay, short, dominant, long, multi, or broadband.");
    }

    std::string Label() const
    {
        std::string base_label = case_name + "_" + axisym + "_h" +
               std::to_string(static_cast<int>(std::round(100.0 * film_ratio))) +
               "_r" + std::to_string(static_cast<int>(std::round(resolution)));
        if (fiber_profile == "conical" || fiber_profile == "finite-conical")
            base_label += "_cone_q" +
                          std::to_string(static_cast<int>(std::round(100.0 * tip_radius_ratio)));
        if ((fiber_profile == "conical" || fiber_profile == "finite-conical") &&
            reservoir_length_over_a > 0.0)
            base_label += "_res" +
                          std::to_string(static_cast<int>(std::round(reservoir_length_over_a)));
        if (fiber_profile == "finite-conical")
            base_label += "_cap" +
                          std::to_string(static_cast<int>(std::round(finite_cap_length_over_a)));
#ifdef SPHINXSYS_SYMMETRIC_TAPERED_FIBER_DEFAULTS
        base_label += "_symmetric_fine_coarse_fine";
#endif
        if (regime == "bead")
            base_label += "_bead_m" + std::to_string(interface_modes);
        if (case_name == "broadband")
            base_label += "_L" + std::to_string(static_cast<int>(std::round(domain_length_over_a))) +
                          "_m" + std::to_string(broadband_mode_min) + "to" +
                          std::to_string(broadband_mode_max) + "_s" +
                          std::to_string(disturbance_seed);
        else if (wavelength_over_a > 0.0)
            base_label += "_lambda" +
                          std::to_string(static_cast<int>(std::round(100.0 * wavelength_over_a)));
        return base_label;
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
            options.case_name = argument.substr(7);
        else if (StartsWith(argument, "--axisym="))
            options.axisym = argument.substr(9);
        else if (StartsWith(argument, "--regime="))
            options.regime = argument.substr(9);
        else if (StartsWith(argument, "--film-ratio="))
            options.film_ratio = std::stod(argument.substr(13));
        else if (StartsWith(argument, "--amplitude-ratio="))
            options.amplitude_ratio = std::stod(argument.substr(18));
        else if (StartsWith(argument, "--resolution="))
            options.resolution = std::stod(argument.substr(13));
        else if (StartsWith(argument, "--oh="))
            options.oh = std::stod(argument.substr(5));
        else if (StartsWith(argument, "--end-T="))
            options.end_T = std::stod(argument.substr(8));
        else if (StartsWith(argument, "--frames="))
            options.frames = std::stoi(argument.substr(9));
        else if (StartsWith(argument, "--interface-modes="))
            options.interface_modes = std::stoi(argument.substr(18));
        else if (StartsWith(argument, "--domain-length-over-a="))
            options.domain_length_over_a = std::stod(argument.substr(
                std::string("--domain-length-over-a=").size()));
        else if (StartsWith(argument, "--wavelength-over-a="))
            options.wavelength_over_a = std::stod(argument.substr(
                std::string("--wavelength-over-a=").size()));
        else if (StartsWith(argument, "--broadband-mode-min="))
            options.broadband_mode_min = std::stoi(argument.substr(
                std::string("--broadband-mode-min=").size()));
        else if (StartsWith(argument, "--broadband-mode-max="))
            options.broadband_mode_max = std::stoi(argument.substr(
                std::string("--broadband-mode-max=").size()));
        else if (StartsWith(argument, "--disturbance-seed="))
            options.disturbance_seed = std::stoi(argument.substr(
                std::string("--disturbance-seed=").size()));
        else if (StartsWith(argument, "--capillary-scale="))
            options.capillary_scale = std::stod(argument.substr(18));
        else if (StartsWith(argument, "--free-surface-pressure-jump="))
        {
            std::string value = argument.substr(29);
            if (value != "on" && value != "off")
                throw std::runtime_error(
                    "--free-surface-pressure-jump must be on or off.");
            options.free_surface_pressure_jump = value == "on";
        }
        else if (StartsWith(argument, "--surface-formulation="))
            options.surface_formulation = argument.substr(22);
        else if (StartsWith(argument, "--surface-mean="))
            options.surface_mean = argument.substr(15);
        else if (StartsWith(argument, "--surface-delta="))
            options.surface_delta = argument.substr(16);
        else if (StartsWith(argument, "--viscosity-integration="))
            options.viscosity_integration = argument.substr(24);
        else if (StartsWith(argument, "--fiber-profile="))
            options.fiber_profile = argument.substr(16);
        else if (StartsWith(argument, "--tip-radius-ratio="))
            options.tip_radius_ratio = std::stod(argument.substr(19));
        else if (StartsWith(argument, "--reservoir-length-over-a="))
            options.reservoir_length_over_a = std::stod(argument.substr(26));
        else if (StartsWith(argument, "--cone-transition-over-a="))
            options.cone_transition_over_a = std::stod(argument.substr(25));
        else if (StartsWith(argument, "--finite-cap-length-over-a="))
            options.finite_cap_length_over_a = std::stod(argument.substr(27));
        else if (StartsWith(argument, "--end-buffer-fraction="))
            options.end_buffer_fraction = std::stod(argument.substr(22));
        else
            sph_arguments.push_back(argument);
    }

    options.WavelengthFactor();
    options.FullAxisymmetric();
    if (options.case_name == "viscous-decay" && options.FullAxisymmetric())
        throw std::runtime_error(
            "--case=viscous-decay is the planar viscosity benchmark; use --axisym=curvature-only.");
    if (options.film_ratio <= 0.0)
        throw std::runtime_error("--film-ratio must be positive.");
    if (options.regime != "early" && options.regime != "bead")
        throw std::runtime_error("--regime must be early or bead.");
    if (options.amplitude_ratio < 0.0 || options.amplitude_ratio >= 0.95)
        throw std::runtime_error("--amplitude-ratio must be in [0,0.95).");
    if (options.resolution < 8.0)
        throw std::runtime_error("--resolution must be at least 8 particles through the film.");
    if (options.oh <= 0.0)
        throw std::runtime_error("--oh must be positive.");
    if (options.end_T <= 0.0)
        throw std::runtime_error("--end-T must be positive.");
    if (options.frames < 2)
        throw std::runtime_error("--frames must be at least 2.");
    if (options.interface_modes < 4 || options.interface_modes > 64)
        throw std::runtime_error("--interface-modes must be between 4 and 64.");
    if (options.domain_length_over_a < 0.0)
        throw std::runtime_error("--domain-length-over-a must be non-negative.");
    if (options.wavelength_over_a < 0.0)
        throw std::runtime_error("--wavelength-over-a must be non-negative.");
    if (options.case_name == "broadband" && options.domain_length_over_a <= 0.0)
        throw std::runtime_error("--case=broadband requires --domain-length-over-a > 0.");
    if (options.case_name == "broadband" && options.wavelength_over_a > 0.0)
        throw std::runtime_error(
            "--wavelength-over-a is a single-mode option and cannot be combined with --case=broadband.");
    if (options.broadband_mode_min < 1 ||
        options.broadband_mode_max <= options.broadband_mode_min ||
        options.broadband_mode_max > 32)
        throw std::runtime_error(
            "broadband mode range must satisfy 1 <= min < max <= 32.");
    if (options.disturbance_seed < 0)
        throw std::runtime_error("--disturbance-seed must be non-negative.");
    if (options.capillary_scale < 0.0 || options.capillary_scale > 2.0)
        throw std::runtime_error("--capillary-scale must be in [0,2].");
    if (options.surface_formulation != "css" &&
        options.surface_formulation != "balanced-curvature")
        throw std::runtime_error(
            "--surface-formulation must be css or balanced-curvature.");
    if (options.surface_mean != "sampled" && options.surface_mean != "nominal")
        throw std::runtime_error(
            "--surface-mean must be sampled or nominal.");
    if (options.surface_delta != "raw" &&
        options.surface_delta != "column-normalized")
        throw std::runtime_error(
            "--surface-delta must be raw or column-normalized.");
    if (options.viscosity_integration != "explicit" &&
        options.viscosity_integration != "implicit")
        throw std::runtime_error(
            "--viscosity-integration must be explicit or implicit.");
    if (options.fiber_profile != "uniform" &&
        options.fiber_profile != "conical" &&
        options.fiber_profile != "finite-conical")
        throw std::runtime_error(
            "--fiber-profile must be uniform, conical or finite-conical.");
    if (options.tip_radius_ratio <= 0.20 || options.tip_radius_ratio > 1.0)
        throw std::runtime_error("--tip-radius-ratio must be in (0.20,1].");
    if (options.reservoir_length_over_a < 0.0)
        throw std::runtime_error("--reservoir-length-over-a must be non-negative.");
    if (options.cone_transition_over_a < 0.0)
        throw std::runtime_error("--cone-transition-over-a must be non-negative.");
    if (options.finite_cap_length_over_a <= 0.0)
        throw std::runtime_error("--finite-cap-length-over-a must be positive.");
    if (options.end_buffer_fraction < 0.0 || options.end_buffer_fraction > 0.25)
        throw std::runtime_error(
            "--end-buffer-fraction must be in [0,0.25]; zero disables both end buffers.");
    if ((options.fiber_profile == "conical" ||
         options.fiber_profile == "finite-conical") &&
        options.domain_length_over_a <= 0.0)
        throw std::runtime_error(
            "a conical fiber profile requires --domain-length-over-a > 0.");
    if (options.fiber_profile == "finite-conical" &&
        options.reservoir_length_over_a <= options.finite_cap_length_over_a)
        throw std::runtime_error(
            "finite-conical requires --reservoir-length-over-a larger than --finite-cap-length-over-a so a dry shoulder remains between film and cap.");
#ifndef SPHINXSYS_FIBER_FILM_TOPOLOGY_DEFAULTS
    if ((1.0 - options.amplitude_ratio) * options.resolution < 2.0)
        throw std::runtime_error("The initial neck must contain at least two particle layers. "
                                 "Increase --resolution or reduce --amplitude-ratio.");
#endif
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

constexpr Real fibre_radius = 1.0;
constexpr Real reference_density = 1.0;
constexpr Real surface_tension = 1.0;
constexpr Real sound_speed_factor = 30.0;
constexpr size_t maximum_diagnostic_modes = 32;

Real FilmThickness() { return run_options.film_ratio * fibre_radius; }
Real MeanOuterRadius()
{
    Real mean_fibre_radius = (run_options.fiber_profile == "conical" ||
                              run_options.fiber_profile == "finite-conical")
                                 ? 0.5 * fibre_radius *
                                       (1.0 + run_options.tip_radius_ratio)
                                 : fibre_radius;
    return mean_fibre_radius + FilmThickness();
}
Real FastestWavelength() { return 2.0 * Pi * std::sqrt(2.0) * MeanOuterRadius(); }
Real CriticalWavelength() { return 2.0 * Pi * MeanOuterRadius(); }
Real NominalDomainLength()
{
    if (run_options.domain_length_over_a > 0.0)
        return run_options.domain_length_over_a * fibre_radius;
    if (run_options.wavelength_over_a > 0.0)
        return run_options.wavelength_over_a * fibre_radius;
    return run_options.WavelengthFactor() * FastestWavelength();
}
Real ConeSectionLength()
{
    if (run_options.wavelength_over_a > 0.0)
    {
        // Align the periodic seam with the lattice without changing existing
        // predefined cases. The realized wavelength is recorded in the CSV.
        Real particle_spacing = FilmThickness() / run_options.resolution;
        return std::round(NominalDomainLength() / particle_spacing) * particle_spacing;
    }
#ifdef SPHINXSYS_FIBER_FILM_TOPOLOGY_DEFAULTS
    // A periodic particle lattice must contain an integer number of spacings.
    // Otherwise the last-to-first distance differs from the interior spacing,
    // producing a false colour gradient and capillary force at the seam.
    Real particle_spacing = FilmThickness() / run_options.resolution;
    return std::round(NominalDomainLength() / particle_spacing) * particle_spacing;
#else
    return NominalDomainLength();
#endif
}
bool IsConicalFiber()
{
    return run_options.fiber_profile == "conical" ||
           run_options.fiber_profile == "finite-conical";
}
bool IsFiniteFiber() { return run_options.fiber_profile == "finite-conical"; }
bool UseConicalEndBuffer()
{
    // A zero fraction explicitly requests fully free finite ends: no
    // position/density relaxation and no capillary-force suppression.
    if (!IsConicalFiber() || run_options.end_buffer_fraction <= TinyReal)
        return false;
#ifdef SPHINXSYS_FINITE_TAPERED_FULL_COATING_DEFAULTS
    // The closed end caps are numerical completion regions, not part of the
    // tapered RP observation section.  Holding them near their reference
    // state prevents cap retraction from launching particles into the cone.
    return true;
#else
    // Legacy finite-film cases contain physical contact lines at their ends
    // and therefore must remain free to move.
    return !IsFiniteFiber();
#endif
}
Real ReservoirLength()
{
    return IsConicalFiber()
               ? run_options.reservoir_length_over_a * fibre_radius
               : 0.0;
}
Real DomainLength() { return ConeSectionLength() + 2.0 * ReservoirLength(); }
Real DomainLowerBound() { return -0.5 * DomainLength(); }
Real DomainUpperBound() { return 0.5 * DomainLength(); }
Real ConeLowerBound() { return -0.5 * ConeSectionLength(); }
Real ConeUpperBound() { return 0.5 * ConeSectionLength(); }
Real ConeTransitionLength()
{
    return std::min(run_options.cone_transition_over_a * fibre_radius,
                    0.45 * ConeSectionLength());
}
Real FiniteCapLength()
{
    return run_options.finite_cap_length_over_a * fibre_radius;
}
Real FiniteCoarseRoundCapLength()
{
    // A circular coarse head has axial radius equal to its radial radius.
    return fibre_radius;
}
Real SmoothStep01(Real value)
{
    Real s = std::clamp(value, Real(0.0), Real(1.0));
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s));
}
Real SmoothStep01FirstDerivative(Real value)
{
    Real s = std::clamp(value, Real(0.0), Real(1.0));
    return (value <= 0.0 || value >= 1.0)
               ? 0.0
               : 30.0 * s * s * (1.0 - s) * (1.0 - s);
}
Real SmoothStep01SecondDerivative(Real value)
{
    Real s = std::clamp(value, Real(0.0), Real(1.0));
    return (value <= 0.0 || value >= 1.0)
               ? 0.0
               : 60.0 * s * (2.0 * s * s - 3.0 * s + 1.0);
}
Real ConeSlopeWeight(Real x)
{
    Real y = x - ConeLowerBound();
    Real length = ConeSectionLength();
    Real transition = ConeTransitionLength();
    if (transition <= TinyReal)
        return (y > 0.0 && y < length) ? 1.0 : 0.0;
    if (y <= 0.0 || y >= length)
        return 0.0;
    if (y < transition)
    {
        Real s = y / transition;
        return s * s * (3.0 - 2.0 * s);
    }
    if (y <= length - transition)
        return 1.0;
    Real s = (y - (length - transition)) / transition;
    Real smoothstep = s * s * (3.0 - 2.0 * s);
    return 1.0 - smoothstep;
}
Real ConeSlopeWeightDerivative(Real x)
{
    Real y = x - ConeLowerBound();
    Real length = ConeSectionLength();
    Real transition = ConeTransitionLength();
    if (transition <= TinyReal || y <= 0.0 || y >= length)
        return 0.0;
    if (y < transition)
    {
        Real s = y / transition;
        return 6.0 * s * (1.0 - s) / transition;
    }
    if (y <= length - transition)
        return 0.0;
    Real s = (y - (length - transition)) / transition;
    return -6.0 * s * (1.0 - s) / transition;
}
Real ConeProgress(Real x)
{
    Real y = x - ConeLowerBound();
    Real length = ConeSectionLength();
    Real transition = ConeTransitionLength();
    if (y <= 0.0)
        return 0.0;
    if (y >= length)
        return 1.0;
    if (transition <= TinyReal)
        return y / length;

    Real integral = 0.0;
    if (y < transition)
    {
        Real s = y / transition;
        integral = transition * (s * s * s - 0.5 * s * s * s * s);
    }
    else if (y <= length - transition)
    {
        integral = y - 0.5 * transition;
    }
    else
    {
        Real s = (y - (length - transition)) / transition;
        integral = length - 1.5 * transition +
                   transition * (s - s * s * s + 0.5 * s * s * s * s);
    }
    return integral / (length - transition);
}

#ifdef SPHINXSYS_SYMMETRIC_TAPERED_FIBER_DEFAULTS
/** Smooth fine-to-coarse progress on either half of the symmetric fibre.
 *
 * Each half retains the former one-way cone length and slope.  The transition
 * length only rounds the joins: the maximum radius is still attained at the
 * single centre point, not over a finite cylindrical plateau.
 */
Real SymmetricHalfSlopeWeight(Real y)
{
    const Real length = 0.5 * ConeSectionLength();
    const Real transition = std::min(ConeTransitionLength(), 0.45 * length);
    if (y <= 0.0 || y >= length)
        return 0.0;
    if (transition <= TinyReal)
        return 1.0;
    if (y < transition)
    {
        Real s = y / transition;
        return s * s * (3.0 - 2.0 * s);
    }
    if (y <= length - transition)
        return 1.0;
    Real s = (y - (length - transition)) / transition;
    return 1.0 - s * s * (3.0 - 2.0 * s);
}
Real SymmetricHalfSlopeWeightDerivative(Real y)
{
    const Real length = 0.5 * ConeSectionLength();
    const Real transition = std::min(ConeTransitionLength(), 0.45 * length);
    if (transition <= TinyReal || y <= 0.0 || y >= length)
        return 0.0;
    if (y < transition)
    {
        Real s = y / transition;
        return 6.0 * s * (1.0 - s) / transition;
    }
    if (y <= length - transition)
        return 0.0;
    Real s = (y - (length - transition)) / transition;
    return -6.0 * s * (1.0 - s) / transition;
}
Real SymmetricConeProgress(Real x)
{
    const Real half_length = 0.5 * ConeSectionLength();
    const Real transition = std::min(ConeTransitionLength(), 0.45 * half_length);
    const Real y = half_length - std::abs(x);
    if (y <= 0.0)
        return 0.0;
    if (y >= half_length)
        return 1.0;
    if (transition <= TinyReal)
        return y / half_length;

    Real integral = 0.0;
    if (y < transition)
    {
        Real s = y / transition;
        integral = transition * (s * s * s - 0.5 * s * s * s * s);
    }
    else if (y <= half_length - transition)
        integral = y - 0.5 * transition;
    else
    {
        Real s = (y - (half_length - transition)) / transition;
        integral = half_length - 1.5 * transition +
                   transition * (s - s * s * s + 0.5 * s * s * s * s);
    }
    return integral / (half_length - transition);
}
#endif
Real FiberRadius(Real x)
{
    if (!IsConicalFiber())
        return fibre_radius;
    if (IsFiniteFiber())
    {
        Real coarse_cap_length =
#ifdef SPHINXSYS_SYMMETRIC_TAPERED_FIBER_DEFAULTS
            fibre_radius * run_options.tip_radius_ratio;
#else
            FiniteCoarseRoundCapLength();
#endif
        Real fine_cap_length =
#ifdef SPHINXSYS_FINITE_TAPERED_FULL_COATING_DEFAULTS
            fibre_radius * run_options.tip_radius_ratio;
#else
            FiniteCapLength();
#endif
        if (x <= DomainLowerBound() || x >= DomainUpperBound())
            return 0.0;
        if (x < DomainLowerBound() + coarse_cap_length)
        {
#ifdef SPHINXSYS_SYMMETRIC_TAPERED_FIBER_DEFAULTS
            const Real radius = fibre_radius * run_options.tip_radius_ratio;
            const Real centre = DomainLowerBound() + radius;
            const Real axial = x - centre;
            return std::sqrt(std::max(Real(0.0),
                                      radius * radius - axial * axial));
#else
            // Left half of a circle of radius a.  Unlike the former smooth
            // radius ramp, this has a vertical tangent at the axis and is
            // therefore geometrically round after mirroring/revolution.
            Real s = (x - DomainLowerBound()) / coarse_cap_length;
            return fibre_radius *
                   std::sqrt(std::max(Real(0.0), s * (2.0 - s)));
#endif
        }
        if (x > DomainUpperBound() - fine_cap_length)
#ifdef SPHINXSYS_FINITE_TAPERED_FULL_COATING_DEFAULTS
        {
            // The fully coated finite-fibre prototype uses a hemispherical
            // fine end.  A round end has a regular parallel free surface,
            // whereas offsetting the former pointed smooth-step cap would
            // create a curvature singularity at its tip.
            const Real radius = fibre_radius * run_options.tip_radius_ratio;
            const Real centre = DomainUpperBound() - radius;
            const Real axial = x - centre;
            return std::sqrt(std::max(Real(0.0),
                                      radius * radius - axial * axial));
        }
#else
            return fibre_radius * run_options.tip_radius_ratio *
                   SmoothStep01((DomainUpperBound() - x) / fine_cap_length);
#endif
    }
#ifdef SPHINXSYS_SYMMETRIC_TAPERED_FIBER_DEFAULTS
    return fibre_radius *
           (run_options.tip_radius_ratio +
            (1.0 - run_options.tip_radius_ratio) * SymmetricConeProgress(x));
#else
    return fibre_radius * (1.0 - (1.0 - run_options.tip_radius_ratio) *
                                     ConeProgress(x));
#endif
}
Real FiberRadiusFirstDerivative(Real x)
{
    if (!IsConicalFiber())
        return 0.0;
    if (IsFiniteFiber())
    {
        Real coarse_cap_length =
#ifdef SPHINXSYS_SYMMETRIC_TAPERED_FIBER_DEFAULTS
            fibre_radius * run_options.tip_radius_ratio;
#else
            FiniteCoarseRoundCapLength();
#endif
        Real fine_cap_length =
#ifdef SPHINXSYS_FINITE_TAPERED_FULL_COATING_DEFAULTS
            fibre_radius * run_options.tip_radius_ratio;
#else
            FiniteCapLength();
#endif
        if (x <= DomainLowerBound() || x >= DomainUpperBound())
            return 0.0;
        if (x < DomainLowerBound() + coarse_cap_length)
        {
#ifdef SPHINXSYS_SYMMETRIC_TAPERED_FIBER_DEFAULTS
            const Real radius = fibre_radius * run_options.tip_radius_ratio;
            const Real centre = DomainLowerBound() + radius;
            const Real axial = x - centre;
            const Real radial = std::sqrt(std::max(
                Real(TinyReal), radius * radius - axial * axial));
            return -axial / radial;
#else
            Real s = (x - DomainLowerBound()) / coarse_cap_length;
            Real radial_factor = std::sqrt(std::max(
                Real(TinyReal), s * (2.0 - s)));
            return fibre_radius * (1.0 - s) /
                   (coarse_cap_length * radial_factor);
#endif
        }
        if (x > DomainUpperBound() - fine_cap_length)
#ifdef SPHINXSYS_FINITE_TAPERED_FULL_COATING_DEFAULTS
        {
            const Real radius = fibre_radius * run_options.tip_radius_ratio;
            const Real centre = DomainUpperBound() - radius;
            const Real axial = x - centre;
            const Real radial = std::sqrt(std::max(
                Real(TinyReal), radius * radius - axial * axial));
            return -axial / radial;
        }
#else
            return -fibre_radius * run_options.tip_radius_ratio /
                   fine_cap_length *
                   SmoothStep01FirstDerivative(
                       (DomainUpperBound() - x) / fine_cap_length);
#endif
    }
#ifdef SPHINXSYS_SYMMETRIC_TAPERED_FIBER_DEFAULTS
    const Real half_length = 0.5 * ConeSectionLength();
    const Real y = half_length - std::abs(x);
    const Real denominator = half_length -
                             std::min(ConeTransitionLength(), 0.45 * half_length);
    const Real sign = x < 0.0 ? 1.0 : (x > 0.0 ? -1.0 : 0.0);
    return sign * fibre_radius * (1.0 - run_options.tip_radius_ratio) *
           SymmetricHalfSlopeWeight(y) / (denominator + TinyReal);
#else
    Real denominator = ConeSectionLength() - ConeTransitionLength();
    return -fibre_radius * (1.0 - run_options.tip_radius_ratio) *
           ConeSlopeWeight(x) / (denominator + TinyReal);
#endif
}
Real FiberRadiusSecondDerivative(Real x)
{
    if (!IsConicalFiber())
        return 0.0;
    if (IsFiniteFiber())
    {
        Real coarse_cap_length =
#ifdef SPHINXSYS_SYMMETRIC_TAPERED_FIBER_DEFAULTS
            fibre_radius * run_options.tip_radius_ratio;
#else
            FiniteCoarseRoundCapLength();
#endif
        Real fine_cap_length =
#ifdef SPHINXSYS_FINITE_TAPERED_FULL_COATING_DEFAULTS
            fibre_radius * run_options.tip_radius_ratio;
#else
            FiniteCapLength();
#endif
        if (x <= DomainLowerBound() || x >= DomainUpperBound())
            return 0.0;
        if (x < DomainLowerBound() + coarse_cap_length)
        {
#ifdef SPHINXSYS_SYMMETRIC_TAPERED_FIBER_DEFAULTS
            const Real radius = fibre_radius * run_options.tip_radius_ratio;
            const Real centre = DomainLowerBound() + radius;
            const Real axial = x - centre;
            const Real radial = std::sqrt(std::max(
                Real(TinyReal), radius * radius - axial * axial));
            return -radius * radius / (radial * radial * radial);
#else
            Real s = (x - DomainLowerBound()) / coarse_cap_length;
            Real radial_factor = std::sqrt(std::max(
                Real(TinyReal), s * (2.0 - s)));
            return -fibre_radius /
                   (coarse_cap_length * coarse_cap_length *
                    radial_factor * radial_factor * radial_factor);
#endif
        }
        if (x > DomainUpperBound() - fine_cap_length)
#ifdef SPHINXSYS_FINITE_TAPERED_FULL_COATING_DEFAULTS
        {
            const Real radius = fibre_radius * run_options.tip_radius_ratio;
            const Real centre = DomainUpperBound() - radius;
            const Real axial = x - centre;
            const Real radial = std::sqrt(std::max(
                Real(TinyReal), radius * radius - axial * axial));
            return -radius * radius /
                   (radial * radial * radial);
        }
#else
            return fibre_radius * run_options.tip_radius_ratio /
                   (fine_cap_length * fine_cap_length) *
                   SmoothStep01SecondDerivative(
                       (DomainUpperBound() - x) / fine_cap_length);
#endif
    }
#ifdef SPHINXSYS_SYMMETRIC_TAPERED_FIBER_DEFAULTS
    const Real half_length = 0.5 * ConeSectionLength();
    const Real y = half_length - std::abs(x);
    const Real denominator = half_length -
                             std::min(ConeTransitionLength(), 0.45 * half_length);
    return fibre_radius * (1.0 - run_options.tip_radius_ratio) *
           SymmetricHalfSlopeWeightDerivative(y) /
           (denominator + TinyReal);
#else
    Real denominator = ConeSectionLength() - ConeTransitionLength();
    return -fibre_radius * (1.0 - run_options.tip_radius_ratio) *
           ConeSlopeWeightDerivative(x) / (denominator + TinyReal);
#endif
}
Real MeanFiberRadius()
{
    return IsConicalFiber()
               ? 0.5 * fibre_radius * (1.0 + run_options.tip_radius_ratio)
               : fibre_radius;
}
Real MinimumFiberRadius()
{
    return IsConicalFiber() ? fibre_radius * run_options.tip_radius_ratio
                            : fibre_radius;
}
Real MaximumFiberRadius() { return fibre_radius; }
Real FiberNormalDistance(const Vecd &position)
{
    Real slope = FiberRadiusFirstDerivative(position[0]);
    return (position[1] - FiberRadius(position[0])) /
           std::sqrt(1.0 + slope * slope);
}
Vecd FiberOutwardNormal(Real x)
{
    Real slope = FiberRadiusFirstDerivative(x);
    return Vecd(-slope, 1.0) / std::sqrt(1.0 + slope * slope);
}
Real FiberAxisymmetricCurvature(Real x)
{
    Real radius = FiberRadius(x);
    Real slope = FiberRadiusFirstDerivative(x);
    Real second = FiberRadiusSecondDerivative(x);
    Real metric = std::sqrt(1.0 + slope * slope);
    return -second / (metric * metric * metric) +
           1.0 / ((radius + TinyReal) * metric);
}
bool InEndBuffer(Real x)
{
    if (!UseConicalEndBuffer())
        return false;
    Real buffer_width = run_options.end_buffer_fraction * DomainLength();
    return x <= DomainLowerBound() + buffer_width ||
           x >= DomainUpperBound() - buffer_width;
}
Real InitialAmplitude()
{
    return (run_options.case_name == "static" ||
            run_options.case_name == "viscous-decay")
               ? 0.0
               : run_options.amplitude_ratio * FilmThickness();
}
Real ViscousDecayWaveNumber() { return 0.5 * Pi / FilmThickness(); }
Real BasicWaveNumber()
{
    // A finite conical segment needs two independent axial scales: the total
    // reservoir-to-reservoir length and the imposed disturbance wavelength.
    // When wavelength_over_a is supplied, do not silently replace it by L.
    return run_options.wavelength_over_a > 0.0
               ? 2.0 * Pi / (run_options.wavelength_over_a * fibre_radius)
               : 2.0 * Pi / DomainLength();
}
size_t ReferenceMode()
{
    if (run_options.case_name == "broadband")
        return static_cast<size_t>(run_options.broadband_mode_min);
    if (run_options.wavelength_over_a > 0.0 &&
        run_options.domain_length_over_a > 0.0)
        return static_cast<size_t>(std::max(
            Real(1.0), std::round(DomainLength() /
                                  (run_options.wavelength_over_a * fibre_radius))));
    return run_options.case_name == "multi" ? 3 : 1;
}
Real ReferenceWaveNumber()
{
    if (run_options.wavelength_over_a > 0.0)
        return BasicWaveNumber();
    return static_cast<Real>(ReferenceMode()) * BasicWaveNumber();
}
Real InputWavelength() { return 2.0 * Pi / ReferenceWaveNumber(); }
Real ParticleSpacing() { return FilmThickness() / run_options.resolution; }
Real BoundaryWidth() { return 4.0 * ParticleSpacing(); }
Real RadialDomainUpperBound()
{
    if (IsConicalFiber())
    {
        // Long-time RP evolution can collect several wavelengths of liquid
        // into a bead whose radius is far larger than the initial film.  The
        // former initial-film-sized box silently lost cell-linked-list support
        // once such a bead grew.  Reserve a nonlinear-morphology envelope.
        return MaximumFiberRadius() + 6.0 * FilmThickness() +
               2.0 * BoundaryWidth();
    }
    return MaximumFiberRadius() + FilmThickness() +
           InitialAmplitude() + BoundaryWidth();
}
Real DynamicViscosity()
{
    return run_options.oh * std::sqrt(reference_density * surface_tension * MeanOuterRadius());
}
Real ViscousDecayRate()
{
    return DynamicViscosity() / reference_density *
           ViscousDecayWaveNumber() * ViscousDecayWaveNumber();
}
Real CapillaryVelocity() { return surface_tension / DynamicViscosity(); }
Real InertialCapillaryVelocity()
{
    return std::sqrt(surface_tension / (reference_density * MeanOuterRadius()));
}
Real AcousticReferenceVelocity()
{
    // gamma/mu is the evolution velocity in the viscous limit, but it becomes
    // arbitrarily small as Oh increases.  It is therefore not a safe acoustic
    // scale for a WCSPH equation of state that must also support the Laplace
    // pressure.  The inertial-capillary velocity provides the required lower
    // bound, while max(...) preserves the previous low-Oh setup.
    return std::max(CapillaryVelocity(), InertialCapillaryVelocity());
}
Real SoundSpeed() { return sound_speed_factor * AcousticReferenceVelocity(); }
Real PhysicalEndTime() { return run_options.end_T * DynamicViscosity() * fibre_radius / surface_tension; }
Real BroadbandPhase(int mode)
{
    // Seed zero exactly preserves the original deterministic broadband field.
    // Other integer seeds change only modal phases, not modal amplitudes.
    Real seed = static_cast<Real>(run_options.disturbance_seed);
    return std::fmod(0.37 * static_cast<Real>(mode * mode + 3 * mode) +
                         0.731 * seed * static_cast<Real>(mode) + 0.193 * seed * seed,
                     2.0 * Pi);
}
Real RawSurfacePattern(Real x)
{
    if (run_options.case_name == "broadband")
    {
        Real theta = BasicWaveNumber() * x;
        Real value = 0.0;
        const Real count = static_cast<Real>(run_options.broadband_mode_max -
                                             run_options.broadband_mode_min + 1);
        for (int mode = run_options.broadband_mode_min;
             mode <= run_options.broadband_mode_max; ++mode)
        {
            Real phase = BroadbandPhase(mode);
            value += std::cos(static_cast<Real>(mode) * theta + phase) / count;
        }
        return value;
    }
    if (run_options.case_name != "multi")
        return std::cos(BasicWaveNumber() * x);

    Real theta = BasicWaveNumber() * x;
    // Equal-amplitude broadband seed: no candidate wavelength is favoured
    // before the physical evolution starts.  The three wavelengths are
    // L/2=1.5 lambda_max, L/3=lambda_max and L/4=0.75 lambda_max.
    return (std::cos(2.0 * theta) +
            std::cos(3.0 * theta) +
            std::cos(4.0 * theta)) /
           3.0;
}
Real RawSurfacePatternFirstDerivative(Real x)
{
    Real k = BasicWaveNumber();
    if (run_options.case_name == "broadband")
    {
        Real theta = k * x;
        Real value = 0.0;
        const Real count = static_cast<Real>(run_options.broadband_mode_max -
                                             run_options.broadband_mode_min + 1);
        for (int mode = run_options.broadband_mode_min;
             mode <= run_options.broadband_mode_max; ++mode)
        {
            Real phase = BroadbandPhase(mode);
            value -= static_cast<Real>(mode) * k *
                     std::sin(static_cast<Real>(mode) * theta + phase) / count;
        }
        return value;
    }
    if (run_options.case_name != "multi")
        return -k * std::sin(k * x);

    Real theta = k * x;
    return (-2.0 * k * std::sin(2.0 * theta) -
            3.0 * k * std::sin(3.0 * theta) -
            4.0 * k * std::sin(4.0 * theta)) /
           3.0;
}
Real RawSurfacePatternSecondDerivative(Real x)
{
    Real k = BasicWaveNumber();
    if (run_options.case_name == "broadband")
    {
        Real theta = k * x;
        Real value = 0.0;
        const Real count = static_cast<Real>(run_options.broadband_mode_max -
                                             run_options.broadband_mode_min + 1);
        for (int mode = run_options.broadband_mode_min;
             mode <= run_options.broadband_mode_max; ++mode)
        {
            Real phase = BroadbandPhase(mode);
            Real wave_number = static_cast<Real>(mode) * k;
            value -= wave_number * wave_number *
                     std::cos(static_cast<Real>(mode) * theta + phase) / count;
        }
        return value;
    }
    if (run_options.case_name != "multi")
        return -k * k * std::cos(k * x);

    Real theta = k * x;
    return (-std::pow(2.0 * k, 2.0) * std::cos(2.0 * theta) -
            std::pow(3.0 * k, 2.0) * std::cos(3.0 * theta) -
            std::pow(4.0 * k, 2.0) * std::cos(4.0 * theta)) /
           3.0;
}
Real DisturbanceEnvelope(Real x)
{
    if (!IsConicalFiber() || ReservoirLength() <= TinyReal)
        return 1.0;
    Real transition = ConeTransitionLength();
    if (x <= ConeLowerBound() || x >= ConeUpperBound())
        return 0.0;
    if (transition <= TinyReal)
        return 1.0;
    if (x < ConeLowerBound() + transition)
    {
        Real s = (x - ConeLowerBound()) / transition;
        return s * s * (3.0 - 2.0 * s);
    }
    if (x > ConeUpperBound() - transition)
    {
        Real s = (x - (ConeUpperBound() - transition)) / transition;
        return 1.0 - s * s * (3.0 - 2.0 * s);
    }
    return 1.0;
}
Real DisturbanceEnvelopeFirstDerivative(Real x)
{
    if (!IsConicalFiber() || ReservoirLength() <= TinyReal)
        return 0.0;
    Real transition = ConeTransitionLength();
    if (transition <= TinyReal || x <= ConeLowerBound() || x >= ConeUpperBound())
        return 0.0;
    if (x < ConeLowerBound() + transition)
    {
        Real s = (x - ConeLowerBound()) / transition;
        return 6.0 * s * (1.0 - s) / transition;
    }
    if (x > ConeUpperBound() - transition)
    {
        Real s = (x - (ConeUpperBound() - transition)) / transition;
        return -6.0 * s * (1.0 - s) / transition;
    }
    return 0.0;
}
Real DisturbanceEnvelopeSecondDerivative(Real x)
{
    if (!IsConicalFiber() || ReservoirLength() <= TinyReal)
        return 0.0;
    Real transition = ConeTransitionLength();
    if (transition <= TinyReal || x <= ConeLowerBound() || x >= ConeUpperBound())
        return 0.0;
    if (x < ConeLowerBound() + transition)
    {
        Real s = (x - ConeLowerBound()) / transition;
        return (6.0 - 12.0 * s) / (transition * transition);
    }
    if (x > ConeUpperBound() - transition)
    {
        Real s = (x - (ConeUpperBound() - transition)) / transition;
        return -(6.0 - 12.0 * s) / (transition * transition);
    }
    return 0.0;
}
Real SurfacePattern(Real x)
{
    return DisturbanceEnvelope(x) * RawSurfacePattern(x);
}
Real SurfacePatternFirstDerivative(Real x)
{
    return DisturbanceEnvelopeFirstDerivative(x) * RawSurfacePattern(x) +
           DisturbanceEnvelope(x) * RawSurfacePatternFirstDerivative(x);
}
Real SurfacePatternSecondDerivative(Real x)
{
    return DisturbanceEnvelopeSecondDerivative(x) * RawSurfacePattern(x) +
           2.0 * DisturbanceEnvelopeFirstDerivative(x) *
               RawSurfacePatternFirstDerivative(x) +
           DisturbanceEnvelope(x) * RawSurfacePatternSecondDerivative(x);
}
Real SurfaceRadius(Real x)
{
    return FiberRadius(x) + FilmThickness() +
           InitialAmplitude() * SurfacePattern(x);
}

class LiquidFilmShape : public MultiPolygonShape
{
  public:
    explicit LiquidFilmShape(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        std::vector<Vecd> polygon;
        const int samples = std::max(200, static_cast<int>(std::ceil(DomainLength() / ParticleSpacing())));
        for (int i = 0; i <= samples; ++i)
        {
            Real x = DomainLowerBound() + DomainLength() *
                     static_cast<Real>(i) / static_cast<Real>(samples);
            polygon.push_back(Vecd(x, FiberRadius(x)));
        }
        for (int i = samples; i >= 0; --i)
        {
            Real x = DomainLowerBound() + DomainLength() *
                     static_cast<Real>(i) / static_cast<Real>(samples);
            // Generate a uniform reference lattice first. The disturbance is
            // imposed continuously in InitialLiquidState below.  If the
            // wavy boundary is used directly for lattice filling, a small
            // disturbance (for example epsilon/h=0.01) is rounded away when
            // it is smaller than one particle spacing.
            polygon.push_back(Vecd(x, FiberRadius(x) + FilmThickness()));
        }
        polygon.push_back(Vecd(DomainLowerBound(), FiberRadius(DomainLowerBound())));
        multi_polygon_.addPolygon(polygon, GeometricOps::add);
    }
};

class RigidFiberShape : public MultiPolygonShape
{
  public:
    explicit RigidFiberShape(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        std::vector<Vecd> polygon;
        Real lower = IsFiniteFiber()
                         ? DomainLowerBound()
                         : DomainLowerBound() - BoundaryWidth();
        Real upper = IsFiniteFiber()
                         ? DomainUpperBound()
                         : DomainUpperBound() + BoundaryWidth();
        polygon.push_back(Vecd(lower, 0.0));
        polygon.push_back(Vecd(upper, 0.0));
        const int samples = std::max(
            200, static_cast<int>(std::ceil((upper - lower) / ParticleSpacing())));
        // A finite cap already meets the symmetry axis at both endpoints.
        // Skip those duplicate axis vertices so the polygon stays simple.
        const int first_surface_index = IsFiniteFiber() ? samples - 1 : samples;
        const int last_surface_index = IsFiniteFiber() ? 1 : 0;
        for (int i = first_surface_index; i >= last_surface_index; --i)
        {
            Real x = lower + (upper - lower) *
                                 static_cast<Real>(i) / static_cast<Real>(samples);
            polygon.push_back(Vecd(x, FiberRadius(x)));
        }
        // Boost polygon rings are explicitly closed by repeating the first
        // point once; the cap endpoints themselves are not repeated.
        polygon.push_back(Vecd(lower, 0.0));
        multi_polygon_.addPolygon(polygon, GeometricOps::add);
    }
};

class InitialLiquidState : public LocalDynamics
{
  public:
    explicit InitialLiquidState(SPHBody &sph_body)
        : LocalDynamics(sph_body),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          rho_(particles_->getVariableDataByName<Real>("Density")),
          pressure_(particles_->registerStateVariableData<Real>("Pressure")),
          velocity_(particles_->registerStateVariableData<Vecd>("Velocity")) {}

    void update(size_t index_i, Real dt)
    {
        Real x = pos_[index_i][0];
        // Continuously map the flat reference lattice onto the prescribed
        // film profile while keeping the no-slip fibre surface fixed.  This
        // makes sub-particle initial amplitudes measurable without increasing
        // their physical size.
        Real radial_scale = 1.0 + InitialAmplitude() * SurfacePattern(x) /
                                      (FilmThickness() + TinyReal);
        Real wall_radius = FiberRadius(x);
        pos_[index_i][1] = wall_radius +
                           (pos_[index_i][1] - wall_radius) * radial_scale;
        Real slope = FiberRadiusFirstDerivative(x) +
                     InitialAmplitude() * SurfacePatternFirstDerivative(x);
        Real second_derivative = FiberRadiusSecondDerivative(x) +
                                 InitialAmplitude() * SurfacePatternSecondDerivative(x);
        Real metric = std::sqrt(1.0 + slope * slope);
        Real kappa_axial = -second_derivative / (metric * metric * metric);
        Real kappa_hoop = 1.0 / (SurfaceRadius(x) * metric);
        // In the balanced-curvature formulation the constant cylindrical
        // Laplace jump is a pressure gauge.  The one-phase particle set has
        // no exterior particles with which to discretize that jump, so only
        // the constant reference pressure is initialized here; curvature
        // departures are introduced consistently by the capillary force.
        Real p0 = run_options.surface_formulation == "balanced-curvature" &&
                          !IsConicalFiber()
                      ? surface_tension / MeanOuterRadius()
                      : surface_tension * (kappa_axial + kappa_hoop);
        rho_[index_i] = reference_density *
                        (1.0 + p0 / (reference_density * SoundSpeed() * SoundSpeed()));
        pressure_[index_i] = p0;
        velocity_[index_i] = Vecd::Zero();
        if (run_options.case_name == "viscous-decay")
        {
            Real wall_distance = std::clamp(
                FiberNormalDistance(pos_[index_i]), Real(0.0), FilmThickness());
            // Exact planar diffusion eigenmode for u(a)=0 and du/dr(a+h)=0.
            // It must decay as exp[-nu*(pi/(2h))^2*t].
            velocity_[index_i][0] =
                std::sin(ViscousDecayWaveNumber() * wall_distance);
        }
    }

  private:
    Vecd *pos_, *velocity_;
    Real *rho_, *pressure_;
};

/** Hold short liquid bands at both ends of a finite cone at their initial
 *  coated-film state. They act as numerical reservoirs and keep end-cap
 *  retraction away from the central observation region. */
class ConicalEndBuffer
{
  public:
    explicit ConicalEndBuffer(BaseParticles &particles)
        : particles_(particles),
          pos_(particles.getVariableDataByName<Vecd>("Position")),
          velocity_(particles.getVariableDataByName<Vecd>("Velocity")),
          density_(particles.getVariableDataByName<Real>("Density")),
          pressure_(particles.getVariableDataByName<Real>("Pressure")),
          weight_(particles.registerStateVariableData<Real>("ConicalEndBufferWeight")),
          initial_pos_(particles.registerStateVariableData<Vecd>(
              "ConicalEndBufferInitialPosition")),
          initial_density_(particles.registerStateVariableData<Real>(
              "ConicalEndBufferInitialDensity"))
    {
        particles.addVariableToWrite<Real>("ConicalEndBufferWeight");
    }

    void capture()
    {
        const size_t total = particles_.TotalRealParticles();
        if (!UseConicalEndBuffer())
        {
            for (size_t i = 0; i < total; ++i)
            {
                initial_pos_[i] = pos_[i];
                initial_density_[i] = density_[i];
                weight_[i] = 0.0;
            }
            return;
        }
        const Real buffer_width = run_options.end_buffer_fraction * DomainLength();
        for (size_t i = 0; i < total; ++i)
        {
            // Store the reference state as particle variables rather than in
            // external std::vectors.  SPHinXsys particle sorting then keeps
            // the reference state attached to the same physical particle.
            initial_pos_[i] = pos_[i];
            initial_density_[i] = density_[i];
            Real distance_to_end = std::min(
                pos_[i][0] - DomainLowerBound(),
                DomainUpperBound() - pos_[i][0]);
            Real inward_fraction = std::clamp(
                distance_to_end / (buffer_width + TinyReal), Real(0.0), Real(1.0));
            // C1-continuous reverse smoothstep: one at the finite end and
            // exactly zero (with zero slope) at the active-domain interface.
            Real smoothstep = inward_fraction * inward_fraction *
                              (3.0 - 2.0 * inward_fraction);
            weight_[i] = 1.0 - smoothstep;
        }
    }

    void exec(Real dt)
    {
        if (!UseConicalEndBuffer() || dt <= 0.0)
            return;
        const Real relaxation_time =
            0.05 * DynamicViscosity() * fibre_radius / surface_tension;
        for (size_t i = 0; i < particles_.TotalRealParticles(); ++i)
        {
            if (weight_[i] <= 0.0)
                continue;
            Real blend = 1.0 - std::exp(
                -weight_[i] * dt / (relaxation_time + TinyReal));
            pos_[i] += blend * (initial_pos_[i] - pos_[i]);
            velocity_[i] *= 1.0 - blend;
            density_[i] += blend * (initial_density_[i] - density_[i]);
            pressure_[i] = SoundSpeed() * SoundSpeed() *
                           (density_[i] - reference_density);
        }
    }

  private:
    BaseParticles &particles_;
    Vecd *pos_, *velocity_;
    Real *density_, *pressure_;
    Real *weight_;
    Vecd *initial_pos_;
    Real *initial_density_;
};

/**
 * Low-pass reconstruction of the free surface followed by a one-sided CSF
 * force. Interface particles are still identified from missing kernel support;
 * the periodic Fourier reconstruction only supplies robust derivatives for
 * curvature and removes particle-scale staircase modes.
 */
class ReconstructedAxisymmetricCapillaryForce : public ForcePrior
{
  public:
    explicit ReconstructedAxisymmetricCapillaryForce(SPHBody &sph_body)
        : ForcePrior(sph_body, "ReconstructedCapillaryForce"),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          indicator_(particles_->getVariableDataByName<int>("Indicator")),
          rho_(particles_->getVariableDataByName<Real>("Density")),
          mass_(particles_->getVariableDataByName<Real>("Mass")),
          gradient_(particles_->registerStateVariableData<Vecd>("ColorGradient")),
          normal_(particles_->registerStateVariableData<Vecd>("NormDirection")),
          stress_(particles_->registerStateVariableData<Matd>("SurfaceTensionStress")),
          delta_(particles_->registerStateVariableData<Real>("SurfaceDelta")),
          meridional_(particles_->registerStateVariableData<Real>("MeridionalCurvature")),
          hoop_(particles_->registerStateVariableData<Real>("HoopCurvature")),
          total_(particles_->registerStateVariableData<Real>("AxisymmetricCurvature")),
          end_buffer_weight_(particles_->registerStateVariableData<Real>(
              "ConicalEndBufferWeight")),
          number_of_bins_(std::max<size_t>(32, static_cast<size_t>(std::round(DomainLength() / ParticleSpacing())))),
          force_scale_(1.0)
    {
        particles_->registerSingleVariable<Real>("SurfaceTensionCoef", surface_tension);
        particles_->addEvolvingVariable<Vecd>("ColorGradient");
        particles_->addEvolvingVariable<Vecd>("NormDirection");
        particles_->addEvolvingVariable<Matd>("SurfaceTensionStress");
        particles_->addEvolvingVariable<Real>("SurfaceDelta");
        particles_->addEvolvingVariable<Real>("MeridionalCurvature");
        particles_->addEvolvingVariable<Real>("HoopCurvature");
        particles_->addEvolvingVariable<Real>("AxisymmetricCurvature");
        particles_->addVariableToWrite<Vecd>("ColorGradient");
        particles_->addVariableToWrite<Vecd>("NormDirection");
        particles_->addVariableToWrite<Matd>("SurfaceTensionStress");
        particles_->addVariableToWrite<Real>("SurfaceDelta");
        particles_->addVariableToWrite<Real>("MeridionalCurvature");
        particles_->addVariableToWrite<Real>("HoopCurvature");
        particles_->addVariableToWrite<Real>("AxisymmetricCurvature");
        particles_->addVariableToWrite<Vecd>("ReconstructedCapillaryForce");
    }

    void setForceScale(Real scale)
    {
        force_scale_ = std::clamp(scale, Real(0.0), Real(2.0));
    }

    void exec(Real dt = 0.0)
    {
        // Reconstruct film thickness rather than absolute radius.  The
        // conical wall has unequal radii at the two finite-domain ends, while
        // its initially uniform coating has equal thickness there.  Removing
        // the known wall geometry before the Fourier fit prevents a false
        // curvature spike caused by wrapping the cone as if it were periodic.
        std::vector<Real> thickness(number_of_bins_, -std::numeric_limits<Real>::infinity());
        for (size_t i = 0; i < particles_->TotalRealParticles(); ++i)
        {
            if (indicator_[i] != 1 ||
                pos_[i][1] < FiberRadius(pos_[i][0]) + 0.25 * FilmThickness())
                continue;
            Real shifted_x = pos_[i][0] - DomainLowerBound();
            Real x = shifted_x - std::floor(shifted_x / DomainLength()) * DomainLength();
            size_t bin = std::min(number_of_bins_ - 1,
                                  static_cast<size_t>(std::floor(x / DomainLength() * number_of_bins_)));
            thickness[bin] = std::max(
                thickness[bin], pos_[i][1] + 0.5 * ParticleSpacing() -
                                    FiberRadius(pos_[i][0]));
        }

        size_t valid = 0;
        for (Real value : thickness)
            valid += std::isfinite(value) ? 1 : 0;
        if (valid == 0)
        {
            for (size_t i = 0; i < particles_->TotalRealParticles(); ++i)
            {
                current_force_[i] = Vecd::Zero();
                ForcePrior::update(i, dt);
            }
            return;
        }

        for (size_t i = 0; i < number_of_bins_; ++i)
        {
            if (std::isfinite(thickness[i]))
                continue;
            size_t left = i, right = i;
            do
                left = (left + number_of_bins_ - 1) % number_of_bins_;
            while (left != i && !std::isfinite(thickness[left]));
            do
                right = (right + 1) % number_of_bins_;
            while (right != i && !std::isfinite(thickness[right]));
            thickness[i] = 0.5 * (thickness[left] + thickness[right]);
        }

        Real sampled_mean_thickness = 0.0;
        for (Real value : thickness)
            sampled_mean_thickness += value;
        sampled_mean_thickness /= static_cast<Real>(number_of_bins_);
        // This is a sub-particle geometric calibration, not a change to the
        // physical film thickness.  It removes resolution-dependent vertical
        // lattice phase from the zero Fourier mode while retaining every
        // non-zero Fourier coefficient measured from the particles.
        Real mean_thickness = run_options.surface_mean == "nominal"
                                  ? FilmThickness()
                                  : sampled_mean_thickness;

        size_t retained_modes = std::min<size_t>(
            static_cast<size_t>(run_options.interface_modes), number_of_bins_ / 8);
        std::vector<Real> cosine_coefficient(retained_modes + 1, 0.0);
        std::vector<Real> sine_coefficient(retained_modes + 1, 0.0);
        for (size_t mode = 1; mode <= retained_modes; ++mode)
        {
            for (size_t i = 0; i < number_of_bins_; ++i)
            {
                Real angle = 2.0 * Pi * static_cast<Real>(mode) *
                             (static_cast<Real>(i) + 0.5) / static_cast<Real>(number_of_bins_);
                Real departure = thickness[i] - mean_thickness;
                cosine_coefficient[mode] += departure * std::cos(angle);
                sine_coefficient[mode] += departure * std::sin(angle);
            }
            cosine_coefficient[mode] *= 2.0 / static_cast<Real>(number_of_bins_);
            sine_coefficient[mode] *= 2.0 / static_cast<Real>(number_of_bins_);
            if (run_options.regime == "bead")
            {
                // Smooth only the shortest reconstructed waves. This preserves
                // nonlinear neck geometry while suppressing lattice-scale
                // curvature noise amplified by the second derivative.
                Real normalized_mode = static_cast<Real>(mode) /
                                       static_cast<Real>(retained_modes);
                Real filter = std::exp(-8.0 * std::pow(normalized_mode, 8.0));
                cosine_coefficient[mode] *= filter;
                sine_coefficient[mode] *= filter;
            }
        }

        Real basic_wave_number = 2.0 * Pi / DomainLength();
        Real interface_width = 2.0 * ParticleSpacing();
        // A spatially constant curvature only changes the Laplace-pressure
        // gauge and must not accelerate a uniformly coated periodic fibre.
        // Subtracting 1/mean_radius leaves the physical curvature departure
        // -r'' - delta_r/r0^2 that drives the RP mode at linear order.
        Real reference_curvature =
            run_options.surface_formulation == "balanced-curvature"
                ? 1.0 / std::max(MeanOuterRadius(),
                                 MinimumFiberRadius() + ParticleSpacing())
                : 0.0;
        size_t total_particles = particles_->TotalRealParticles();
        std::vector<size_t> particle_bin(total_particles, number_of_bins_);
        std::vector<Real> column_delta_integral(number_of_bins_, 0.0);
        for (size_t i = 0; i < total_particles; ++i)
        {
            Real shifted_x = pos_[i][0] - DomainLowerBound();
            Real x = shifted_x - std::floor(shifted_x / DomainLength()) * DomainLength();
            Real theta = basic_wave_number * x;
            Real reconstructed_radius = FiberRadius(pos_[i][0]) + mean_thickness;
            Real first_derivative = FiberRadiusFirstDerivative(pos_[i][0]);
            Real second_derivative = FiberRadiusSecondDerivative(pos_[i][0]);
            for (size_t mode = 1; mode <= retained_modes; ++mode)
            {
                Real mode_angle = static_cast<Real>(mode) * theta;
                Real harmonic = cosine_coefficient[mode] * std::cos(mode_angle) +
                                sine_coefficient[mode] * std::sin(mode_angle);
                Real mode_wave_number = static_cast<Real>(mode) * basic_wave_number;
                reconstructed_radius += harmonic;
                first_derivative += mode_wave_number *
                                    (-cosine_coefficient[mode] * std::sin(mode_angle) +
                                     sine_coefficient[mode] * std::cos(mode_angle));
                second_derivative -= mode_wave_number * mode_wave_number * harmonic;
            }

            Real metric = std::sqrt(1.0 + first_derivative * first_derivative);
            Vecd normal(-first_derivative / metric, 1.0 / metric);
            Real kappa_meridional = -second_derivative / (metric * metric * metric);
            Real kappa_hoop = 1.0 /
                (std::max(reconstructed_radius,
                          FiberRadius(pos_[i][0]) + ParticleSpacing()) * metric);
            Real kappa = kappa_meridional + kappa_hoop;
            Real depth = reconstructed_radius - pos_[i][1];
            Real surface_delta = 0.0;
            if (depth >= -0.5 * ParticleSpacing() && depth <= interface_width)
            {
                Real nonnegative_depth = std::max(depth, Real(0.0));
                surface_delta = (1.0 + std::cos(Pi * nonnegative_depth / interface_width)) /
                                interface_width;
            }

            normal_[i] = Vecd::Zero();
            if (surface_delta > 0.0)
            {
                normal_[i] = normal;
                size_t bin = std::min(number_of_bins_ - 1,
                                      static_cast<size_t>(std::floor(x / DomainLength() * number_of_bins_)));
                particle_bin[i] = bin;
                column_delta_integral[bin] += surface_delta * ParticleSpacing();
            }
            delta_[i] = surface_delta;
            meridional_[i] = surface_delta > 0.0 ? kappa_meridional : 0.0;
            hoop_[i] = surface_delta > 0.0 ? kappa_hoop : 0.0;
            total_[i] = surface_delta > 0.0 ? kappa : 0.0;
        }

        for (size_t i = 0; i < total_particles; ++i)
        {
            Real surface_delta = delta_[i];
            if (surface_delta > 0.0 &&
                run_options.surface_delta == "column-normalized")
            {
                size_t bin = particle_bin[i];
                Real integral = bin < number_of_bins_
                                    ? column_delta_integral[bin]
                                    : 0.0;
                if (integral > TinyReal)
                    surface_delta /= integral;
            }
            delta_[i] = surface_delta;
            gradient_[i] = surface_delta * normal_[i];
            stress_[i] = surface_tension *
                         (Matd::Identity() - normal_[i] * normal_[i].transpose()) * surface_delta;
            Real driving_curvature = total_[i] - reference_curvature;
            Real active_weight = IsConicalFiber()
                                     ? 1.0 - std::clamp(end_buffer_weight_[i],
                                                        Real(0.0), Real(1.0))
                                     : 1.0;
            current_force_[i] = -active_weight * force_scale_ * mass_[i] * surface_tension *
                                driving_curvature * surface_delta * normal_[i] /
                                (rho_[i] + TinyReal);
            ForcePrior::update(i, dt);
        }
    }

  private:
    Vecd *pos_, *gradient_, *normal_;
    int *indicator_;
    Real *rho_, *mass_, *delta_, *meridional_, *hoop_, *total_,
        *end_buffer_weight_;
    Matd *stress_;
    size_t number_of_bins_;
    Real force_scale_;
};

class InitializeAxisymmetricWeights : public LocalDynamics
{
  public:
    explicit InitializeAxisymmetricWeights(SPHBody &sph_body)
        : LocalDynamics(sph_body), pos_(particles_->getVariableDataByName<Vecd>("Position")),
          mass_(particles_->getVariableDataByName<Real>("Mass")),
          ring_mass_(particles_->registerStateVariableData<Real>("AxisymmetricRingMass"))
    {
        particles_->addEvolvingVariable<Real>("AxisymmetricRingMass");
        particles_->addVariableToWrite<Real>("AxisymmetricRingMass");
    }
    void update(size_t index_i, Real dt)
    {
        ring_mass_[index_i] = mass_[index_i] * std::max(pos_[index_i][1], 0.25 * ParticleSpacing());
    }

  private:
    Vecd *pos_;
    Real *mass_, *ring_mass_;
};

class UpdateAxisymmetricWeights : public LocalDynamics
{
  public:
    explicit UpdateAxisymmetricWeights(SPHBody &sph_body)
        : LocalDynamics(sph_body), pos_(particles_->getVariableDataByName<Vecd>("Position")),
          rho_(particles_->getVariableDataByName<Real>("Density")),
          mass_(particles_->getVariableDataByName<Real>("Mass")),
          Vol_(particles_->getVariableDataByName<Real>("VolumetricMeasure")),
          ring_mass_(particles_->getVariableDataByName<Real>("AxisymmetricRingMass")) {}
    void update(size_t index_i, Real dt)
    {
        Real r = std::max(pos_[index_i][1], 0.25 * ParticleSpacing());
        mass_[index_i] = ring_mass_[index_i] / r;
        Vol_[index_i] = mass_[index_i] / (rho_[index_i] + TinyReal);
    }

  private:
    Vecd *pos_;
    Real *rho_, *mass_, *Vol_, *ring_mass_;
};

class AxisymmetricContinuityCorrection : public LocalDynamics
{
  public:
    explicit AxisymmetricContinuityCorrection(SPHBody &sph_body)
        : LocalDynamics(sph_body), pos_(particles_->getVariableDataByName<Vecd>("Position")),
          vel_(particles_->getVariableDataByName<Vecd>("Velocity")),
          rho_(particles_->getVariableDataByName<Real>("Density")) {}
    void update(size_t index_i, Real dt)
    {
        Real r = std::max(pos_[index_i][1], 0.25 * ParticleSpacing());
        Real factor = std::exp(-vel_[index_i][1] * dt / r);
        rho_[index_i] *= factor;
    }

  private:
    Vecd *pos_, *vel_;
    Real *rho_;
};

class AxisymmetricViscousForce : public ForcePrior, public DataDelegateInner
{
  public:
    explicit AxisymmetricViscousForce(BaseInnerRelation &inner_relation)
        : ForcePrior(inner_relation.getSPHBody(), "AxisymmetricViscousForce"),
          DataDelegateInner(inner_relation),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          vel_(particles_->getVariableDataByName<Vecd>("Velocity")),
          rho_(particles_->getVariableDataByName<Real>("Density")),
          mass_(particles_->getVariableDataByName<Real>("Mass")),
          Vol_(particles_->getVariableDataByName<Real>("VolumetricMeasure")),
          mu_(DynamicViscosity())
    {
        particles_->addVariableToWrite<Vecd>("AxisymmetricViscousForce");
    }

    void interaction(size_t index_i, Real dt)
    {
        Vecd radial_derivative = Vecd::Zero();
        const Neighborhood &neighborhood = inner_configuration_[index_i];
        for (size_t n = 0; n != neighborhood.current_size_; ++n)
        {
            size_t index_j = neighborhood.j_[n];
            radial_derivative -= (vel_[index_i] - vel_[index_j]) *
                                 neighborhood.e_ij_[n][1] * neighborhood.dW_ij_[n] * Vol_[index_j];
        }
        Real r = std::max(pos_[index_i][1], 0.25 * ParticleSpacing());
        Vecd acceleration(radial_derivative[0] / r,
                          radial_derivative[1] / r - vel_[index_i][1] / (r * r));
        current_force_[index_i] = mass_[index_i] * mu_ * acceleration /
                                  (rho_[index_i] + TinyReal);
    }

    void update(size_t index_i, Real dt) { ForcePrior::update(index_i, dt); }

  private:
    Vecd *pos_, *vel_;
    Real *rho_, *mass_, *Vol_;
    Real mu_;
};

void WriteParameterSummary(const std::string &output_folder)
{
    std::ofstream file(output_folder + "/parameters.csv");
    file << "parameter,value,description\n" << std::scientific << std::setprecision(10);
    file << "case," << run_options.case_name << ",wavelength case\n";
    file << "axisym," << run_options.axisym << ",axisymmetric correction level\n";
    file << "regime," << run_options.regime << ",early-wave or nonlinear bead mode\n";
    file << "a," << fibre_radius << ",fibre radius\n";
    file << "fiber_profile," << run_options.fiber_profile
         << ",uniform cylinder or finite conical frustum\n";
    file << "tip_radius_ratio," << run_options.tip_radius_ratio
         << ",thin-end radius divided by thick-end radius\n";
    file << "a_thin," << MinimumFiberRadius() << ",thin-end fibre radius\n";
    file << "cone_slope," << FiberRadiusFirstDerivative(0.0)
         << ",da/dx on the conical wall\n";
    file << "cone_section_length_over_a," << ConeSectionLength() / fibre_radius
         << ",central tapered research-section length\n";
    file << "reservoir_length_over_a," << ReservoirLength() / fibre_radius
         << ",straight coated extension at each end\n";
    file << "cone_transition_over_a," << ConeTransitionLength() / fibre_radius
         << ",smooth slope-ramp length at each cone shoulder\n";
    file << "finite_cap_length_over_a," << FiniteCapLength() / fibre_radius
         << ",fine-end smooth cap length; active only for finite-conical\n";
    file << "finite_coarse_round_cap_length_over_a,"
         << FiniteCoarseRoundCapLength() / fibre_radius
         << ",coarse circular-head axial length (one coarse radius)\n";
    file << "end_buffer_fraction," << run_options.end_buffer_fraction
         << ",fixed far-end buffer fraction of total domain at each end\n";
    file << "h_over_a," << run_options.film_ratio << ",film thickness ratio\n";
    file << "r0," << MeanOuterRadius() << ",mean outer radius\n";
    file << "rho," << reference_density << ",liquid reference density\n";
    file << "gamma," << surface_tension << ",surface tension\n";
    file << "Oh," << run_options.oh << ",Ohnesorge number based on r0\n";
    file << "mu," << DynamicViscosity() << ",dynamic viscosity\n";
    file << "viscosity_integration," << run_options.viscosity_integration
         << ",explicit force or pairwise implicit damping\n";
    file << "capillary_scale," << run_options.capillary_scale
         << ",multiplier applied to reconstructed capillary force\n";
    file << "surface_formulation," << run_options.surface_formulation
         << ",css uses total curvature; balanced-curvature removes the constant cylindrical jump\n";
    file << "surface_mean," << run_options.surface_mean
         << ",sampled particle staircase or nominal prescribed zero Fourier mode\n";
    file << "surface_delta," << run_options.surface_delta
         << ",raw or per-column unit-integral regularized delta\n";
    file << "capillary_velocity," << CapillaryVelocity() << ",gamma divided by mu\n";
    file << "inertial_capillary_velocity," << InertialCapillaryVelocity()
         << ",sqrt(gamma/(rho*r0))\n";
    file << "acoustic_reference_velocity," << AcousticReferenceVelocity()
         << ",max(viscous and inertial capillary velocity)\n";
    file << "sound_speed," << SoundSpeed() << ",weakly compressible sound speed\n";
    file << "epsilon_over_h," << (FilmThickness() > 0.0 ? InitialAmplitude() / FilmThickness() : 0.0)
         << ",initial disturbance ratio\n";
    file << "lambda_c," << CriticalWavelength() << ",classical critical wavelength\n";
    file << "lambda_max," << FastestWavelength() << ",classical fastest wavelength\n";
    file << "lambda," << InputWavelength() << ",imposed reference wavelength\n";
    file << "requested_wavelength_over_a," << run_options.wavelength_over_a
         << ",zero means use the predefined case wavelength\n";
    file << "lambda_over_lambda_max," << InputWavelength() / FastestWavelength() << ",case wavelength ratio\n";
    file << "reference_mode," << ReferenceMode() << ",mode tracked for amplitude diagnostics\n";
    std::string initial_pattern = "single_cosine";
    if (run_options.case_name == "multi")
        initial_pattern = "equal_modes_2_3_4";
    else if (run_options.case_name == "broadband")
        initial_pattern = "equal_broadband_modes_" +
                          std::to_string(run_options.broadband_mode_min) + "_to_" +
                          std::to_string(run_options.broadband_mode_max);
    else if (run_options.case_name == "viscous-decay")
        initial_pattern = "planar_half_sine_velocity_eigenmode";
    file << "initial_pattern," << initial_pattern << ",initial interface disturbance\n";
    file << "domain_length_over_a," << DomainLength() / fibre_radius
         << ",total axial domain including both straight reservoirs\n";
    file << "broadband_mode_min," << run_options.broadband_mode_min
         << ",smallest seeded Fourier mode\n";
    file << "broadband_mode_max," << run_options.broadband_mode_max
         << ",largest seeded Fourier mode\n";
    file << "disturbance_seed," << run_options.disturbance_seed
         << ",phase seed; modal amplitudes remain equal\n";
    file << "resolution," << run_options.resolution << ",particles through mean film thickness\n";
    file << "particle_spacing," << ParticleSpacing() << ",reference particle spacing\n";
    file << "end_T," << run_options.end_T << ",dimensionless end time gamma*t/(mu*a)\n";
    file << "end_time," << PhysicalEndTime() << ",solver physical time\n";
    file << "frames," << run_options.frames << ",requested output intervals\n";
    if (run_options.case_name == "viscous-decay")
    {
        file << "viscous_decay_wave_number," << ViscousDecayWaveNumber()
             << ",pi divided by twice the film thickness\n";
        file << "viscous_decay_rate," << ViscousDecayRate()
             << ",analytic physical-time exponential decay rate\n";
    }
    file << "free_surface_method,reconstructed_one_sided_CSF,support-deficiency interface with periodic Fourier derivatives\n";
    file << "retained_interface_modes," << run_options.interface_modes
         << ",maximum low-pass modes used for curvature\n";
    file << "diagnostic_mode_max,"
         << (run_options.case_name == "broadband"
                 ? run_options.broadband_mode_max
                 : 8)
         << ",largest Fourier mode used for dominant-wavelength diagnostics\n";
    file << "topology_model,single_valued_radius_with_residual_film,beads may be separated by a thin precursor film but not detached\n";
    file << "axisymmetric_source_scope,user_example_only,no SPHinXsys public API changes\n";
    file << "air_particles,0,air phase intentionally omitted\n";
    file << "gravity,0,gravity disabled\n";
}

struct InterfaceMetrics
{
    Real mean_radius = 0.0;
    Real minimum_radius = 0.0;
    Real maximum_radius = 0.0;
    Real minimum_thickness = 0.0;
    Real maximum_thickness = 0.0;
    Real amplitude = 0.0;
    Real dominant_wavelength = 0.0;
    size_t dominant_mode = 1;
    size_t number_of_modes = 0;
    std::array<Real, maximum_diagnostic_modes> mode_amplitudes{};
    size_t valid_bins = 0;
};

InterfaceMetrics MeasureInterface(BaseParticles &particles, Real *positions_flat,
                                  int *indicator, size_t number_of_bins)
{
    Vecd *positions = reinterpret_cast<Vecd *>(positions_flat);
    std::vector<Real> radius(number_of_bins, -std::numeric_limits<Real>::infinity());
    std::vector<Real> thickness(number_of_bins, -std::numeric_limits<Real>::infinity());
    for (size_t i = 0; i < particles.TotalRealParticles(); ++i)
    {
        if (indicator[i] != 1 ||
            positions[i][1] < FiberRadius(positions[i][0]) + 0.25 * FilmThickness())
            continue;
        Real shifted_x = positions[i][0] - DomainLowerBound();
        Real x = shifted_x - std::floor(shifted_x / DomainLength()) * DomainLength();
        size_t bin = std::min(number_of_bins - 1,
                              static_cast<size_t>(std::floor(x / DomainLength() * number_of_bins)));
        // Lattice particles represent cell centres; add half a spacing to
        // estimate the geometric free-surface location.
        Real surface_radius = positions[i][1] + 0.5 * ParticleSpacing();
        radius[bin] = std::max(radius[bin], surface_radius);
        thickness[bin] = std::max(
            thickness[bin], surface_radius - FiberRadius(positions[i][0]));
    }

    size_t valid = 0;
    for (Real value : radius)
        if (std::isfinite(value))
            ++valid;
    if (valid == 0)
        return InterfaceMetrics{};

    for (size_t i = 0; i < number_of_bins; ++i)
    {
        if (std::isfinite(radius[i]))
            continue;
        size_t left = i, right = i;
        do
            left = (left + number_of_bins - 1) % number_of_bins;
        while (left != i && !std::isfinite(radius[left]));
        do
            right = (right + 1) % number_of_bins;
        while (right != i && !std::isfinite(radius[right]));
        radius[i] = 0.5 * (radius[left] + radius[right]);
        thickness[i] = 0.5 * (thickness[left] + thickness[right]);
    }

    InterfaceMetrics metrics;
    metrics.valid_bins = valid;
    metrics.minimum_radius = std::numeric_limits<Real>::max();
    metrics.maximum_radius = -std::numeric_limits<Real>::max();
    metrics.minimum_thickness = std::numeric_limits<Real>::max();
    metrics.maximum_thickness = -std::numeric_limits<Real>::max();
    for (size_t i = 0; i < number_of_bins; ++i)
    {
        Real value = radius[i];
        metrics.mean_radius += value;
        metrics.minimum_radius = std::min(metrics.minimum_radius, value);
        metrics.maximum_radius = std::max(metrics.maximum_radius, value);
        metrics.minimum_thickness =
            std::min(metrics.minimum_thickness, thickness[i]);
        metrics.maximum_thickness =
            std::max(metrics.maximum_thickness, thickness[i]);
    }
    metrics.mean_radius /= static_cast<Real>(number_of_bins);

    Real cosine = 0.0, sine = 0.0;
    for (size_t i = 0; i < number_of_bins; ++i)
    {
        Real x = DomainLength() * (static_cast<Real>(i) + 0.5) / static_cast<Real>(number_of_bins);
        Real departure = thickness[i] - FilmThickness();
        cosine += departure * std::cos(ReferenceWaveNumber() * x);
        sine += departure * std::sin(ReferenceWaveNumber() * x);
    }
    metrics.amplitude = 2.0 * std::sqrt(cosine * cosine + sine * sine) /
                        static_cast<Real>(number_of_bins);

    Real largest_mode = -1.0;
    size_t dominant_mode = 1;
    size_t requested_maximum_mode = run_options.case_name == "broadband"
                                        ? static_cast<size_t>(run_options.broadband_mode_max)
                                        : 8;
    size_t maximum_mode = std::min(
        {maximum_diagnostic_modes, requested_maximum_mode, number_of_bins / 4});
    for (size_t mode = 1; mode <= maximum_mode; ++mode)
    {
        Real c = 0.0, s = 0.0;
        for (size_t i = 0; i < number_of_bins; ++i)
        {
            Real angle = 2.0 * Pi * static_cast<Real>(mode) *
                         (static_cast<Real>(i) + 0.5) / static_cast<Real>(number_of_bins);
            Real departure = thickness[i] - FilmThickness();
            c += departure * std::cos(angle);
            s += departure * std::sin(angle);
        }
        Real mode_amplitude = 2.0 * std::sqrt(c * c + s * s) /
                              static_cast<Real>(number_of_bins);
        metrics.mode_amplitudes[mode - 1] = mode_amplitude;
        if (mode_amplitude > largest_mode)
        {
            largest_mode = mode_amplitude;
            dominant_mode = mode;
        }
    }
    metrics.number_of_modes = maximum_mode;
    metrics.dominant_mode = dominant_mode;
    metrics.dominant_wavelength = DomainLength() / static_cast<Real>(dominant_mode);
    return metrics;
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

    Real spacing = ParticleSpacing();
    Real boundary_width = BoundaryWidth();
    BoundingBoxd bounds(Vec2d(DomainLowerBound() - boundary_width, -boundary_width),
                        Vec2d(DomainUpperBound() + boundary_width,
                              RadialDomainUpperBound()));
    SPHSystem sph_system(bounds, spacing);
    sph_system.handleCommandlineOptions(static_cast<int>(sph_argument_pointers.size()),
                                        sph_argument_pointers.data());
    IO::getEnvironment().appendOutputFolder(run_options.Label());

    std::cout << "Case=" << run_options.case_name
              << " axisym=" << run_options.axisym
              << " regime=" << run_options.regime
              << " h/a=" << run_options.film_ratio
              << " fiber=" << run_options.fiber_profile
              << " a_tip/a=" << run_options.tip_radius_ratio
              << " input_lambda/a=" << InputWavelength() / fibre_radius
              << " lambda/lambda_max=" << InputWavelength() / FastestWavelength()
              << " resolution=" << run_options.resolution << std::endl;
    std::cout << "Oh=" << run_options.oh << " mu=" << DynamicViscosity()
              << " end_T=" << run_options.end_T
              << " solver_end_time=" << PhysicalEndTime() << std::endl;

    FluidBody liquid(sph_system, makeShared<LiquidFilmShape>("LiquidFilmHalf"));
    liquid.defineMatterMaterial<WeaklyCompressibleFluid>(reference_density, SoundSpeed());
    liquid.addMaterialProperty<Viscosity>(DynamicViscosity());
    liquid.generateParticles<BaseParticles, Lattice>();

    SolidBody fibre(sph_system, makeShared<RigidFiberShape>("RigidFiberHalf"));
    fibre.defineMatterMaterial<Solid>();
    fibre.generateParticles<BaseParticles, Lattice>();

    InnerRelation liquid_inner(liquid);
    ContactRelation liquid_fibre_contact(liquid, {&fibre});
    ComplexRelation liquid_complex(liquid_inner, liquid_fibre_contact);

    SimpleDynamics<NormalDirectionFromBodyShape> fibre_normals(fibre);
    InteractionWithUpdate<FreeSurfaceIndicationComplex> indicate_free_surface(
        liquid_inner, liquid_fibre_contact);
    ReconstructedAxisymmetricCapillaryForce reconstructed_capillary_force(liquid);
    // Keep the diagnostic switch connected to the actual force.  Earlier
    // versions parsed --capillary-scale but left the reconstructed force at 1.
    reconstructed_capillary_force.setForceScale(run_options.capillary_scale);

    // Pressure/density integration registers Velocity, Pressure and Force.
    // Construct it before viscosity dynamics, which reads those variables.
    Dynamics1Level<fluid_dynamics::Integration1stHalfWithWallRiemann>
        pressure_relaxation(liquid_inner, liquid_fibre_contact);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallRiemann>
        density_relaxation(liquid_inner, liquid_fibre_contact);

    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall>
        viscous_force(liquid_inner, liquid_fibre_contact);
    InteractionSplit<DampingPairwiseWithWall<Vec2d, FixedDampingRate>>
        implicit_viscous_damping(
            DynamicsArgs(liquid_inner, "Velocity", DynamicViscosity()),
            DynamicsArgs(liquid_fibre_contact, "Velocity", DynamicViscosity()));
    InteractionWithUpdate<AxisymmetricViscousForce> axisymmetric_viscous_force(liquid_inner);

    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep>
        advection_time_step(liquid, CapillaryVelocity(), 0.15);
    ReduceDynamics<fluid_dynamics::AdvectionTimeStep>
        implicit_advection_time_step(liquid, CapillaryVelocity(), 0.15);
    ReduceDynamics<fluid_dynamics::SurfaceTensionTimeStep>
        acoustic_surface_time_step(liquid, 0.4);

    PeriodicAlongAxis periodic_axis(liquid.getSPHBodyBounds(), xAxis);
    PeriodicConditionUsingCellLinkedList periodic_condition(liquid, periodic_axis);
    ParticleSorting particle_sorting(liquid);

    SimpleDynamics<InitialLiquidState> initialize_liquid(liquid);
    ConicalEndBuffer conical_end_buffer(liquid.getBaseParticles());
    SimpleDynamics<InitializeAxisymmetricWeights> initialize_axisymmetric_weights(liquid);
    SimpleDynamics<UpdateAxisymmetricWeights> update_axisymmetric_weights(liquid);
    SimpleDynamics<AxisymmetricContinuityCorrection> axisymmetric_continuity(liquid);

    BodyStatesRecordingToVtp write_states(sph_system);
    write_states.addToWrite<Real>(liquid, "Density");
    write_states.addToWrite<Real>(liquid, "Pressure");
    write_states.addToWrite<int>(liquid, "Indicator");
    write_states.addToWrite<Vecd>(liquid, "Velocity");
    write_states.addToWrite<Vecd>(liquid, "ReconstructedCapillaryForce");
    write_states.addToWrite<Vecd>(liquid, "ViscousForce");
    write_states.addToWrite<Vecd>(liquid, "AxisymmetricViscousForce");

    initialize_liquid.exec();
    conical_end_buffer.capture();
    initialize_axisymmetric_weights.exec();
    if (run_options.FullAxisymmetric())
        update_axisymmetric_weights.exec();

    sph_system.initializeSystemCellLinkedLists();
    if (!IsConicalFiber())
        periodic_condition.update_cell_linked_list_.exec();
    sph_system.initializeSystemConfigurations();
    fibre_normals.exec();
    indicate_free_surface.exec();
    reconstructed_capillary_force.exec();

    std::string output_folder = IO::getEnvironment().OutputFolder();
    WriteParameterSummary(output_folder);
    std::ofstream interface_file(output_folder + "/interface_modes.csv");
    interface_file << "time,T,mean_radius,min_radius,max_radius,min_film_thickness,"
                      "max_film_thickness,neck_thickness_over_h,bead_thickness_over_h,"
                      "amplitude,amplitude_over_h,log_A_over_A0,"
                      "dominant_mode,dominant_wavelength,mode1,mode2,mode3,mode4,"
                      "mode5,mode6,mode7,mode8,valid_surface_bins,ring_mass,ring_mass_error,"
                      "max_speed,min_density,max_density\n";
    interface_file << std::scientific << std::setprecision(10);
    std::ofstream stability_file(output_folder + "/stability_diagnostics.csv");
    stability_file << "time,T,finite_state,out_of_domain_particles,max_speed,"
                      "min_density,max_density,density_relative_deviation\n";
    stability_file << std::scientific << std::setprecision(10);
    std::ofstream modal_spectrum_file(output_folder + "/modal_spectrum.csv");
    modal_spectrum_file
        << "time,T,mode,wavelength,wavelength_over_a,amplitude,amplitude_over_h\n"
        << std::scientific << std::setprecision(10);
    std::ofstream viscous_decay_file;
    if (run_options.case_name == "viscous-decay")
    {
        viscous_decay_file.open(output_folder + "/viscous_decay.csv");
        viscous_decay_file
            << "time,T,numerical_amplitude,analytic_amplitude,"
               "relative_amplitude_error,profile_l2_error,max_radial_velocity\n"
            << std::scientific << std::setprecision(10);
    }

    BaseParticles &particles = liquid.getBaseParticles();
    Vecd *positions = particles.getVariableDataByName<Vecd>("Position");
    Vecd *velocities = particles.getVariableDataByName<Vecd>("Velocity");
    Real *densities = particles.getVariableDataByName<Real>("Density");
    Real *masses = particles.getVariableDataByName<Real>("Mass");
    Real *ring_masses = particles.getVariableDataByName<Real>("AxisymmetricRingMass");
    int *indicators = particles.getVariableDataByName<int>("Indicator");
    const size_t bins = std::max<size_t>(32, static_cast<size_t>(std::round(DomainLength() / spacing)));
    Real initial_amplitude_measured = -1.0;
    Real last_diagnostic_time = -1.0;
    Real initial_ring_mass = 0.0;
    for (size_t i = 0; i < particles.TotalRealParticles(); ++i)
        initial_ring_mass += 2.0 * Pi * ring_masses[i];

    Real initial_viscous_projection = 1.0;
    if (run_options.case_name == "viscous-decay")
    {
        Real numerator = 0.0, denominator = 0.0;
        for (size_t i = 0; i < particles.TotalRealParticles(); ++i)
        {
            Real wall_distance = std::clamp(
                FiberNormalDistance(positions[i]), Real(0.0), FilmThickness());
            Real basis = std::sin(ViscousDecayWaveNumber() * wall_distance);
            numerator += masses[i] * velocities[i][0] * basis;
            denominator += masses[i] * basis * basis;
        }
        initial_viscous_projection = numerator / (denominator + TinyReal);
    }

    auto write_diagnostics = [&](Real time)
    {
        if (last_diagnostic_time >= 0.0 &&
            std::abs(time - last_diagnostic_time) < 1.0e-10 * std::max(Real(1.0), PhysicalEndTime()))
            return true;
        indicate_free_surface.exec();
        reconstructed_capillary_force.exec();
        InterfaceMetrics metrics = MeasureInterface(
            particles, reinterpret_cast<Real *>(positions), indicators, bins);
        if (initial_amplitude_measured < 0.0)
            initial_amplitude_measured = std::max(metrics.amplitude, 1.0e-12 * FilmThickness());

        bool finite = true;
        size_t escaped = 0;
        Real max_speed = 0.0;
        Real min_density = std::numeric_limits<Real>::max();
        Real max_density = -std::numeric_limits<Real>::max();
        Real ring_mass = 0.0;
        for (size_t i = 0; i < particles.TotalRealParticles(); ++i)
        {
            finite = finite && std::isfinite(densities[i]) &&
                     std::isfinite(positions[i][0]) && std::isfinite(positions[i][1]) &&
                     std::isfinite(velocities[i][0]) && std::isfinite(velocities[i][1]);
            if (positions[i][1] < FiberRadius(positions[i][0]) - spacing ||
                positions[i][1] > RadialDomainUpperBound() - boundary_width)
                ++escaped;
            max_speed = std::max(max_speed, velocities[i].norm());
            min_density = std::min(min_density, densities[i]);
            max_density = std::max(max_density, densities[i]);
            ring_mass += run_options.FullAxisymmetric()
                             ? 2.0 * Pi * positions[i][1] * masses[i]
                             : 2.0 * Pi * ring_masses[i];
        }
        Real T = surface_tension * time / (DynamicViscosity() * fibre_radius);
        Real ring_error = (ring_mass - initial_ring_mass) / (initial_ring_mass + TinyReal);
        Real amplitude_floor = 1.0e-12 * FilmThickness();
        Real log_ratio = std::log(std::max(metrics.amplitude, amplitude_floor) /
                                  initial_amplitude_measured);
        Real minimum_thickness = metrics.minimum_thickness;
        Real maximum_thickness = metrics.maximum_thickness;
        interface_file << time << "," << T << "," << metrics.mean_radius << ","
                       << metrics.minimum_radius << "," << metrics.maximum_radius << ","
                       << minimum_thickness << "," << maximum_thickness << ","
                       << minimum_thickness / FilmThickness() << ","
                       << maximum_thickness / FilmThickness() << ","
                       << metrics.amplitude << "," << metrics.amplitude / FilmThickness() << ","
                       << log_ratio << "," << metrics.dominant_mode << ","
                       << metrics.dominant_wavelength << ",";
        for (size_t mode = 0; mode < 8; ++mode)
            interface_file << metrics.mode_amplitudes[mode] << ",";
        interface_file
                       << metrics.valid_bins << "," << ring_mass << "," << ring_error << ","
                       << max_speed << "," << min_density << "," << max_density << "\n";
        interface_file.flush();
        for (size_t mode = 1; mode <= metrics.number_of_modes; ++mode)
        {
            Real wavelength = DomainLength() / static_cast<Real>(mode);
            Real mode_amplitude = metrics.mode_amplitudes[mode - 1];
            modal_spectrum_file << time << "," << T << "," << mode << ","
                                << wavelength << "," << wavelength / fibre_radius << ","
                                << mode_amplitude << ","
                                << mode_amplitude / FilmThickness() << "\n";
        }
        modal_spectrum_file.flush();
        Real density_deviation = std::max(std::abs(min_density / reference_density - 1.0),
                                          std::abs(max_density / reference_density - 1.0));
        stability_file << time << "," << T << "," << (finite ? 1 : 0) << ","
                       << escaped << "," << max_speed << "," << min_density << ","
                       << max_density << "," << density_deviation << "\n";
        stability_file.flush();
        if (run_options.case_name == "viscous-decay")
        {
            Real numerator = 0.0, denominator = 0.0;
            Real error_norm = 0.0, reference_norm = 0.0;
            Real maximum_radial_velocity = 0.0;
            Real analytic_amplitude =
                initial_viscous_projection * std::exp(-ViscousDecayRate() * time);
            for (size_t i = 0; i < particles.TotalRealParticles(); ++i)
            {
                Real wall_distance = std::clamp(
                    FiberNormalDistance(positions[i]), Real(0.0), FilmThickness());
                Real basis = std::sin(ViscousDecayWaveNumber() * wall_distance);
                numerator += masses[i] * velocities[i][0] * basis;
                denominator += masses[i] * basis * basis;
                Real exact_velocity = analytic_amplitude * basis;
                Real difference = velocities[i][0] - exact_velocity;
                error_norm += masses[i] * difference * difference;
                reference_norm += masses[i] * exact_velocity * exact_velocity;
                maximum_radial_velocity =
                    std::max(maximum_radial_velocity, std::abs(velocities[i][1]));
            }
            Real numerical_amplitude = numerator / (denominator + TinyReal);
            viscous_decay_file
                << time << "," << T << "," << numerical_amplitude << ","
                << analytic_amplitude << ","
                << (numerical_amplitude - analytic_amplitude) /
                       (std::abs(analytic_amplitude) + TinyReal)
                << "," << std::sqrt(error_norm / (reference_norm + TinyReal))
                << "," << maximum_radial_velocity << "\n";
            viscous_decay_file.flush();
        }
        last_diagnostic_time = time;
        return finite && escaped == 0;
    };

    Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
    Real output_interval = PhysicalEndTime() / static_cast<Real>(run_options.frames);
    Real dt = 0.0;
    size_t iteration = 0;
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
            // Preserve the initialized Laplace-pressure equilibrium. Density
            // evolves through the continuity equation; summation reset would
            // otherwise erase the single-phase free-surface pressure offset.
            if (run_options.viscosity_integration == "explicit")
                viscous_force.exec();
            if (run_options.FullAxisymmetric())
                axisymmetric_viscous_force.exec();

            Real Dt = run_options.viscosity_integration == "implicit"
                          ? implicit_advection_time_step.exec()
                          : advection_time_step.exec();
            Real relaxation_time = 0.0;
            const Real time_tolerance =
                100.0 * std::numeric_limits<Real>::epsilon() *
                std::max(Real(1.0), PhysicalEndTime());
            while (relaxation_time < Dt && integration_time < output_interval &&
                   physical_time < PhysicalEndTime())
            {
                Real remaining_advection = Dt - relaxation_time;
                Real remaining_output = output_interval - integration_time;
                Real remaining_total = PhysicalEndTime() - physical_time;
                if (remaining_advection <= time_tolerance)
                    break;
                if (remaining_output <= time_tolerance)
                {
                    integration_time = output_interval;
                    break;
                }
                if (remaining_total <= time_tolerance)
                {
                    physical_time = PhysicalEndTime();
                    break;
                }
                reconstructed_capillary_force.exec();

                Real acoustic_dt = acoustic_surface_time_step.exec();
                if (!std::isfinite(acoustic_dt) || acoustic_dt <= 0.0)
                    throw std::runtime_error(
                        "Acoustic/capillary time-step estimator returned a non-positive value.");
                dt = std::min({acoustic_dt, remaining_advection,
                               remaining_output, remaining_total});
                if (dt <= time_tolerance)
                    throw std::runtime_error(
                        "Time step fell below the floating-point progress tolerance.");
                if (run_options.viscosity_integration == "implicit")
                    implicit_viscous_damping.exec(dt);
                pressure_relaxation.exec(dt);
                density_relaxation.exec(dt);
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

            ++iteration;
            if (!IsConicalFiber())
                periodic_condition.bounding_.exec();
            // The finite conical case carries a material reference state for
            // the end reservoirs.  Keep particle indices fixed until every
            // custom buffer/force variable has a verified sorting delegate;
            // cell-linked lists are still rebuilt every advection step, so
            // neighbour search and the physical equations remain active.
            if (!IsConicalFiber() && iteration % 100 == 0)
                particle_sorting.exec();
            liquid.updateCellLinkedList();
            if (!IsConicalFiber())
                periodic_condition.update_cell_linked_list_.exec();
            liquid_complex.updateConfiguration();

            if (iteration % 50 == 0)
                std::cout << std::fixed << std::setprecision(7)
                          << "N=" << iteration << " time=" << physical_time
                          << " T=" << surface_tension * physical_time /
                                          (DynamicViscosity() * fibre_radius)
                          << " dt=" << dt << std::endl;
        }

        if (!write_diagnostics(physical_time))
        {
            std::cerr << "Non-finite or escaped-particle state detected at time="
                      << physical_time << "." << std::endl;
            return 3;
        }
        // The fibre is static, so explicitly mark it for each requested frame.
        fibre.setNewlyUpdated();
        write_states.writeToFile();
    }

    TimeInterval elapsed = TickCount::now() - start;
    std::cout << "Completed " << run_options.Label() << " in "
              << elapsed.seconds() << " seconds. Output: " << output_folder << std::endl;
    return 0;
}
