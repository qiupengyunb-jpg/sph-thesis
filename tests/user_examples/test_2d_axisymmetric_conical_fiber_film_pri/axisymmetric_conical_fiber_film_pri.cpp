/**
 * @file axisymmetric_conical_fiber_film_pri.cpp
 * @brief Finite half-axisymmetric liquid film on a rigid conical frustum.
 *
 * The implementation reuses the validated axisymmetric film equations while
 * enabling the conical geometry defaults.  The thick end has radius a=1 and
 * the thin-end radius is selected by --tip-radius-ratio.  Unlike the uniform
 * fibre case, unequal-radius ends are not connected periodically.  Short
 * liquid end bands are fixed at their initial state as numerical reservoirs.
 */
#define SPHINXSYS_CONICAL_FIBER_DEFAULTS
#include "../test_2d_axisymmetric_fiber_film_pri/axisymmetric_fiber_film_pri.cpp"
