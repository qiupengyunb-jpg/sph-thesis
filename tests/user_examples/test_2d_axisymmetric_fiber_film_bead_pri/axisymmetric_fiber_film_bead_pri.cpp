/**
 * @file axisymmetric_fiber_film_bead_pri.cpp
 * @brief Nonlinear bead-on-fibre trial built on the validated early-stage PRI
 *        implementation without changing the SPHinXsys public API.
 *
 * The shared implementation is compiled with nonlinear defaults. The original
 * early-stage executable retains its former defaults and output label.
 */
#define SPHINXSYS_FIBER_FILM_BEAD_DEFAULTS
#include "../test_2d_axisymmetric_fiber_film_pri/axisymmetric_fiber_film_pri.cpp"
