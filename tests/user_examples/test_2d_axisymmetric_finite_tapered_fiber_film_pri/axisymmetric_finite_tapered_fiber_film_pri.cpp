/**
 * @file axisymmetric_finite_tapered_fiber_film_pri.cpp
 * @brief Closed finite tapered fibre carrying a finite liquid coating.
 *
 * The central sidewall is the tapered research section.  Dry straight
 * shoulders and C2-smooth physical caps close the rigid fibre at both ends.
 * The liquid has no inlet/outlet and therefore remains a closed material
 * system.  Default settings deliberately stop before topology change.
 */
#define SPHINXSYS_FINITE_TAPERED_FIBER_DEFAULTS
#include "../test_2d_axisymmetric_fiber_film_breakup_pri/axisymmetric_fiber_film_breakup_pri.cpp"
