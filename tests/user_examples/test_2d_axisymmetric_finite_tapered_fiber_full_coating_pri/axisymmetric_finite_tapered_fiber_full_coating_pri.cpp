/**
 * @file axisymmetric_finite_tapered_fiber_full_coating_pri.cpp
 * @brief Half-axisymmetric finite tapered fibre fully covered by liquid.
 *
 * The liquid forms a closed shell around a rigid fibre with round thick and
 * thin ends.  There is no initial dry fibre and no initial three-phase
 * contact line.  The first validation stage keeps rupture, disjoining
 * pressure and post-breakup wetting disabled so that geometry, axis support
 * and capillary stability can be assessed independently.
 */
#define SPHINXSYS_FINITE_TAPERED_FIBER_DEFAULTS
#define SPHINXSYS_FINITE_TAPERED_FULL_COATING_DEFAULTS
#include "../test_2d_axisymmetric_fiber_film_breakup_pri/axisymmetric_fiber_film_breakup_pri.cpp"
