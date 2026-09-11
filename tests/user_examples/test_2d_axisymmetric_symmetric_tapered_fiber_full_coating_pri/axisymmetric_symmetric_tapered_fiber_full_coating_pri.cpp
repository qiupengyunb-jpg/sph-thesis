/**
 * @file axisymmetric_symmetric_tapered_fiber_full_coating_pri.cpp
 * @brief Half-axisymmetric fine--coarse--fine rigid fibre with a complete film.
 *
 * This target mirrors the retained one-way taper about its thick end.  Each
 * half therefore keeps the same radius ratio and taper slope.  The thickest
 * radius occurs at x=0, while both physical ends retain the existing rounded
 * fine cap.  It is an independent user example and does not replace the
 * earlier coarse--fine target.
 */
#define SPHINXSYS_FINITE_TAPERED_FIBER_DEFAULTS
#define SPHINXSYS_FINITE_TAPERED_FULL_COATING_DEFAULTS
#define SPHINXSYS_SYMMETRIC_TAPERED_FIBER_DEFAULTS
#include "../test_2d_axisymmetric_fiber_film_breakup_pri/axisymmetric_fiber_film_breakup_pri.cpp"
