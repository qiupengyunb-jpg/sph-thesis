/**
 * @file axisymmetric_conical_fiber_film_breakup_pri.cpp
 * @brief Topology-capable second-generation liquid-film model on a finite
 *        conical fibre.
 *
 * This target deliberately uses the breakup/JFM/adhesion implementation as
 * its main program and enables the conical geometry supplied by the shared
 * axisymmetric base.  It is separate from the retained-film conical trial.
 */
#define SPHINXSYS_CONICAL_FIBER_DEFAULTS
#include "../test_2d_axisymmetric_fiber_film_breakup_pri/axisymmetric_fiber_film_breakup_pri.cpp"
