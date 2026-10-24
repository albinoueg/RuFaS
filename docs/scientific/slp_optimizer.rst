Sequential linear ration optimization
====================================

The ration optimizer supports two backends: the existing SLSQP method and a
sequential linear programming (SLP) method implemented with SciPy/HiGHS.
SLSQP remains the default. The SLP implementation does not call SLSQP internally.

Configuration
-------------

In the feed input JSON, select the backend with::

    "ration_formulation_parameters": {
        "optimization_method": "SLP"
    }

Add this key to the existing object; retain its other parameters. Valid values
are ``SLSQP`` and ``SLP``. Omitting the key selects SLSQP. No additional solver
dependency is required beyond the SciPy dependency already used by RuFaS.

Formulation and acceptance
--------------------------

Decision variables are kg of ingredient dry matter per animal per day. The
objective is the linear sum of ingredient purchase costs. Both backends use
the same ingredient bounds and original nutrient constraint functions.

At each iterate, SLP approximates the inequalities ``g(x) >= 0`` with their
first-order expansions. Each subproblem minimizes the incremental linear cost
plus penalized nonnegative elastic slacks, subject to the linearized
constraints, ingredient bounds, and a box trust region. Finite differences
are central in the interior and one-sided at bounds. A merit line search and
trust-region updates control approximation error. Elastic slacks support
infeasible starts; the penalty increases if feasibility restoration stalls.

A result is successful only after local convergence and a fresh check of
all original inequalities and bounds at absolute tolerance ``1e-7`` in each
constraint's native units. The existing nutritional evaluator still runs
after the solver. A failed SLP attempt returns ``success=False`` to the existing
ration management flow, which retains its normal retry and fallback behavior.

This method solves a sequence of linear programs. It is a local approximation,
not a globally equivalent single-LP reformulation of NASEM. It provides no
certificate of global optimality and can fail on other datasets or nonsmooth
regions. A strict single-LP requirement would require a different approximation
with explicitly bounded domains and approximation errors.

Diagnostics include constraint margins, the maximum violation, iteration
history, and the last LP's inequality marginals. These marginals describe the
last linear subproblem and its trust region; they are not global shadow prices
for the nonlinear NASEM model.

Validation and sensitivity study
-------------------------------

The implementation was evaluated on a local NASEM farm scenario with a
2015--2019 simulation, 1,826 days, and random seed 44:

* 123 formulation problems were captured from the SLSQP reference.
* A deterministic sample of 15 problems covered the three adult categories.
  SLP and the locally corrected SLSQP both passed all original nonlinear
  constraint checks. Their costs differed by less than 0.000003 percent.
* Both integrated simulations completed without errors, with 123 successful
  formulations and 123 category-appropriate nutritional evaluations. Their
  total milk production was 6,846,331.509 kg.
* Sensitivity analysis used one frozen lactating-cow problem, seven parameters,
  and five levels (-20, -10, 0, +10, +20 percent), giving 105 reoptimizations
  across SLP, default SLSQP, and SLSQP with tighter tolerance.
* Protein requirement and soybean meal price had the largest central cost
  elasticities: 0.7000 and 0.5206, respectively, over the +/-10 percent interval.
  These are finite-interval, scenario-specific estimates, not universal values.

The downloaded upstream constraint functions were also evaluated separately
at commit ``beac7f321761652d46c8442fbf0ae2df951cff70``. The main solver comparison
uses the same locally corrected equations for both methods. Earlier corrections
to individual-animal aggregation, nutritional evaluation, and ration precision
must not be attributed to the SLP backend itself.

The local farm inputs, raw outputs, backups, and full study report are deliberately
kept outside the source repository. The tests below use synthetic data and do
not depend on those local files. Reproducing the exact five-year study requires
the external scenario and experimental artifacts.

Tests
-----

From the repository root, using the project's Python environment::

    python -m unittest discover -s tests/test_biophysical/test_animal/test_ration -p test_linearized_optimizer.py -v
    python -m unittest discover -s tests/test_biophysical/test_animal/test_ration -p test_nutrition_regressions.py -v

The four SLP tests cover a known linear optimum, a curved constraint checked
against its original equation, an infeasible problem, and a fixed ingredient.
The 14 nutritional regression tests cover shared equations, evaluation flags,
precision, calf protein units, and feed selection when repopulating an empty pen.

See the `SciPy/HiGHS documentation <https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html>`_
for LP result fields and the interpretation of inequality marginals.
