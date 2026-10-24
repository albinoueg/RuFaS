"""Sequential linear programming with explicit checks of the original inequalities.

Every optimization subproblem is an elastic LP solved by HiGHS. This is a local
approximation, not an exact or globally optimal reformulation of NASEM.
"""
import numpy as np
from scipy.optimize import OptimizeResult, linprog


def solve_linearized(cost, x0, bounds, constraints, maxiter=150, tolerance=1e-7):
    """Minimize a linear cost subject to scipy-style inequalities g(x) >= 0.

    Constraint scaling is fixed at the starting point. A trust region and merit
    line search control errors in the first-order Taylor approximation. Elastic
    slacks allow restoration from infeasible starts but never relax final checks.
    """
    c = np.asarray(cost, dtype=float)
    lower = np.array([0.0 if b[0] is None else b[0] for b in bounds])
    upper = np.array([np.inf if b[1] is None else b[1] for b in bounds])
    x = np.clip(np.asarray(x0, dtype=float), lower, upper)
    n, m = len(x), len(constraints)
    evaluations = 0

    def raw(z):
        nonlocal evaluations
        evaluations += 1
        return np.array([q['fun'](z, *q.get('args', ())) for q in constraints], dtype=float)

    g = raw(x)
    scale = np.maximum(np.abs(g), 1.0)
    radius = max(1.0, float(np.sum(x)) * 0.25)
    penalty = max(10.0, float(np.max(np.abs(c))) * 10)
    history = []
    last_lp = None
    message = 'Iteration limit reached'
    for iteration in range(maxiter):
        jac = np.empty((m, n))
        for j in range(n):
            h = 1e-5 * max(1.0, abs(x[j]))
            if x[j] - lower[j] >= h and upper[j] - x[j] >= h:
                plus, minus = x.copy(), x.copy()
                plus[j] += h
                minus[j] -= h
                jac[:, j] = (raw(plus) - raw(minus)) / (2 * h) / scale
                continue
            step = min(h, upper[j] - x[j])
            if step < 1e-10:
                step = -min(h, x[j] - lower[j])
            if abs(step) < 1e-10:
                jac[:, j] = 0
            else:
                shifted = x.copy()
                shifted[j] += step
                jac[:, j] = (raw(shifted) - g) / step / scale
        scaled = g / scale
        # g(x) + J d + s >= 0, s >= 0.
        lp = linprog(np.r_[c, np.full(m, penalty)],
                     A_ub=np.c_[-jac, -np.eye(m)], b_ub=scaled,
                     bounds=list(zip(np.maximum(lower - x, -radius),
                                     np.minimum(upper - x, radius))) + [(0, None)] * m,
                     method='highs', options={'primal_feasibility_tolerance': 1e-9,
                                              'dual_feasibility_tolerance': 1e-9})
        last_lp = lp
        if not lp.success:
            message = 'Linear subproblem failed: ' + lp.message
            break
        direction = lp.x[:n]
        predicted_improvement = penalty * np.maximum(-scaled, 0).sum() - lp.fun
        if np.maximum(-g, 0).max(initial=0) <= tolerance and predicted_improvement <= 1e-7:
            message = 'Feasible stationary point of sequential LP approximation'
            break
        merit = float(c @ x + penalty * np.maximum(-scaled, 0).sum())
        accepted = False
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
            candidate = np.clip(x + alpha * direction, lower, upper)
            candidate_g = raw(candidate)
            candidate_merit = float(c @ candidate + penalty * np.maximum(-candidate_g / scale, 0).sum())
            if candidate_merit < merit - 1e-10:
                predicted_merit = float(c @ candidate + penalty * np.maximum(
                    -(scaled + jac @ (candidate - x)), 0).sum())
                ratio = (merit - candidate_merit) / max(merit - predicted_merit, 1e-12)
                x, g = candidate, candidate_g
                accepted = True
                break
        violation = float(np.maximum(-g, 0).max(initial=0))
        history.append({'iteration': iteration + 1, 'cost': float(c @ x),
                        'max_violation': violation, 'radius': radius})
        if not accepted:
            radius *= 0.3
            if violation > tolerance and radius < 1e-4 and penalty < 1e7:
                penalty *= 10
                radius = max(1.0, float(np.sum(x)) * 0.1)
        elif alpha < 0.5 or ratio < 0.25:
            radius *= 0.5
        elif alpha == 1.0 and ratio > 0.75:
            radius = min(radius * 1.5, 50.0)
        if violation <= tolerance and (np.max(np.abs(direction)) < 1e-5 or radius < 1e-6):
            message = 'Feasible stationary point of sequential LP approximation'
            break
        if radius < 1e-8:
            message = 'Trust region exhausted'
            break
    final_g = raw(x)
    violation = float(np.maximum(-final_g, 0).max(initial=0))
    feasible = bool(np.isfinite(final_g).all() and violation <= tolerance
                    and np.all(x >= lower - tolerance) and np.all(x <= upper + tolerance))
    converged = message.startswith('Feasible stationary')
    return OptimizeResult(x=x, fun=float(c @ x), success=feasible and converged,
                          status=0 if feasible and converged else 1,
                          message=message, nit=len(history), nfev=evaluations,
                          max_constraint_violation=violation, constraint_margins=final_g,
                          history=history, backend='SLP-HiGHS',
                          lp_marginals=None if last_lp is None or not last_lp.success
                          else last_lp.ineqlin.marginals / scale)
