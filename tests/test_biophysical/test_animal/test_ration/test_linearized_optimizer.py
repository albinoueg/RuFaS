import unittest
import numpy as np
from RUFAS.biophysical.animal.ration.linearized_optimizer import solve_linearized


class SequentialLPTests(unittest.TestCase):
    def test_known_linear_optimum_from_infeasible_start(self):
        result = solve_linearized([1, 2], [0.2, 0.2], [(0, 5), (0, 5)],
                                  [{'fun': lambda x: x.sum() - 3}])
        self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(result.x, [3, 0], atol=1e-6)

    def test_curved_constraint_is_checked_at_final_point(self):
        result = solve_linearized([-1, -1], [0.2, 0.2], [(0, 2), (0, 2)],
                                  [{'fun': lambda x: 1 - x @ x}])
        self.assertTrue(result.success, result.message)
        self.assertGreaterEqual(1 - result.x @ result.x, -1e-7)
        self.assertAlmostEqual(result.fun, -np.sqrt(2), places=5)

    def test_infeasible_problem_never_reports_success(self):
        result = solve_linearized([1], [0.5], [(0, 1)], [{'fun': lambda x: x[0] - 2}])
        self.assertFalse(result.success)
        self.assertGreater(result.max_constraint_violation, 0.9)

    def test_fixed_ingredient_bound(self):
        result = solve_linearized([1, 2], [0.5, 0.5], [(1, 1), (0, 3)],
                                  [{'fun': lambda x: x.sum() - 2}])
        self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(result.x, [1, 1], atol=1e-6)
