"""Numerical and combinatorial checks independent of any vendor evaluator."""
import math
import random
import unittest
from mission.baselines import displacement_set, project_to_grid
from mission.dynamics import field_and_jacobian, integrate, transition_matrices, MU
from mission.problems import DRO, HALO, MissionProblem, load_problem
from mission.optimizer import initial_population, evolve


class BaselineTests(unittest.TestCase):
    def test_exact_support(self):
        self.assertEqual(displacement_set(set()), set())
        self.assertEqual(displacement_set({(0, 0)}), {(0, 0)})
        self.assertEqual(displacement_set({(0, 0), (2, 1)}), {(0, 0), (2, 1), (-2, -1)})
        self.assertEqual(len(displacement_set({(0, 0), (1, 0), (2, 0)})), 5)

    def test_crop_and_quantization(self):
        points = [(-1, 0, 0), (1, 0, 0), (0, 0, 0), (-0.99, 0.99, 0)]
        self.assertEqual(project_to_grid(points, 11), [(5, 5, 5), (0, 10, 5)])


class DynamicsTests(unittest.TestCase):
    def test_jacobian_against_finite_difference(self):
        state = [0.8, 0.1, -0.15, 0.02, 0.3, -0.01]
        _, jac = field_and_jacobian(state)
        h = 1e-6
        for j in range(6):
            plus, minus = state[:], state[:]
            plus[j] += h
            minus[j] -= h
            fp, _ = field_and_jacobian(plus)
            fm, _ = field_and_jacobian(minus)
            for i in range(6):
                self.assertAlmostEqual(jac[i][j], (fp[i] - fm[i]) / (2*h), delta=1e-7)

    def test_integrator_known_solution(self):
        result = integrate(lambda x: [x[0]], [1.], [0., 1.])[-1][0]
        self.assertAlmostEqual(result, math.e, delta=1e-8)

    def test_transition_predicts_perturbed_trajectory(self):
        state = DRO[0]
        phi = transition_matrices(state, [0., 0.2])[-1]
        eps = 1e-6
        for j in range(6):
            plus, minus = state[:], state[:]
            plus[j] += eps
            minus[j] -= eps
            f = lambda s: field_and_jacobian(s)[0]
            xp = integrate(f, plus, [0., 0.2])[-1]
            xm = integrate(f, minus, [0., 0.2])[-1]
            for i in range(6):
                self.assertAlmostEqual(phi[i][j], (xp[i]-xm[i])/(2*eps), delta=2e-6)

    def test_jacobi_constant(self):
        def invariant(s):
            x,y,z,vx,vy,vz = s
            r1 = math.sqrt((x+MU)**2+y*y+z*z)
            r2 = math.sqrt((x-1+MU)**2+y*y+z*z)
            return x*x+y*y+2*((1-MU)/r1+MU/r2)-vx*vx-vy*vy-vz*vz
        for initial, period in [DRO, HALO]:
            final = integrate(lambda s: field_and_jacobian(s)[0], initial, [0., period])[-1]
            self.assertAlmostEqual(invariant(initial), invariant(final), delta=1e-7)

    def test_halo_transition_converges_with_tighter_tolerance(self):
        initial, period = HALO
        a = transition_matrices(initial, [0., period, 2*period], tolerance=1e-9)
        b = transition_matrices(initial, [0., period, 2*period], tolerance=1e-11)
        error = max(abs(x-y)/(1+abs(y)) for ma,mb in zip(a,b)
                    for ra,rb in zip(ma,mb) for x,y in zip(ra,rb))
        self.assertLess(error, 2e-6)


class MissionTests(unittest.TestCase):
    def test_collocated_formation_and_bounds(self):
        problem = load_problem('easy')
        self.assertAlmostEqual(problem.fitness([0.]*30)[0], -3/441)
        with self.assertRaises(ValueError):
            problem.fitness([0.]*29)
        with self.assertRaises(ValueError):
            problem.fitness([math.nan]*30)

    def test_worst_observation_objective(self):
        problem = MissionProblem.__new__(MissionProblem)
        problem.observations = lambda x: [{'total': 0.4}, {'total': 0.1}, {'total': 0.3}]
        self.assertEqual(problem.fitness([]), [-0.1])

    def test_optimizer_monotonic_and_reproducible(self):
        problem = load_problem('easy')
        runs = []
        for _ in range(2):
            rng = random.Random(4)
            pop = initial_population(problem, 5, rng)
            result = evolve(problem, pop, 2, rng)
            self.assertTrue(all(b <= a for a,b in zip(result[2], result[2][1:])))
            self.assertEqual(result[3], 15)
            problem.decode(result[0])
            runs.append(result)
        self.assertEqual(runs[0], runs[1])


if __name__ == '__main__':
    unittest.main()
