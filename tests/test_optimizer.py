"""Check local moves against propagated limits and optimizer invariants."""
import contextlib
import io
import random
import unittest

from mission.optimizer import evolve, initial_population, mutate_satellite
from mission.problems import load_problem


class LocalMutationTests(unittest.TestCase):
    def test_one_satellite_moves_without_modifying_parent(self):
        problem = load_problem('easy')
        parent = [0.] * (6 * problem.n_sat)
        trial = mutate_satellite(problem, parent, random.Random(11))
        changed = {index % problem.n_sat for index, (a, b) in enumerate(zip(parent, trial)) if a != b}
        self.assertEqual(len(changed), 1)
        self.assertEqual(parent, [0.] * len(parent))
        problem.decode(trial)

    def test_displacement_limit_for_all_scenarios_and_boundaries(self):
        for case in ['easy', 'medium', 'hard']:
            problem = load_problem(case)
            rng = random.Random(8)
            lower, upper = problem.get_bounds()
            parents = initial_population(problem, 5, rng) + [lower, upper]
            for parent in parents:
                trial = mutate_satellite(problem, parent, rng, max_displacement=0.12)
                problem.decode(trial)
                delta = [b - a for a, b in zip(parent, trial)]
                for satellite in range(problem.n_sat):
                    state = [delta[j * problem.n_sat + satellite] for j in range(6)]
                    for epoch, matrix in enumerate(problem.stms):
                        scale = 1 if epoch == 0 else problem.inflation_factor
                        for row in matrix[:3]:
                            movement = abs(sum(a*b for a,b in zip(row, state))) / scale
                            self.assertLessEqual(movement, 0.12 + 1e-10)

    def test_invalid_step_size(self):
        problem = load_problem('easy')
        for step in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                mutate_satellite(problem, [0.] * 30, random.Random(2), step)


class HybridSearchTests(unittest.TestCase):
    def test_reproducibility_budget_and_parent_preservation(self):
        problem = load_problem('easy')
        population = initial_population(problem, 8, random.Random(4))
        original = [row[:] for row in population]
        with contextlib.redirect_stdout(io.StringIO()):
            first = evolve(problem, population, 5, random.Random(6))
            second = evolve(problem, population, 5, random.Random(6))
        self.assertEqual(first, second)
        self.assertEqual(population, original)
        x, score, history, evaluations = first
        self.assertEqual(evaluations, 48)
        self.assertEqual(len(history), 6)
        self.assertTrue(all(b <= a for a,b in zip(history, history[1:])))
        self.assertEqual(problem.fitness(x)[0], score)

    def test_zero_generations_and_invalid_arguments(self):
        problem = load_problem('easy')
        population = initial_population(problem, 5, random.Random(3))
        result = evolve(problem, population, 0, random.Random(7))
        self.assertEqual(result[3], 5)
        self.assertEqual(len(result[2]), 1)
        with self.assertRaises(ValueError):
            evolve(problem, population[:4], 1, random.Random(7))
        with self.assertRaises(ValueError):
            evolve(problem, population, -1, random.Random(7))


if __name__ == '__main__':
    unittest.main()
