"""Check Metropolis acceptance, archive safety, cooling and reproducibility."""
import math
import random
import unittest
from unittest.mock import patch

from mission import annealing
from mission.optimizer import initial_population
from mission.problems import load_problem


class ProbabilityTests(unittest.TestCase):
    def test_probability_rule_and_temperature_effect(self):
        self.assertEqual(annealing.acceptance_probability(-1, 0.1), 1)
        self.assertEqual(annealing.acceptance_probability(0, 0.1), 1)
        self.assertAlmostEqual(annealing.acceptance_probability(0.2, 0.1), math.exp(-2))
        self.assertGreater(annealing.acceptance_probability(0.2, 1),
                           annealing.acceptance_probability(0.2, 0.01))
        self.assertEqual(annealing.acceptance_probability(1000, 1e-10), 0.)

    def test_invalid_temperatures_and_delta(self):
        for value in [0, -1, True, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                annealing.acceptance_probability(1, value)
        for value in [True, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                annealing.acceptance_probability(value, 1)
        with self.assertRaises(ValueError):
            annealing.validate_temperatures(0.1, 1)


class FakeState:
    def __init__(self, problem, x):
        self.x, self.key = list(x), (x[0], 0.)

    def replace(self, x):
        return FakeState(None, x)


class ArchiveTests(unittest.TestCase):
    def test_accepted_uphill_move_keeps_archive_then_escapes(self):
        proposals = iter([[1.], [-1.]])
        rng = random.Random(1)
        with patch.object(annealing, 'FormationState', FakeState), patch.object(rng, 'random', return_value=0.):
            result = annealing.search(None, [[0.]], 2, rng,
                                      proposal=lambda *args: next(proposals))
        self.assertEqual(result['history'], [0., 0., -1.])
        self.assertEqual(result['annealing']['current_history'], [0., 1., -1.])
        self.assertEqual(result['annealing']['uphill_moves'], 1)
        self.assertEqual(result['x'], [-1.])
        self.assertEqual(result['evaluations'], 3)

    def test_rejected_uphill_move_does_not_change_current_state(self):
        rng = random.Random(1)
        with patch.object(annealing, 'FormationState', FakeState), patch.object(rng, 'random', return_value=0.99):
            result = annealing.search(None, [[0.]], 1, rng, proposal=lambda *args: [1.])
        self.assertEqual(result['annealing']['current_history'], [0., 0.])
        self.assertEqual(result['annealing']['accepted_moves'], 0)


class RealSearchTests(unittest.TestCase):
    def test_reproducibility_bounds_and_parent_preservation(self):
        problem = load_problem('easy')
        population = initial_population(problem, 5, random.Random(3))
        before = [x[:] for x in population]
        first = annealing.search(problem, population, 20, random.Random(7), record_every=5)
        second = annealing.search(problem, population, 20, random.Random(7), record_every=5)
        self.assertEqual(first, second)
        self.assertEqual(population, before)
        self.assertEqual(first['evaluations'], 25)
        self.assertEqual(len(first['history']), 5)
        self.assertTrue(all(b <= a for a,b in zip(first['history'], first['history'][1:])))
        self.assertEqual(problem.fitness(first['x'])[0], first['fitness'])
        temperatures = first['annealing']['temperatures']
        self.assertTrue(all(b <= a + 1e-15 for a,b in zip(temperatures, temperatures[1:])))
        self.assertAlmostEqual(temperatures[-1], 0.0001)

    def test_zero_iterations_and_partial_record_interval(self):
        problem = load_problem('easy')
        population = initial_population(problem, 5, random.Random(2))
        zero = annealing.search(problem, population, 0, random.Random(1))
        self.assertEqual(zero['evaluations'], 5)
        self.assertEqual(len(zero['history']), 1)
        partial = annealing.search(problem, population, 7, random.Random(1), record_every=5)
        self.assertEqual(len(partial['history']), 3)
        self.assertEqual(partial['evaluations'], 12)

    def test_invalid_search_parameters(self):
        for population, steps, interval in [([], 1, 1), ([[0.] * 30], -1, 1),
                                             ([[0.] * 30], True, 1), ([[0.] * 30], 1, 0)]:
            with self.assertRaises(ValueError):
                annealing.search(None, population, steps, random.Random(1), record_every=interval)


if __name__ == '__main__':
    unittest.main()
