"""Correctness checks for greedy construction and finite-pool local search."""
import contextlib
import copy
import io
import json
from pathlib import Path
import random
import tempfile
import unittest

from mission.greedy import addition_key, candidate_cells, chromosome, make_pool, search
from mission.optimizer import quality_key
from mission.problems import MissionProblem, load_problem
from scripts.run_greedy import main
from scripts.verify_solution import verify_record


def small_problem(count=3):
    problem = MissionProblem.__new__(MissionProblem)
    problem.case, problem.n_sat, problem.grid_size = 'easy', count, 7
    problem.n_meas, problem.inflation_factor = 3, 1.
    identity = [[float(i == j) for j in range(6)] for i in range(6)]
    problem.stms = [identity for _ in range(3)]
    problem.times = [0., 1., 2.]
    return problem


class GreedyTests(unittest.TestCase):
    def setUp(self):
        self.problem = small_problem()
        self.pool = [[-0.8, -0.5, 0.4, 0, 0, 0], [0.6, -0.2, -0.5, 0, 0, 0],
                     [-0.3, 0.7, -0.1, 0, 0, 0], [0.1, 0.1, 0.1, 0, 0, 0],
                     [0.8, 0.5, 0.7, 0, 0, 0], [-0.6, 0.2, -0.8, 0, 0, 0]]

    def test_addition_counts_match_full_partial_formation(self):
        occupied = [[] for _ in range(3)]
        supports = [[set() for _ in range(3)] for _ in range(3)]
        for count, state in enumerate(self.pool[:3], 1):
            cells = candidate_cells(self.problem, state)
            key, additions = addition_key(self.problem, occupied, supports, cells)
            partial_problem = small_problem(count)
            x = chromosome(self.pool, list(range(count)))
            self.assertEqual(key, quality_key(partial_problem, x))
            for epoch in range(3):
                occupied[epoch].append(cells[epoch])
                for plane in range(3):
                    supports[epoch][plane].update(additions[epoch][plane])

    def test_construction_is_reproducible_unique_and_does_not_mutate_pool(self):
        original = copy.deepcopy(self.pool)
        first = search(self.problem, self.pool, max_evaluations=15, max_sweeps=0)
        second = search(self.problem, self.pool, max_evaluations=15, max_sweeps=0)
        self.assertEqual(first, second)
        self.assertEqual(self.pool, original)
        self.assertEqual(len(set(first['selected_indices'])), 3)
        self.assertEqual(first['construction_evaluations'], 15)
        self.assertEqual(first['refinement_evaluations'], 0)
        self.assertEqual(self.problem.fitness(first['x'])[0], first['fitness'])
        trace = [item['fitness'] for item in first['construction_trace']]
        self.assertTrue(all(b <= a for a,b in zip(trace, trace[1:])))

    def test_reported_local_optimum_has_no_improving_pool_replacement(self):
        result = search(self.problem, self.pool, max_evaluations=500, max_sweeps=100)
        self.assertEqual(result['stop_reason'], 'pool_local_optimum')
        selected = result['selected_indices']
        key = quality_key(self.problem, result['x'])
        for position in range(3):
            for index in range(len(self.pool)):
                if index in selected:
                    continue
                trial = selected[:]
                trial[position] = index
                self.assertGreaterEqual(quality_key(self.problem, chromosome(self.pool, trial)), key)
        self.assertLessEqual(result['fitness'], result['initial_best'])

    def test_refinement_stops_at_exact_score_budget(self):
        result = search(self.problem, [[0.] * 6 for _ in range(6)], max_evaluations=16, max_sweeps=10)
        self.assertEqual(result['evaluations'], 16)
        self.assertEqual(result['refinement_evaluations'], 1)
        self.assertEqual(result['stop_reason'], 'evaluation_limit')

    def test_ties_keep_first_available_pool_index(self):
        result = search(self.problem, [[0.] * 6 for _ in range(6)])
        self.assertEqual(result['selected_indices'], [0, 1, 2])
        self.assertEqual(result['accepted_replacements'], 0)

    def test_invalid_pool_and_budgets(self):
        for pool in [self.pool[:2], [[2.] * 6] * 6, [[True] * 6] * 6,
                     [[float('nan')] * 6] * 6, [[0.] * 5] * 6]:
            with self.subTest(pool=pool), self.assertRaises(ValueError):
                search(self.problem, pool)
        with self.assertRaises(ValueError):
            search(self.problem, self.pool, max_evaluations=14)
        with self.assertRaises(ValueError):
            search(self.problem, self.pool, max_sweeps=-1)
        with self.assertRaises(ValueError):
            make_pool(self.problem, 2, random.Random(1))

    def test_real_easy_orbit_result_matches_unchanged_objective(self):
        problem = load_problem('easy')
        pool = make_pool(problem, 20, random.Random(2))
        result = search(problem, pool, max_evaluations=200, max_sweeps=2)
        self.assertEqual(problem.fitness(result['x'])[0], result['fitness'])
        self.assertTrue(all(item['remaining'] == 5 for item in problem.observations(result['x'])))


class GreedyCliTests(unittest.TestCase):
    def test_saved_record_verifies_and_existing_file_is_protected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'easy.json'
            args = ['--case', 'easy', '--pool-size', '10', '--max-evaluations', '100',
                    '--max-sweeps', '1', '--output', str(path)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), path)
            record = json.loads(path.read_text(encoding='utf-8'))
            self.assertTrue(verify_record(record)['valid'])
            self.assertEqual(record['algorithm'], 'greedy-baseline-local-search-v1')
            self.assertEqual(record['evaluations'], record['construction_evaluations'] +
                             record['refinement_evaluations'])
            original = path.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
