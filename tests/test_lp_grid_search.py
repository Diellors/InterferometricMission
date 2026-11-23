import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest

from mission.alns import remove
from mission.lp_grid_search import CellFeasibilityLP, plane_tables, search
from mission.problems import MissionProblem, load_problem
from mission.sampling import sample_state
from mission.search_state import FormationState
from scripts.verify_solution import load_record, verify_record

try:
    from ortools.linear_solver import pywraplp
except ImportError:
    pywraplp = None


@unittest.skipIf(pywraplp is None, 'Optional OR-Tools LP solver is not installed')
class LpGridTests(unittest.TestCase):
    def test_cell_solver_recovers_exact_cells_and_feasible_states(self):
        for case in ['easy', 'hard']:
            problem = load_problem(case)
            rng = random.Random(91)
            state = sample_state(problem, rng)
            cache = FormationState(problem, [state[j] for j in range(6) for _ in range(problem.n_sat)])
            for epoch in range(3):
                model = CellFeasibilityLP(problem, epoch)
                cell = cache._cell(state, epoch)
                moved = model.solve(cell, reference=state)
                self.assertIsNotNone(moved)
                self.assertEqual(cache._cell(moved, epoch), cell)
                self.assertTrue(all(cache._cell(moved, t) is not None for t in range(3)))

    def test_tables_match_independent_full_baseline_sets(self):
        problem = load_problem('easy')
        rng = random.Random(92)
        states = [sample_state(problem, rng) for _ in range(problem.n_sat)]
        x = [states[i][j] for j in range(6) for i in range(problem.n_sat)]
        cache = FormationState(problem, x)
        partial = remove(cache, [2])
        moved = sample_state(problem, rng)
        trial = x[:]
        for j in range(6):
            trial[j*problem.n_sat+2] = moved[j]
        full = problem.observations(trial)
        for epoch in range(3):
            tables = plane_tables(partial, epoch)
            cell = cache._cell(moved, epoch)
            for plane, (a, b) in enumerate([(0, 1), (0, 2), (1, 2)]):
                self.assertEqual(tables[plane][cell[a], cell[b]], len(full[epoch]['planes'][plane]['baselines']))
        single = MissionProblem('easy')
        single.n_sat = 1
        partial = remove(FormationState(single, [0.]*6), [0])
        self.assertTrue(all(value == 1 for table in plane_tables(partial, 0) for value in table.values()))

    def test_tiny_budget_preserves_archive(self):
        problem = load_problem('easy')
        x = [0.]*30
        result = search(problem, x, max_lp_calls=10)
        self.assertEqual(result['x'], x)
        self.assertEqual(result['fitness'], problem.fitness(x)[0])
        self.assertEqual(result['lp_grid_search']['lp_calls'], 10)
        self.assertEqual(result['lp_grid_search']['stop_reason'], 'lp_limit')

    def test_hard_regression_finds_a_real_improvement_from_saved_incumbent(self):
        problem = load_problem('hard')
        path = Path(__file__).resolve().parents[1] / 'solutions/starts/hard_1_522308.json'
        source = load_record(path)
        result = search(problem, source['x'])
        self.assertLess(result['fitness'], source['fitness']-0.01)
        self.assertEqual(result['fitness'], problem.fitness(result['x'])[0])
        self.assertEqual(len(result['lp_grid_search']['changes']), 1)
        self.assertEqual(result['lp_grid_search']['stop_reason'], 'improvement_limit')

    def test_validation_and_existing_output_protection(self):
        from scripts.run_lp_grid_search import main
        problem = load_problem('easy')
        with self.assertRaises(ValueError):
            CellFeasibilityLP(problem, 3)
        with self.assertRaises(ValueError):
            CellFeasibilityLP(problem, 0, epsilon=0.)
        model = CellFeasibilityLP(problem, 0)
        with self.assertRaises(ValueError):
            model.solve((0, 0, 99))
        with self.assertRaises(ValueError):
            search(problem, [0.]*30, max_improvements=0)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source.json'
            source.write_text(json.dumps({'problem': 'tiny-observatory-in-dro', 'x': [0.]*30,
                                         'fitness': -3/441}), encoding='utf-8')
            output = Path(folder) / 'run'
            args = ['--start', str(source), '--lp-limit', '10', '--output', str(output)]
            with contextlib.redirect_stdout(io.StringIO()):
                main(args)
            path = output / 'result.json'
            self.assertTrue(verify_record(load_record(path))['valid'])
            original = path.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
