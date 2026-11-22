"""Check grid transitions, reverse-memory/aspiration and archived fitness."""
import contextlib
import copy
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from mission import tabu
from mission.optimizer import initial_population, quality_key
from mission.problems import load_problem
from mission.search_state import FormationState
from scripts.run_tabu import main
from scripts.verify_solution import load_record, verify_record


class GridNeighborTests(unittest.TestCase):
    def test_neighbors_really_change_cells_and_keep_other_satellites(self):
        for case in ['easy', 'medium', 'hard']:
            problem = load_problem(case)
            rng = random.Random(12)
            x = initial_population(problem, 5, rng)[0]
            cache = FormationState(problem, x)
            successful = 0
            for step in range(30):
                satellite = step % problem.n_sat
                result = tabu.grid_neighbor(problem, x, satellite, rng, interval_limit=6 if step % 2 else 24)
                if result is None:
                    continue
                trial, before, after = result
                successful += 1
                self.assertNotEqual(before, after)
                problem.decode(trial)
                changed = {i % problem.n_sat for i,(a,b) in enumerate(zip(x,trial)) if a != b}
                self.assertEqual(changed, {satellite})
                next_cache = cache.replace(trial)
                self.assertEqual(next_cache.key, quality_key(problem, trial))
                self.assertTrue(all(o['remaining'] == problem.n_sat for o in problem.observations(trial)))
                x, cache = trial, next_cache
            self.assertGreater(successful, 0)

    def test_aspiration_override_rule(self):
        self.assertFalse(tabu.allowed(True, (0, 0), (-1, 0)))
        self.assertFalse(tabu.allowed(True, (-1, 0), (-1, 0)))
        self.assertTrue(tabu.allowed(True, (-2, 0), (-1, 0)))
        self.assertTrue(tabu.allowed(False, (2, 0), (-1, 0)))

    def test_axis_neighbor_changes_exactly_the_requested_cell(self):
        problem = load_problem('hard')
        rng = random.Random(9)
        x = initial_population(problem, 5, rng)[0]
        successful = 0
        for satellite in range(6):
            for axis in range(3):
                result = tabu.adjacent_neighbor(problem, x, satellite, 1, axis, 1)
                if result is None:
                    continue
                trial, before, after = result
                changed = [(epoch, component) for epoch in range(3) for component in range(3)
                           if before[epoch][component] != after[epoch][component]]
                self.assertEqual(changed, [(1, axis)])
                self.assertEqual(after[1][axis], before[1][axis] + 1)
                self.assertEqual(FormationState(problem, x).replace(trial).key, quality_key(problem, trial))
                successful += 1
        self.assertGreater(successful, 0)

    def test_relaxed_axis_proposal_still_matches_target_epoch(self):
        problem = load_problem('hard')
        x = initial_population(problem, 5, random.Random(19))[0]
        successful = 0
        for satellite in range(5):
            result = tabu.adjacent_neighbor(problem, x, satellite, 1, 0, 1, other_epoch_slack=1)
            if result is None:
                continue
            trial, before, after = result
            self.assertEqual(after[1], (before[1][0]+1, before[1][1], before[1][2]))
            self.assertTrue(all(abs(after[t][j]-before[t][j]) <= 1 for t in [0, 2] for j in range(3)))
            self.assertEqual(FormationState(problem, x).replace(trial).key, quality_key(problem, trial))
            successful += 1
        self.assertGreater(successful, 0)

    def test_coordinated_offsets_are_checked(self):
        problem = load_problem('hard')
        x = initial_population(problem, 5, random.Random(19))[0]
        successful = 0
        for satellite in range(5):
            result = tabu.adjacent_neighbor(problem, x, satellite, 1, 0, 1,
                                             other_epoch_slack=2, target_offsets=[1, 0, 1])
            if result is None:
                continue
            trial, before, after = result
            self.assertEqual(after[1], (before[1][0]+1, before[1][1], before[1][2]+1))
            self.assertEqual(FormationState(problem, x).replace(trial).key, quality_key(problem, trial))
            successful += 1
        self.assertGreater(successful, 0)


class FakeState:
    def __init__(self, problem, x):
        self.x, self.key = list(x), (x[0], 0)

    def replace(self, x):
        return FakeState(None, x)


class ToyProblem:
    n_sat = 1

    def fitness(self, x):
        return [x[0]]


class TabuSearchTests(unittest.TestCase):
    def test_worse_current_reversal_block_and_aspiration_keep_archive(self):
        moves = [([1.], 'A', 'B'), ([0.], 'B', 'A'), ([-1.], 'B', 'A')]
        with patch.object(tabu, 'FormationState', FakeState), \
                patch.object(tabu, 'grid_neighbor', side_effect=moves):
            result = tabu.search(ToyProblem(), [0.], 4, random.Random(1), neighborhood_size=1)
        self.assertEqual(result['history'], [0., 0., 0., -1.])
        self.assertEqual(result['tabu']['current_history'], [0., 1., 1., -1.])
        self.assertEqual(result['tabu']['tabu_rejections'], 1)
        self.assertEqual(result['tabu']['aspirations'], 1)
        self.assertEqual(result['tabu']['uphill_moves'], 1)
        self.assertEqual(result['evaluations'], 4)

    def test_real_reproducibility_bounds_and_input_preservation(self):
        problem = load_problem('easy')
        x = initial_population(problem, 5, random.Random(3))[0]
        original = copy.deepcopy(x)
        a = tabu.search(problem, x, 41, random.Random(7), neighborhood_size=8)
        b = tabu.search(problem, x, 41, random.Random(7), neighborhood_size=8)
        self.assertEqual(a, b)
        self.assertEqual(x, original)
        self.assertEqual(a['fitness'], problem.fitness(a['x'])[0])
        self.assertLessEqual(a['fitness'], problem.fitness(x)[0])
        self.assertTrue(all(b <= a for a,b in zip(a['history'], a['history'][1:])))
        self.assertLessEqual(a['evaluations'], 41)
        self.assertEqual(a['tabu']['checkpoint_evaluations'][-1], a['evaluations'])

    def test_already_reached_target_and_no_generated_neighbor(self):
        problem = load_problem('easy')
        x = [0.] * 30
        result = tabu.search(problem, x, 20, random.Random(1), target=0.)
        self.assertEqual(result['evaluations'], 1)
        self.assertEqual(result['tabu']['stop_reason'], 'target_reached')
        with patch.object(tabu, 'grid_neighbor', return_value=None):
            result = tabu.search(problem, x, 20, random.Random(1), neighborhood_size=2)
        self.assertEqual(result['evaluations'], 1)
        self.assertEqual(result['tabu']['stop_reason'], 'no_grid_neighbors')

    def test_invalid_parameters(self):
        for kwargs in [{'max_evaluations': 0}, {'max_evaluations': True}, {'neighborhood_size': 0},
                       {'tenure': 0}, {'interval_limit': 0}, {'radius': float('nan')}, {'target': float('inf')}]:
            arguments = {'max_evaluations': 10}
            arguments.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                tabu.search(None, [], rng=random.Random(1), **arguments)

    def test_cli_record_verifies_and_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            start = root / 'start.json'
            start.write_text(json.dumps({'problem': 'tiny-observatory-in-dro',
                                         'x': [0.] * 30, 'fitness': -3/441}), encoding='utf-8')
            output = root / 'result.json'
            args = ['--start', str(start), '--target', '-0.1', '--evaluations', '10', '--output', str(output)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), output)
            self.assertTrue(verify_record(load_record(output))['valid'])
            original = output.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            self.assertEqual(output.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
