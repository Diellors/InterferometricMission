"""Verify paired feasibility, exact hybrid budgets and archive preservation."""
import copy
import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest

from mission.hybrid import propose_pair, search
from mission.optimizer import initial_population, quality_key
from mission.problems import load_problem
from mission.search_state import FormationState
from scripts.compare_hybrid import main as compare
from scripts.verify_solution import load_record, verify_record


class PairedMoveTests(unittest.TestCase):
    def test_two_distinct_satellites_and_centroid_preservation(self):
        problem = load_problem('easy')
        x = [0.] * 30
        trial = propose_pair(problem, x, random.Random(4), 0.2)
        changed = {i % 5 for i,(a,b) in enumerate(zip(x,trial)) if a != b}
        self.assertEqual(len(changed), 2)
        for j in range(6):
            self.assertAlmostEqual(sum(trial[j*5:(j+1)*5]), 0.)
        self.assertEqual(x, [0.] * 30)

    def test_feasible_paired_moves_and_reference_scores(self):
        for case in ['easy', 'medium', 'hard']:
            problem = load_problem(case)
            rng = random.Random(6)
            x = initial_population(problem, 5, rng)[0]
            cached = FormationState(problem, x)
            for _ in range(20):
                trial = propose_pair(problem, x, rng, 0.3)
                old_key = cached.key
                updated = cached.replace(trial)
                self.assertEqual(updated.key, quality_key(problem, trial))
                self.assertEqual(cached.key, old_key)
                problem.decode(trial)
                self.assertTrue(all(o['remaining'] == problem.n_sat for o in problem.observations(trial)))
                for j in range(6):
                    start, stop = j*problem.n_sat, (j+1)*problem.n_sat
                    self.assertAlmostEqual(sum(x[start:stop]), sum(trial[start:stop]), delta=1e-10)
                x = trial
                cached = updated


class HybridSearchTests(unittest.TestCase):
    def test_budget_reproducibility_and_nonworsening(self):
        problem = load_problem('easy')
        x = initial_population(problem, 5, random.Random(9))[0]
        original = copy.deepcopy(x)
        a = search(problem, x, 106, random.Random(7), record_every=10)
        b = search(problem, x, 106, random.Random(7), record_every=10)
        self.assertEqual(a, b)
        self.assertEqual(x, original)
        self.assertEqual(a['evaluations'], 106)
        phases = a['hybrid']
        self.assertEqual(phases['annealing_steps'] + phases['local_steps'] + phases['setup_evaluations'], 106)
        self.assertLessEqual(a['fitness'], problem.fitness(x)[0])
        self.assertEqual(a['fitness'], problem.fitness(a['x'])[0])
        self.assertTrue(all(b <= a for a,b in zip(a['history'], a['history'][1:])))

    def test_empty_phases_and_invalid_budget(self):
        problem = load_problem('easy')
        x = [0.] * 30
        result = search(problem, x, 6, random.Random(1))
        self.assertEqual(result['evaluations'], 6)
        self.assertEqual(result['x'], x)
        for fraction in [0., 1.]:
            result = search(problem, x, 16, random.Random(1), annealing_fraction=fraction)
            self.assertEqual(result['evaluations'], 16)
        with self.assertRaises(ValueError):
            search(problem, x, 5, random.Random(1))
        with self.assertRaises(ValueError):
            search(problem, x, 16, random.Random(1), annealing_fraction=2)

    def test_comparison_cli_saves_both_methods_with_equal_budgets(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            start = root / 'start.json'
            start.write_text(json.dumps({'problem': 'tiny-observatory-in-dro',
                                         'x': [0.] * 30, 'fitness': -3 / 441}), encoding='utf-8')
            before = start.read_bytes()
            output = root / 'comparison'
            args = ['--start', str(start), '--seeds', '1', '--evaluations', '20', '--output', str(output)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(compare(args), output)
            records = [load_record(output / f'{method}-seed-1.json') for method in ['annealing', 'hybrid']]
            self.assertTrue(all(record['evaluations'] == 20 for record in records))
            self.assertTrue(all(verify_record(record)['valid'] for record in records))
            self.assertEqual(records[0]['initial_best'], records[1]['initial_best'])
            self.assertEqual(start.read_bytes(), before)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                compare(args)


if __name__ == '__main__':
    unittest.main()
