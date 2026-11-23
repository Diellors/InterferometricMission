import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest

from mission.alns import block_state, insert, remove, search, supports_block_recombination
from mission.optimizer import initial_population
from mission.problems import load_problem
from mission.search_state import FormationState
from scripts.run_alns import main
from scripts.verify_solution import load_record, verify_record


class AlnsTests(unittest.TestCase):
    def test_medium_block_recombination_preserves_orbital_feasibility(self):
        problem = load_problem('medium')
        rng = random.Random(42)
        x = initial_population(problem, 5, rng)[0]
        states = problem.decode(x)
        original = FormationState(problem, x)
        self.assertTrue(supports_block_recombination(problem))
        for _ in range(20):
            state = block_state(problem, states, rng)
            for epoch in range(3):
                self.assertIsNotNone(original._cell(state, epoch))
            trial = x[:]
            for j in range(6):
                trial[j*problem.n_sat] = state[j]
            problem.decode(trial)
            self.assertEqual(original.replace(trial).key, FormationState(problem, trial).key)

    def test_block_recombination_is_rejected_for_coupled_hard_model(self):
        problem = load_problem('hard')
        self.assertFalse(supports_block_recombination(problem))
        with self.assertRaises(ValueError):
            search(problem, [0.]*240, 10, random.Random(1), block_probability=0.5)

    def test_medium_search_with_block_proposals_matches_full_score(self):
        problem = load_problem('medium')
        x = initial_population(problem, 5, random.Random(18))[0]
        result = search(problem, x, 81, random.Random(19), min_destroy=2, max_destroy=3,
                        pool_size=4, block_probability=0.5, target=-2.)
        self.assertEqual(result['fitness'], problem.fitness(result['x'])[0])
        self.assertLessEqual(result['fitness'], problem.fitness(x)[0])
        self.assertEqual(result['alns']['block_probability'], 0.5)
        self.assertLessEqual(result['evaluations'], 81)

    def test_partial_baselines_and_restore_match_full_evaluator(self):
        for case in ['easy', 'hard']:
            problem = load_problem(case)
            x = initial_population(problem, 5, random.Random(32))[0]
            original = FormationState(problem, x)
            group = [0, 1, 3]
            partial = remove(original, group)
            for epoch in range(3):
                points = [p for i, p in enumerate(original.cells[epoch]) if i not in group and p is not None]
                for plane, (a, b) in enumerate([(0, 1), (0, 2), (1, 2)]):
                    expected = {(p[a]-q[a], p[b]-q[b]) for p in points for q in points}
                    self.assertEqual(set(partial.counts[epoch][plane]), expected)
            for satellite in reversed(group):
                partial = insert(partial, satellite, original.states[satellite])
            self.assertEqual(partial.counts, original.counts)
            self.assertEqual(partial.key, original.key)
            self.assertEqual(partial.x, x)
            self.assertEqual(original.key[0], problem.fitness(x)[0])

    def test_budget_reproducibility_archive_and_adaptation(self):
        problem = load_problem('easy')
        x = [0.]*30
        before = x[:]
        kwargs = dict(pool_size=3, target=-0.15, segment_length=2, restart_every=3)
        a = search(problem, x, 301, random.Random(4), **kwargs)
        b = search(problem, x, 301, random.Random(4), **kwargs)
        self.assertEqual(a, b)
        self.assertEqual(x, before)
        self.assertLessEqual(a['evaluations'], 301)
        self.assertEqual(a['alns']['partial_scores']+1, a['evaluations'])
        self.assertEqual(a['fitness'], problem.fitness(a['x'])[0])
        self.assertLessEqual(a['fitness'], problem.fitness(x)[0])
        self.assertTrue(all(v <= u for u, v in zip(a['history'], a['history'][1:])))
        self.assertEqual(sum(a['alns']['operator_uses']), a['alns']['iterations'])
        self.assertGreater(a['alns']['restarts'], 0)
        self.assertNotEqual(a['alns']['weights'], [1.]*6)

    def test_empty_partial_and_changed_insertions(self):
        problem = load_problem('easy')
        rng = random.Random(2)
        x, y = initial_population(problem, 5, rng)[:2]
        state = FormationState(problem, x)
        partial = remove(state, list(range(problem.n_sat)))
        self.assertEqual(partial.key, (0., 0.))
        target = problem.decode(y)
        for i in range(problem.n_sat):
            partial = insert(partial, i, target[i])
        self.assertEqual(partial.counts, FormationState(problem, y).counts)
        self.assertEqual(partial.key[0], problem.fitness(y)[0])

    def test_target_and_validation(self):
        problem = load_problem('easy')
        result = search(problem, [0.]*30, 100, random.Random(1), target=0.)
        self.assertEqual(result['evaluations'], 1)
        for kwargs in [{'pool_size': 1}, {'max_evaluations': True}, {'target': float('nan')},
                       {'restart_every': 0}, {'segment_length': 0}, {'min_destroy': 9, 'max_destroy': 4},
                       {'block_probability': -0.1}, {'block_probability': float('nan')}]:
            args = dict(max_evaluations=10)
            args.update(kwargs)
            with self.assertRaises(ValueError):
                search(problem, [0.]*30, rng=random.Random(1), **args)

    def test_cli_verifies_result_and_refuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.json'
            source.write_text(json.dumps({'problem': 'tiny-observatory-in-dro',
                                         'x': [0.]*30, 'fitness': -3/441}), encoding='utf-8')
            output = root / 'run'
            args = ['--start', str(source), '--output', str(output), '--target', '-0.1',
                    '--evaluations', '31', '--pool-size', '3']
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), output)
            path = output / 'result.json'
            self.assertTrue(verify_record(load_record(path))['valid'])
            original = path.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
