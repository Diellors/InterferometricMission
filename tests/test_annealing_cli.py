"""Check SA commands, experiment budgets, metadata and independent validation."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

from mission import optimizer
from mission.problems import load_problem
from scripts import run_experiments as experiments
from scripts.verify_solution import verify_record


class AnnealingIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def run_main(self, *extra):
        arguments = ['main.py', '--case', 'easy', '--population', '5', '--generations', '2', '--seed', '1', *extra]
        with patch.object(optimizer, 'ROOT', self.root), patch.object(sys, 'argv', arguments), \
                contextlib.redirect_stdout(io.StringIO()):
            optimizer.main()

    def test_default_sa_save_path_does_not_overwrite_orbit_results(self):
        old = self.root / 'results/easy/seed-1/easy.json'
        old.parent.mkdir(parents=True)
        old.write_text('keep existing record', encoding='utf-8')
        self.run_main()
        path = self.root / 'results/annealing/easy/seed-1/easy.json'
        record = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(record['algorithm'], 'simulated-annealing-v1')
        self.assertEqual(record['evaluations'], 15)
        self.assertEqual(len(record['history']), 3)
        self.assertTrue(verify_record(record)['valid'])
        self.assertEqual(old.read_text(encoding='utf-8'), 'keep existing record')

    def test_cli_overrides_config_and_orbit_remains_available(self):
        config = self.root / 'config.json'
        config.write_text(json.dumps({'algorithm': 'orbit', 'initial_temperature': 0.2,
                                      'final_temperature': 0.001}), encoding='utf-8')
        self.run_main('--config', str(config), '--algorithm', 'annealing', '--initial-temperature', '0.1')
        record = json.loads((self.root / 'results/annealing/easy/seed-1/easy.json').read_text())
        self.assertEqual(record['annealing']['initial_temperature'], 0.1)
        self.assertEqual(record['annealing']['final_temperature'], 0.001)
        self.run_main('--algorithm', 'orbit')
        legacy = json.loads((self.root / 'results/easy/seed-1/easy.json').read_text())
        self.assertEqual(legacy['algorithm'], 'feasible-satellite-search-v2')
        self.assertTrue(verify_record(legacy)['valid'])

    def test_invalid_temperature_rejected_before_search(self):
        with patch.object(optimizer, 'solve') as solve, contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit):
            self.run_main('--initial-temperature', '0')
        solve.assert_not_called()

    def test_both_methods_have_equal_budget_and_same_initial_best(self):
        problem = load_problem('easy')
        population = optimizer.initial_population(problem, 5, random.Random(3))
        with contextlib.redirect_stdout(io.StringIO()):
            sa = optimizer.run_search(problem, population, 3, random.Random(4), algorithm='annealing')
            orbit = optimizer.run_search(problem, population, 3, random.Random(4), algorithm='orbit')
        self.assertEqual(sa['initial_best'], orbit['initial_best'])
        self.assertEqual(sa['evaluations'], 20)
        self.assertEqual(sa['evaluations'], orbit['evaluations'])

    def test_experiments_save_algorithm_and_valid_metadata(self):
        with contextlib.redirect_stdout(io.StringIO()):
            rows = experiments.run_experiments(['easy'], [1], 5, 1, self.root / 'batch', algorithm='annealing')
        self.assertEqual(rows[0]['algorithm'], 'simulated-annealing-v1')
        record = json.loads((self.root / 'batch' / rows[0]['result_file']).read_text())
        self.assertTrue(verify_record(record)['valid'])
        with (self.root / 'batch/summary.csv').open(newline='', encoding='utf-8') as stream:
            self.assertEqual(list(csv.DictReader(stream))[0]['algorithm'], 'simulated-annealing-v1')

    def test_summaries_do_not_mix_algorithms(self):
        rows = [{'case': 'easy', 'algorithm': method, 'seed': 1, 'fitness': score,
                 'search_seconds': 1., 'evaluations': 10, 'result_file': method}
                for method, score in [('simulated-annealing-v1', -0.1), ('feasible-satellite-search-v2', -0.12)]]
        self.assertEqual(len(experiments.summarize(rows)), 2)

    def test_validator_rejects_inconsistent_sa_metadata(self):
        self.run_main()
        record = json.loads((self.root / 'results/annealing/easy/seed-1/easy.json').read_text())
        variants = [('iterations', 9), ('record_every', 2), ('accepted_moves', 99),
                    ('uphill_moves', 99), ('initial_temperature', 0), ('temperatures', [0.1]),
                    ('current_history', [0.])]
        for field, value in variants:
            broken = copy.deepcopy(record)
            broken['annealing'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_record(broken)


if __name__ == '__main__':
    unittest.main()
