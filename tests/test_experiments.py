"""Check batch output, seed reproducibility, summaries and failure recovery."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_experiments as runner
from mission.problems import load_problem


class SummaryTests(unittest.TestCase):
    def test_minimized_fitness_grouping_and_statistics(self):
        rows = [
            {'case': 'easy', 'seed': 1, 'fitness': -0.1, 'search_seconds': 2.,
             'evaluations': 10, 'result_file': 'easy/one.json'},
            {'case': 'easy', 'seed': 2, 'fitness': -0.3, 'search_seconds': 4.,
             'evaluations': 10, 'result_file': 'easy/two.json'},
            {'case': 'hard', 'seed': 1, 'fitness': -0.05, 'search_seconds': 8.,
             'evaluations': 20, 'result_file': 'hard/one.json'},
        ]
        easy, hard = runner.summarize(rows)
        self.assertEqual(easy['best_seed'], 2)
        self.assertEqual(easy['best_result_file'], 'easy/two.json')
        self.assertAlmostEqual(easy['mean_fitness'], -0.2)
        self.assertAlmostEqual(easy['median_fitness'], -0.2)
        self.assertAlmostEqual(easy['stdev_fitness'], 0.02 ** 0.5)
        self.assertEqual(easy['mean_search_seconds'], 3.)
        self.assertEqual(easy['total_evaluations'], 20)
        self.assertEqual(hard['runs'], 1)
        self.assertEqual(hard['stdev_fitness'], 0.)
        self.assertEqual(runner.summarize([]), [])

    def test_tied_fitness_keeps_first_seed(self):
        rows = [{'case': 'easy', 'seed': seed, 'fitness': -0.2, 'search_seconds': 1.,
                 'evaluations': 5, 'result_file': str(seed)} for seed in [8, 3]]
        self.assertEqual(runner.summarize(rows)[0]['best_seed'], 8)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_batch(self, output, seeds=(1, 2), generations=1):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.run_experiments(['easy'], list(seeds), 5, generations, output)

    def test_real_batch_csv_json_and_reproducibility(self):
        first = self.run_batch(self.root / 'first')
        second = self.run_batch(self.root / 'second')
        self.assertEqual(len(first), 2)
        with (self.root / 'first' / 'runs.csv').open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        for a, b, csv_row in zip(first, second, rows):
            self.assertEqual(a['fitness'], b['fitness'])
            self.assertEqual(a['evaluations'], 10)
            self.assertGreaterEqual(a['search_seconds'], 0)
            path = self.root / 'first' / a['result_file']
            record = json.loads(path.read_text(encoding='utf-8'))
            other = json.loads((self.root / 'second' / b['result_file']).read_text(encoding='utf-8'))
            self.assertEqual(record['x'], other['x'])
            self.assertEqual(record['history'], other['history'])
            self.assertEqual(len(record['history']), 2)
            self.assertEqual(record['fitness'], load_problem('easy').fitness(record['x'])[0])
            self.assertEqual(float(csv_row['fitness']), record['fitness'])
            self.assertEqual(csv_row['result_file'], a['result_file'])
        with (self.root / 'first' / 'summary.csv').open(newline='', encoding='utf-8') as stream:
            summary = list(csv.DictReader(stream))
        self.assertEqual(len(summary), 1)
        best = min(first, key=lambda row: row['fitness'])
        self.assertEqual(int(summary[0]['best_seed']), best['seed'])

    def test_zero_generations(self):
        row = self.run_batch(self.root / 'zero', seeds=[0], generations=0)[0]
        self.assertEqual(row['evaluations'], 5)
        self.assertEqual(row['improvement'], 0)

    def test_existing_batch_is_not_overwritten(self):
        folder = self.root / 'existing'
        folder.mkdir()
        sentinel = folder / 'keep.txt'
        sentinel.write_text('keep', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            self.run_batch(folder)
        self.assertEqual(sentinel.read_text(encoding='utf-8'), 'keep')
        self.assertEqual(list(folder.iterdir()), [sentinel])

    def test_invalid_arguments_do_not_create_output(self):
        cases = [([], [1], 5, 1), (['unknown'], [1], 5, 1),
                 (['easy'], [], 5, 1), (['easy'], [True], 5, 1),
                 (['easy'], [1, 1], 5, 1), (['easy', 'easy'], [1], 5, 1),
                 (['easy'], [1], 4, 1), (['easy'], [1], 5, -1)]
        for index, arguments in enumerate(cases):
            output = self.root / str(index)
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                runner.run_experiments(*arguments, output)
            self.assertFalse(output.exists())

    def test_completed_run_survives_later_failure(self):
        original = runner.run_search
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('Interrupted second run')
            return original(*args, **kwargs)

        folder = self.root / 'partial'
        with patch.object(runner, 'run_search', side_effect=fail_second), self.assertRaises(RuntimeError):
            self.run_batch(folder)
        self.assertTrue((folder / 'easy' / 'seed-1' / 'easy.json').is_file())
        with (folder / 'runs.csv').open(newline='', encoding='utf-8') as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 1)
        with (folder / 'summary.csv').open(newline='', encoding='utf-8') as stream:
            self.assertEqual(list(csv.DictReader(stream))[0]['runs'], '1')

    def test_cli_uses_requested_output(self):
        folder = self.root / 'cli'
        with contextlib.redirect_stdout(io.StringIO()):
            actual = runner.main(['--cases', 'easy', '--seeds', '9', '--population', '5',
                                  '--generations', '0', '--output', str(folder)])
        self.assertEqual(actual, folder)
        self.assertTrue((folder / 'summary.csv').is_file())


if __name__ == '__main__':
    unittest.main()
