"""Test warm-start target runs, restart budgets and checkpoint preservation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import seek_targets as runner
from scripts.verify_solution import load_record, verify_record


class TargetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.start = self.root / 'start.json'
        self.record = {'problem': 'tiny-observatory-in-dro', 'x': [0.] * 30, 'fitness': -3 / 441}
        self.start.write_text(json.dumps(self.record), encoding='utf-8')

    def run_search(self, name='out', **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.run_target_search('easy', self.start, -1 / 7, self.root / name,
                                             attempts=2, steps=5, **kwargs)

    def test_reproducible_results_budget_and_verification(self):
        first = self.run_search('first', seed=7)
        second = self.run_search('second', seed=7)
        a, b = load_record(self.root / 'first/best.json'), load_record(self.root / 'second/best.json')
        self.assertEqual(a['x'], b['x'])
        self.assertEqual(a['history'], b['history'])
        self.assertLessEqual(a['fitness'], self.record['fitness'])
        self.assertEqual(first['stage_evaluations'], first['attempts_completed'] * 10)
        self.assertEqual(first['best_fitness'], second['best_fitness'])
        self.assertTrue(verify_record(a)['valid'])

    def test_already_reached_target_does_not_run_search(self):
        with patch.object(runner, 'search') as search:
            summary = runner.run_target_search('easy', self.start, 0., self.root / 'done', steps=5)
        search.assert_not_called()
        self.assertTrue(summary['target_reached'])
        self.assertEqual(summary['attempts_completed'], 0)
        self.assertEqual(load_record(self.root / 'done/best.json')['x'], self.record['x'])

    def test_zero_attempts_preserves_input_and_saves_summary(self):
        original = self.start.read_bytes()
        summary = runner.run_target_search('easy', self.start, -1 / 7, self.root / 'zero', attempts=0, steps=5)
        self.assertFalse(summary['target_reached'])
        self.assertEqual(summary['stage_evaluations'], 0)
        self.assertEqual(self.start.read_bytes(), original)
        self.assertTrue((self.root / 'zero/summary.json').exists())

    def test_failed_attempt_keeps_last_checkpoint(self):
        original = runner.search
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('Interrupted attempt')
            return original(*args, **kwargs)

        with patch.object(runner, 'search', side_effect=fail_second), self.assertRaises(RuntimeError):
            self.run_search('partial', seed=7)
        summary = json.loads((self.root / 'partial/summary.json').read_text())
        self.assertEqual(summary['attempts_completed'], 1)
        self.assertTrue(verify_record(load_record(self.root / 'partial/best.json'))['valid'])

    def test_invalid_inputs_do_not_create_output(self):
        for index, kwargs in enumerate([{'steps': 7}, {'steps': 0}, {'attempts': -1},
                                        {'target': 1}, {'target': -1.5}, {'target': float('nan')},
                                        {'initial_temperature': 0}, {'seed': True}]):
            args = {'steps': 5, 'target': -1 / 7}
            args.update(kwargs)
            target = args.pop('target')
            output = self.root / f'bad-{index}'
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                runner.run_target_search('easy', self.start, target, output, **args)
            self.assertFalse(output.exists())

    def test_case_mismatch_and_existing_output_are_rejected(self):
        with self.assertRaises(ValueError):
            runner.run_target_search('hard', self.start, -1.5, self.root / 'wrong', steps=5)
        existing = self.root / 'existing'
        existing.mkdir()
        with self.assertRaises(FileExistsError):
            runner.run_target_search('easy', self.start, -1 / 7, existing, steps=5)

    def test_cli_writes_a_new_output(self):
        with contextlib.redirect_stdout(io.StringIO()):
            summary = runner.main(['--case', 'easy', '--start', str(self.start), '--target', '0',
                                   '--steps', '5', '--output', str(self.root / 'cli')])
        self.assertTrue(summary['target_reached'])
        self.assertTrue((self.root / 'cli/best.json').exists())


if __name__ == '__main__':
    unittest.main()
