"""Check result schemas, numerical re-evaluation and verification errors."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import verify_solution as verifier


def valid_record(case='easy'):
    count = 5 if case == 'easy' else 40
    size = 11 if case == 'easy' else 21
    return {'problem': verifier.CASES[case][1], 'x': [0.] * (6 * count),
            'fitness': -3 / (2 * size - 1) ** 2}


class SchemaTests(unittest.TestCase):
    def test_accepts_three_scenarios(self):
        for case in verifier.CASES:
            self.assertEqual(verifier.validate_record(valid_record(case)), case)

    def test_rejects_missing_fields_and_unknown_problem(self):
        for record in [[], None, {}, {'problem': 'unknown'}]:
            with self.subTest(record=record), self.assertRaises(ValueError):
                verifier.validate_record(record)
        for field in ['x', 'fitness']:
            record = valid_record()
            del record[field]
            with self.assertRaises(ValueError):
                verifier.validate_record(record)

    def test_rejects_bad_chromosome_shape_and_bounds(self):
        for x in [[0.] * 29, [0.] * 31, 'bad', None]:
            record = valid_record()
            record['x'] = x
            with self.assertRaises(ValueError):
                verifier.validate_record(record)
        for index, value in [(0, 1.01), (14, -1.01), (15, 10.01), (29, -10.01)]:
            record = valid_record()
            record['x'][index] = value
            with self.assertRaises(ValueError):
                verifier.validate_record(record)

    def test_rejects_nonfinite_and_nonnumeric_values(self):
        for value in [True, None, '0', float('nan'), float('inf'), 10 ** 1000]:
            for field in ['x', 'fitness']:
                record = valid_record()
                if field == 'x':
                    record['x'][0] = value
                else:
                    record['fitness'] = value
                with self.subTest(field=field, value=str(value)[:20]), self.assertRaises(ValueError):
                    verifier.validate_record(record)

    def test_actual_fitness_and_mismatch(self):
        record = valid_record()
        report = verifier.verify_record(record)
        self.assertTrue(report['valid'])
        self.assertEqual(report['case'], 'easy')
        self.assertAlmostEqual(report['fitness'], record['fitness'])
        record['fitness'] = -0.5
        with self.assertRaisesRegex(ValueError, 'Fitness mismatch'):
            verifier.verify_record(record)

    def test_schema_failure_does_not_construct_model(self):
        with patch.object(verifier, 'load_problem') as loader:
            with self.assertRaises(ValueError):
                verifier.verify_record({})
        loader.assert_not_called()


class FileAndCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.path = self.folder / 'result.json'

    def test_utf8_bom_and_input_not_modified(self):
        self.path.write_text(json.dumps(valid_record()), encoding='utf-8-sig')
        original = self.path.read_bytes()
        self.assertEqual(verifier.load_record(self.path), valid_record())
        with contextlib.redirect_stdout(io.StringIO()) as output:
            verifier.main([str(self.path)])
        self.assertIn('VALID: easy', output.getvalue())
        self.assertEqual(self.path.read_bytes(), original)

    def test_missing_malformed_and_invalid_files_have_clean_cli_errors(self):
        for text in [None, '{broken', '[]', '{"problem":"unknown"}']:
            if text is not None:
                self.path.write_text(text, encoding='utf-8')
            with contextlib.redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit) as stopped:
                verifier.main([str(self.path)])
            self.assertEqual(stopped.exception.code, 2)
            self.assertIn('error:', error.getvalue())
            self.assertNotIn('Traceback', error.getvalue())


if __name__ == '__main__':
    unittest.main()
