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


class MetadataAndObservationTests(unittest.TestCase):
    def setUp(self):
        self.record = valid_record()
        self.record.update({'history': [0., self.record['fitness']], 'initial_best': 0.,
                            'generations': 1, 'population': 5, 'evaluations': 10,
                            'seed': 42, 'search_seconds': 0.,
                            'algorithm': 'hybrid-de-single-satellite-v1',
                            'observation_totals': [-self.record['fitness']] * 3})

    def test_complete_record_details_and_no_mutation(self):
        original = copy.deepcopy(self.record)
        report = verifier.verify_record(self.record)
        self.assertEqual(self.record, original)
        self.assertEqual(report['not_recorded'], [])
        self.assertEqual(report['worst_epoch'], 0)
        self.assertEqual(len(report['observations']), 3)
        for item in report['observations']:
            self.assertEqual(item['remaining_satellites'], 5)
            self.assertAlmostEqual(item['total'], -self.record['fitness'])
            for plane in item['planes'].values():
                self.assertEqual(plane['occupied_cells'], 1)
                self.assertEqual(plane['baselines'], 1)
                self.assertAlmostEqual(plane['fill_factor'], 1 / 441)

    def test_minimal_records_all_scenarios_report_missing_metadata(self):
        for case in verifier.CASES:
            report = verifier.verify_record(valid_record(case))
            self.assertEqual(report['case'], case)
            self.assertIn('history', report['not_recorded'])
            self.assertIn('observation_totals', report['not_recorded'])

    def test_rejects_inconsistent_history(self):
        for history in [[], None, 'bad', [float('nan')], [True], [0., -0.2, -0.1],
                        [0., -0.5], [-0.001, self.record['fitness']], [self.record['fitness']]]:
            record = copy.deepcopy(self.record)
            record['history'] = history
            with self.subTest(history=history), self.assertRaises(ValueError):
                verifier.validate_record(record)

    def test_rejects_invalid_metadata_and_budget(self):
        for field, value in [('generations', -1), ('generations', True), ('population', 4),
                             ('evaluations', -1), ('evaluations', 11), ('seed', '42'),
                             ('search_seconds', -1), ('search_seconds', float('inf')),
                             ('initial_best', -1), ('initial_best', 'bad')]:
            record = copy.deepcopy(self.record)
            record[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                verifier.validate_record(record)

    def test_unknown_algorithm_does_not_assume_de_budget(self):
        self.record['algorithm'] = 'another-search'
        self.record['evaluations'] = 14
        self.assertEqual(verifier.validate_record(self.record), 'easy')

    def test_rejects_invalid_or_mismatched_observation_totals(self):
        for totals in [[], [0., 0.], [0., 0., 0., 0.], [0., True, 0.],
                       [0., float('nan'), 0.], [0., 0., 0.]]:
            record = copy.deepcopy(self.record)
            record['observation_totals'] = totals
            with self.subTest(totals=totals), self.assertRaises(ValueError):
                verifier.verify_record(record)

    def test_zero_generation_metadata(self):
        score = self.record['fitness']
        self.record.update({'history': [score], 'initial_best': score,
                            'generations': 0, 'evaluations': 5})
        self.assertTrue(verifier.verify_record(self.record)['valid'])

    def test_cli_json_report_is_parseable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'result.json'
            path.write_text(json.dumps(self.record), encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()) as output:
                report = verifier.main([str(path), '--json'])
            self.assertEqual(json.loads(output.getvalue()), report)


if __name__ == '__main__':
    unittest.main()
