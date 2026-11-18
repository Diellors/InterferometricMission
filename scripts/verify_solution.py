"""Validate result files and re-evaluate them with our independent model."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mission.problems import CASES, load_problem

TOLERANCE = 1e-10
OPTIONAL_FIELDS = ['history', 'initial_best', 'generations', 'population', 'evaluations',
                   'seed', 'search_seconds', 'observation_totals']


def finite_number(value, field):
    if type(value) not in (int, float):
        raise ValueError(f'{field} must be a finite number')
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f'{field} must be a finite number')
    return value


def validate_record(record):
    """Check required fields and bounds without constructing an orbit model."""
    if not isinstance(record, dict):
        raise ValueError('Result JSON must be an object')
    case = next((key for key, value in CASES.items() if value[1] == record.get('problem')), None)
    if case is None:
        raise ValueError('Unknown or missing problem identifier')
    count = 5 if case == 'easy' else 40
    x = record.get('x')
    if not isinstance(x, list) or len(x) != 6 * count:
        raise ValueError(f'x must be a list of {6 * count} numbers for {case}')
    for index, value in enumerate(x):
        bound = 1 if index < 3 * count else 10
        finite_number(value, f'x[{index}]')
        if abs(value) > bound:
            raise ValueError(f'x[{index}] must be inside [-{bound}, {bound}]')
    finite_number(record.get('fitness'), 'fitness')
    validate_metadata(record)
    return case


def require_match(actual, expected, field):
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=TOLERANCE):
        raise ValueError(f'{field} mismatch: computed={expected}, stored={actual}')


def validate_metadata(record):
    """Check the consistency of optional metadata; old minimal results remain valid.

    These checks cannot establish that intermediate generations actually ran;
    only the saved final chromosome can be re-evaluated independently here.
    """
    for field, minimum in [('generations', 0), ('population', 5), ('evaluations', 0)]:
        if field in record and (type(record[field]) is not int or record[field] < minimum):
            raise ValueError(f'{field} must be an integer >= {minimum}')
    if 'seed' in record and type(record['seed']) is not int:
        raise ValueError('seed must be an integer')
    if 'search_seconds' in record:
        if finite_number(record['search_seconds'], 'search_seconds') < 0:
            raise ValueError('search_seconds must be nonnegative')
    if 'initial_best' in record:
        initial = finite_number(record['initial_best'], 'initial_best')
        if record['fitness'] > initial + TOLERANCE:
            raise ValueError('Final fitness is worse than initial_best')
    if 'history' in record:
        history = record['history']
        if not isinstance(history, list) or not history:
            raise ValueError('history must be a nonempty list')
        for index, value in enumerate(history):
            finite_number(value, f'history[{index}]')
        if any(b > a + TOLERANCE for a, b in zip(history, history[1:])):
            raise ValueError('Best-so-far history must not increase')
        require_match(history[-1], record['fitness'], 'history final fitness')
        if 'initial_best' in record:
            require_match(history[0], record['initial_best'], 'history initial fitness')
        if 'generations' in record and len(history) != record['generations'] + 1:
            raise ValueError('history length must equal generations + 1')
    if (record.get('algorithm') == 'hybrid-de-single-satellite-v1' and
            all(field in record for field in ['evaluations', 'population', 'generations'])):
        expected = record['population'] * (record['generations'] + 1)
        if record['evaluations'] != expected:
            raise ValueError(f'evaluations must equal population * (generations + 1) = {expected}')
    if 'observation_totals' in record:
        totals = record['observation_totals']
        if not isinstance(totals, list) or len(totals) != 3:
            raise ValueError('observation_totals must contain three numbers')
        for index, value in enumerate(totals):
            finite_number(value, f'observation_totals[{index}]')


def load_record(path):
    record = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    validate_record(record)
    return record


def verify_record(record):
    """Return a report only after the chromosome and fitness pass validation."""
    case = validate_record(record)
    problem = load_problem(case)
    observations = problem.observations(record['x'])
    totals = [item['total'] for item in observations]
    score = -min(totals)
    if not math.isfinite(score) or not math.isclose(score, record['fitness'], rel_tol=0, abs_tol=TOLERANCE):
        raise ValueError(f'Fitness mismatch: computed={score}, stored={record["fitness"]}')
    if 'observation_totals' in record:
        for index, (stored, calculated) in enumerate(zip(record['observation_totals'], totals)):
            require_match(stored, calculated, f'observation_totals[{index}]')
    details = []
    for epoch, item in enumerate(observations):
        planes = {}
        for name, data in zip(['XY', 'XZ', 'YZ'], item['planes']):
            planes[name] = {'occupied_cells': len(data['cells']),
                            'baselines': len(data['baselines']), 'fill_factor': data['fill']}
        details.append({'epoch': epoch, 'time': problem.times[epoch],
                        'remaining_satellites': item['remaining'], 'total': item['total'],
                        'planes': planes})
    return {'valid': True, 'case': case, 'fitness': score, 'satellites': problem.n_sat,
            'worst_epoch': min(range(len(totals)), key=totals.__getitem__),
            'observations': details,
            'verified_fields': ['x', 'fitness'] + [key for key in OPTIONAL_FIELDS if key in record],
            'not_recorded': [key for key in OPTIONAL_FIELDS if key not in record]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    parser.add_argument('--json', action='store_true', help='Print the verification report as JSON')
    args = parser.parse_args(argv)
    try:
        report = verify_record(load_record(args.result))
    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print(f'VALID: {report["case"]}, fitness={report["fitness"]:.12f}')
        for item in report['observations']:
            fills = ', '.join(f'{name}={data["fill_factor"]:.6f}'
                              for name, data in item['planes'].items())
            print(f't={item["epoch"]}T: {item["remaining_satellites"]}/{report["satellites"]} '
                  f'satellites retained, total={item["total"]:.6f}; {fills}')
        print(f'Worst observation: t={report["worst_epoch"]}T')
        if report['not_recorded']:
            print('Metadata not recorded (not checked): ' + ', '.join(report['not_recorded']))
    return report


if __name__ == '__main__':
    main()
