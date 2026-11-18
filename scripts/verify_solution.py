"""Validate result files and re-evaluate them with our independent model."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mission.problems import CASES, load_problem

TOLERANCE = 1e-10


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
    return case


def load_record(path):
    record = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    validate_record(record)
    return record


def verify_record(record):
    """Return a report only after the chromosome and fitness pass validation."""
    case = validate_record(record)
    problem = load_problem(case)
    score = problem.fitness(record['x'])[0]
    if not math.isfinite(score) or not math.isclose(score, record['fitness'], rel_tol=0, abs_tol=TOLERANCE):
        raise ValueError(f'Fitness mismatch: computed={score}, stored={record["fitness"]}')
    return {'valid': True, 'case': case, 'fitness': score}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    args = parser.parse_args(argv)
    try:
        report = verify_record(load_record(args.result))
    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f'VALID: {report["case"]}, fitness={report["fitness"]:.12f}')
    return report


if __name__ == '__main__':
    main()
