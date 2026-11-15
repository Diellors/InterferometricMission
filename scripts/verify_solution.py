"""Re-evaluate a local result with our independent model."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math
from mission.problems import CASES, load_problem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    args = parser.parse_args()
    record = json.loads(args.result.read_text(encoding='utf-8'))
    case = next((key for key, value in CASES.items() if value[1] == record['problem']), None)
    if case is None:
        raise ValueError('Unknown problem identifier')
    udp = load_problem(case)
    x = record['x']
    udp.decode(x)
    score = float(udp.fitness(x)[0])
    if not math.isfinite(score) or not math.isclose(score, record['fitness'], rel_tol=0, abs_tol=1e-10):
        raise ValueError(f'Fitness mismatch: computed={score}, stored={record["fitness"]}')
    print(f'VALID: {case}, fitness={score:.12f}')


if __name__ == '__main__':
    main()
