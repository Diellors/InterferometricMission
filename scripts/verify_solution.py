"""Re-evaluate a local result with the official evaluator."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
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
    x = np.asarray(record['x'], dtype=float)
    lo, hi = np.asarray(udp.get_bounds())
    if x.shape != lo.shape or not np.all(np.isfinite(x)):
        raise ValueError('Invalid chromosome shape or non-finite values')
    if np.any(x < lo) or np.any(x > hi):
        raise ValueError('Chromosome outside official bounds')
    score = float(udp.fitness(x)[0])
    if not np.isfinite(score) or not np.isclose(score, record['fitness'], rtol=0, atol=1e-10):
        raise ValueError(f'Fitness mismatch: official={score}, stored={record["fitness"]}')
    print(f'VALID: {case}, fitness={score:.12f}')


if __name__ == '__main__':
    main()
