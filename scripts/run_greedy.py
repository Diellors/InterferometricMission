"""Run our greedy/local algorithm: python scripts/run_greedy.py --case easy."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from time import perf_counter
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mission.greedy import ALGORITHM, make_pool, search
from mission.problems import CASES, ROOT, load_problem


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=list(CASES), default='easy')
    parser.add_argument('--pool-size', type=int, default=80)
    parser.add_argument('--max-evaluations', type=int, default=10000)
    parser.add_argument('--max-sweeps', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, help='New JSON file; existing files are not overwritten')
    args = parser.parse_args(argv)
    count = 5 if args.case == 'easy' else 40
    minimum = count * args.pool_size - count * (count - 1) // 2
    if args.pool_size < count or args.max_sweeps < 0 or args.max_evaluations < minimum:
        parser.error(f'pool-size >= {count}, max-sweeps >= 0 and max-evaluations >= {minimum} required')
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target = args.output if args.output is not None else (
        ROOT / 'results' / 'greedy' / f'{timestamp}-{uuid4().hex[:8]}' / f'{args.case}.json')
    if target.exists():
        parser.error(f'Output already exists: {target}')
    try:
        problem = load_problem(args.case)
        started = perf_counter()
        pool = make_pool(problem, args.pool_size, random.Random(args.seed))
        result = search(problem, pool, max_evaluations=args.max_evaluations, max_sweeps=args.max_sweeps)
        result.update({'problem': CASES[args.case][1], 'seed': args.seed, 'algorithm': ALGORITHM,
                       'model': 'independent-cr3bp-linearized-rk4-v1',
                       'search_seconds': perf_counter() - started,
                       'observation_totals': [o['total'] for o in problem.observations(result['x'])]})
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f'{args.case}: fitness={result["fitness"]:.9f}, '
          f'construction scores={result["construction_evaluations"]}, '
          f'full refinement scores={result["refinement_evaluations"]}, '
          f'stop={result["stop_reason"]}')
    print(f'Saved {target}')
    return target


if __name__ == '__main__':
    main()
