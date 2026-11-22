"""Run grid-aware Tabu Search from a verified saved formation."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import random
import sys
from time import perf_counter
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mission.problems import CASES, ROOT, load_problem
from mission.tabu import search
from scripts.seek_targets import write_json
from scripts.verify_solution import load_record, verify_record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=Path, default=ROOT / 'solutions/hard.json')
    parser.add_argument('--target', type=float, default=-1.515)
    parser.add_argument('--evaluations', type=int, default=10000)
    parser.add_argument('--neighbors', type=int, default=32)
    parser.add_argument('--tenure', type=int, default=7)
    parser.add_argument('--radius', type=float, default=0.3)
    parser.add_argument('--intervals', type=int, default=6, help='Number of nearest changed-cell intervals to sample')
    parser.add_argument('--axis-probability', type=float, default=0., help='Probability of an adjacent-cell feasibility proposal')
    parser.add_argument('--axis-slack', type=int, default=0, help='Allowed cell displacement at other observation epochs')
    parser.add_argument('--diagonal-probability', type=float, default=0., help='Probability of coordinated cell shifts at the target epoch')
    parser.add_argument('--seed', type=int, default=900)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output if args.output is not None else ROOT / 'results/tabu' / f'{stamp}-{uuid4().hex[:8]}.json'
    if output.exists():
        parser.error(f'Output already exists: {output}')
    try:
        record = load_record(args.start)
        case = verify_record(record)['case']
        problem = load_problem(case)
        if args.target < problem.fitness_lower_bound() - 1e-12:
            raise ValueError('Target is below the counting bound for this case')
        started = perf_counter()
        result = search(problem, record['x'], args.evaluations, random.Random(args.seed),
                        neighborhood_size=args.neighbors, tenure=args.tenure,
                        radius=args.radius, target=args.target, interval_limit=args.intervals,
                        axis_probability=args.axis_probability, axis_slack=args.axis_slack,
                        diagonal_probability=args.diagonal_probability)
        result.update({'problem': CASES[case][1], 'seed': args.seed,
                       'search_seconds': perf_counter()-started,
                       'model': 'independent-cr3bp-linearized-rk4-v1',
                       'resumed_from': str(args.start),
                       'observation_totals': [o['total'] for o in problem.observations(result['x'])]})
        verify_record(result)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, result)
    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f'{case}: fitness={result["fitness"]:.12f}, evaluations={result["evaluations"]}, '
          f'stop={result["tabu"]["stop_reason"]}', flush=True)
    print(f'Saved {output}', flush=True)
    return output


if __name__ == '__main__':
    main()
