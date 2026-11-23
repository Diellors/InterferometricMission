"""Run adaptive large neighborhood search from a verified formation."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import random
import sys
from time import perf_counter
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mission.alns import search
from mission.problems import CASES, ROOT, load_problem
from scripts.seek_targets import write_json
from scripts.verify_solution import load_record, verify_record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=Path, default=ROOT / 'solutions/hard.json')
    parser.add_argument('--evaluations', type=int, default=30000)
    parser.add_argument('--pool-size', type=int, default=6)
    parser.add_argument('--min-destroy', type=int, default=4)
    parser.add_argument('--max-destroy', type=int, default=8)
    parser.add_argument('--block-probability', type=float, default=0.,
                        help='Planar/vertical donor recombination, only for block-separated models')
    parser.add_argument('--seed', type=int, default=1500)
    parser.add_argument('--target', type=float, default=-1.55)
    parser.add_argument('--initial-temperature', type=float, default=0.003)
    parser.add_argument('--final-temperature', type=float, default=0.00005)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    output = args.output or ROOT / 'results/alns' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8])
    try:
        source = load_record(args.start)
        case = verify_record(source)['case']
        problem = load_problem(case)
        if args.target < problem.fitness_lower_bound()-1e-12:
            raise ValueError('Target is below the counting bound')
        output.mkdir(parents=True, exist_ok=False)
        def progress(best, evaluations, iteration):
            record = {'problem': CASES[case][1], 'x': best.x, 'fitness': best.key[0],
                      'seed': args.seed, 'algorithm': 'alns-checkpoint',
                      'evaluations': evaluations}
            verify_record(record)
            write_json(output / 'checkpoint.json', record)
            print(f'iteration={iteration}, scores={evaluations}, best={best.key[0]:.12f}', flush=True)
        started = perf_counter()
        result = search(problem, source['x'], args.evaluations, random.Random(args.seed),
                        pool_size=args.pool_size, target=args.target,
                        min_destroy=args.min_destroy, max_destroy=args.max_destroy,
                        block_probability=args.block_probability,
                        initial_temperature=args.initial_temperature,
                        final_temperature=args.final_temperature, progress=progress)
        result.update({'problem': CASES[case][1], 'seed': args.seed,
                       'resumed_from': str(args.start), 'search_seconds': perf_counter()-started,
                       'model': 'independent-cr3bp-linearized-rk4-v1',
                       'observation_totals': [o['total'] for o in problem.observations(result['x'])]})
        verify_record(result)
        write_json(output / 'result.json', result)
    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f'ALNS {case}: fitness={result["fitness"]:.12f}; saved {output}', flush=True)
    return output


if __name__ == '__main__':
    main()
