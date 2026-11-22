"""Compare SA and SA/local/paired search from an identical saved incumbent."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import sys
from time import perf_counter
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mission import annealing, hybrid
from mission.problems import CASES, ROOT, load_problem
from scripts.seek_targets import write_json
from scripts.verify_solution import load_record, verify_record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=Path, default=ROOT / 'solutions/hard.json')
    parser.add_argument('--seeds', type=int, nargs='+', default=[701, 702, 703])
    parser.add_argument('--evaluations', type=int, default=10005)
    parser.add_argument('--initial-temperature', type=float, default=0.0005)
    parser.add_argument('--final-temperature', type=float, default=0.00001)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.evaluations < 10 or args.evaluations % 5:
        parser.error('evaluations must be a multiple of five and at least ten')
    if len(args.seeds) != len(set(args.seeds)):
        parser.error('Seeds must be distinct')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output if args.output is not None else ROOT / 'results/hybrid_comparison' / f'{stamp}-{uuid4().hex[:8]}'
    try:
        annealing.validate_temperatures(args.initial_temperature, args.final_temperature)
        source = load_record(args.start)
        case = verify_record(source)['case']
        problem = load_problem(case)
        output.mkdir(parents=True, exist_ok=False)
        rows = []
        for seed in args.seeds:
            for method in ['annealing', 'hybrid']:
                rng = random.Random(seed)
                started = perf_counter()
                if method == 'annealing':
                    iterations = args.evaluations - 5
                    result = annealing.search(problem, [source['x'][:] for _ in range(5)],
                                               iterations, rng, initial_temperature=args.initial_temperature,
                                               final_temperature=args.final_temperature, record_every=5)
                    result.update({'algorithm': annealing.ALGORITHM, 'population': 5,
                                   'generations': iterations // 5})
                else:
                    result = hybrid.search(problem, source['x'], args.evaluations, rng,
                                           initial_temperature=args.initial_temperature,
                                           final_temperature=args.final_temperature)
                result.update({'problem': CASES[case][1], 'seed': seed,
                               'model': 'independent-cr3bp-linearized-rk4-v1',
                               'search_seconds': perf_counter() - started,
                               'resumed_from': str(args.start),
                               'observation_totals': [o['total'] for o in problem.observations(result['x'])]})
                verify_record(result)
                result['verification_evaluations'] = result.get('verification_evaluations', 0) + 1
                if result['fitness'] > source['fitness'] + 1e-12:
                    raise RuntimeError('Search discarded its incumbent')
                path = output / f'{method}-seed-{seed}.json'
                write_json(path, result)
                rows.append({'method': method, 'seed': seed, 'fitness': result['fitness'],
                             'initial_best': result['initial_best'], 'evaluations': result['evaluations'],
                             'seconds': result['search_seconds'], 'record': path.name})
                with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                print(f'{method}, seed={seed}, fitness={result["fitness"]:.9f}, '
                      f'evaluations={result["evaluations"]}, seconds={result["search_seconds"]:.2f}', flush=True)
        summary = {method: {'best': min(r['fitness'] for r in rows if r['method'] == method),
                            'mean': statistics.mean(r['fitness'] for r in rows if r['method'] == method)}
                   for method in ['annealing', 'hybrid']}
        write_json(output / 'summary.json', summary)
    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2))
    print(f'Saved {output}')
    return output


if __name__ == '__main__':
    main()
