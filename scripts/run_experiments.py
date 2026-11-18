"""Compare seeds: python scripts/run_experiments.py --cases easy --seeds 1 2 3.

Use --cases easy medium hard to run all scenarios. Every batch creates a new
output folder containing individual JSON results, runs.csv and summary.csv.
Search timings exclude model construction, result validation and disk writes.
"""
import argparse
import contextlib
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
from time import perf_counter
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mission.optimizer import ALGORITHM, evolve, initial_population
from mission.problems import CASES, ROOT, load_problem

RUN_FIELDS = ['case', 'seed', 'population', 'generations', 'fitness', 'initial_best',
              'improvement', 'evaluations', 'search_seconds', 'result_file']
SUMMARY_FIELDS = ['case', 'runs', 'best_seed', 'best_fitness', 'mean_fitness',
                  'median_fitness', 'stdev_fitness', 'mean_search_seconds',
                  'total_evaluations', 'best_result_file']


def summarize(rows):
    """Compare only runs within each scenario; minimized fitness selects best."""
    summaries = []
    for case in dict.fromkeys(row['case'] for row in rows):
        group = [row for row in rows if row['case'] == case]
        best = min(group, key=lambda row: row['fitness'])
        scores = [row['fitness'] for row in group]
        summaries.append({
            'case': case, 'runs': len(group), 'best_seed': best['seed'],
            'best_fitness': best['fitness'], 'mean_fitness': statistics.mean(scores),
            'median_fitness': statistics.median(scores),
            'stdev_fitness': statistics.stdev(scores) if len(scores) > 1 else 0.,
            'mean_search_seconds': statistics.mean(row['search_seconds'] for row in group),
            'total_evaluations': sum(row['evaluations'] for row in group),
            'best_result_file': best['result_file'],
        })
    return summaries


def write_summary(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summarize(rows))


def run_experiments(cases, seeds, population, generations, output, *, verbose=False):
    """Run a serial batch, preserving completed results if a later run fails.

    Output must not exist, so earlier batches and single-run results are safe.
    Duplicate cases/seeds are rejected rather than repeating or overwriting runs.
    The CSV is flushed after every completed run; summary.csv is updated too.
    """
    if not cases or any(case not in CASES for case in cases):
        raise ValueError('Select at least one known case: easy, medium, hard')
    if not seeds or any(type(seed) is not int for seed in seeds):
        raise ValueError('Select at least one integer seed')
    if len(set(cases)) != len(cases) or len(set(seeds)) != len(seeds):
        raise ValueError('Duplicate cases or seeds are not allowed')
    if type(population) is not int or population < 5:
        raise ValueError('population must be an integer >= 5')
    if type(generations) is not int or generations < 0:
        raise ValueError('generations must be a nonnegative integer')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    with (output / 'runs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=RUN_FIELDS)
        writer.writeheader()
        stream.flush()
        for case in cases:
            # Construct/retrieve the orbit model before timing each seed.
            problem = load_problem(case)
            for seed in seeds:
                rng = random.Random(seed)
                started = perf_counter()
                initial = initial_population(problem, population, rng)
                with contextlib.ExitStack() as stack:
                    if not verbose:
                        sink = stack.enter_context(open(os.devnull, 'w'))
                        stack.enter_context(contextlib.redirect_stdout(sink))
                    x, score, history, evaluations = evolve(problem, initial, generations, rng)
                seconds = perf_counter() - started
                checked = problem.fitness(x)[0]
                if (not math.isfinite(score) or not math.isclose(score, checked, rel_tol=0, abs_tol=1e-12)
                        or score > history[0] + 1e-12):
                    raise RuntimeError(f'Result validation failed for {case}, seed={seed}')
                relative = Path(case) / f'seed-{seed}' / f'{case}.json'
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                record = {
                    'problem': CASES[case][1], 'x': x, 'fitness': score,
                    'initial_best': history[0], 'population': population,
                    'seed': seed, 'generations': generations, 'history': history,
                    'evaluations': evaluations, 'search_seconds': seconds,
                    'algorithm': ALGORITHM,
                    'model': 'independent-cr3bp-linearized-rk4-v1',
                    'observation_totals': [item['total'] for item in problem.observations(x)],
                }
                target.write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
                row = {
                    'case': case, 'seed': seed, 'population': population,
                    'generations': generations, 'fitness': score, 'initial_best': history[0],
                    'improvement': history[0] - score, 'evaluations': evaluations,
                    'search_seconds': seconds, 'result_file': relative.as_posix(),
                }
                rows.append(row)
                writer.writerow(row)
                stream.flush()
                write_summary(output / 'summary.csv', rows)
                print(f'{case}, seed={seed}: fitness={score:.9f}, '
                      f'evaluations={evaluations}, search={seconds:.3f}s', flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cases', nargs='+', choices=list(CASES), default=['easy'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--population', type=int, default=20)
    parser.add_argument('--generations', type=int, default=10)
    parser.add_argument('--output', type=Path, help='New output folder; must not already exist')
    parser.add_argument('--verbose', action='store_true', help='Print progress for every generation')
    args = parser.parse_args(argv)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output if args.output is not None else (
        ROOT / 'results' / 'experiments' / f'{timestamp}-{uuid4().hex[:8]}')
    try:
        rows = run_experiments(args.cases, args.seeds, args.population, args.generations,
                               output, verbose=args.verbose)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f'Batch saved: {output}')
    for item in summarize(rows):
        print(f'{item["case"]}: best seed={item["best_seed"]}, '
              f'best={item["best_fitness"]:.9f}, mean={item["mean_fitness"]:.9f}')
    return output


if __name__ == '__main__':
    main()
