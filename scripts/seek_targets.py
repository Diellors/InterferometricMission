"""Warm-start SA until a target is reached or a finite restart budget is used."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
from time import perf_counter, sleep
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mission.annealing import ALGORITHM, search, validate_temperatures
from mission.optimizer import mutate_satellite
from mission.problems import CASES, ROOT, load_problem
from scripts.verify_solution import load_record, verify_record


def write_json(path, record):
    """Replace a checkpoint atomically so interruption does not leave half JSON."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
    # Windows readers can briefly deny replacing an open checkpoint. Retain
    # atomic publication and retry only this transient permission failure.
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            sleep(0.02 * (attempt + 1))


def run_target_search(case, start, target, output, *, attempts=12, steps=10000,
                      initial_temperature=0.003, final_temperature=0.00001, seed=100):
    if case not in CASES:
        raise ValueError('Select easy, medium or hard')
    if type(target) not in (int, float) or not math.isfinite(target) or target > 0:
        raise ValueError('target must be a finite nonpositive fitness')
    if type(attempts) is not int or attempts < 0:
        raise ValueError('attempts must be a nonnegative integer')
    if type(steps) is not int or steps < 5 or steps % 5:
        raise ValueError('steps must be a positive multiple of five')
    if type(seed) is not int:
        raise ValueError('seed must be an integer')
    validate_temperatures(initial_temperature, final_temperature)
    incumbent = load_record(start)
    if incumbent['problem'] != CASES[case][1]:
        raise ValueError('Start result does not match the selected case')
    problem = load_problem(case)
    if target < problem.fitness_lower_bound() - 1e-12:
        raise ValueError(f'Target is below the counting bound {problem.fitness_lower_bound():.12f}')
    verify_record(incumbent)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    starting_fitness = incumbent['fitness']
    previous = str(start)
    completed, stage_evaluations = 0, 0
    write_json(output / 'best.json', incumbent)

    def save_summary():
        summary = {'case': case, 'target': target, 'starting_fitness': starting_fitness,
                   'best_fitness': incumbent['fitness'], 'target_reached': incumbent['fitness'] <= target,
                   'attempts_completed': completed, 'stage_evaluations': stage_evaluations,
                   'best_file': 'best.json', 'start_file': str(start),
                   'steps_per_attempt': steps, 'initial_temperature': initial_temperature,
                   'final_temperature': final_temperature, 'first_seed': seed}
        write_json(output / 'summary.json', summary)
        return summary

    save_summary()
    for attempt in range(attempts):
        if incumbent['fitness'] <= target:
            break
        attempt_seed = seed + attempt
        rng = random.Random(attempt_seed)
        x = incumbent['x']
        # Match the target-reaching experiment: incumbent + four nearby states.
        population = [x[:]] + [mutate_satellite(problem, x, rng, 0.08) for _ in range(4)]
        started = perf_counter()
        result = search(problem, population, steps, rng,
                        initial_temperature=initial_temperature, final_temperature=final_temperature,
                        record_every=5)
        result.update({'problem': CASES[case][1], 'algorithm': ALGORITHM,
                       'model': 'independent-cr3bp-linearized-rk4-v1',
                       'population': 5, 'generations': steps // 5, 'seed': attempt_seed,
                       'search_seconds': perf_counter() - started, 'target': target,
                       'resumed_from': previous,
                       'observation_totals': [o['total'] for o in problem.observations(result['x'])]})
        verify_record(result)
        if result['fitness'] > incumbent['fitness'] + 1e-12:
            raise RuntimeError('Search attempted to discard a better incumbent')
        path = output / f'attempt-{attempt + 1:02d}.json'
        write_json(path, result)
        write_json(output / 'best.json', result)
        incumbent, previous = result, str(path)
        completed += 1
        stage_evaluations += result['evaluations']
        save_summary()
        print(f'{case}, attempt={completed}, seed={attempt_seed}, fitness={result["fitness"]:.9f}', flush=True)
    return save_summary()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=list(CASES), required=True)
    parser.add_argument('--start', type=Path, required=True)
    parser.add_argument('--target', type=float, required=True)
    parser.add_argument('--attempts', type=int, default=12)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--initial-temperature', type=float, default=0.003)
    parser.add_argument('--final-temperature', type=float, default=0.00001)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--output', type=Path, help='New folder; existing folders are protected')
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output if args.output is not None else ROOT / 'results/targets' / f'{args.case}-{stamp}-{uuid4().hex[:8]}'
    try:
        summary = run_target_search(args.case, args.start, args.target, output,
                                    attempts=args.attempts, steps=args.steps,
                                    initial_temperature=args.initial_temperature,
                                    final_temperature=args.final_temperature, seed=args.seed)
    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f'Best={summary["best_fitness"]:.9f}; target reached={summary["target_reached"]}; {output}')
    return summary


if __name__ == '__main__':
    main()
