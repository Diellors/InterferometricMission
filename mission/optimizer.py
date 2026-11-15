"""Optimize the three ESA instances using their unmodified official evaluator."""
import argparse
import json
from .problems import CASES, ROOT, load_problem

import numpy as np
from scipy.optimize import differential_evolution

def initial_population(udp, size, rng):
    """Sample states with limited displacement at all three observations.

    SVD suppresses unstable directions, particularly useful for the HALO orbit.
    This is a starting heuristic, not a constraint or an optimality guarantee.
    """
    lo, hi = np.asarray(udp.get_bounds())
    scales = np.ones(udp.n_meas)
    scales[1:] = udp.inflation_factor
    mapping = (udp.stms[:, :3, :] / scales[:, None, None]).reshape(-1, 6)
    _, singular, vt = np.linalg.svd(mapping, full_matrices=False)
    transform = vt.T / np.maximum(singular, 1.0)
    population = []
    for _ in range(size):
        states = rng.normal(size=(udp.n_sat, 6)) @ transform.T
        positions = np.einsum('tij,nj->tni', udp.stms[:, :3, :], states)
        extent = np.max(np.abs(positions / scales[:, None, None]), axis=(0, 2))
        states *= (rng.uniform(0.3, 0.95, udp.n_sat) / np.maximum(extent, 1e-12))[:, None]
        population.append(np.clip(states.T.reshape(-1), lo, hi))
    return np.asarray(population)


def solve(case, args):
    udp = load_problem(case)
    rng = np.random.default_rng(args.seed)
    initial = initial_population(udp, args.population, rng)
    baseline = min(float(udp.fitness(x)[0]) for x in initial)
    print(f'{case}: {6 * udp.n_sat} variables; initial best = {baseline:.9f}', flush=True)
    # Grid occupancy makes the objective discontinuous: skip gradient polishing.
    result = differential_evolution(
        lambda x: float(udp.fitness(x)[0]),
        list(zip(*udp.get_bounds())), init=initial,
        maxiter=args.generations, seed=args.seed, polish=False,
        mutation=(0.4, 1.0), recombination=0.7, tol=0, atol=0,
        disp=True,
    )
    x = result.x.tolist()
    score = float(udp.fitness(x)[0])
    if not np.isfinite(score) or score > baseline + 1e-12:
        raise RuntimeError('Final evaluation failed validation.')
    out = ROOT / 'results' / case / f'seed-{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{case}.json'
    path.write_text(json.dumps({
        'problem': CASES[case][1], 'x': x, 'fitness': score,
        'initial_best': baseline, 'seed': args.seed,
        'evaluations': result.nfev, 'generations': result.nit,
        'optimizer_message': str(result.message),
    }, indent=2), encoding='utf-8')
    print(f'Saved {path}; official fitness = {score:.9f}', flush=True)
    if args.plot:
        import matplotlib.pyplot as plt
        udp.plot(x)
        plt.savefig(out / f'{case}.png', bbox_inches='tight', dpi=150)
        plt.close('all')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=[*CASES, 'all'])
    parser.add_argument('--config', type=str, help='JSON with case, population, generations, seed')
    parser.add_argument('--population', type=int)
    parser.add_argument('--generations', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    if args.config:
        from pathlib import Path
        config = json.loads(Path(args.config).read_text(encoding='utf-8'))
        if not isinstance(config, dict):
            parser.error('Config must be a JSON object')
    else:
        config = {}
    args.case = args.case if args.case is not None else config.get('case', 'easy')
    if args.case not in [*CASES, 'all']:
        parser.error('Invalid case in config')
    for key, default in [('population', 80), ('generations', 100), ('seed', 42)]:
        if getattr(args, key) is None:
            setattr(args, key, config.get(key, default))
        if type(getattr(args, key)) is not int:
            parser.error(f'{key} must be an integer')
    if args.population < 5 or args.generations < 0:
        parser.error('population >= 5 and generations >= 0 required')
    for case in CASES if args.case == 'all' else [args.case]:
        solve(case, args)
