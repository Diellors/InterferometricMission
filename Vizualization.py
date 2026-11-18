"""Plot a result: python Vizualization.py results/easy/seed-42/easy.json --show"""
import argparse
import json
import math
from pathlib import Path

from mission.problems import CASES, load_problem
from mission.plotting import plot_convergence, plot_formation, validate_history


def load_result(path):
    """Validate a result before importing optional plotting packages."""
    record = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(record, dict):
        raise ValueError('Result JSON must be an object')
    case = next((key for key, value in CASES.items() if value[1] == record.get('problem')), None)
    if case is None:
        raise ValueError('Unknown or missing problem identifier')
    x = record.get('x')
    count = 5 if case == 'easy' else 40
    if not isinstance(x, list) or len(x) != 6 * count:
        raise ValueError(f'x must be a list of {6 * count} numbers for {case}')
    for index, value in enumerate(x):
        bound = 1 if index < 3 * count else 10
        if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > bound:
            raise ValueError(f'x[{index}] must be finite and inside [-{bound}, {bound}]')
    fitness = record.get('fitness')
    if type(fitness) not in (int, float) or not math.isfinite(fitness):
        raise ValueError('fitness must be a finite number')
    if 'history' in record:
        history = validate_history(record['history'])
        if not math.isclose(history[-1], fitness, rel_tol=0, abs_tol=1e-10):
            raise ValueError('Final history value does not match fitness')
        if 'generations' in record and (type(record['generations']) is not int or
                                        record['generations'] != len(history) - 1):
            raise ValueError('generations does not match history length')
    return case, record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    parser.add_argument('--kind', choices=['all', 'formation', 'convergence'], default='all')
    parser.add_argument('--output-dir', type=Path, help='Default: same folder as result JSON')
    parser.add_argument('--show', action='store_true', help='Also open interactive figure windows')
    parser.add_argument('--dpi', type=int, default=150)
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error('--dpi must be positive')
    figures = []
    plt = None
    try:
        case, record = load_result(args.result)
        if args.kind == 'convergence' and 'history' not in record:
            raise ValueError('This result has no history; rerun the optimizer to record convergence')
        problem = None
        if args.kind in ('all', 'formation'):
            problem = load_problem(case)
            score = problem.fitness(record['x'])[0]
            if not math.isclose(score, record['fitness'], rel_tol=0, abs_tol=1e-10):
                raise ValueError(f'Stored fitness differs from recomputed fitness ({score:.9f})')
        import matplotlib
        if not args.show:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        output = args.output_dir if args.output_dir is not None else args.result.parent
        output.mkdir(parents=True, exist_ok=True)
        if problem is not None:
            figures.append((plot_formation(problem, record['x']), output / f'{args.result.stem}.png'))
        if args.kind in ('all', 'convergence'):
            if 'history' in record:
                figures.append((plot_convergence(record['history'], title=f'{case.upper()} convergence'),
                                output / f'{args.result.stem}_convergence.png'))
            else:
                print('Convergence skipped: this older result has no history.')
        for figure, target in figures:
            figure.savefig(target, bbox_inches='tight', dpi=args.dpi)
            print(f'Saved {target}')
        if args.show:
            plt.show()
    except ModuleNotFoundError as error:
        parser.error(f'Missing plotting dependency ({error.name}). Run: python -m pip install -r requirements.txt')
    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    finally:
        if plt is not None:
            for figure, _ in figures:
                plt.close(figure)


if __name__ == '__main__':
    main()
