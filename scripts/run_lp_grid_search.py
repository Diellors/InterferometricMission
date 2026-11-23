"""Run coverage-ranked grid best response with LP feasibility."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if (ROOT / '.runtime/cpsat').is_dir():
    sys.path.insert(0, str(ROOT / '.runtime/cpsat'))
from mission.lp_grid_search import search
from mission.problems import CASES, load_problem
from scripts.seek_targets import write_json
from scripts.verify_solution import load_record, verify_record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=Path, default=ROOT / 'solutions/hard.json')
    parser.add_argument('--improvements', type=int, default=1)
    parser.add_argument('--lp-limit', type=int, default=50000)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    output = args.output or ROOT / 'results/lp_grid_search' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8])
    try:
        source = load_record(args.start)
        case = verify_record(source)['case']
        problem = load_problem(case)
        output.mkdir(parents=True, exist_ok=False)
        def progress(best, stats):
            record = {'problem': CASES[case][1], 'x': best.x, 'fitness': best.key[0],
                      'algorithm': 'lp-grid-checkpoint', 'lp_progress': stats}
            verify_record(record)
            write_json(output / 'checkpoint.json', record)
            print(f'{stats}, best={best.key[0]:.12f}', flush=True)
        started = perf_counter()
        result = search(problem, source['x'], max_improvements=args.improvements,
                        max_lp_calls=args.lp_limit, epsilon=args.epsilon, progress=progress)
        import ortools
        result.update(problem=CASES[case][1], resumed_from=str(args.start),
                      solver_version=ortools.__version__, search_seconds=perf_counter()-started,
                      model='independent-cr3bp-linearized-rk4-v1',
                      observation_totals=[o['total'] for o in problem.observations(result['x'])])
        verify_record(result)
        write_json(output / 'result.json', result)
    except (OSError, UnicodeError, ValueError, RuntimeError, ImportError) as error:
        parser.error(str(error))
    print(f'LP grid search {case}: {result["fitness"]:.12f}; saved {output}', flush=True)
    return output


if __name__ == '__main__':
    main()
