"""Greedy construction followed by first-improvement one-satellite search.

The search space is a finite pool of feasible orbital states. All choices and
comparisons are implemented here; no optimization library is used.
"""
import math

from .sampling import sample_state
from .search_state import FormationState

ALGORITHM = 'greedy-baseline-local-search-v1'
PLANES = ((0, 1), (0, 2), (1, 2))


def candidate_cells(problem, state):
    cells = []
    for epoch, matrix in enumerate(problem.stms):
        scale = 1 if epoch == 0 else problem.inflation_factor
        point = [sum(a*b for a,b in zip(row, state)) / scale for row in matrix[:3]]
        if not all(math.isfinite(value) and -1 < value < 1 for value in point):
            raise ValueError('Every pool candidate must remain inside all observation grids')
        cells.append(tuple(int(value * problem.grid_size / 2 + problem.grid_size // 2)
                           for value in point))
    return cells


def addition_key(problem, selected_cells, baseline_sets, candidate):
    """Count only new baseline displacements introduced by this candidate."""
    increments, totals = [], []
    area = (2 * problem.grid_size - 1) ** 2
    for epoch, cell in enumerate(candidate):
        plane_increments, fills = [], []
        for plane, (a, b) in enumerate(PLANES):
            added = {(0, 0)}
            for other in selected_cells[epoch]:
                dx, dy = cell[a] - other[a], cell[b] - other[b]
                added.add((dx, dy))
                added.add((-dx, -dy))
            added = {difference for difference in added if difference not in baseline_sets[epoch][plane]}
            plane_increments.append(added)
            fills.append((len(baseline_sets[epoch][plane]) + len(added)) / area)
        increments.append(plane_increments)
        totals.append(sum(fills))
    return (-min(totals), -sum(totals)), increments


def chromosome(pool, selected):
    return [pool[index][component] for component in range(6) for index in selected]


def search(problem, pool, *, max_evaluations=10000, max_sweeps=5):
    """Choose N distinct pool indices, then improve by single replacements.

    Each partial construction score and each full refinement score counts once.
    These two costs are reported separately: a partial formation is cheaper to
    score and is not a full-N objective evaluation. Final verification is extra.
    """
    count = problem.n_sat
    if len(pool) < count:
        raise ValueError('Candidate pool must contain at least N states')
    if type(max_sweeps) is not int or max_sweeps < 0:
        raise ValueError('max_sweeps must be a nonnegative integer')
    construction_budget = count * len(pool) - count * (count - 1) // 2
    if type(max_evaluations) is not int or max_evaluations < construction_budget:
        raise ValueError(f'max_evaluations must be at least {construction_budget} for construction')
    for state in pool:
        if (not isinstance(state, (list, tuple)) or len(state) != 6 or
                any(type(v) not in (int, float) or not math.isfinite(v) or
                    abs(v) > (1 if j < 3 else 10) for j, v in enumerate(state))):
            raise ValueError('Pool states must contain six finite coordinates within decision bounds')
    cells = [candidate_cells(problem, state) for state in pool]
    selected = []
    used = set()
    occupied = [[] for _ in range(problem.n_meas)]
    supports = [[set() for _ in PLANES] for _ in range(problem.n_meas)]
    construction_trace = []
    evaluations = 0
    for _ in range(count):
        best_index, best_key, best_additions = None, None, None
        for index, candidate in enumerate(cells):
            if index in used:
                continue
            key, additions = addition_key(problem, occupied, supports, candidate)
            evaluations += 1
            if best_key is None or key < best_key:
                best_index, best_key, best_additions = index, key, additions
        selected.append(best_index)
        used.add(best_index)
        for epoch in range(problem.n_meas):
            occupied[epoch].append(cells[best_index][epoch])
            for plane in range(3):
                supports[epoch][plane].update(best_additions[epoch][plane])
        construction_trace.append({'satellites': len(selected), 'fitness': best_key[0],
                                   'sum_observation_totals': -best_key[1],
                                   'candidate_scores': evaluations})
    current = FormationState(problem, chromosome(pool, selected))
    if current.key != best_key:
        raise RuntimeError('Greedy baseline counts do not match the full evaluator')
    initial = current.key[0]
    improvement_trace = [{'candidate_scores': evaluations, 'fitness': initial,
                          'sum_observation_totals': -current.key[1]}]
    sweeps, accepted = 0, 0
    reason = 'sweep_limit'
    while sweeps < max_sweeps:
        improved, exhausted = False, False
        for position in range(count):
            for index in range(len(pool)):
                if index in used:
                    continue
                if evaluations >= max_evaluations:
                    exhausted = True
                    break
                trial_indices = selected[:]
                trial_indices[position] = index
                candidate = current.replace(chromosome(pool, trial_indices))
                evaluations += 1
                if candidate.key < current.key:
                    used.remove(selected[position])
                    used.add(index)
                    selected = trial_indices
                    current = candidate
                    accepted += 1
                    improved = True
                    improvement_trace.append({'candidate_scores': evaluations,
                                              'fitness': current.key[0],
                                              'sum_observation_totals': -current.key[1]})
                    break
            if exhausted:
                break
        if exhausted:
            reason = 'evaluation_limit'
            break
        sweeps += 1
        if not improved:
            reason = 'pool_local_optimum'
            break
    verified = problem.fitness(current.x)[0]
    if verified != current.key[0]:
        raise RuntimeError('Final greedy result failed re-evaluation')
    return {'x': current.x[:], 'fitness': verified, 'initial_best': initial,
            'selected_indices': selected, 'pool_size': len(pool),
            'evaluations': evaluations, 'construction_evaluations': construction_budget,
            'verification_evaluations': 1,
            'refinement_evaluations': evaluations - construction_budget,
            'sweeps': sweeps, 'accepted_replacements': accepted, 'stop_reason': reason,
            'construction_trace': construction_trace, 'improvement_trace': improvement_trace}


def make_pool(problem, size, rng):
    if type(size) is not int or size < problem.n_sat:
        raise ValueError('pool_size must be an integer >= N')
    return [sample_state(problem, rng) for _ in range(size)]
