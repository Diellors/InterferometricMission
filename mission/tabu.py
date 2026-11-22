"""Grid-aware Tabu Search with reverse-move memory and aspiration."""
import math

from .sampling import MARGIN, direction, geometry, sample_state
from .search_state import FormationState

ALGORITHM = 'grid-aware-tabu-search-v1'


def signature(problem, state):
    """Projected cells define a phenotype; raw coordinate changes may not."""
    cells = []
    for epoch, matrix in enumerate(problem.stms):
        scale = 1 if epoch == 0 else problem.inflation_factor
        point = [sum(a*b for a,b in zip(row, state)) / scale for row in matrix[:3]]
        if not all(-1 < v < 1 for v in point):
            return None
        cells.append(tuple(int(v * problem.grid_size / 2 + problem.grid_size // 2) for v in point))
    return tuple(cells)


def grid_neighbor(problem, x, satellite, rng, radius=0.3, interval_limit=6):
    """Select a feasible interval that changes at least one projected cell.

    Grid-transition points partition a chord into constant-cell intervals.
    Sampling interval interiors avoids scoring moves on a flat unchanged grid.
    No objective evaluations occur in this geometric candidate generation.
    """
    state = [x[j * problem.n_sat + satellite] for j in range(6)]
    before = signature(problem, state)
    rows, _ = geometry(problem)
    values = [sum(a*b for a,b in zip(row, state)) for row in rows]
    if before is None or any(abs(v) >= MARGIN for v in values):
        moved = sample_state(problem, rng)
        after = signature(problem, moved)
        if after == before:
            return None
    else:
        delta = direction(problem, rng, targeted=rng.random() < 0.5)
        slopes = [sum(a*b for a,b in zip(row, delta)) for row in rows]
        extent = max(max(abs(v) for v in slopes), 1e-15)
        delta = [v / extent for v in delta]
        slopes = [v / extent for v in slopes]
        lower, upper = -radius, radius
        for value, slope in zip(values, slopes):
            if abs(slope) > 1e-15:
                a, b = (-MARGIN - value) / slope, (MARGIN - value) / slope
                lower, upper = max(lower, min(a,b)), min(upper, max(a,b))
        if lower >= upper:
            return None
        cuts = {lower, upper}
        size = problem.grid_size
        for value, slope in zip(values[:3 * problem.n_meas], slopes[:3 * problem.n_meas]):
            if abs(slope) < 1e-15:
                continue
            # int truncates toward zero. Cell zero extends to the negative
            # cube edge, so internal transitions are at integers 1,...,G-1.
            for cell in range(1, size):
                boundary = 2 * (cell - size // 2) / size
                distance = (boundary - value) / slope
                if lower < distance < upper:
                    cuts.add(distance)
        cuts = sorted(cuts)
        distances = [(a+b)/2 for a,b in zip(cuts, cuts[1:]) if b-a > 1e-12]
        distances.sort(key=abs)
        options = []
        seen = set()
        for distance in distances:
            moved = [v + distance*d for v,d in zip(state, delta)]
            after = signature(problem, moved)
            if after is not None and after != before and after not in seen:
                seen.add(after)
                options.append((moved, after))
                if len(options) == interval_limit:
                    break
        if not options:
            return None
        moved, after = rng.choice(options)
    trial = x[:]
    for component, value in enumerate(moved):
        trial[component * problem.n_sat + satellite] = value
    return trial, before, after


def allowed(is_tabu, candidate_key, best_key):
    """Aspiration overrides tabu only for a strict archive improvement."""
    return not is_tabu or candidate_key < best_key


def adjacent_neighbor(problem, x, satellite, epoch, axis, shift, *, cycles=128,
                      other_epoch_slack=0, target_offsets=None):
    """Seek one adjacent projected cell by cyclic metric projections.

    This is a bounded feasibility heuristic, not an exact LP solver. Returning
    None does not prove infeasibility. A result is returned only after checking
    every desired cell and all decision bounds.
    """
    state = [x[j * problem.n_sat + satellite] for j in range(6)]
    before = signature(problem, state)
    if before is None:
        return None
    desired = [list(cell) for cell in before]
    if target_offsets is None:
        desired[epoch][axis] += shift
    else:
        desired[epoch] = [value + offset for value, offset in zip(before[epoch], target_offsets)]
    if any(not 0 <= value < problem.grid_size for value in desired[epoch]):
        return None
    rows, lower = geometry(problem)
    intervals = []
    epsilon = 1e-7
    size, center = problem.grid_size, problem.grid_size // 2
    for time, point in enumerate(desired):
        for cell in point:
            slack = other_epoch_slack if time != epoch else 0
            first_cell, last_cell = max(0, cell-slack), min(size-1, cell+slack)
            lo = -MARGIN if first_cell == 0 else 2 * (first_cell-center) / size + epsilon
            hi = min(MARGIN, 2 * (last_cell+1-center) / size - epsilon)
            intervals.append((lo, hi))
    intervals.extend([(-MARGIN, MARGIN)] * 6)
    normals = []
    for row in rows:
        intermediate = [0.] * 6
        for i in range(6):
            intermediate[i] = (row[i] - sum(lower[i][j]*intermediate[j] for j in range(i))) / lower[i][i]
        normal = [0.] * 6
        for i in range(5, -1, -1):
            normal[i] = (intermediate[i] - sum(lower[j][i]*normal[j] for j in range(i+1, 6))) / lower[i][i]
        denominator = sum(a*b for a,b in zip(row, normal))
        normals.append([v / max(denominator, 1e-15) for v in normal])
    moved = state[:]
    for _ in range(cycles):
        maximum_error = 0.
        for row, normal, (lo, hi) in zip(rows, normals, intervals):
            value = sum(a*b for a,b in zip(row, moved))
            correction = lo-value if value < lo else hi-value if value > hi else 0.
            maximum_error = max(maximum_error, abs(correction))
            if correction:
                moved = [v + correction*d for v,d in zip(moved, normal)]
        if maximum_error < 1e-10:
            break
    for row, (lo, hi) in zip(rows, intervals):
        value = sum(a*b for a,b in zip(row, moved))
        if not lo-1e-10 <= value <= hi+1e-10:
            return None
    after = signature(problem, moved)
    if after is None or after[epoch] != tuple(desired[epoch]):
        return None
    if any(abs(after[t][j]-before[t][j]) > other_epoch_slack
           for t in range(problem.n_meas) if t != epoch for j in range(3)):
        return None
    trial = x[:]
    for component, value in enumerate(moved):
        trial[component * problem.n_sat + satellite] = value
    # Enforce public decision bounds as well as the projected-cell bands.
    problem.decode(trial)
    return trial, before, after


