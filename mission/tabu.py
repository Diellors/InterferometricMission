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


