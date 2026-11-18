"""Sample relative states using the geometry of all observation constraints."""
from functools import lru_cache
import math

MARGIN = 1 - 1e-9


@lru_cache(maxsize=3)
def geometry(problem):
    """Whiten state directions with A^T A, where |A state| < 1 is feasible.

    Include all epochs and decision bounds. This compensates for unstable
    directions without compressing stable directions to the same tiny scale.
    A positive pivot floor protects the Cholesky factorization.
    """
    rows = []
    for epoch, matrix in enumerate(problem.stms):
        scale = 1 if epoch == 0 else problem.inflation_factor
        rows.extend([[value / scale for value in row] for row in matrix[:3]])
    _, upper = problem.get_bounds()
    for component in range(6):
        row = [0.] * 6
        row[component] = 1 / upper[component * problem.n_sat]
        rows.append(row)
    gram = [[sum(row[i] * row[j] for row in rows) for j in range(6)] for i in range(6)]
    lower = [[0.] * 6 for _ in range(6)]
    for i in range(6):
        for j in range(i + 1):
            value = gram[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                lower[i][j] = math.sqrt(max(value, 1e-12))
            else:
                lower[i][j] = value / lower[j][j]
    return rows, lower


def direction(problem, rng, *, targeted=False):
    """Solve L^T d = normal(0,I), with A^T A = L L^T."""
    rows, lower = geometry(problem)
    if targeted:
        # Q^-1 times a selected observation row favors changing that coordinate
        # while limiting the squared displacement over the remaining rows.
        target = rng.choice(rows[:3 * problem.n_meas])
        normal = [0.] * 6
        for i in range(6):
            normal[i] = (target[i] - sum(lower[i][k] * normal[k] for k in range(i))) / lower[i][i]
    else:
        normal = [rng.gauss(0, 1) for _ in range(6)]
    state = [0.] * 6
    for i in range(5, -1, -1):
        state[i] = (normal[i] - sum(lower[k][i] * state[k] for k in range(i + 1, 6))) / lower[i][i]
    return state


def fit_state(problem, state, radius=MARGIN):
    rows, _ = geometry(problem)
    extent = max(abs(sum(a*b for a,b in zip(row, state))) for row in rows)
    scale = min(1., radius / max(extent, 1e-15))
    return [value * scale for value in state]


def sample_state(problem, rng):
    state = direction(problem, rng)
    rows, _ = geometry(problem)
    extent = max(abs(sum(a*b for a,b in zip(row, state))) for row in rows)
    scale = rng.uniform(0.65, 0.995) / max(extent, 1e-15)
    return [value * scale for value in state]


def local_state(problem, state, rng, max_displacement=0.2, *, boundary=False, targeted=False):
    """Move along a feasible chord, bounded in every normalized observation.

    Return None for an infeasible starting point; callers can use a fallback.
    Unlike clipping one coordinate, a chord respects all orbit constraints.
    """
    rows, _ = geometry(problem)
    projections = [sum(a*b for a,b in zip(row, state)) for row in rows]
    if any(abs(value) >= MARGIN for value in projections):
        return None
    delta = direction(problem, rng, targeted=targeted)
    slopes = [sum(a*b for a,b in zip(row, delta)) for row in rows]
    extent = max(max(abs(value) for value in slopes), 1e-15)
    delta = [value / extent for value in delta]
    slopes = [value / extent for value in slopes]
    lower, upper = -max_displacement, max_displacement
    for value, slope in zip(projections, slopes):
        if abs(slope) < 1e-15:
            continue
        a, b = (-MARGIN - value) / slope, (MARGIN - value) / slope
        lower, upper = max(lower, min(a, b)), min(upper, max(a, b))
    if lower > upper:
        return None
    # Stay slightly inside the interval to avoid numerical boundary crossings.
    if boundary:
        distance = (lower if rng.random() < 0.5 else upper) * 0.999
    else:
        distance = rng.uniform(lower, upper) * 0.999
    return [value + distance * change for value, change in zip(state, delta)]


def sample_boundary_state(problem, rng):
    state = sample_state(problem, rng)
    for _ in range(3):
        moved = local_state(problem, state, rng, max_displacement=2., boundary=True)
        if moved is not None:
            state = moved
    return state
