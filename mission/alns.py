"""Adaptive destroy/repair search over feasible satellite formations."""
import math

from .annealing import acceptance_probability, validate_temperatures
from .sampling import sample_boundary_state, sample_state
from .search_state import FormationState
from .tabu import grid_neighbor

ALGORITHM = 'adaptive-large-neighborhood-search-v1'


def remove(state, satellites):
    """Return a partial formation; missing satellites contribute no baselines."""
    result = FormationState.__new__(FormationState)
    result.problem, result.x, result.states = state.problem, state.x[:], state.states[:]
    result.cells = [cells[:] for cells in state.cells]
    result.counts = [[counts.copy() for counts in planes] for planes in state.counts]
    for satellite in satellites:
        for epoch, cells in enumerate(result.cells):
            point = cells[satellite]
            if point is None:
                continue
            for counts, (a, b) in zip(result.counts[epoch], [(0, 1), (0, 2), (1, 2)]):
                result._adjust(counts, (0, 0), -1)
                for index, other in enumerate(cells):
                    if index == satellite or other is None:
                        continue
                    difference = point[a]-other[a], point[b]-other[b]
                    result._adjust(counts, difference, -1)
                    result._adjust(counts, (-difference[0], -difference[1]), -1)
            cells[satellite] = None
    result._score()
    return result


def insert(partial, satellite, state):
    """Score an insertion without reintroducing any other removed satellite."""
    if any(cells[satellite] is not None for cells in partial.cells):
        raise ValueError('Satellite must be removed before insertion')
    result = FormationState.__new__(FormationState)
    result.problem = partial.problem
    result.x, result.states = partial.x[:], partial.states[:]
    result.states[satellite] = list(state)
    for component, value in enumerate(state):
        result.x[component * result.problem.n_sat + satellite] = value
    result.problem.decode(result.x)
    result.cells = [cells[:] for cells in partial.cells]
    result.counts = [[counts.copy() for counts in planes] for planes in partial.counts]
    for epoch, cells in enumerate(result.cells):
        point = result._cell(state, epoch)
        if point is None:
            continue
        for counts, (a, b) in zip(result.counts[epoch], [(0, 1), (0, 2), (1, 2)]):
            result._adjust(counts, (0, 0), 1)
            for other in cells:
                if other is not None:
                    difference = point[a]-other[a], point[b]-other[b]
                    result._adjust(counts, difference, 1)
                    result._adjust(counts, (-difference[0], -difference[1]), 1)
        cells[satellite] = point
    result._score()
    return result


def select_group(current, size, mode, rng):
    population = list(range(current.problem.n_sat))
    if mode == 'random':
        return rng.sample(population, size)
    epoch = min(range(current.problem.n_meas), key=current.totals.__getitem__)
    cells = current.cells[epoch]
    if mode == 'related':
        anchor = cells[rng.randrange(len(cells))]
        if anchor is None:
            return rng.sample(population, size)
        population.sort(key=lambda i: sum(abs(a-b) for a, b in zip(cells[i], anchor))
                        + rng.random()*3 if cells[i] is not None else -1)
    else:
        # Low exclusive-baseline contribution is a cheap destroy heuristic.
        def contribution(i):
            point = cells[i]
            if point is None:
                return -1
            unique = 0
            for counts, (a, b) in zip(current.counts[epoch], [(0, 1), (0, 2), (1, 2)]):
                for j, other in enumerate(cells):
                    if j != i and other is not None:
                        unique += counts[(point[a]-other[a], point[b]-other[b])] == 1
            return unique + rng.random()*5
        population.sort(key=contribution)
    return population[:size]


