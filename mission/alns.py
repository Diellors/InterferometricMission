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


def search(problem, incumbent, max_evaluations, rng, *, pool_size=6,
           initial_temperature=0.003, final_temperature=0.00005, target=-1.55,
           segment_length=25, restart_every=150, min_destroy=4, max_destroy=8, progress=None):
    """Budget counts initial, destroy and partial insertion scores, not just finals.

    Incomplete repairs are discarded. A final independent full verification is
    outside the score budget. Progress callbacks receive complete archived states.
    """
    for name, value, minimum in [('max_evaluations', max_evaluations, 1),
                                 ('pool_size', pool_size, 2),
                                 ('segment_length', segment_length, 1),
                                 ('restart_every', restart_every, 1),
                                 ('min_destroy', min_destroy, 1), ('max_destroy', max_destroy, 1)]:
        if type(value) is not int or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    validate_temperatures(initial_temperature, final_temperature)
    if max_destroy < min_destroy:
        raise ValueError('max_destroy must be >= min_destroy')
    if type(target) not in (int, float) or not math.isfinite(target) or target > 0:
        raise ValueError('target must be finite and nonpositive')
    current = best = FormationState(problem, incumbent)
    operators = [(mode, repair) for mode in ['random', 'related', 'redundant']
                 for repair in ['local', 'wide']]
    weights, rewards, uses = [1.]*6, [0.]*6, [0]*6
    total_uses = [0]*6
    history, checkpoints = [best.key[0]], [1]
    evaluations, iteration, accepted, uphill, restarts = 1, 0, 0, 0, 0
    partial_scores = 0
    while evaluations < max_evaluations and best.key[0] > target:
        if iteration and iteration % restart_every == 0:
            current = best
            restarts += 1
        index = rng.choices(range(6), weights=weights)[0]
        mode, repair = operators[index]
        group = select_group(current, rng.randint(min(min_destroy, problem.n_sat),
                                                 min(max_destroy, problem.n_sat)), mode, rng)
        rng.shuffle(group)
        candidate = remove(current, group)
        evaluations += 1
        partial_scores += 1
        complete = True
        for satellite in group:
            if max_evaluations-evaluations < pool_size:
                complete = False
                break
            options = [current.states[satellite]]
            # The original state is always an option; the other options can
            # rebuild the satellite rather than merely undoing destruction.
            for _ in range(pool_size-1):
                if rng.random() < (0.9 if repair == 'local' else 0.55):
                    proposal = grid_neighbor(problem, current.x, satellite, rng,
                                             radius=0.25 if repair == 'local' else 1.2,
                                             interval_limit=12 if repair == 'local' else 80)
                    if proposal is not None:
                        options.append([proposal[0][j*problem.n_sat+satellite] for j in range(6)])
                        continue
                options.append(sample_boundary_state(problem, rng) if rng.random() < 0.5
                               else sample_state(problem, rng))
            candidate = min((insert(candidate, satellite, state) for state in options), key=lambda s: s.key)
            evaluations += len(options)
            partial_scores += len(options)
        if not complete:
            break
        previous_key = current.key
        fraction = (iteration % restart_every) / max(1, restart_every-1)
        temperature = initial_temperature * (final_temperature/initial_temperature)**fraction
        delta = candidate.key[0]-current.key[0]
        accept = delta <= 0 or rng.random() < acceptance_probability(delta, temperature)
        reward = 0.05
        if accept:
            current = candidate
            accepted += 1
            uphill += delta > 0
            reward = 2. if candidate.key < previous_key else 0.5
        if candidate.key < best.key:
            best = candidate
            reward = 5.
        uses[index] += 1
        total_uses[index] += 1
        rewards[index] += reward
        iteration += 1
        history.append(best.key[0])
        checkpoints.append(evaluations)
        if iteration % segment_length == 0:
            weights = [max(0.1, 0.8*w+0.2*r/u) if u else w
                       for w, r, u in zip(weights, rewards, uses)]
            rewards, uses = [0.]*6, [0]*6
            if progress is not None:
                progress(best, evaluations, iteration)
    if problem.fitness(best.x)[0] != best.key[0]:
        raise RuntimeError('ALNS archive failed full verification')
    return {'x': best.x[:], 'fitness': best.key[0], 'initial_best': history[0],
            'history': history, 'evaluations': evaluations, 'verification_evaluations': 1,
            'algorithm': ALGORITHM, 'alns': {'iterations': iteration, 'pool_size': pool_size,
                'initial_temperature': initial_temperature, 'final_temperature': final_temperature,
                'target': target, 'segment_length': segment_length, 'restart_every': restart_every,
                'min_destroy': min_destroy, 'max_destroy': max_destroy,
                'accepted': accepted, 'uphill_moves': uphill, 'restarts': restarts,
                'partial_scores': partial_scores, 'weights': weights, 'operator_uses': total_uses,
                'operators': [f'{a}/{b}' for a, b in operators],
                'checkpoint_evaluations': checkpoints,
                'stop_reason': 'target_reached' if best.key[0] <= target else 'evaluation_limit'}}
