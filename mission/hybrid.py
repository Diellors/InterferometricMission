"""SA exploration followed by greedy local refinement, with paired moves."""
import math

from .annealing import propose, search as anneal, validate_temperatures
from .sampling import MARGIN, direction, geometry, local_state, sample_state
from .search_state import FormationState

ALGORITHM = 'sa-local-pair-search-v1'


def propose_pair(problem, x, rng, step_size):
    """Change two distinct satellites while keeping their mean state fixed.

    For interior states, a common direction and its opposite preserve the mean
    and define a chord constrained by both satellites at every observation.
    At a feasibility face, the fallback uses two independent feasible moves;
    that fallback does not promise centroid preservation.
    """
    if problem.n_sat < 2:
        return propose(problem, x, rng, step_size)
    if not math.isfinite(step_size) or step_size <= 0:
        raise ValueError('step_size must be finite and positive')
    first, second = rng.sample(range(problem.n_sat), 2)
    a = [x[j * problem.n_sat + first] for j in range(6)]
    b = [x[j * problem.n_sat + second] for j in range(6)]
    rows, _ = geometry(problem)
    delta = direction(problem, rng, targeted=rng.random() < 0.4)
    slopes = [sum(p*q for p,q in zip(row, delta)) for row in rows]
    extent = max(max(abs(value) for value in slopes), 1e-15)
    delta = [value / extent for value in delta]
    lower, upper = -step_size, step_size
    for state, sign in [(a, 1), (b, -1)]:
        for row, slope in zip(rows, slopes):
            value = sum(p*q for p,q in zip(row, state))
            slope = sign * slope / extent
            if abs(value) >= MARGIN:
                # Robust fallback for an incumbent on/outside a feasibility face.
                trial = x[:]
                for satellite, original in [(first, a), (second, b)]:
                    moved = local_state(problem, original, rng, step_size)
                    moved = sample_state(problem, rng) if moved is None else moved
                    for j, coordinate in enumerate(moved):
                        trial[j * problem.n_sat + satellite] = coordinate
                return trial
            if abs(slope) > 1e-15:
                u, v = (-MARGIN - value) / slope, (MARGIN - value) / slope
                lower, upper = max(lower, min(u, v)), min(upper, max(u, v))
    if lower > upper:
        return x[:]
    distance = rng.uniform(lower, upper) * 0.999
    trial = x[:]
    for j, change in enumerate(delta):
        trial[j * problem.n_sat + first] = a[j] + distance * change
        trial[j * problem.n_sat + second] = b[j] - distance * change
    return trial


def mixed_proposal(problem, x, rng, step_size):
    if rng.random() < 0.3:
        return propose_pair(problem, x, rng, step_size)
    return propose(problem, x, rng, step_size)


def search(problem, incumbent, max_evaluations, rng, *, annealing_fraction=0.7,
           initial_temperature=0.0005, final_temperature=0.00001, record_every=100):
    """Spend an exact candidate-score budget, including both phase setups.

    Five identical initial candidates match the SA comparison setup. Rebuilding
    the local-phase cache counts as one more evaluation. Final re-evaluation
    is additional, as in the other project search implementations.
    """
    if type(max_evaluations) is not int or max_evaluations < 6:
        raise ValueError('max_evaluations must be an integer >= 6')
    if type(record_every) is not int or record_every < 1:
        raise ValueError('record_every must be a positive integer')
    if not math.isfinite(annealing_fraction) or not 0 <= annealing_fraction <= 1:
        raise ValueError('annealing_fraction must be between zero and one')
    validate_temperatures(initial_temperature, final_temperature)
    problem.decode(incumbent)
    annealing_steps = int((max_evaluations - 6) * annealing_fraction)
    local_steps = max_evaluations - 6 - annealing_steps
    result = anneal(problem, [incumbent[:] for _ in range(5)], annealing_steps, rng,
                    initial_temperature=initial_temperature, final_temperature=final_temperature,
                    record_every=record_every, proposal=mixed_proposal)
    current = FormationState(problem, result['x'])
    history = result['history'][:]
    accepted = 0
    for step in range(local_steps):
        radius = rng.choice([0.005, 0.01, 0.02, 0.04, 0.08, 0.16])
        trial = mixed_proposal(problem, current.x, rng, radius)
        candidate = current.replace(trial)
        if candidate.key < current.key:
            current = candidate
            accepted += 1
        if (step + 1) % record_every == 0 or step + 1 == local_steps:
            history.append(current.key[0])
    if problem.fitness(current.x)[0] != current.key[0]:
        raise RuntimeError('Hybrid result failed full evaluation')
    return {'x': current.x[:], 'fitness': current.key[0], 'initial_best': history[0],
            'history': history, 'evaluations': max_evaluations, 'verification_evaluations': 1,
            'algorithm': ALGORITHM, 'hybrid': {
                'annealing_fraction': annealing_fraction, 'annealing_steps': annealing_steps,
                'local_steps': local_steps, 'setup_evaluations': 6,
                'accepted_local_moves': accepted, 'annealing': result['annealing']}}
