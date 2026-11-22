"""Simulated Annealing with orbit-feasible proposals and a best-so-far archive.

Only Python's standard library is used. Current state may worsen; the returned
archive never does. The objective and baseline normalization are unchanged.
"""
import math

from .sampling import local_state, sample_boundary_state, sample_state
from .search_state import FormationState

ALGORITHM = 'simulated-annealing-v1'


def validate_temperatures(initial, final):
    for name, value in [('initial_temperature', initial), ('final_temperature', final)]:
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if final > initial:
        raise ValueError('final_temperature must not exceed initial_temperature')


def acceptance_probability(delta, temperature):
    """Metropolis probability for a minimized objective."""
    if type(delta) not in (int, float) or not math.isfinite(delta):
        raise ValueError('delta must be finite')
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    return 1. if delta <= 0 else math.exp(-delta / temperature)


def propose(problem, x, rng, step_size):
    """Change one satellite using a feasible chord or a new feasible state."""
    satellite = rng.randrange(problem.n_sat)
    state = [x[j * problem.n_sat + satellite] for j in range(6)]
    choice = rng.random()
    if choice < 0.85:
        moved = local_state(problem, state, rng, step_size, targeted=rng.random() < 0.4)
        state = moved if moved is not None else sample_state(problem, rng)
    elif choice < 0.95:
        state = sample_state(problem, rng)
    else:
        state = sample_boundary_state(problem, rng)
    trial = x[:]
    for component, value in enumerate(state):
        trial[component * problem.n_sat + satellite] = value
    return trial


def search(problem, population, iterations, rng, *, initial_temperature=0.02,
           final_temperature=0.0001, record_every=1, proposal=None):
    """Run one chain initialized from the best of the provided candidates.

    Exactly len(population) + iterations candidate scores are computed. History
    stores the archive initially, every record_every trials, and at the final
    trial. A proposal injection is available for controlled algorithm tests.
    """
    validate_temperatures(initial_temperature, final_temperature)
    if not population:
        raise ValueError('At least one initial candidate is required')
    if type(iterations) is not int or iterations < 0:
        raise ValueError('iterations must be a nonnegative integer')
    if type(record_every) is not int or record_every <= 0:
        raise ValueError('record_every must be a positive integer')
    current = min((FormationState(problem, x) for x in population), key=lambda state: state.key)
    best = current
    history, current_history = [best.key[0]], [current.key[0]]
    temperatures = [initial_temperature]
    accepted, uphill = 0, 0
    make_trial = propose if proposal is None else proposal
    for step in range(iterations):
        fraction = step / max(1, iterations - 1)
        temperature = math.exp(math.log(initial_temperature) + fraction *
                               (math.log(final_temperature) - math.log(initial_temperature)))
        radius = 0.03 + 0.47 * math.sqrt(temperature / initial_temperature)
        candidate = current.replace(make_trial(problem, current.x, rng, radius))
        delta = candidate.key[0] - current.key[0]
        accept = delta <= 0 or rng.random() < acceptance_probability(delta, temperature)
        if accept:
            current = candidate
            accepted += 1
            uphill += int(delta > 0)
        if candidate.key < best.key:
            best = candidate
        if (step + 1) % record_every == 0 or step + 1 == iterations:
            history.append(best.key[0])
            current_history.append(current.key[0])
            temperatures.append(temperature)
    return {'x': best.x[:], 'fitness': best.key[0], 'initial_best': history[0],
            'history': history, 'evaluations': len(population) + iterations,
            'annealing': {'iterations': iterations, 'record_every': record_every,
                          'initial_temperature': initial_temperature,
                          'final_temperature': final_temperature,
                          'accepted_moves': accepted, 'uphill_moves': uphill,
                          'current_history': current_history, 'temperatures': temperatures}}
