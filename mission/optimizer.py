"""Run Simulated Annealing or our orbit-aware formation search."""
import argparse
import json
import math
import random
from .problems import CASES, ROOT, load_problem
from .sampling import direction, fit_state, local_state, sample_boundary_state, sample_state
from .search_state import FormationState

ALGORITHM = 'feasible-satellite-search-v2'

def initial_population(problem,size,rng):
    population = []
    for _ in range(size):
        states = []
        for satellite in range(problem.n_sat):
            states.append(sample_state(problem, rng))
        population.append([states[i][j] for j in range(6) for i in range(problem.n_sat)])
    return population


def mutate_satellite(problem, target, rng, max_displacement=0.2):
    """Move one satellite, limiting displacement at every observation epoch.

    Work in the same normalized coordinates as the evaluator. Scaling by
    propagated displacement avoids huge moves along unstable orbital directions.
    The remaining satellites are copied unchanged.
    """
    if not math.isfinite(max_displacement) or max_displacement <= 0:
        raise ValueError('max_displacement must be finite and positive')
    satellite = rng.randrange(problem.n_sat)
    state = [target[j * problem.n_sat + satellite] for j in range(6)]
    moved = local_state(problem, state, rng, max_displacement, targeted=rng.random() < 0.4)
    if moved is not None:
        trial = target[:]
        for component, value in enumerate(moved):
            trial[component * problem.n_sat + satellite] = value
        return trial
    delta_state = direction(problem, rng)
    extent = max(abs(sum(row[j] * delta_state[j] for j in range(6))) /
                 (1 if epoch == 0 else problem.inflation_factor)
                 for epoch, matrix in enumerate(problem.stms) for row in matrix[:3])
    scale = max_displacement / max(extent, 1e-12)
    lower, upper = problem.get_bounds()
    # Shorten the entire step at a bound, preserving the direction and its
    # propagated displacement limit. Clipping coordinates separately would not.
    for component, delta in enumerate(delta_state):
        index = component * problem.n_sat + satellite
        if delta > 0:
            scale = min(scale, (upper[index] - target[index]) / delta)
        elif delta < 0:
            scale = min(scale, (lower[index] - target[index]) / delta)
    trial = target[:]
    for component, delta in enumerate(delta_state):
        index = component * problem.n_sat + satellite
        trial[index] = min(upper[index], max(lower[index], target[index] + scale * delta))
    return trial


def replace_satellite(problem, target, state, rng):
    satellite = rng.randrange(problem.n_sat)
    trial = target[:]
    for component, value in enumerate(state):
        trial[component * problem.n_sat + satellite] = value
    return trial


def quality_key(problem, x):
    totals = [item['total'] for item in problem.observations(x)]
    # Keep the original worst-observation objective. Only ties use the other
    # observations, so improvements there can escape a flat objective plateau.
    return -min(totals), -sum(totals)


def resume_population(problem, population, path, rng):
    """Seed a new search with a verified incumbent and nearby formations."""
    from pathlib import Path
    record = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(record, dict) or record.get('problem') != CASES[problem.case][1]:
        raise ValueError('Resume result does not match the selected case')
    x = record.get('x')
    if not isinstance(x, list) or any(type(value) not in (int, float) for value in x):
        raise ValueError('Resume chromosome must be a list of numbers')
    problem.decode(x)
    score = record.get('fitness')
    if (type(score) not in (int, float) or not math.isfinite(score) or
            not math.isclose(problem.fitness(x)[0], score, rel_tol=0, abs_tol=1e-10)):
        raise ValueError('Resume fitness does not match the chromosome')
    result = [row[:] for row in population]
    result[0] = x[:]
    for index in range(1, max(1, len(result) // 2)):
        result[index] = mutate_satellite(problem, x, rng, max_displacement=0.3)
    return result


def evolve(problem,population,generations,rng):
    """Orbit-aware satellite search with shared elites and plateau tie-breaking.

    Every trial costs one evaluation. Total = population * (generations + 1).
    The primary objective and grid normalization are unchanged.
    """
    if len(population) < 5:
        raise ValueError('At least five individuals are required')
    if type(generations) is not int or generations < 0:
        raise ValueError('generations must be a nonnegative integer')
    caches = [FormationState(problem, x) for x in population]
    scores = [state.key for state in caches]
    history,evaluations = [min(scores)[0]],len(population)
    for generation in range(generations):
        updated,new_scores,new_caches = [],[],[]
        elite_index = min(range(len(scores)), key=scores.__getitem__)
        for i,target in enumerate(population):
            parent_index = elite_index if rng.random() < 0.35 else i
            parent = population[parent_index]
            choice = rng.random()
            if choice < 0.65:
                step = rng.choice([0.03, 0.06, 0.12, 0.25, 0.5, 0.9])
                trial = mutate_satellite(problem, parent, rng, max_displacement=step)
            elif choice < 0.8:
                trial = replace_satellite(problem, parent, sample_state(problem, rng), rng)
            elif choice < 0.9:
                trial = replace_satellite(problem, parent, sample_boundary_state(problem, rng), rng)
            else:
                a,b,c = rng.sample([j for j in range(len(population)) if j!=i],3)
                indices = [rng.randrange(problem.n_sat) for _ in range(3)]
                states = [[population[p][j * problem.n_sat + s] for j in range(6)]
                          for p, s in zip([a,b,c], indices)]
                factor = rng.uniform(0.3, 0.8)
                state = [x + factor * (y-z) for x,y,z in zip(*states)]
                trial = replace_satellite(problem, parent, fit_state(problem, state), rng)
            candidate = caches[parent_index].replace(trial)
            score = candidate.key
            evaluations += 1
            accept = score<=scores[i]
            updated.append(trial if accept else target)
            new_scores.append(score if accept else scores[i])
            new_caches.append(candidate if accept else caches[i])
        population,scores = updated,new_scores
        caches = new_caches
        history.append(min(scores)[0])
        print(f"Generation {generation+1}: {history[-1]:.9f}",flush=True)
    best = min(range(len(scores)),key=scores.__getitem__)
    return population[best],scores[best][0],history,evaluations

def run_search(problem, population, generations, rng, *, algorithm='annealing',
               initial_temperature=0.02, final_temperature=0.0001):
    """Use P*(G+1) scores for both methods; SA uses one chain and P trials/block."""
    if type(generations) is not int or generations < 0:
        raise ValueError('generations must be a nonnegative integer')
    if algorithm == 'annealing':
        from .annealing import ALGORITHM as SA_ALGORITHM, search
        result = search(problem, population, len(population) * generations, rng,
                        initial_temperature=initial_temperature, final_temperature=final_temperature,
                        record_every=len(population))
        result['algorithm'] = SA_ALGORITHM
        return result
    if algorithm == 'orbit':
        x, score, history, evaluations = evolve(problem, population, generations, rng)
        return {'x': x, 'fitness': score, 'initial_best': history[0], 'history': history,
                'evaluations': evaluations, 'algorithm': ALGORITHM}
    raise ValueError('algorithm must be annealing or orbit')


def solve(case,args):
    problem = load_problem(case)
    rng = random.Random(args.seed)
    initial = initial_population(problem,args.population,rng)
    if getattr(args, 'resume', None):
        initial = resume_population(problem, initial, args.resume, rng)
    result = run_search(problem, initial, args.generations, rng,
                        algorithm=getattr(args, 'algorithm', 'annealing'),
                        initial_temperature=getattr(args, 'initial_temperature', 0.02),
                        final_temperature=getattr(args, 'final_temperature', 0.0001))
    x, score, history = result['x'], result['fitness'], result['history']
    checked = problem.fitness(x)[0]
    if not math.isfinite(checked) or checked!=score or score>history[0]:
        raise RuntimeError("Result did not pass re-evaluation")
    out = ROOT / 'results'
    if result['algorithm'] != ALGORITHM:
        out = out / 'annealing'
    out = out / case / f'seed-{args.seed}'
    out.mkdir(parents=True,exist_ok=True)
    record = {"problem":CASES[case][1],"x":x,"fitness":score,
              "initial_best":history[0],"seed":args.seed,"population":args.population,
              "evaluations":result['evaluations'],
              "generations":args.generations,"history":history,
              "algorithm":result['algorithm'],
              "model":"independent-cr3bp-linearized-rk4-v1",
              "observation_totals":[o["total"] for o in problem.observations(x)]}
    if 'annealing' in result:
        record['annealing'] = result['annealing']
    if getattr(args, 'resume', None):
        record['resumed_from'] = str(args.resume)
    path = out/f"{case}.json"
    path.write_text(json.dumps(record,indent=2,allow_nan=False),encoding="utf-8")
    print(f"Saved {path}; fitness={score:.9f}",flush=True)
    if args.plot:
        try:
            fig = problem.plot(x)
        except ModuleNotFoundError as error:
            raise SystemExit("Result saved. Install matplotlib to generate plots.") from error
        fig.savefig(out/f"{case}.png",bbox_inches="tight",dpi=150)
        import matplotlib.pyplot as plt
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",choices=[*CASES,"all"])
    parser.add_argument("--config")
    parser.add_argument("--population",type=int)
    parser.add_argument("--generations",type=int)
    parser.add_argument("--seed",type=int)
    parser.add_argument('--algorithm', choices=['annealing', 'orbit'])
    parser.add_argument('--initial-temperature', type=float)
    parser.add_argument('--final-temperature', type=float)
    parser.add_argument('--resume', help='Verified result JSON to use as an incumbent')
    parser.add_argument("--plot",action="store_true")
    args = parser.parse_args()
    config = {}
    if args.config:
        from pathlib import Path
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if not isinstance(config,dict):
            parser.error("Configuration must be an object")
    args.case = args.case if args.case is not None else config.get("case","easy")
    if args.case not in [*CASES,"all"]:
        parser.error("Invalid case")
    args.algorithm = args.algorithm if args.algorithm is not None else config.get('algorithm', 'annealing')
    if args.algorithm not in ('annealing', 'orbit'):
        parser.error('algorithm must be annealing or orbit')
    for key, default in [('initial_temperature', 0.02), ('final_temperature', 0.0001)]:
        if getattr(args, key) is None:
            setattr(args, key, config.get(key, default))
    if args.algorithm == 'annealing':
        from .annealing import validate_temperatures
        try:
            validate_temperatures(args.initial_temperature, args.final_temperature)
        except ValueError as error:
            parser.error(str(error))
    for key,default in [("population",80),("generations",100),("seed",42)]:
        value = getattr(args,key)
        value = config.get(key,default) if value is None else value
        if type(value) is not int:
            parser.error(f"{key} must be an integer")
        setattr(args,key,value)
    if args.population<5 or args.generations<0:
        parser.error("population >= 5 and generations >= 0 required")
    if args.resume and args.case == 'all':
        parser.error('--resume requires a single selected case')
    for case in CASES if args.case=="all" else [args.case]:
        try:
            solve(case,args)
        except (OSError, UnicodeError, ValueError, RuntimeError) as error:
            parser.error(str(error))
