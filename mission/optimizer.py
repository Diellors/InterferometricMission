"""Our differential evolution optimizer for interferometric formations."""
import argparse
import json
import math
import random
from .problems import CASES, ROOT, load_problem

def initial_population(problem,size,rng):
    population = []
    for _ in range(size):
        states = []
        for satellite in range(problem.n_sat):
            state = [rng.uniform(-1,1) for _ in range(6)]
            extent = max(abs(sum(row[j]*state[j] for j in range(6))) /
                         (1 if epoch==0 else problem.inflation_factor)
                         for epoch,matrix in enumerate(problem.stms) for row in matrix[:3])
            scale = rng.uniform(0.3,0.95)/max(extent,1.)
            states.append([v*scale for v in state])
        population.append([states[i][j] for j in range(6) for i in range(problem.n_sat)])
    return population

def evolve(problem,population,generations,rng):
    """DE/rand/1/bin: mutation, forced crossover, synchronous greedy selection."""
    lo,hi = problem.get_bounds()
    scores = [problem.fitness(x)[0] for x in population]
    history,evaluations = [min(scores)],len(population)
    for generation in range(generations):
        updated,new_scores = [],[]
        for i,target in enumerate(population):
            a,b,c = rng.sample([j for j in range(len(population)) if j!=i],3)
            factor,forced = rng.uniform(0.4,0.9),rng.randrange(len(target))
            trial = [min(hi[j],max(lo[j],population[a][j]+factor*(population[b][j]-population[c][j])))
                     if j==forced or rng.random()<0.7 else v for j,v in enumerate(target)]
            score = problem.fitness(trial)[0]
            evaluations += 1
            accept = score<=scores[i]
            updated.append(trial if accept else target)
            new_scores.append(score if accept else scores[i])
        population,scores = updated,new_scores
        history.append(min(scores))
        print(f"Generation {generation+1}: {history[-1]:.9f}",flush=True)
    best = min(range(len(scores)),key=scores.__getitem__)
    return population[best],scores[best],history,evaluations

def solve(case,args):
    problem = load_problem(case)
    rng = random.Random(args.seed)
    initial = initial_population(problem,args.population,rng)
    x,score,history,evaluations = evolve(problem,initial,args.generations,rng)
    checked = problem.fitness(x)[0]
    if not math.isfinite(checked) or checked!=score or score>history[0]:
        raise RuntimeError("Result did not pass re-evaluation")
    out = ROOT/"results"/case/f"seed-{args.seed}"
    out.mkdir(parents=True,exist_ok=True)
    record = {"problem":CASES[case][1],"x":x,"fitness":score,
              "initial_best":history[0],"seed":args.seed,"evaluations":evaluations,
              "generations":args.generations,"history":history,
              "model":"independent-cr3bp-linearized-rk4-v1",
              "observation_totals":[o["total"] for o in problem.observations(x)]}
    path = out/f"{case}.json"
    path.write_text(json.dumps(record,indent=2),encoding="utf-8")
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
    for key,default in [("population",80),("generations",100),("seed",42)]:
        value = getattr(args,key)
        value = config.get(key,default) if value is None else value
        if type(value) is not int:
            parser.error(f"{key} must be an integer")
        setattr(args,key,value)
    if args.population<5 or args.generations<0:
        parser.error("population >= 5 and generations >= 0 required")
    for case in CASES if args.case=="all" else [args.case]:
        solve(case,args)
