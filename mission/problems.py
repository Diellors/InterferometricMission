"""Scenario inputs and our independent formation evaluator."""
from pathlib import Path
from functools import lru_cache
from math import isfinite
from .dynamics import transition_matrices
from .baselines import project_to_grid,displacement_set
ROOT = Path(__file__).resolve().parents[1]
CASES = {"easy":("dro","tiny-observatory-in-dro"),
         "medium":("dro","large-observatory-in-dro"),
         "hard":("halo","large-observatory-in-halo")}
# Scenario data supplied by the challenge, not independently discovered orbits.
DRO = ([0.8965084609449406,0.,0.,1.3951082e-14,0.47481794884853445,0.],2.6905181697222775)
HALO = ([1.0829551779304256,-6.923280193602759e-27,-0.20231744561698364,
         9.78887918274808e-15,-0.20102644884016105,2.4744866465838825e-14],2.383491010514447)

class MissionProblem:
    def __init__(self,case):
        self.case = case
        self.n_sat = 5 if case=="easy" else 40
        self.grid_size = 11 if case=="easy" else 21
        self.n_meas,self.inflation_factor = 3,1.23
        self.ic,self.period = HALO if case=="hard" else DRO
        self.times = [0.,self.period,2*self.period]
        self.stms = transition_matrices(self.ic,self.times)

    def get_bounds(self):
        hi = [1.]*(3*self.n_sat)+[10.]*(3*self.n_sat)
        return [-v for v in hi],hi

    def decode(self,x):
        lo,hi = self.get_bounds()
        if len(x)!=len(lo) or any(not isfinite(v) or v<a or v>b for v,a,b in zip(x,lo,hi)):
            raise ValueError("Expected a finite chromosome of length 6*N within bounds")
        return [[x[j*self.n_sat+i] for j in range(6)] for i in range(self.n_sat)]

    def observations(self,x):
        states = self.decode(x)
        observations = []
        for epoch,matrix in enumerate(self.stms):
            scale = 1. if epoch==0 else self.inflation_factor
            points = [[sum(row[j]*s[j] for j in range(6))/scale for row in matrix[:3]] for s in states]
            cells = project_to_grid(points,self.grid_size)
            planes = []
            for a,b in [(0,1),(0,2),(1,2)]:
                occupied = {(p[a],p[b]) for p in cells}
                differences = displacement_set(occupied)
                planes.append({"cells":occupied,"baselines":differences,
                               "fill":len(differences)/(2*self.grid_size-1)**2})
            observations.append({"remaining":len(cells),"planes":planes,
                                 "total":sum(p["fill"] for p in planes)})
        return observations

    def fitness(self,x):
        return [-min(o["total"] for o in self.observations(x))]

    def plot(self,x):
        from .plotting import plot_formation
        return plot_formation(self,x)

@lru_cache(maxsize=3)
def load_problem(case):
    if case not in CASES:
        raise ValueError("Unknown scenario")
    return MissionProblem(case)
