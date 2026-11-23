"""Coverage-ranked critical-grid search with linear-programming feasibility."""
from collections import Counter
from itertools import product
import math

from .alns import remove
from .sampling import geometry
from .search_state import FormationState

ALGORITHM = 'lp-guided-grid-best-response-v1'
PLANES = [(0, 1), (0, 2), (1, 2)]


class CellFeasibilityLP:
    """Find a feasible continuous state in one specified observation cell.

    Whitening improves conditioning. For a reference state, minimize the maximum
    projected displacement at all other observation epochs. Interior margins and
    recovered-state checks protect against grid-boundary numerical errors.
    """
    def __init__(self, problem, epoch, epsilon=1e-5):
        if type(epoch) is not int or not 0 <= epoch < problem.n_meas:
            raise ValueError('Invalid observation epoch')
        if not math.isfinite(epsilon) or not 0 < epsilon < 1/(4*problem.grid_size):
            raise ValueError('epsilon must be positive and smaller than 1/(4*grid_size)')
        try:
            from ortools.linear_solver import pywraplp
        except ImportError as error:
            raise ImportError('LP grid search requires: python -m pip install -r requirements-lp.txt') from error
        self.problem, self.epoch, self.epsilon = problem, epoch, epsilon
        self.rows, lower = geometry(problem)
        inverse = [[0.]*6 for _ in range(6)]
        for j in range(6):
            for i in range(5, -1, -1):
                inverse[i][j] = ((i == j)-sum(lower[k][i]*inverse[k][j]
                                             for k in range(i+1, 6)))/lower[i][i]
        self.inverse = inverse
        transformed = [[sum(row[k]*inverse[k][j] for k in range(6)) for j in range(6)]
                       for row in self.rows]
        self.solver = pywraplp.Solver.CreateSolver('GLOP')
        if self.solver is None:
            raise RuntimeError('The installed OR-Tools does not provide GLOP')
        self.optimal = pywraplp.Solver.OPTIMAL
        infinity = self.solver.infinity()
        self.variables = [self.solver.NumVar(-infinity, infinity, f'u{i}') for i in range(6)]
        distance = self.solver.NumVar(0., infinity, 'distance')
        self.targets, self.proximity = [], []
        def row_constraint(row, lo, hi):
            constraint = self.solver.RowConstraint(lo, hi, '')
            for variable, value in zip(self.variables, row):
                constraint.SetCoefficient(variable, value)
            return constraint
        for row in transformed:
            row_constraint(row, -1+epsilon, 1-epsilon)
        for row in transformed[3*epoch:3*epoch+3]:
            self.targets.append(row_constraint(row, -1., 1.))
        for index, row in enumerate(transformed[:3*problem.n_meas]):
            if index//3 == epoch:
                continue
            pair = []
            for sign in [1, -1]:
                constraint = row_constraint([sign*v for v in row], -infinity, infinity)
                constraint.SetCoefficient(distance, -1.)
                pair.append(constraint)
            self.proximity.append((index, pair))
        self.solver.Objective().SetCoefficient(distance, 1.)
        self.solver.Objective().SetMinimization()
        self.calls, self.rejected_recoveries = 0, 0
        self.statuses = Counter()

    def solve(self, cell, reference=None):
        if len(cell) != 3 or any(type(c) is not int or not 0 <= c < self.problem.grid_size for c in cell):
            raise ValueError('Expected three grid cell indices')
        if reference is not None and (len(reference) != 6 or not all(math.isfinite(v) for v in reference)):
            raise ValueError('Reference state must contain six finite values')
        size, center = self.problem.grid_size, self.problem.grid_size//2
        epsilon = self.epsilon
        for constraint, index in zip(self.targets, cell):
            lo = -1+epsilon if index == 0 else 2*(index-center)/size+epsilon
            hi = min(1-epsilon, 2*(index+1-center)/size-epsilon)
            constraint.SetBounds(lo, hi)
        for index, pair in self.proximity:
            value = sum(a*b for a, b in zip(self.rows[index], reference)) if reference is not None else None
            pair[0].SetUb(value if value is not None else self.solver.infinity())
            pair[1].SetUb(-value if value is not None else self.solver.infinity())
        self.calls += 1
        status = self.solver.Solve()
        self.statuses[status] += 1
        if status != self.optimal:
            return None
        values = [variable.solution_value() for variable in self.variables]
        state = [sum(a*b for a, b in zip(row, values)) for row in self.inverse]
        projected = [sum(a*b for a, b in zip(row, state)) for row in self.rows]
        if not all(math.isfinite(v) and abs(v) < 1 for v in projected):
            self.rejected_recoveries += 1
            return None
        point = projected[3*self.epoch:3*self.epoch+3]
        recovered_cell = tuple(int(v*size/2+center) for v in point)
        if recovered_cell != tuple(cell):
            self.rejected_recoveries += 1
            return None
        return state


