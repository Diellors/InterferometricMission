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


def plane_tables(partial, epoch):
    """Exact single-insertion baseline counts at every plane cell."""
    points = [point for point in partial.cells[epoch] if point is not None]
    tables = []
    size = partial.problem.grid_size
    for plane, (a, b) in enumerate(PLANES):
        base = set(partial.counts[epoch][plane])
        table = {}
        for x, y in product(range(size), repeat=2):
            additions = {(0, 0)}
            for point in points:
                dx, dy = x-point[a], y-point[b]
                additions.update([(dx, dy), (-dx, -dy)])
            table[x, y] = len(base | additions)
        tables.append(table)
    return tables


class _BudgetExhausted(Exception):
    pass


def search(problem, incumbent, *, max_improvements=1, max_lp_calls=50000, epsilon=1e-5,
           progress=None):
    for name, value in [('max_improvements', max_improvements), ('max_lp_calls', max_lp_calls)]:
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    best = FormationState(problem, incumbent)
    initial_best = best.key[0]
    history, changes = [best.key[0]], []
    models, catalogs = {}, {}
    lp_calls, formation_scores, satellites_tested = 0, 1, 0
    reason = 'improvement_limit'
    def query(model, cell, reference=None):
        nonlocal lp_calls
        if lp_calls >= max_lp_calls:
            raise _BudgetExhausted
        lp_calls += 1
        return model.solve(cell, reference)
    try:
        while len(changes) < max_improvements:
            epoch = min(range(problem.n_meas), key=best.totals.__getitem__)
            if epoch not in catalogs:
                model = CellFeasibilityLP(problem, epoch, epsilon)
                models[epoch] = model
                catalog = {}
                for cell in product(range(problem.grid_size), repeat=3):
                    state = query(model, cell)
                    if state is not None:
                        catalog[cell] = state
                    if progress is not None and lp_calls % 1000 == 0:
                        progress(best, {'phase': 'catalog', 'lp_calls': lp_calls, 'epoch': epoch})
                catalogs[epoch] = catalog
            model, catalog = models[epoch], catalogs[epoch]
            minimum_count = min(sum(len(c) for c in planes) for planes in best.counts)
            improved = False
            for satellite in range(problem.n_sat):
                satellites_tested += 1
                partial = remove(best, [satellite])
                tables = plane_tables(partial, epoch)
                ranked = sorted(((tables[0][x, y]+tables[1][x, z]+tables[2][y, z], (x, y, z))
                                 for x, y, z in catalog), reverse=True)
                for score, cell in ranked:
                    if score <= minimum_count:
                        break
                    closest = query(model, cell, best.states[satellite])
                    for moved in [closest, catalog[cell]]:
                        if moved is None:
                            continue
                        trial = best.x[:]
                        for j, value in enumerate(moved):
                            trial[j*problem.n_sat+satellite] = value
                        candidate = best.replace(trial)
                        formation_scores += 1
                        if sum(len(c) for c in candidate.counts[epoch]) != score:
                            raise RuntimeError('Grid insertion table disagrees with the formation score')
                        if candidate.key[0] < best.key[0]-1e-12:
                            if problem.fitness(trial)[0] != candidate.key[0]:
                                raise RuntimeError('LP candidate failed full verification')
                            best = candidate
                            history.append(best.key[0])
                            changes.append({'satellite': satellite, 'epoch': epoch, 'cell': list(cell),
                                            'fitness': best.key[0], 'lp_calls': lp_calls})
                            improved = True
                            if progress is not None:
                                progress(best, {'phase': 'improved', 'lp_calls': lp_calls,
                                                'satellite': satellite, 'epoch': epoch})
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                reason = 'no_improvement_in_catalog'
                break
    except _BudgetExhausted:
        reason = 'lp_limit'
    if problem.fitness(best.x)[0] != best.key[0]:
        raise RuntimeError('LP search archive failed full verification')
    return {'x': best.x[:], 'fitness': best.key[0], 'initial_best': initial_best,
            'history': history, 'algorithm': ALGORITHM, 'evaluations': formation_scores,
            'verification_evaluations': 1+len(changes),
            'lp_grid_search': {'lp_calls': lp_calls, 'max_lp_calls': max_lp_calls,
                'max_improvements': max_improvements, 'epsilon': epsilon,
                'satellites_tested': satellites_tested, 'changes': changes,
                'catalog_sizes': {str(e): len(c) for e, c in catalogs.items()},
                'rejected_recoveries': sum(model.rejected_recoveries for model in models.values()),
                'stop_reason': reason}}
