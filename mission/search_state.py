"""Incremental baseline counts for single-satellite search moves.

This accelerates the existing objective; it does not rescale or replace it.
Counts include duplicate projected satellites. Their multiplicities change the
counts but not the support, which is identical to the evaluator's occupied grid.
"""
from collections import Counter


class FormationState:
    def __init__(self, problem, x):
        self.problem = problem
        self.x = list(x)
        self.states = problem.decode(x)
        self.cells = [[self._cell(state, epoch) for state in self.states]
                      for epoch in range(problem.n_meas)]
        self.counts = []
        for cells in self.cells:
            planes = []
            for a, b in [(0, 1), (0, 2), (1, 2)]:
                points = [(cell[a], cell[b]) for cell in cells if cell is not None]
                planes.append(Counter((p[0]-q[0], p[1]-q[1]) for p in points for q in points))
            self.counts.append(planes)
        self._score()

    def _cell(self, state, epoch):
        problem = self.problem
        scale = 1 if epoch == 0 else problem.inflation_factor
        point = [sum(row[j]*state[j] for j in range(6)) / scale
                 for row in problem.stms[epoch][:3]]
        if not all(-1 < value < 1 for value in point):
            return None
        return tuple(int(value * problem.grid_size / 2 + problem.grid_size // 2) for value in point)

    def _score(self):
        denominator = (2 * self.problem.grid_size - 1) ** 2
        # Match the evaluator's floating point operation order exactly.
        self.totals = [sum(len(counts) / denominator for counts in planes) for planes in self.counts]
        self.key = -min(self.totals), -sum(self.totals)

    @staticmethod
    def _adjust(counts, difference, delta):
        value = counts.get(difference, 0) + delta
        if value:
            counts[difference] = value
        else:
            counts.pop(difference, None)

    def replace(self, x):
        """Return a new cache; never mutate a parent's baseline counts."""
        states = self.problem.decode(x)
        changed = [index for index, (old, new) in enumerate(zip(self.states, states)) if old != new]
        if not changed:
            return self
        if len(changed) == 2:
            # Reuse the single-satellite update twice. Rebuilding the entire
            # formation would discard the speed advantage for paired proposals.
            result = self
            for satellite in changed:
                intermediate = result.x[:]
                for component in range(6):
                    index = component * self.problem.n_sat + satellite
                    intermediate[index] = x[index]
                result = result.replace(intermediate)
            return result
        if len(changed) != 1:
            return FormationState(self.problem, x)
        satellite = changed[0]
        result = FormationState.__new__(FormationState)
        result.problem, result.x, result.states = self.problem, list(x), states
        result.cells, result.counts = [], []
        for epoch, cells in enumerate(self.cells):
            old, new = cells[satellite], self._cell(states[satellite], epoch)
            updated_cells = cells[:]
            updated_cells[satellite] = new
            result.cells.append(updated_cells)
            if old == new:
                result.counts.append(self.counts[epoch])
                continue
            planes = []
            for plane, (a, b) in enumerate([(0, 1), (0, 2), (1, 2)]):
                counts = self.counts[epoch][plane].copy()
                for point, delta in [(old, -1), (new, 1)]:
                    if point is None:
                        continue
                    self._adjust(counts, (0, 0), delta)
                    for index, other in enumerate(cells):
                        if index == satellite or other is None:
                            continue
                        difference = (point[a]-other[a], point[b]-other[b])
                        self._adjust(counts, difference, delta)
                        self._adjust(counts, (-difference[0], -difference[1]), delta)
                planes.append(counts)
            result.counts.append(planes)
        result._score()
        return result
