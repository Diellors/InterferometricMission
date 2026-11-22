"""Grid-aware Tabu Search with reverse-move memory and aspiration."""
import math

from .sampling import MARGIN, direction, geometry, sample_state
from .search_state import FormationState

ALGORITHM = 'grid-aware-tabu-search-v1'


def signature(problem, state):
    """Projected cells define a phenotype; raw coordinate changes may not."""
    cells = []
    for epoch, matrix in enumerate(problem.stms):
        scale = 1 if epoch == 0 else problem.inflation_factor
        point = [sum(a*b for a,b in zip(row, state)) / scale for row in matrix[:3]]
        if not all(-1 < v < 1 for v in point):
            return None
        cells.append(tuple(int(v * problem.grid_size / 2 + problem.grid_size // 2) for v in point))
    return tuple(cells)


