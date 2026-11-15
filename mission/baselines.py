"""Exact directed baseline support and grid quantization."""
def project_to_grid(points, size):
    if size < 3 or size % 2 != 1:
        raise ValueError("Grid must have an odd size >= 3")
    return [tuple(int(v * size / 2 + size // 2) for v in p)
            for p in points if all(-1 < v < 1 for v in p)]

def displacement_set(cells):
    return {(a-c, b-d) for a,b in cells for c,d in cells}
