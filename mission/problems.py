"""Load the unmodified ESA problem definitions."""
import importlib.util
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ROOT / 'vendor' / 'SpOC3-main' / 'Challenge 2 Interferometric Mission' / 'problems'
CASES = {
    'easy': ('golomb_simple', 'tiny-observatory-in-dro'),
    'medium': ('golomb_medium', 'large-observatory-in-dro'),
    'hard': ('golomb_hard', 'large-observatory-in-halo'),
}


@lru_cache(maxsize=3)
def load_problem(case):
    name, _ = CASES[case]
    path = PROBLEMS / f'{name}.py'
    if not path.is_file():
        raise FileNotFoundError(f'ESA problem definition missing: {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.udp
