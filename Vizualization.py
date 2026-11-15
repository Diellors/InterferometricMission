"""Plot an existing solution: python Vizualization.py results/easy/seed-42/easy.json"""
import argparse
import json
from pathlib import Path

from mission.problems import CASES, load_problem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    args = parser.parse_args()
    record = json.loads(args.result.read_text(encoding='utf-8'))
    case = next(key for key, value in CASES.items() if value[1] == record['problem'])
    import matplotlib.pyplot as plt
    load_problem(case).plot(record['x'])
    target = args.result.with_suffix('.png')
    plt.savefig(target, bbox_inches='tight', dpi=150)
    print(target)
    plt.close('all')


if __name__ == '__main__':
    main()
