"""Cost analysis for support ticket routing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.import_utils import load_config
from shared.cost import compute_cost_comparison

CONFIG = load_config(__file__)


def main():
    report = compute_cost_comparison(CONFIG)
    print(report)


if __name__ == "__main__":
    main()
