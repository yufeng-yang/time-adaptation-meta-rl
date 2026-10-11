"""Final entry point: resume the retained ~55w model through ~64.5w."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tidyup_plan2.baseline3_gate_meta_rl.release_0_64p5w.reproduce import continue_training


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('episode', 'step'), default='episode')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    continue_training(args.mode, args.output_dir)
