"""Version 2: pure PPO from random initialization, with no BC at any point."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import main


if __name__ == '__main__':
    main(['--no-bc', '--ppo-updates', '200', '--batch-trials', '32',
          '--eval-trials', '64', '--eval-every', '20', *sys.argv[1:]])
