"""Separate from-scratch entry: success before deadline, not ASAP.

Same architecture/tasks/BC routes as train.py, with discount-compensated terminal
step refunds and success rewards. Does not overwrite the original 80w result.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tidyup_plan2.baseline3_gate_meta_rl.episode_transformer_bc import train


def parse_args(argv=None):
    tokens = list(sys.argv[1:] if argv is None else argv)
    args = train.parse_args(tokens)
    # Only the new reward entry changes defaults; historical runs keep gamma=.99.
    if not any(t == "--gamma" or t.startswith("--gamma=") for t in tokens):
        args.gamma = 1.
    if not 0 < args.gamma <= 1:
        raise ValueError("Discount compensation requires 0 < gamma <= 1")
    if not any(t == "--output-dir" or t.startswith("--output-dir=") for t in tokens):
        args.output_dir = str(HERE / "runs/seed0")
    args.experiment_name = "episode_transformer_deadline_reach"
    args.training_source_paths = [str(HERE / "train.py"), str(HERE / "reward.py")]
    args.discount_compensated_terminal_refund = True
    args.report_discounted_return = True
    args.reward_definition = (
        "Per-step -.01 retained; at episode end refund all discounted step costs; "
        "compensate success bonus discount; danger/collision/timeout unchanged"
    )
    args.return_definition = (
        "Episode return with gamma=1 (no discount); step costs refunded; "
        "safe success=1, danger success=.8, collision-free timeout=-1"
        if args.gamma == 1 else
        "Discounted episode return; safe success=1 regardless of arrival step"
    )
    return args


if __name__ == "__main__":
    train.train(parse_args())
