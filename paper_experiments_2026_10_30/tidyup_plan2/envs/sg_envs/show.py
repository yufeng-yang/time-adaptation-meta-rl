"""Render a short random-action smoke test for the continuous gate map."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from point_gate_goal import PointThreeRouteGateEnv
else:
    from .point_gate_goal import PointThreeRouteGateEnv


def main() -> None:
    env = PointThreeRouteGateEnv(render_mode="human")
    observation, info = env.reset(
        seed=0,
        options={"new_trial": True, "gate_state": "closed", "deadline": 200},
    )
    print("observation shape:", observation.shape)
    print("reset info:", info)

    for _ in range(info["deadline"]):
        observation, reward, cost, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        if terminated or truncated:
            break

    print("final info:", info)
    env.close()


if __name__ == "__main__":
    main()
