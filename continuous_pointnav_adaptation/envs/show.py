"""Render a short random-action smoke test for TimeGoalV2."""

from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from continuous_pointnav_adaptation.envs.time_goal_v2 import TimeGoalV2Env


def main() -> None:
    env = TimeGoalV2Env(render_mode="human")
    observation, info = env.reset(seed=0, options={"deadline": 275})
    env.render()
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
