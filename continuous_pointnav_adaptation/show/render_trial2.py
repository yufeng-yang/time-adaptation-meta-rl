"""Load the Trial 2 PPO weights and render one TimeGoalV2 episode."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from continuous_pointnav_adaptation.train_time_goal_v2 import (  # noqa: E402
    DEADLINE,
    RewardOnlyWrapper,
)
from continuous_pointnav_adaptation.envs import TimeGoalV2Env  # noqa: E402


TRIAL_DIR = Path(__file__).resolve().parents[1] / "trial2"


def main() -> None:
    with (TRIAL_DIR / "vec_normalize.pkl").open("rb") as file:
        normalizer = pickle.load(file)
    normalizer.training = False
    normalizer.norm_reward = False
    model = PPO.load(TRIAL_DIR / "final_model")

    env = RewardOnlyWrapper(
        TimeGoalV2Env(
            render_mode="human",
            deadlines=(DEADLINE,),
            collision_penalty=0.1,
            progress_reward_scale=0.2,
        )
    )
    observation, info = env.reset(seed=0, options={"deadline": DEADLINE})
    env.render()
    print("deadline:", info["deadline"])

    total_reward = 0.0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        normalized = normalizer.normalize_obs(
            np.asarray(observation, dtype=np.float32)
        )
        action, _ = model.predict(normalized, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

    print(
        f"steps={steps} success={info['success']} timeout={info['timeout']} "
        f"danger_visits={info['danger_visits']} return={total_reward:.3f}"
    )
    env.close()


if __name__ == "__main__":
    main()
