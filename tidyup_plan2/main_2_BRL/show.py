"""Render the five-episode TEST.py policy trial (closed gate, D=17)."""

from __future__ import annotations

# ----------------------------- display settings -----------------------------
MODEL_PATH = (
    "/home/yufeng.yang/codespace/time_adaptation/tidyup_plan2/"
    "main_2_BRL/artifacts/test_closed_d17/brl_closed_d17.zip"
)
DETERMINISTIC = True
STEP_DELAY = 0.25
EPISODE_DELAY = 0.8
SHOW_LOCAL_OBS = True
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 1000
SEED = 0
# ---------------------------------------------------------------------------

import sys
import time
from pathlib import Path

import pygame
from stable_baselines3 import PPO


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
WORKSPACE_ROOT = PACKAGE_DIR.parent
MINIGRID_DIR = WORKSPACE_ROOT / "Minigrid"
for path in (WORKSPACE_ROOT, MINIGRID_DIR, PACKAGE_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from belief_trial_env import ExactBeliefTrialEnv
from tidyup_plan2.envs import ThreeRouteHiddenGateEnvV3


ACTION_NAMES = ("left", "right", "forward")
GATE_STATE = "closed"
DEADLINE = 17
EPISODES_PER_TRIAL = 5
PRIOR_OPEN = 0.5
DANGER_PENALTY = 0.1

# Observation layout: 150 semantic cells + 4 direction values, then these.
REMAINING_TIME_INDEX = 154
BELIEF_INDEX = 155
EPISODE_BOUNDARY_INDEX = 156
EPISODES_REMAINING_INDEX = 157


def make_env() -> ExactBeliefTrialEnv:
    base_env = ThreeRouteHiddenGateEnvV3(
        render_mode="human",
        min_deadline=DEADLINE,
        max_deadline=DEADLINE,
        step_penalty=0.01,
        progress_scale=0.02,
        danger_penalty=DANGER_PENALTY,
        defer_danger_penalty=True,
    )
    return ExactBeliefTrialEnv(
        base_env,
        deadlines=(DEADLINE,),
        episodes_per_trial=EPISODES_PER_TRIAL,
        prior_open=PRIOR_OPEN,
        fixed_gate_state=GATE_STATE,
    )


def classify_route(positions: list[tuple[int, int]]) -> str:
    if any(y == 6 for _, y in positions):
        return "detour"
    if any(y == 1 for _, y in positions):
        return "danger"
    if any(y == 5 for _, y in positions):
        return "direct"
    return "other"


def print_trial_header(observation, info: dict) -> None:
    print(
        "\nNew TEST trial\n"
        f"  gate={info['gate_state']}  deadline={info['deadline']}\n"
        f"  belief={observation[BELIEF_INDEX]:.1f}\n"
        f"  episodes={EPISODES_PER_TRIAL}\n"
        "Controls: Space=pause/resume, N=single step, R=new trial, Esc=quit"
    )


def main() -> None:
    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")

    env = make_env()
    base = env.unwrapped
    base.show_local_obs = SHOW_LOCAL_OBS
    base.window_width = WINDOW_WIDTH
    base.window_height = WINDOW_HEIGHT
    base.screen_size = WINDOW_WIDTH
    model = PPO.load(str(model_file), env=env, device="auto")

    observation, info = env.reset(seed=SEED)
    positions = [tuple(map(int, base.agent_pos))]
    current_episode = 1
    trial_done = False
    paused = False
    single_step = False
    env.render()
    print_trial_header(observation, info)

    clock = pygame.time.Clock()
    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        print("Paused" if paused else "Running")
                    elif event.key == pygame.K_n:
                        paused = True
                        single_step = True
                    elif event.key == pygame.K_r:
                        observation, info = env.reset(seed=SEED)
                        positions = [tuple(map(int, base.agent_pos))]
                        current_episode = 1
                        trial_done = False
                        paused = False
                        single_step = False
                        env.render()
                        print_trial_header(observation, info)

            if not running:
                break
            if trial_done or (paused and not single_step):
                clock.tick(30)
                continue
            single_step = False

            belief_before = float(observation[BELIEF_INDEX])
            episodes_remaining = int(observation[EPISODES_REMAINING_INDEX])
            action, _ = model.predict(
                observation,
                deterministic=DETERMINISTIC,
            )
            action = int(action)
            next_observation, reward, terminated, truncated, next_info = env.step(
                action
            )
            subepisode_done = bool(next_info.get("subepisode_done", False))

            if not subepisode_done:
                positions.append(tuple(map(int, base.agent_pos)))
                print(
                    f"episode={current_episode}/{EPISODES_PER_TRIAL} "
                    f"step={base.step_count:02d}/{DEADLINE} "
                    f"action={ACTION_NAMES[action]:<7s} "
                    f"pos={tuple(map(int, base.agent_pos))} "
                    f"reward={reward:+.2f} "
                    f"belief={belief_before:.1f}->{next_observation[BELIEF_INDEX]:.1f} "
                    f"episodes_left={episodes_remaining}"
                )
            else:
                if terminated or truncated:
                    summary = next_info["subepisodes"][-1]
                else:
                    summary = next_info["completed_subepisode"]
                if summary["success"]:
                    positions.append(tuple(map(int, base.GOAL_POSITION)))
                print(
                    f"EPISODE {current_episode} END: "
                    f"success={summary['success']} "
                    f"timeout={summary['timeout']} "
                    f"route={classify_route(positions)} "
                    f"return={summary['return']:.3f} "
                    f"belief={summary['belief_open']:.1f}"
                )

            observation, info = next_observation, next_info
            env.render()
            pygame.display.set_caption(
                "TEST BRL | "
                f"episode {current_episode}/{EPISODES_PER_TRIAL} | "
                f"belief={observation[BELIEF_INDEX]:.1f} | "
                f"D-t={int(observation[REMAINING_TIME_INDEX])}"
            )

            if subepisode_done:
                if terminated or truncated:
                    trial_done = True
                    print(
                        f"TRIAL END: return={next_info['trial_return']:.3f}. "
                        "Press R to replay or Esc to quit."
                    )
                else:
                    current_episode += 1
                    positions = [tuple(map(int, base.agent_pos))]
                    time.sleep(EPISODE_DELAY)
            else:
                time.sleep(STEP_DELAY)
    except KeyboardInterrupt:
        print("\nExited")
    finally:
        env.close()


if __name__ == "__main__":
    main()
