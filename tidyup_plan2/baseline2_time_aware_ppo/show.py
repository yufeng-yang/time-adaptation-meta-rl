from __future__ import annotations

# ========================= =====================
GATE_STATE = "open"
DEADLINE = 19
MODEL_PATH = "/home/yufeng.yang/codespace/time_adaptation/tidyup_plan2/baseline2_time_aware_ppo/artifacts/final_model.zip"
DETERMINISTIC = True
STEP_DELAY = 0.2
SHOW_LOCAL_OBS = True
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 1000
# ==================================================

import sys
import time
from pathlib import Path

import gymnasium as gym
import pygame
from stable_baselines3 import PPO

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
WORKSPACE_ROOT = PACKAGE_DIR.parent
MINIGRID_DIR = WORKSPACE_ROOT / "Minigrid"
for path in (WORKSPACE_ROOT, MINIGRID_DIR, PACKAGE_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tidyup_plan2.envs import ThreeRouteHiddenGateEnvV4
from train import DOnlyObservation, EVAL_DEADLINES, classify_route


ACTION_NAMES = ("left", "right", "forward")
MODEL_FILE = Path(MODEL_PATH)


def make_env() -> gym.Env:
    env: gym.Env = ThreeRouteHiddenGateEnvV4(
        render_mode="human",
        min_deadline=min(EVAL_DEADLINES),
        max_deadline=max(EVAL_DEADLINES),
        step_penalty=0.01,
        progress_scale=0.0,
    )
    return DOnlyObservation(env)


def reset_episode(env):
    return env.reset(
        options={
            "new_trial": True,
            "deadline": DEADLINE,
            "gate_state": GATE_STATE,
        }
    )


def main() -> None:
    model_file = MODEL_FILE.with_suffix(".zip")
    if not model_file.exists():
        raise FileNotFoundError(f"找不到模型：{model_file}")

    env = make_env()
    base = env.unwrapped
    base.show_local_obs = SHOW_LOCAL_OBS
    base.window_width = WINDOW_WIDTH
    base.window_height = WINDOW_HEIGHT
    base.screen_size = WINDOW_WIDTH

    model = PPO.load(str(MODEL_FILE), env=env, device="auto")
    observation, info = reset_episode(env)
    positions = [tuple(map(int, base.agent_pos))]
    env.render()

    print(
        f"加载 {model_file}\n"
        f"deadline={info['deadline']}  gate={info['gate_state']}\n"
        "策略自动走  R 重开  Esc 退出"
    )

    clock = pygame.time.Clock()
    episode_done = False
    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        observation, info = reset_episode(env)
                        positions = [tuple(map(int, base.agent_pos))]
                        episode_done = False
                        print(
                            f"重置：deadline={info['deadline']} "
                            f"gate={info['gate_state']}"
                        )

            if not running:
                break
            if episode_done:
                clock.tick(30)
                continue

            action, _ = model.predict(observation, deterministic=DETERMINISTIC)
            observation, reward, terminated, truncated, info = env.step(int(action))
            positions.append(tuple(map(int, base.agent_pos)))
            print(
                f"step={base.step_count} "
                f"action={ACTION_NAMES[int(action)]:<7} "
                f"pos={tuple(map(int, base.agent_pos))} "
                f"reward={reward:>6.2f} "
                f"remaining={info['remaining_steps']} "
                f"return={info['episode_return']:.2f}"
            )
            if terminated or truncated:
                episode_done = True
                print(
                    f"结束：success={info['success']} "
                    f"timeout={info['timeout']} "
                    f"route={classify_route(positions)} "
                    f"return={info['episode_return']:.3f}"
                )
                print("按 R 重新开始，或 Esc 退出")
            time.sleep(STEP_DELAY)
    except KeyboardInterrupt:
        print("\n已退出")
    finally:
        env.close()


if __name__ == "__main__":
    main()
