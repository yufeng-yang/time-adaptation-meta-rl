from __future__ import annotations

# ===================== 超参数 =====================
ENV_NAME = "ThreeRouteHiddenGateEnvV4"
DEADLINE = 17
GATE_STATE = "open"  # "open" / "closed"
SHOW_LOCAL_OBS = True
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 1000
# ==================================================

import sys
from pathlib import Path

import pygame

SCRIPT_DIR = Path(__file__).resolve().parent
TIDYUP_DIR = SCRIPT_DIR.parent
WORKSPACE_ROOT = TIDYUP_DIR.parent
MINIGRID_DIR = WORKSPACE_ROOT / "Minigrid"
for path in (WORKSPACE_ROOT, MINIGRID_DIR, TIDYUP_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import tidyup_plan2.envs as envs

EnvCls = getattr(envs, ENV_NAME)

ACTION_NAMES = ("left", "right", "forward")
KEY_TO_ACTION = {
    pygame.K_LEFT: 0,
    pygame.K_a: 0,
    pygame.K_RIGHT: 1,
    pygame.K_d: 1,
    pygame.K_UP: 2,
    pygame.K_w: 2,
}


def reset_env(env):
    return env.reset(
        options={
            "new_trial": True,
            "deadline": DEADLINE,
            "gate_state": GATE_STATE,
        }
    )


def print_status(env, observation: dict, info: dict, prefix: str = "") -> None:
    print(
        f"{prefix}"
        f"step={env.step_count} "
        f"pos={tuple(map(int, env.agent_pos))} "
        f"dir={int(env.agent_dir)} "
        f"obs.image={tuple(observation['image'].shape)} "
        f"remaining={info['remaining_steps']} "
        f"return={info['episode_return']:.2f} "
        f"gate={info['gate_state']}"
    )


def main() -> None:
    env = EnvCls(render_mode="human")
    env.show_local_obs = SHOW_LOCAL_OBS
    env.window_width = WINDOW_WIDTH
    env.window_height = WINDOW_HEIGHT
    env.screen_size = WINDOW_WIDTH
    observation, info = reset_env(env)
    env.render()
    pygame.key.set_repeat(150, 100)
    clock = pygame.time.Clock()

    print(
        f"{ENV_NAME} 手动控制\n"
        f"deadline={info['deadline']}  gate={info['gate_state']}\n"
        "↑/W 前进  ←/A 左转  →/D 右转  R 重置  Esc 退出"
    )
    print_status(env, observation, info, prefix="开局：")

    episode_done = False
    running = True
    try:
        while running:
            action = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        observation, info = reset_env(env)
                        episode_done = False
                        print_status(env, observation, info, prefix="重置：")
                    elif event.key in KEY_TO_ACTION and not episode_done:
                        action = KEY_TO_ACTION[event.key]

            if action is None:
                clock.tick(30)
                continue

            observation, reward, terminated, truncated, info = env.step(
                int(action)
            )
            print(
                f"action={ACTION_NAMES[action]:<7} "
                f"reward={reward:>5.2f}  "
                f"pos={tuple(map(int, env.agent_pos))}  "
                f"remaining={info['remaining_steps']}  "
                f"return={info['episode_return']:.2f}"
            )
            if terminated or truncated:
                episode_done = True
                print(
                    f"结束：success={info['success']} "
                    f"timeout={info['timeout']} "
                    f"danger={info['danger_visits']} "
                    f"return={info['episode_return']:.3f}"
                )
                print("按 R 重新开始，或 Esc 退出")
    except KeyboardInterrupt:
        print("\n已退出")
    finally:
        env.close()


if __name__ == "__main__":
    main()
