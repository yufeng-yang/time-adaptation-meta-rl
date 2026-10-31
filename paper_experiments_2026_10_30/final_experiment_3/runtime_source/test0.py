#!/home/yufeng.yang/Downloads/enter/envs/time_adapt_robosuite/bin/python
"""查看 Panda 的圆周 Pick-Place 自定义场景：内圈一球，外圈随机四球。

运行：python test0.py
随机动作：python test0.py --random-actions
原生场景：python test0.py --original
复现布局：python test0.py --seed 0

脚本自动使用 time_adapt_robosuite 环境。当前只搭建物理场景，
尚未添加隐藏价值、deadline、跨局记忆或训练。
"""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time


CONDA_PYTHON = Path(
    "/home/yufeng.yang/Downloads/enter/envs/time_adapt_robosuite/bin/python"
)


def main():
    # 即使从其他 Python 环境启动，也切换到本实验指定的解释器。
    if Path(sys.prefix).resolve() != CONDA_PYTHON.parent.parent.resolve():
        if not CONDA_PYTHON.is_file():
            raise SystemExit(f"找不到指定环境的 Python：{CONDA_PYTHON}")
        os.execv(
            str(CONDA_PYTHON),
            [str(CONDA_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--random-actions", action="store_true", help="让机械臂执行随机动作"
    )
    parser.add_argument(
        "--check", action="store_true", help="只检查创建、重置和执行动作，不打开窗口"
    )
    parser.add_argument("--original", action="store_true", help="查看原生 PickPlace 场景")
    parser.add_argument("--seed", type=int, default=None, help="固定随机种子；默认每次运行生成新布局")
    args = parser.parse_args()

    suite_version = version("robosuite")
    mujoco_version = version("mujoco")
    mujoco_release = tuple(int(part) for part in mujoco_version.split(".")[:2])
    if suite_version == "1.5.2" and mujoco_release >= (3, 10):
        raise SystemExit(
            f"版本不兼容：robosuite {suite_version} + MuJoCo {mujoco_version}。\n"
            "请在 time_adapt_robosuite 环境中运行：\n"
            f'  {CONDA_PYTHON} -m pip install "mujoco==3.9.0"\n'
            "然后重新运行此脚本，可先加 --check 做无窗口验证。"
        )

    # 窗口预览使用 GLFW；必须在导入 MuJoCo / robosuite 之前设置。
    if not args.check:
        os.environ["MUJOCO_GL"] = "glfw"

    import numpy as np
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config
    from radial_pick_place import RadialPickPlace

    print(f"Python: {sys.executable}")
    print(f"robosuite: {suite.__version__}")
    print(f"MuJoCo: {mujoco_version}")
    if not args.check:
        print("关闭窗口或按 Ctrl+C 退出。")

    control_freq = 20
    env_type = "PickPlace" if args.original else "RadialPickPlace"
    env = suite.make(
        env_name=env_type,
        robots="Panda",
        controller_configs=load_composite_controller_config(robot="Panda"),
        has_renderer=not args.check,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        **({"use_object_obs": True} if args.original else {}),
        renderer="mjviewer",
        render_camera=None,
        control_freq=control_freq,
        ignore_done=True,
        initialization_noise=None,
        seed=args.seed,
        renderer_config={"cam_config": {
            "lookat": [0, 0, 0.85], "distance": 1.65,
            "azimuth": 90, "elevation": -60,
        }},
    )
    try:
        env.reset()
        if isinstance(env, RadialPickPlace):
            print("金色 S：18 cm 内圈保底球；蓝色 A/B/C/D：36 cm 外圈随机四球；绿色浅盘：原点投放区。")
            print(json.dumps(env.layout_summary(), ensure_ascii=False, indent=2))
        low, high = env.action_spec
        hold_action = np.zeros_like(low)
        hold_action[-1] = low[-1]  # 保持夹爪打开。
        rng = np.random.default_rng(args.seed)

        if args.check:
            for _ in range(10):
                obs, reward, done, info = env.step(hold_action)
                if not all(np.all(np.isfinite(value)) for value in obs.values()):
                    raise RuntimeError("观测中出现非有限数值。")
                if not np.isfinite(reward):
                    raise RuntimeError("奖励中出现非有限数值。")
            print(f"检查通过：Panda + {env_type} 创建、重置和 10 步动作正常，未打开渲染窗口。")
            return

        while True:
            start = time.monotonic()
            action = (
                rng.uniform(low, high) if args.random_actions else hold_action
            )
            env.step(action)
            env.render()

            # robosuite 1.5.2 在第一次 step 时创建原生 MuJoCo 窗口。
            if not env.viewer.viewer.is_running():
                break
            time.sleep(max(0.0, 1.0 / control_freq - (time.monotonic() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
