"""Render a real policy rollout in the ideal Cartesian environment, headlessly.

This is a top-down state visualization, not a MuJoCo arm/controller rollout.
Frames reconstruct the exact planned movement and are checked against env.step.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from env import ACTIONS, LABELS, PickPlaceTrial, TaskConfig
from model import TransformerMetaPolicy


def trace_option(env, action, known):
    """Expand one option for display without changing the actual environment."""
    eef, balls = env.eef.copy(), env.balls.copy()
    remaining, holding = env.remaining, env.holding
    delivered = False
    frames = []

    def capture(phase, pause=1):
        frames.append(dict(episode=env.episode + 1, remaining=remaining,
                           eef=eef.copy(), balls=balls.copy(), holding=holding,
                           known=list(known), phase=phase, action=action,
                           delivered=delivered, repeats=pause))

    for kind, value in env.plan(action):
        if kind in ('move', 'wait'):
            start = eef.copy()
            distance = np.linalg.norm(value - start) if kind == 'move' else 0.
            needed = int(np.ceil(max(0., distance - 1e-10) / env.config.movement_m_per_step)) if kind == 'move' else value
            used = min(needed, remaining)
            for index in range(used):
                if kind == 'move' and distance:
                    eef = start + min(1., (index + 1) * env.config.movement_m_per_step / distance) * (value - start)
                    if holding >= 0:
                        balls[holding] = eef - [0, 0, .005]
                remaining -= 1
                phase = ('水平移动' if abs(value[2] - start[2]) < 1e-9 else
                         ('抬起夹爪' if value[2] > start[2] else '下降夹爪')) if kind == 'move' else '抓取 / 释放等待'
                capture(phase)
            if used < needed:
                break
        elif kind == 'contact':
            known[value] = float(env._scores[value])
            capture(f'接触 {LABELS[value]}，发现 {known[value]:g} 分', pause=8)
        elif kind == 'grasp':
            holding = value
            capture(f'抓住 {LABELS[value]}', pause=3)
        elif kind == 'release':
            balls[value] = env.plate.copy()
            holding = -1
            delivered = True
            capture(f'投放成功：{known[value]:g} 分，本局结束', pause=14)
    return frames


def rollout(checkpoint, deadline, seed):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    task = dict(saved['config']['task']['task'])
    task.setdefault('score_sampling', 'independent')
    env = PickPlaceTrial(TaskConfig(**task), seed=seed)
    obs = env.reset(deadline=deadline)
    model = TransformerMetaPolicy().eval()
    model.load_state_dict(saved['model'])
    hidden = model.initial_hidden()
    observations = [obs]
    known = [env.config.safe_score, None, None, None, None]
    frames = []

    def initial_frame():
        return dict(episode=env.episode + 1, remaining=env.remaining,
                    eef=env.eef.copy(), balls=env.balls.copy(), holding=-1,
                    known=list(known), phase=('新 Trial 开始：跨局记忆清零' if env.episode == 0 else
                                             '位置和时间重置；跨局记忆保留'),
                    action=None, delivered=False, repeats=14, history=copy.deepcopy(env.records))

    frames.append(initial_frame())
    with torch.no_grad():
        while True:
            tensor = torch.from_numpy(np.array(observations))[None]
            logits, _, _ = model.forward_episode(tensor, torch.tensor([tensor.shape[1]]), hidden)
            action = int(logits[0, -1].argmax())
            previous_records = copy.deepcopy(env.records)
            duration_before = env.remaining
            option_frames = trace_option(env, action, known)
            obs, reward, done, info = env.step(action)
            terminal = info.get('terminal_observation', obs)
            last = option_frames[-1]
            np.testing.assert_allclose(last['eef'], terminal[:3] * .5, atol=1e-7)
            np.testing.assert_allclose(last['balls'].ravel(), terminal[6:21] * .5, atol=1e-7)
            np.testing.assert_allclose(last['remaining'], terminal[21] * env.config.time_scale, atol=2e-5)
            assert last['holding'] + 1 == int(terminal[43:49].argmax())
            assert duration_before - last['remaining'] == info['duration']
            for frame in option_frames:
                frame['history'] = copy.deepcopy(env.records if frame['delivered'] else previous_records)
            if info['episode_done'] and not last['delivered']:
                last['history'] = copy.deepcopy(env.records)
                last['phase'] = '时间耗尽：本局未投放，得 0 分'
                last['repeats'] = 14
            frames.extend(option_frames)
            observations.append(terminal)
            if info['episode_done']:
                tensor = torch.from_numpy(np.array(observations))[None]
                _, _, hidden = model.forward_episode(tensor, torch.tensor([tensor.shape[1]]), hidden)
                observations = [obs]
                if not done:
                    frames.append(initial_frame())
            if done:
                break
    frames[-1]['repeats'] = 26
    metadata = dict(checkpoint=str(checkpoint.resolve()), checkpoint_phase=saved['phase'],
                    checkpoint_update=saved['update'], seed=seed, deadline=deadline,
                    scores=dict(zip(LABELS, env._scores.tolist())), positions=env.initial_balls.tolist(),
                    records=env.records, total_low_steps=env.total_low_steps,
                    rendering='Top-down ideal Cartesian state visualization; no MuJoCo joints, collisions or physical grasp simulation',
                    note='Score/known-value display is diagnostic only and is not an extra policy input. Playback pauses at contacts and episode boundaries.')
    return frames, metadata, env.layout


class Painter:
    size = (1180, 800)

    def __init__(self, metadata, layout):
        self.metadata, self.layout = metadata, layout
        self.regular = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        self.bold = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
        self.fonts = {}

    def font(self, size, bold=False):
        key = (size, bold)
        if key not in self.fonts:
            self.fonts[key] = ImageFont.truetype(self.bold if bold else self.regular, size, index=2)
        return self.fonts[key]

    def point(self, position):
        return (355 + float(position[0]) * 630, 430 - float(position[1]) * 630)

    def paint(self, frame):
        image = Image.new('RGB', self.size, '#F8FAFC')
        draw = ImageDraw.Draw(image)

        def text(x, y, value, size=19, color='#263241', bold=False, anchor=None):
            draw.text((x, y), str(value), fill=color, font=self.font(size, bold), anchor=anchor)

        text(38, 20, '无 BC 版本二 · 一个完整 Trial', 29, bold=True)
        text(40, 69, f'最佳检查点：PPO {self.metadata["checkpoint_update"]} 次更新  |  '
             f'deadline = {self.metadata["deadline"]} 步  |  seed = {self.metadata["seed"]}', 18, '#546477')
        draw.rounded_rectangle((64, 130, 648, 726), radius=8, fill='#FFFFFF', outline='#D9E0E6', width=2)
        for radius in (.18, .36):
            r = radius * 630
            for angle in range(0, 360, 9):
                draw.arc((355-r, 430-r, 355+r, 430+r), angle, angle+4, fill='#C6CED7', width=1)
        for index in range(-2, 3):
            value = index * .2
            x, y = self.point([value, value])
            text(x, 731, f'{value:g}', 13, '#7E8A97', anchor='mt')
            text(57, y, f'{value:g}', 13, '#7E8A97', anchor='rm')
        text(357, 759, '桌面中心坐标 · x / m', 15, '#657587', anchor='mt')
        text(65, 105, '俯视：球与夹爪的实际位置轨迹', 17, '#657587')
        r = self.layout.tray_radius * 630
        draw.ellipse((355-r, 430-r, 355+r, 430+r), fill='#DBEBDF', outline='#457C56', width=3)
        text(355, 483, '中央投放盘', 16, '#457C56', anchor='mt')
        for index, (position, original) in enumerate(zip(frame['balls'], self.metadata['positions'])):
            if np.linalg.norm(np.array(position[:2]) - original[:2]) > .01:
                gx, gy = self.point(original)
                draw.ellipse((gx-15, gy-15, gx+15, gy+15), outline='#CCD5DF', width=1)
            x, y = self.point(position)
            radius = 16
            color = '#B98A27' if index == 0 else '#39719F'
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color, outline='#FFFFFF', width=2)
            text(x, y-1, LABELS[index], 17, '#FFFFFF', True, anchor='mm')
            value = frame['known'][index]
            text(x, y+22, '?' if value is None else f'{value:g} 分', 17, '#263241', anchor='mt')
        x, y = self.point(frame['eef'])
        gap = 8 if frame['holding'] >= 0 else 16
        for side in (-1, 1):
            xx = x + side * gap
            draw.line((xx, y-18, xx, y+14), fill='#B3473E', width=4)
            draw.line((xx, y-18, x, y-18), fill='#B3473E', width=4)
        draw.ellipse((x-3, y-3, x+3, y+3), fill='#B3473E')
        text(708, 132, f'第 {frame["episode"]} / 3 局', 28, bold=True)
        used = self.metadata['deadline'] - frame['remaining']
        text(708, 185, f'本局用时  {used} / {self.metadata["deadline"]} 步', 22)
        draw.rounded_rectangle((709, 225, 1125, 236), radius=4, fill='#E1E7EC')
        if used:
            draw.rounded_rectangle((709, 225, 709 + 416 * used / self.metadata['deadline'], 236), radius=4, fill='#39719F')
        action = frame['action']
        label = '尚未选择' if action is None else (
            f'检查 {LABELS[action+1]}' if action < 4 else f'取回 {LABELS[action-4]}')
        text(708, 254, f'上层动作：{label}', 21)
        text(708, 290, frame['phase'], 17, '#B3473E' if '接触' in frame['phase'] else '#546477')
        text(708, 325, f'夹爪高度：{frame["eef"][2] * 100:.1f} cm（桌面上方）', 17, '#546477')
        draw.line((708, 369, 1125, 369), fill='#D9E0E6', width=1)
        text(708, 388, '已观察分值（回放辅助显示）', 19, bold=True)
        for index, value in enumerate(frame['known']):
            xx = 708 + (index % 3) * 143
            yy = 435 + (index // 3) * 47
            text(xx, yy, f'{LABELS[index]}：' + ('未知' if value is None else f'{value:g} 分'), 20)
        text(708, 540, f'Trial 已获得：{sum(record["score"] for record in frame["history"]):g} 分', 24, bold=True)
        for index in range(3):
            if index < len(frame['history']):
                record = frame['history'][index]
                detail = f'{record["ball"] or "超时"} · {record["score"]:g} 分 · {record["low_steps"]} 步'
            else:
                detail = '进行中' if index + 1 == frame['episode'] else '尚未开始'
            text(708, 589 + index * 35, f'第 {index+1} 局：{detail}', 19, '#546477')
        text(708, 718, '红色符号 = 夹爪；蓝色 = 外球', 16, '#657587')
        text(40, 779, '理想下层轨迹示意；不含机械臂关节、碰撞或真实抓取仿真。接触与局间重置处放慢播放。', 14, '#657587')
        return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path(__file__).resolve().parent / 'runs/v2_no_bc_seed0_1000/best.pt')
    parser.add_argument('--deadline', type=int, default=150)
    parser.add_argument('--seed', type=int, default=300000)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--fps', type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output_dir or args.checkpoint.parent / 'renders' / f'trial_D{args.deadline}_seed{args.seed}'
    output.mkdir(parents=True, exist_ok=True)
    frames, metadata, layout = rollout(args.checkpoint, args.deadline, args.seed)
    painter = Painter(metadata, layout)
    images = [painter.paint(frame) for frame in frames]
    durations = [round(1000 * frame['repeats'] / args.fps) for frame in frames]
    images[-1].save(output / 'poster.png')
    images[0].save(output / 'trial.gif', save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=True, disposal=2)
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        process = subprocess.Popen([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                                    '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1180x800',
                                    '-r', str(args.fps), '-i', '-', '-an', '-c:v', 'libx264',
                                    '-crf', '21', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                                    str(output / 'trial.mp4')], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for image, frame in zip(images, frames):
            data = image.tobytes()
            for _ in range(frame['repeats']):
                process.stdin.write(data)
        process.stdin.close()
        error = process.stderr.read().decode()
        if process.wait():
            raise RuntimeError(error)
    metadata['playback_fps'] = args.fps
    metadata['playback_seconds'] = sum(frame['repeats'] for frame in frames) / args.fps
    metadata['trace_frames_checked'] = len(frames)
    (output / 'trial.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(output=str(output.resolve()), **metadata), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
