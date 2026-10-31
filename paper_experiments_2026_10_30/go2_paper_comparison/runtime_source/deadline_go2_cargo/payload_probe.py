"""MuJoCo-only payload test of a frozen local Go2 locomotion policy.

Never imports the Unitree SDK, connects hardware, trains weights, or sets a
mass-dependent speed limit. Payloads have identical box size but real varying
mass/inertia, rigidly attached above the torso. Matched trials share exactly
the same empty-robot warmup state and action history before loading.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import time
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import torch
from torch import nn

LOCAL_ROOT = Path('/home/yufeng.yang/codespace/unitree_sdk2/example/unitree_go2')
DEFAULT_POLICY = LOCAL_ROOT / 'jit_artifacts/policy_jit.pt'
DEFAULT_CONFIG = LOCAL_ROOT / 'onpolicy/runs/onpolicy/full_20260424_103256_091127/cfgs.pkl'
CONTROL_DT = .02
PHYSICS_DT = .002


def load_policy(path):
    if path == DEFAULT_POLICY or 'jit' in path.name:
        return torch.jit.load(str(path), map_location='cpu').eval()
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = checkpoint['actor_state_dict']
    keys = sorted([k for k in state if k.startswith('mlp.') and k.endswith('.weight')],
                  key=lambda k: int(k.split('.')[1]))
    layers = []
    for i, key in enumerate(keys):
        weight = state[key]
        layers.append(nn.Linear(weight.shape[1], weight.shape[0]))
        if i < len(keys) - 1:
            layers.append(nn.ELU())
    model = nn.Sequential(*layers)
    model.load_state_dict({k[len('mlp.'):]:v for k,v in state.items() if k.startswith('mlp.')}, strict=True)
    return model.eval()


def build_model(payload_kg):
    asset = Path(__file__).parent / 'assets/go2'
    root = ET.parse(asset / 'go2.xml').getroot()
    root.find('compiler').set('meshdir', str((asset / 'assets').resolve()))
    ET.SubElement(root, 'option', timestep=str(PHYSICS_DT), integrator='implicitfast',
                  gravity='0 0 -9.81', iterations='50')
    ET.SubElement(root, 'visual').append(ET.Element('global', offwidth='960', offheight='640'))
    collision = root.find("default/default/default[@class='collision']/geom")
    # Robot-ground contacts only: match training's disabled self-collisions.
    collision.set('contype', '2')
    collision.set('conaffinity', '1')
    collision.set('friction', '1 .005 .0001')
    world = root.find('worldbody')
    ET.SubElement(world, 'light', pos='0 0 6', dir='0 0 -1', castshadow='false')
    ET.SubElement(world, 'geom', name='ground', type='plane', size='50 50 .1',
                  rgba='.32 .35 .38 1', friction='1 .005 .0001', contype='1', conaffinity='2')
    if payload_kg > 0:
        base = world.find("body[@name='base_link']")
        body = ET.SubElement(base, 'body', name='payload', pos='0 0 .14')
        ET.SubElement(body, 'geom', name='cargo', type='box', size='.14 .11 .07',
                      mass=str(payload_kg), rgba='.2 .55 .9 1', contype='0', conaffinity='0')
    # Put the unchanged PD law inside MuJoCo's implicit integrator. Applying
    # -kd*qvel externally as qfrc_applied makes stiff leg damping explicit and
    # can create numerical instability unrelated to controller/load quality.
    actuators = ET.SubElement(root, 'actuator')
    for leg in ('FR','FL','RR','RL'):
        for part in ('hip','thigh','calf'):
            limit = 35.55 if part == 'calf' else 23.7
            ET.SubElement(actuators, 'position', name=f'{leg}_{part}_pd', joint=f'{leg}_{part}_joint',
                          kp='20', kv='.5', forcerange=f'{-limit} {limit}')
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


class FrozenController:
    def __init__(self, model, policy, env_cfg, obs_cfg):
        self.model, self.policy, self.cfg = model, policy, env_cfg
        self.scales = obs_cfg['obs_scales']
        names = env_cfg['joint_names']
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        if any(i < 0 for i in joints):
            raise ValueError('Robot and policy joint names do not match')
        self.qpos_ids = model.jnt_qposadr[joints]
        self.qvel_ids = model.jnt_dofadr[joints]
        self.default = np.array([env_cfg['default_joint_angles'][n] for n in names])
        # Limits from the local Go2 URDF. All conditions use these same limits.
        self.torque_limits = np.array([35.55 if 'calf' in n else 23.7 for n in names])
        self.actuator_ids = [mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,n.replace('_joint','_pd')) for n in names]
        # Controller gains come from this checkpoint's configuration, not a load schedule.
        for aid in self.actuator_ids:
            model.actuator_gainprm[aid,0] = env_cfg['kp']
            model.actuator_biasprm[aid,1:3] = [-env_cfg['kp'],-env_cfg['kd']]
        self.last_action = np.zeros(12)

    def reset(self, data, seed):
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, data)
        data.qpos[:7] = [*self.cfg['base_init_pos'], *self.cfg['base_init_quat']]
        data.qpos[self.qpos_ids] = self.default + rng.normal(0, .005, 12)
        self.last_action[:] = 0
        mujoco.mj_forward(self.model, data)

    @torch.inference_mode()
    def step(self, data, command):
        rotation = data.xmat[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')].reshape(3,3)
        # Free-joint angular velocity is expressed in the local body frame.
        angular = data.qvel[3:6]
        gravity = rotation.T @ np.array([0., 0., -1.])
        scales = self.scales
        obs = np.concatenate([angular * scales['ang_vel'], gravity,
                              np.asarray(command) * [scales['lin_vel'], scales['lin_vel'], scales['ang_vel']],
                              (data.qpos[self.qpos_ids] - self.default) * scales['dof_pos'],
                              data.qvel[self.qvel_ids] * scales['dof_vel'], self.last_action]).astype(np.float32)
        action = self.policy(torch.from_numpy(obs)[None])[0].numpy()
        action = np.clip(action, -self.cfg['clip_actions'], self.cfg['clip_actions'])
        executed = self.last_action if self.cfg['simulate_action_latency'] else action
        targets = self.default + self.cfg['action_scale'] * executed
        data.ctrl[self.actuator_ids] = targets
        saturation = []
        for _ in range(round(CONTROL_DT / PHYSICS_DT)):
            requested = self.cfg['kp'] * (targets - data.qpos[self.qpos_ids]) - self.cfg['kd'] * data.qvel[self.qvel_ids]
            saturation.append(np.mean(np.abs(requested) >= self.torque_limits))
            mujoco.mj_step(self.model, data)
        self.last_action = action.copy()
        return float(np.mean(saturation))


def attitude(data):
    w, x, y, z = data.qpos[3:7]
    roll = np.arctan2(2 * (w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2 * (w*y - z*x), -1, 1))
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2*(y*y + z*z))
    return np.degrees([roll, pitch, yaw])


def matched_trial(policy, cfgs, seed, masses, seconds, command):
    env_cfg, obs_cfg = cfgs[:2]
    warmup_model = build_model(0.)
    warmup = mujoco.MjData(warmup_model)
    low = FrozenController(warmup_model, policy, env_cfg, obs_cfg)
    low.reset(warmup, seed)
    for _ in range(150):  # Three seconds of empty walking before all payloads.
        low.step(warmup, command)
    warmup_attitude = attitude(warmup)
    if warmup.qpos[2] < .15 or np.max(np.abs(warmup_attitude[:2])) > 45:
        return dict(seed=seed, warmup_failed=True, warmup_pose=warmup.qpos[:7].tolist()), []
    matched = []
    traces = []
    for mass in masses:
        model = build_model(mass)
        data = mujoco.MjData(model)
        controller = FrozenController(model, policy, env_cfg, obs_cfg)
        data.qpos[:] = warmup.qpos
        data.qvel[:] = warmup.qvel
        data.time = warmup.time
        controller.last_action = low.last_action.copy()
        mujoco.mj_forward(model, data)
        initial = data.qpos[:2].copy()
        heading = np.radians(warmup_attitude[2])
        direction = np.array([np.cos(heading), np.sin(heading)])
        lateral_direction = np.array([-direction[1], direction[0]])
        rows = []
        failure_time = None
        bad_steps = 0
        arrivals = {2: None, 6: None}
        for index in range(round(seconds / CONTROL_DT)):
            saturated = controller.step(data, command)
            elapsed = (index + 1) * CONTROL_DT
            roll, pitch, yaw = attitude(data)
            delta = data.qpos[:2] - initial
            progress = float(delta @ direction)
            lateral = float(delta @ lateral_direction)
            speed = float(data.qvel[:2] @ direction)
            bad = data.qpos[2] < .15 or max(abs(roll), abs(pitch)) > 45
            bad_steps = bad_steps + 1 if bad else 0
            if failure_time is None and bad_steps >= 10:
                failure_time = elapsed
            for distance in arrivals:
                if arrivals[distance] is None and progress >= distance and abs(lateral) <= .75 and failure_time is None:
                    arrivals[distance] = elapsed
            rows.append([elapsed, progress, lateral, speed, data.qpos[2], roll, pitch, yaw, saturated])
            if not np.isfinite(data.qpos).all():
                raise RuntimeError('Nonfinite simulation state')
        array = np.array(rows)
        steady = array[:, 0] >= 2.
        record = dict(seed=seed, payload_kg=mass, robot_mass_kg=float(warmup_model.body_mass.sum()),
                      total_mass_kg=float(model.body_mass.sum()), failed=failure_time is not None,
                      failure_seconds=failure_time, distance_m=float(array[-1,1]),
                      average_progress_mps=float(array[-1,1] / seconds),
                      steady_progress_mps=float(array[steady,3].mean()),
                      max_lateral_m=float(np.abs(array[:,2]).max()),
                      rms_roll_deg=float(np.sqrt(np.mean(array[steady,5] ** 2))),
                      rms_pitch_deg=float(np.sqrt(np.mean(array[steady,6] ** 2))),
                      torque_saturation_fraction=float(array[:,8].mean()),
                      arrival_2m_seconds=arrivals[2], arrival_6m_seconds=arrivals[6])
        matched.append(record)
        traces.append(array)
    return dict(seed=seed, warmup_failed=False, conditions=matched), traces


def summarize(records, masses):
    summary = []
    for mass in masses:
        selected = [r for r in records if r['payload_kg'] == mass]
        if not selected:
            continue
        row = dict(payload_kg=mass, count=len(selected), failure_count=sum(r['failed'] for r in selected))
        for key in ('average_progress_mps', 'steady_progress_mps', 'rms_roll_deg', 'rms_pitch_deg',
                    'max_lateral_m', 'torque_saturation_fraction'):
            values = [r[key] for r in selected]
            row[key] = dict(mean=float(np.mean(values)), std=float(np.std(values)))
        for distance in (2,6):
            arrivals = [r[f'arrival_{distance}m_seconds'] for r in selected if r[f'arrival_{distance}m_seconds'] is not None]
            row[f'arrival_{distance}m'] = dict(success_count=len(arrivals),
                    mean_seconds=float(np.mean(arrivals)) if arrivals else None,
                    std_seconds=float(np.std(arrivals)) if arrivals else None)
        summary.append(row)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, default=DEFAULT_POLICY)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--masses', type=float, nargs='+', default=[0.,1.,3.,5.,8.,10.])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seconds', type=float, default=20.)
    parser.add_argument('--vx', type=float, default=.5)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; refusing overwrite')
    args.output.mkdir(parents=True)
    shutil.copy2(__file__, args.output / 'payload_probe_source.py')
    shutil.copy2(args.policy, args.output / args.policy.name)
    shutil.copy2(args.config, args.output / 'controller_cfgs.pkl')
    with args.config.open('rb') as file:
        cfgs = pickle.load(file)
    torch.set_num_threads(1)
    policy = load_policy(args.policy)
    config = dict(policy=str(args.policy), policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
                  cfg_source=str(args.config), policy_provenance='existing local Genesis-trained policy; not factory Unitree Sport controller',
                  env_cfg=cfgs[0], obs_cfg=cfgs[1], command=[args.vx,0.,0.],
                  payloads_kg=args.masses, repeats=args.repeats, seconds=args.seconds,
                  physics_dt=PHYSICS_DT, control_dt=CONTROL_DT, mujoco=mujoco.__version__,
                  controller_implementation='position-actuator PD with implicitfast integration; unchanged checkpoint gains',
                  protocol='matched three-second empty warmup, then rigidly attach mass with the same initial qpos/qvel/action history',
                  payload_box_half_size_m=[.14,.11,.07], payload_center_on_torso_m=[0.,0.,.14],
                  failure='base height <0.15m or |roll/pitch| >45 degrees for >=0.2s; no automatic reset',
                  metrics='20-second progress includes failures; speed is not averaged only over surviving conditions',
                  caution='MuJoCo sim-to-sim transfer; small joint-pose perturbations only; not real Go2 payload validation')
    (args.output / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
    start = time.monotonic()
    records, warmups = [], []
    traces = {}
    for seed in range(args.repeats):
        trial, arrays = matched_trial(policy,cfgs,seed,args.masses,args.seconds,[args.vx,0.,0.])
        warmups.append({k:v for k,v in trial.items() if k!='conditions'})
        for record, array in zip(trial.get('conditions',[]), arrays):
            records.append(record)
            traces[f'seed{seed}_mass{record["payload_kg"]:g}'] = array
            print(json.dumps(record), flush=True)
    result = dict(warmups=warmups, summary=summarize(records,args.masses), records=records,
                  elapsed_seconds=time.monotonic()-start)
    (args.output / 'results.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(args.output / 'traces.npz',**traces)
    if records:
        with (args.output / 'measurements.csv').open('w') as file:
            writer=csv.DictWriter(file,fieldnames=records[0].keys());writer.writeheader();writer.writerows(records)
    print('SUMMARY',json.dumps(result['summary']),flush=True)
    print('WARMUPS',json.dumps(warmups),flush=True)


if __name__ == '__main__':
    main()
