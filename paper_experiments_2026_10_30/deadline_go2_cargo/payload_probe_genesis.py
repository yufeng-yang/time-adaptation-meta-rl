"""Matched payload experiment in the local policy's original Genesis engine.

No SDK imports, hardware connections, training, or artificial load-speed rule.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import time
import xml.etree.ElementTree as ET
import numpy as np
import torch
from payload_probe import load_policy, summarize

ARTIFACT = Path('/home/yufeng.yang/codespace/Prepare_to_transfer_final/RTSS_Artifact')
DEFAULT_WEIGHTS = ARTIFACT / 'checkpoints/Unitree Go2 Sim/Full/weights'
ROBOT_URDF = ARTIFACT / 'extra_resources/Genesis/genesis/assets/urdf/go2/urdf/go2.urdf'
EPSILON_MASS = 1e-6


def payload_urdf(output):
    tree = ET.parse(ROBOT_URDF)
    root = tree.getroot()
    for mesh in root.findall('.//mesh'):
        mesh.set('filename', str((ROBOT_URDF.parent / mesh.get('filename')).resolve()))
    body = ET.SubElement(root, 'link', name='test_payload')
    inertia = ET.SubElement(body, 'inertial')
    ET.SubElement(inertia, 'origin', xyz='0 0 0', rpy='0 0 0')
    # Initial 1 kg carrier; every environment changes this to epsilon for warmup.
    ET.SubElement(inertia, 'mass', value='1')
    xx, yy, zz = .28, .22, .14
    ET.SubElement(inertia, 'inertia', ixx=str((yy*yy+zz*zz)/12), iyy=str((xx*xx+zz*zz)/12),
                  izz=str((xx*xx+yy*yy)/12), ixy='0', ixz='0', iyz='0')
    visual = ET.SubElement(body, 'visual')
    ET.SubElement(ET.SubElement(visual, 'geometry'), 'box', size='.28 .22 .14')
    ET.SubElement(ET.SubElement(visual, 'material', name='cargo_blue'), 'color', rgba='.2 .55 .9 1')
    joint = ET.SubElement(root, 'joint', name='test_payload_weld', type='fixed')
    ET.SubElement(joint, 'parent', link='base')
    ET.SubElement(joint, 'child', link='test_payload')
    ET.SubElement(joint, 'origin', xyz='0 0 .14', rpy='0 0 0')
    tree.write(output, encoding='unicode')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy', type=Path, default=DEFAULT_WEIGHTS/'Full_8000.pt')
    parser.add_argument('--config', type=Path, default=DEFAULT_WEIGHTS/'cfgs.pkl')
    parser.add_argument('--masses', type=float, nargs='+', default=[0.,1.,3.,5.,8.,10.])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seconds', type=float, default=20.)
    parser.add_argument('--vx', type=float, default=.5)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite existing experiment')
    args.output.mkdir(parents=True)
    shutil.copy2(__file__,args.output/'payload_probe_genesis_source.py')
    shutil.copy2(Path(__file__).parent/'payload_probe.py',args.output/'payload_probe_common_source.py')
    shutil.copy2(args.policy,args.output/args.policy.name)
    shutil.copy2(args.config,args.output/'controller_cfgs.pkl')
    with args.config.open('rb') as file:
        env_cfg, obs_cfg, _, command_cfg, _ = pickle.load(file)
    torch.set_num_threads(1)
    import genesis as gs
    from genesis.utils.geom import inv_quat, transform_by_quat, quat_to_xyz
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision='32', logging_level='warning', seed=0)
    policy = load_policy(args.policy).to(gs.device)
    urdf = args.output/'payload_go2.urdf'
    payload_urdf(urdf)
    scene=gs.Scene(sim_options=gs.options.SimOptions(dt=.02,substeps=2),
                   rigid_options=gs.options.RigidOptions(enable_self_collision=False, tolerance=1e-5,
                                                          max_collision_pairs=40, batch_links_info=True),
                   show_viewer=False)
    scene.add_entity(gs.morphs.URDF(file='urdf/plane/plane.urdf',fixed=True))
    robot=scene.add_entity(gs.morphs.URDF(file=str(urdf.resolve()),pos=env_cfg['base_init_pos'],
                                        quat=env_cfg['base_init_quat'],links_to_keep=['test_payload']))
    masses=np.tile(args.masses,args.repeats)
    batch=len(masses)
    scene.build(n_envs=batch)
    device=gs.device
    motors=[robot.get_joint(name).dof_start-robot.dof_start for name in env_cfg['joint_names']]
    robot.set_dofs_kp([env_cfg['kp']]*12,motors)
    robot.set_dofs_kv([env_cfg['kd']]*12,motors)
    payload=robot.get_link('test_payload')
    payload.set_mass(np.full(batch,EPSILON_MASS,np.float32))
    default=torch.tensor([env_cfg['default_joint_angles'][name] for name in env_cfg['joint_names']],device=device)
    # The same perturbation is repeated across all mass conditions for each seed.
    noise=np.concatenate([np.repeat(np.random.default_rng(seed).normal(0,.005,(1,12)),len(args.masses),axis=0)
                          for seed in range(args.repeats)])
    positions=default[None]+torch.tensor(noise,dtype=torch.float32,device=device)
    robot.set_dofs_position(positions,motors,zero_velocity=True)
    last_action=torch.zeros(batch,12,device=device)
    scales=obs_cfg['obs_scales']
    command=torch.tensor([args.vx,0.,0.],device=device).expand(batch,-1)
    cmdscale=torch.tensor([scales['lin_vel'],scales['lin_vel'],scales['ang_vel']],device=device)
    gravity=torch.tensor([0.,0.,-1.],device=device).expand(batch,-1)

    @torch.inference_mode()
    def advance():
        nonlocal last_action
        quat=robot.get_quat()
        obs=torch.cat([transform_by_quat(robot.get_ang(),inv_quat(quat))*scales['ang_vel'],
                       transform_by_quat(gravity,inv_quat(quat)),command*cmdscale,
                       (robot.get_dofs_position(motors)-default)*scales['dof_pos'],
                       robot.get_dofs_velocity(motors)*scales['dof_vel'],last_action],dim=-1)
        action=policy(obs).clamp(-env_cfg['clip_actions'],env_cfg['clip_actions'])
        executed=last_action if env_cfg['simulate_action_latency'] else action
        robot.control_dofs_position(default+env_cfg['action_scale']*executed,motors)
        scene.step()
        last_action=action

    start=time.monotonic()
    for _ in range(150):
        advance()
    # Explicitly clone one empty-walking state per seed to all of that seed's
    # payload conditions. This also removes small per-env solver divergence.
    source_ids=torch.repeat_interleave(torch.arange(args.repeats,device=device)*len(args.masses),len(args.masses))
    matched_qpos=robot.get_qpos()[source_ids].clone()
    matched_velocity=robot.get_dofs_velocity()[source_ids].clone()
    last_action=last_action[source_ids].clone()
    robot.set_qpos(matched_qpos,zero_velocity=False)
    robot.set_dofs_velocity(matched_velocity)
    initial=robot.get_pos().clone()
    rpy=quat_to_xyz(robot.get_quat(),rpy=True,degrees=True).cpu().numpy()
    initial_quat=robot.get_quat().cpu().numpy()
    warmup_qpos=robot.get_qpos().cpu().numpy().copy()
    # Batched qpos can contain each environment's layout offset; compare the
    # robot-relative root position returned by get_pos instead.
    warmup_qpos[:,:3]=initial.cpu().numpy()
    state_difference=max(float(np.max(np.abs(group-group[0]))) for group in
                         warmup_qpos.reshape(args.repeats,len(args.masses),-1))
    if state_difference > 1e-5:
        raise RuntimeError('Mass conditions did not have a matched warmup state')
    warmup_failed=(initial[:,2].cpu().numpy()<.15)|(np.abs(rpy[:,:2]).max(1)>45)
    metadata=dict(engine='Genesis',policy=str(args.policy),policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
                  config_source=str(args.config),robot_urdf_source=str(ROBOT_URDF),
                  payloads_kg=args.masses,repeats=args.repeats,seconds=args.seconds,
                  command=[args.vx,0.,0.],env_cfg=env_cfg,obs_cfg=obs_cfg,training_commands=command_cfg,
                  control_dt=.02,physics_substeps=2,base_mass_kg=15.019,
                  protocol='three seconds empty warmup; clone one qpos/qvel/action state per seed to all masses; then update welded payload mass and box inertia together',
                  empty_payload_epsilon_kg=EPSILON_MASS,
                  matched_warmup_max_qpos_difference=state_difference,
                  initial_positions=initial.cpu().numpy().tolist(),initial_quaternions=initial_quat.tolist(),
                  source='existing local Genesis-trained policy, not factory Unitree Sport controller',
                  caution='Current installed Genesis version may differ from training version; no auto-reset or retraining')
    (args.output/'config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('WARMUP',warmup_failed.tolist(),initial.cpu().numpy().tolist(),flush=True)
    if warmup_failed.any():
        (args.output/'results.json').write_text(json.dumps(dict(warmup_failed=warmup_failed.tolist(),
                                                 initial_pos=initial.cpu().numpy().tolist(),initial_rpy=rpy.tolist()),indent=2))
        return
    payload.set_mass(np.maximum(masses,EPSILON_MASS).astype(np.float32))
    np.testing.assert_allclose(payload.get_mass(),np.maximum(masses,EPSILON_MASS),rtol=1e-5)
    metadata['measured_total_masses_kg']=np.asarray(robot.get_mass()).tolist()
    (args.output/'config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    # Verify the mass setter changed all conditions as requested.
    print('PAYLOAD_MASS',payload.get_mass(),flush=True)
    heading=np.radians(rpy[:,2])
    directions=np.stack([np.cos(heading),np.sin(heading)],axis=1)
    laterals=np.stack([-directions[:,1],directions[:,0]],axis=1)
    initial_np=initial[:,:2].cpu().numpy()
    arrays=[]
    for step in range(round(args.seconds/.02)):
        advance()
        elapsed=(step+1)*.02
        pos=robot.get_pos().cpu().numpy()
        vel=robot.get_vel().cpu().numpy()
        body_velocity=transform_by_quat(robot.get_vel(),inv_quat(robot.get_quat())).cpu().numpy()
        rpy=quat_to_xyz(robot.get_quat(),rpy=True,degrees=True).cpu().numpy()
        delta=pos[:,:2]-initial_np
        progress=(delta*directions).sum(1)
        lateral=(delta*laterals).sum(1)
        speed=(vel[:,:2]*directions).sum(1)
        arrays.append(np.column_stack([np.full(batch,elapsed),progress,lateral,speed,pos[:,2],rpy,np.zeros(batch),
                                       body_velocity[:,0],np.linalg.norm(vel[:,:2],axis=1)]))
        if (step+1)%250==0:
            print('SIM_SECONDS',elapsed,'progress',np.round(progress,2).tolist(),flush=True)
    traces=np.stack(arrays,axis=1)
    records=[]
    archive={}
    for i,(mass,array) in enumerate(zip(masses,traces)):
        bad=(array[:,4]<.15)|(np.abs(array[:,5:7]).max(1)>45)
        bad_counts=np.convolve(bad.astype(int),np.ones(10,dtype=int),mode='valid')
        failure_ids=np.flatnonzero(bad_counts>=10)
        failure_time=float(array[failure_ids[0]+9,0]) if len(failure_ids) else None
        arrivals={}
        for dist in (2,6):
            ids=np.flatnonzero((array[:,1]>=dist)&(np.abs(array[:,2])<=.75)
                               & (array[:,0]<(failure_time if failure_time is not None else np.inf)))
            arrivals[dist]=float(array[ids[0],0]) if len(ids) else None
        steady=array[:,0]>=2
        record=dict(seed=i//len(args.masses),payload_kg=float(mass),failed=failure_time is not None,
                    failure_seconds=failure_time,distance_m=float(array[-1,1]),average_progress_mps=float(array[-1,1]/args.seconds),
                    steady_progress_mps=float(array[steady,3].mean()),max_lateral_m=float(np.abs(array[:,2]).max()),
                    steady_body_forward_mps=float(array[steady,9].mean()),
                    steady_planar_speed_mps=float(array[steady,10].mean()),
                    rms_roll_deg=float(np.sqrt(np.mean(array[steady,5]**2))),rms_pitch_deg=float(np.sqrt(np.mean(array[steady,6]**2))),
                    torque_saturation_fraction=0.,arrival_2m_seconds=arrivals[2],arrival_6m_seconds=arrivals[6])
        for distance in (2,6):
            indices=np.flatnonzero(array[:,1]>=distance)
            record[f'crossing_{distance}m_forward_plane_seconds']=float(array[indices[0],0]) if len(indices) else None
        record.pop('torque_saturation_fraction')
        records.append(record)
        archive[f'seed{record["seed"]}_mass{mass:g}']=array
    # Saturation was not sampled in Genesis; do not fabricate a measurement.
    summary=summarize([{**r,'torque_saturation_fraction':float('nan')} for r in records],args.masses)
    for row in summary:
        row.pop('torque_saturation_fraction')
        selected=[r for r in records if r['payload_kg']==row['payload_kg']]
        for key in ('steady_body_forward_mps','steady_planar_speed_mps','crossing_2m_forward_plane_seconds',
                    'crossing_6m_forward_plane_seconds'):
            values=[r[key] for r in selected if r[key] is not None]
            row[key]=dict(count=len(values),mean=float(np.mean(values)) if values else None,
                          std=float(np.std(values)) if values else None)
    result=dict(summary=summary,records=records,elapsed_seconds=time.monotonic()-start,warmup_failed=False)
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(args.output/'traces.npz',**archive)
    print('SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
