"""Frozen, genuinely simulated, unloaded Go2 locomotion in Genesis.

No SDK or hardware access. Every cargo value uses the identical unloaded robot.
The source checkpoint was trained with zero yaw commands; supported navigation
uses body vx/vy with yaw command fixed to zero.
"""
import os
from pathlib import Path
import pickle
import numpy as np
import torch

os.environ.setdefault('GS_CACHE_FILE_PATH','/tmp/time_adapt_genesis_cache')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/time_adapt_cache')
os.environ.setdefault('MPLCONFIGDIR','/tmp/time_adapt_mpl')

from payload_probe import load_policy
from payload_probe_genesis import DEFAULT_WEIGHTS, ROBOT_URDF


class GenesisMotion:
    def __init__(self,batch,cpu=True,map_config=None,scene_builder=None,show_viewer=False):
        import genesis as gs
        from genesis.utils.geom import inv_quat,transform_by_quat
        self.gs,self.inv_quat,self.rotate=gs,inv_quat,transform_by_quat
        if not gs._initialized:
            gs.init(backend=gs.cpu if cpu else gs.gpu,precision='32',logging_level='warning',seed=0)
        with (DEFAULT_WEIGHTS/'cfgs.pkl').open('rb') as file:
            self.env_cfg,self.obs_cfg,*_=pickle.load(file)
        self.batch=batch
        self.device=gs.device
        self.policy=load_policy(DEFAULT_WEIGHTS/'Full_8000.pt').to(self.device)
        self.scene=gs.Scene(sim_options=gs.options.SimOptions(dt=.02,substeps=2),
                            rigid_options=gs.options.RigidOptions(enable_self_collision=False,tolerance=1e-5,
                                                                  max_collision_pairs=30),show_viewer=show_viewer)
        self.scene.add_entity(gs.morphs.URDF(file='urdf/plane/plane.urdf',fixed=True))
        self.robot=self.scene.add_entity(gs.morphs.URDF(file=str(ROBOT_URDF),
                                                      pos=self.env_cfg['base_init_pos'],
                                                      quat=self.env_cfg['base_init_quat']))
        if scene_builder is not None:
            scene_builder(self)
        self.physical_step_hook=None
        self.lidar=None
        self.cargo_geometry=[]
        if map_config is not None:
            # Reserved contact bit keeps these first-hit scan targets out of the
            # robot/ground contact pairs. They remain persistent loading stations.
            sensor_bit=1 << 30
            if any((g.contype | g.conaffinity) & sensor_bit for g in self.robot.geoms):
                raise ValueError('Reserved sensing-geometry contact bit is already in use')
            for x,y in map_config.stations:
                self.cargo_geometry.append(self.scene.add_entity(gs.morphs.Box(
                    size=(.36,.36,.35),pos=(x,y,.175),fixed=True,
                    contype=sensor_bit,conaffinity=0)))
            self.lidar_offset=(.30,0.,.15)
            self.lidar_pattern=gs.sensors.SphericalPattern(n_points=(360,32),fov=(360.,(-45.,15.)))
            self.lidar=self.scene.add_sensor(gs.sensors.Lidar(
                entity_idx=self.robot.idx,pos_offset=self.lidar_offset,
                pattern=self.lidar_pattern,min_range=.15,max_range=map_config.sensor_range,
                return_world_frame=True,no_hit_value=float('nan')))
        self.scene.build(n_envs=batch)
        self.motors=[self.robot.get_joint(name).dof_start-self.robot.dof_start for name in self.env_cfg['joint_names']]
        self.default=torch.tensor([self.env_cfg['default_joint_angles'][name] for name in self.env_cfg['joint_names']],
                                  dtype=torch.float32,device=self.device)
        self.robot.set_dofs_kp([self.env_cfg['kp']]*12,self.motors)
        self.robot.set_dofs_kv([self.env_cfg['kd']]*12,self.motors)
        self.last_action=torch.zeros(batch,12,device=self.device)
        self.gravity=torch.tensor([0.,0.,-1.],device=self.device).expand(batch,-1)
        self.scales=self.obs_cfg['obs_scales']
        self.cmdscale=torch.tensor([self.scales['lin_vel'],self.scales['lin_vel'],self.scales['ang_vel']],device=self.device)
        self.reset()

    @torch.inference_mode()
    def reset(self,indices=None,seeds=None):
        ids=np.arange(self.batch) if indices is None else np.asarray(indices,dtype=int)
        index=torch.as_tensor(ids,device=self.device)
        qpos=self.robot.get_qpos()[index].clone()
        qpos[:,:3]=torch.tensor(self.env_cfg['base_init_pos'],device=self.device)
        qpos[:,3:7]=torch.tensor(self.env_cfg['base_init_quat'],device=self.device)
        for i,name in enumerate(self.env_cfg['joint_names']):
            qid=self.robot.get_joint(name).q_start-self.robot.q_start
            qpos[:,qid]=self.default[i]
        if seeds is not None:
            for row,seed in enumerate(seeds):
                noise=np.random.default_rng(seed).normal(0,.005,12)
                for j,name in enumerate(self.env_cfg['joint_names']):
                    qid=self.robot.get_joint(name).q_start-self.robot.q_start
                    qpos[row,qid]+=float(noise[j])
        self.robot.set_qpos(qpos,envs_idx=index,zero_velocity=True)
        self.last_action[index]=0
        if self.lidar is not None:
            # Genesis 0.4.6 read() returns a cache updated by physics steps;
            # teleports need an explicit refresh, without advancing task time.
            self.scene.sim._sensor_manager.step()

    @torch.inference_mode()
    def lidar_scan(self):
        if self.lidar is None:
            raise RuntimeError('Lidar was not enabled before building this scene')
        data=self.lidar.read()
        quat=self.robot.get_quat()
        offset=torch.tensor(self.lidar_offset,device=self.device).expand(self.batch,-1)
        origin=self.robot.get_pos()+self.rotate(offset,quat)
        dirs=self.lidar_pattern.ray_dirs.reshape(-1,3)
        # Sensor orientation follows full physical roll/pitch/yaw, not just heading.
        directions=self.rotate(dirs[None].expand(self.batch,-1,-1),quat[:,None].expand(-1,len(dirs),-1))
        return dict(points=data.points.reshape(self.batch,-1,3).cpu().numpy(),
                    distances=data.distances.reshape(self.batch,-1).cpu().numpy(),
                    origins=origin.cpu().numpy(),directions=directions.cpu().numpy())

    @torch.inference_mode()
    def advance(self,commands,seconds=.5):
        command=torch.as_tensor(commands,dtype=torch.float32,device=self.device)
        steps=round(seconds/.02)
        if abs(steps*.02-seconds)>1e-5:
            raise ValueError('Control duration must be a multiple of 0.02 seconds')
        for _ in range(steps):
            quat=self.robot.get_quat()
            obs=torch.cat([self.rotate(self.robot.get_ang(),self.inv_quat(quat))*self.scales['ang_vel'],
                           self.rotate(self.gravity,self.inv_quat(quat)),command*self.cmdscale,
                           (self.robot.get_dofs_position(self.motors)-self.default)*self.scales['dof_pos'],
                           self.robot.get_dofs_velocity(self.motors)*self.scales['dof_vel'],self.last_action],dim=-1)
            action=self.policy(obs).clamp(-self.env_cfg['clip_actions'],self.env_cfg['clip_actions'])
            executed=self.last_action if self.env_cfg['simulate_action_latency'] else action
            self.robot.control_dofs_position(self.default+self.env_cfg['action_scale']*executed,self.motors)
            self.scene.step()
            if self.physical_step_hook is not None:
                self.physical_step_hook(self)
            self.last_action=action
        return self.state()

    def state(self):
        from genesis.utils.geom import quat_to_xyz
        position=self.robot.get_pos().cpu().numpy()
        rpy=quat_to_xyz(self.robot.get_quat(),rpy=True,degrees=False).cpu().numpy()
        velocity=self.robot.get_vel().cpu().numpy()
        failed=(position[:,2]<.15)|(np.abs(rpy[:,:2]).max(1)>np.pi/4)
        return dict(pose=np.column_stack([position[:,:2],rpy[:,2]]),position=position,rpy=rpy,
                    velocity=velocity,failed=failed)


def goal_commands(poses,targets,speed=.5):
    delta=np.asarray(targets)-poses[:,:2]
    c,s=np.cos(poses[:,2]),np.sin(poses[:,2])
    x=c*delta[:,0]+s*delta[:,1]
    y=-s*delta[:,0]+c*delta[:,1]
    distance=np.linalg.norm(delta,axis=1)
    magnitude=np.minimum(speed,distance/.8)
    return np.column_stack([np.clip(x/np.maximum(distance,1e-6)*magnitude,-.6,.6),
                            np.clip(y/np.maximum(distance,1e-6)*magnitude,-.4,.4),np.zeros(len(poses))])


def main():
    import argparse,json,time
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',action='store_true')
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Refusing overwrite')
    targets=np.array([[2.,0.],[3.,5.196],[3.,-5.196]])
    sim=GenesisMotion(3,cpu=not args.gpu)
    state=sim.state()
    start=time.monotonic()
    records=[]
    for phase in ('outbound','return'):
        goals=targets if phase=='outbound' else np.zeros((3,2))
        done=np.zeros(3,bool);failure=np.zeros(3,bool);arrivals=[None]*3
        for step in range(160):
            commands=goal_commands(state['pose'],goals)
            commands[done]=0
            state=sim.advance(commands,.5)
            distances=np.linalg.norm(state['pose'][:,:2]-goals,axis=1)
            for i in range(3):
                if not done[i] and distances[i]<.4:
                    arrivals[i]=(step+1)*.5
                    done[i]=True
            failure|=state['failed']
            if done.all() or failure.any():
                break
        records.append(dict(phase=phase,arrival_seconds=arrivals,failed=failure.tolist(),
                            final_pose=state['pose'].tolist()))
        print(records[-1],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(records=records,elapsed_seconds=time.monotonic()-start,
                                         command='body vx/vy; yaw=0; no load'),indent=2)+'\n')


if __name__=='__main__':
    main()
