"""Real unloaded Genesis physics and local first-hit lidar; no hardware API."""
from pathlib import Path
import sys
import numpy as np
import torch
from layout import Config, WALLS, GATES

SHARED = Path(__file__).resolve().parent.parent/'deadline_go2_cargo'
if str(SHARED) not in sys.path:
    sys.path.append(str(SHARED))
from genesis_motion import GenesisMotion, goal_commands
from voxel_map import build_local_voxels


class GateWorld:
    def __init__(self, batch=1, config=Config(), show_viewer=False):
        self.config = config
        self.batch = batch
        self.doors = np.ones((batch, 2), dtype=bool)
        self.maps = {}
        self.scan = None
        self.sim = GenesisMotion(batch, map_config=config, scene_builder=self._build,
                                 show_viewer=show_viewer)
        # Genesis 0.4.6 updates every attached sensor at every physics step and
        # has no per-sensor update period. This scene contains only a zero-delay
        # lidar. Defer its instance-local manager refresh until the 2Hz control
        # boundary; locomotion still integrates every 0.02s with two substeps.
        manager=self.sim.scene.sim._sensor_manager
        self._update_sensor_cache=manager.step
        manager.step=lambda: None
        self.state = self.sim.state()
        self.obstacle_geoms = torch.tensor([g.idx for entity in self.walls+self.gates
                                            for g in entity.geoms], device=self.sim.device)
        self.contacts = np.zeros(batch, bool)
        self.trace = []
        self.sim.physical_step_hook = self._track

    def refresh_sensor(self):
        self._update_sensor_cache()

    def _build(self, sim):
        gs = sim.gs
        def add(box, door=False):
            return sim.scene.add_entity(gs.morphs.Box(
                size=(*box.size, box.height),pos=(*box.xy,box.height/2),fixed=True,
                contype=1,conaffinity=1),
                surface=gs.surfaces.Default(color=(.85,.55,.12) if door else (.36,.42,.48)))
        self.walls = [add(box) for box in WALLS]
        self.gates = [add(box, True) for box in GATES]

    def _invalidate(self):
        self.state = self.sim.state()
        self.maps.clear()
        self.scan = None

    def set_doors(self, index, opened):
        opened = np.asarray(opened, dtype=bool)
        if opened.shape != (2,):
            raise ValueError('Expected two gate states, True=open')
        self.doors[index] = opened
        for entity, box, is_open in zip(self.gates, GATES, opened):
            # Physical gate surfaces disappear below the floor when open.
            # Closed gates retain their real collision mask and ray geometry.
            entity.set_pos((*box.xy, -3. if is_open else box.height/2),envs_idx=[index])
        self.refresh_sensor()
        self._invalidate()

    def reset(self, index, seed):
        self.sim.reset([index], [seed])
        self.refresh_sensor()
        self._invalidate()

    def _track(self, sim):
        contacts = sim.robot.get_contacts(exclude_self_contact=True)
        mask = contacts['valid_mask'] & (torch.isin(contacts['geom_a'], self.obstacle_geoms) |
                                          torch.isin(contacts['geom_b'], self.obstacle_geoms))
        self.contacts |= mask.any(1).cpu().numpy()
        self.trace.append(sim.robot.get_pos().cpu().numpy()[:, :2])

    def advance(self, commands, seconds):
        self.contacts[:] = False
        self.trace = []
        self.sim.advance(commands, seconds)
        self.refresh_sensor()
        self._invalidate()

    def local_map(self, index):
        if index not in self.maps:
            if self.scan is None:
                self.scan = self.sim.lidar_scan()
            scan = {key: value[index] for key, value in self.scan.items()}
            self.maps[index] = build_local_voxels(scan, self.state['pose'][index], self.config)
        return self.maps[index]


class Backend:
    name = 'Genesis 0.4.6 + frozen Full_8000 unloaded locomotion + first-hit lidar'

    def __init__(self, world, index=0, seed=0):
        self.world, self.index, self.seed = world, index, seed

    def set_task(self, doors):
        self.world.set_doors(self.index, doors)

    def reset(self):
        self.world.reset(self.index, self.seed)

    @property
    def pose(self): return self.world.state['pose'][self.index]

    @property
    def velocity(self): return self.world.state['velocity'][self.index]

    @property
    def failed(self): return bool(self.world.state['failed'][self.index])

    @property
    def wall_contact(self): return bool(self.world.contacts[self.index])

    def local_map(self): return self.world.local_map(self.index)

    def step(self, command, seconds):
        if self.world.batch != 1:
            raise RuntimeError('Use step_batch for a shared world')
        self.world.advance(np.asarray(command)[None], seconds)


def make_batch(seeds, config=Config(), show_viewer=False):
    world = GateWorld(len(seeds), config, show_viewer)
    return [Backend(world, i, seed) for i, seed in enumerate(seeds)]


def step_batch(envs, actions, active=None):
    active = np.ones(len(envs), bool) if active is None else np.asarray(active, bool)
    world = envs[0].backend.world
    if any(e.backend.world is not world for e in envs):
        raise ValueError('Batched environments must share a world')
    commands = np.zeros((len(envs),3))
    for i, env in enumerate(envs):
        if active[i]: commands[i,:2] = np.clip(actions[i],-1,1)*np.array(env.config.command_limits[:2])
    world.advance(commands, envs[0].config.dt)
    return [e.step(action, advance_motion=False) if run else None
            for e, action, run in zip(envs, actions, active)]
