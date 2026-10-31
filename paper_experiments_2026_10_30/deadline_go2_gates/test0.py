"""MuJoCo static preview of the same gate geometry; actual locomotion is Genesis."""
import argparse
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from layout import Config, WALLS, GATES


def range_boundary(radius, doors=(False, False), clearance=.035):
    """Initial horizontal visibility boundary, clipped at first wall intersections.

    Illustration geometry only, using the same boxes as the static scene.
    It is not supplied to the policy or substituted for the training lidar.
    """
    center = np.array([.30, 0.])
    angles = np.linspace(0., 2*np.pi, 720, endpoint=False)
    directions = np.c_[np.cos(angles), np.sin(angles)]
    safe_directions = np.where(np.abs(directions)<1e-12, 1e-12, directions)
    distances = np.full(len(angles), radius)
    boxes = list(WALLS) + [box for box, opened in zip(GATES, doors) if not opened]
    for box in boxes:
        # Expand by the drawing margin so even a dash grazing a wall remains
        # fully on the visible side (not only rays directly hitting its face).
        low = np.asarray(box.xy)-np.asarray(box.size)/2-clearance
        high = np.asarray(box.xy)+np.asarray(box.size)/2+clearance
        near = (low-center)/safe_directions
        far = (high-center)/safe_directions
        enter = np.minimum(near, far).max(axis=1)
        leave = np.maximum(near, far).min(axis=1)
        hits = (enter>0) & (leave>=enter) & (enter<distances)
        distances[hits] = enter[hits]
    return center+distances[:,None]*directions


def add_range_overlay(asset, world, radius, doors):
    """Draw wall-clipped horizontal visibility with a dashed outline."""
    center = np.array([.30, 0.])
    points = range_boundary(radius, doors)
    count = len(points)
    for index, a in enumerate(points):
        # Six-degree arcs alternating with six-degree gaps.
        if index % 24 >= 12:
            continue
        b = points[(index+1)%count]
        ET.SubElement(world, 'geom', name=f'lidar_range_{index}', type='capsule',
                      fromto=f'{a[0]} {a[1]} .035 {b[0]} {b[1]} .035', size='.018',
                      rgba='.15 .92 .96 1', contype='0', conaffinity='0', group='5')
    # A triangulated thin polygon replaces the full cylinder, so both the fill
    # and the outline stop before the first wall rather than leaking behind it.
    flat = np.vstack([center, points])
    vertices = np.vstack([np.c_[flat,np.full(count+1,.015)],
                          np.c_[flat,np.full(count+1,.017)]])
    top = count+1
    faces = []
    for index in range(count):
        a, b = index+1, (index+1)%count+1
        faces.extend([(0,b,a),(top,top+a,top+b),
                      (a,b,top+b),(a,top+b,top+a)])
    ET.SubElement(asset, 'mesh', name='lidar_visibility_mesh',
                  vertex=' '.join(map(str,vertices.ravel())),
                  face=' '.join(map(str,np.asarray(faces).ravel())))
    ET.SubElement(world, 'geom', name='lidar_range_projection', type='mesh',
                  mesh='lidar_visibility_mesh', rgba='.15 .85 .94 .10',
                  contype='0', conaffinity='0', group='5')


def build_scene(doors=(False,False), *, goal_xy=None, show_lidar_range=False,
                framebuffer_size=(1440,1080)):
    config=Config()
    asset_dir=Path(__file__).resolve().parent.parent/'deadline_go2_cargo/assets/go2'
    root=ET.parse(asset_dir/'go2.xml').getroot()
    root.find('compiler').set('meshdir',str(asset_dir/'assets'))
    visual=ET.SubElement(root,'visual')
    ET.SubElement(visual,'global',offwidth=str(framebuffer_size[0]),offheight=str(framebuffer_size[1]))
    asset=root.find('asset')
    ET.SubElement(asset,'texture',name='floor_tex',type='2d',builtin='checker',
                  rgb1='.20 .23 .26',rgb2='.25 .28 .31',width='512',height='512')
    ET.SubElement(asset,'material',name='floor_mat',texture='floor_tex',texrepeat='16 16')
    world=root.find('worldbody')
    ET.SubElement(world,'light',pos='3 -1 10',dir='0 0 -1',diffuse='.85 .85 .85',
                  directional='true',castshadow='false')
    ET.SubElement(world,'geom',name='floor',type='plane',size='15 15 .1',material='floor_mat')
    ET.SubElement(world,'geom',name='home',type='cylinder',pos='0 0 .006',size='.45 .006',
                  rgba='.15 .8 .4 1',contype='0',conaffinity='0')
    goal_xy = config.goal if goal_xy is None else tuple(goal_xy)
    ET.SubElement(world,'geom',name='loading_fixture',type='box',
                  pos=f'{goal_xy[0]} {goal_xy[1]} .175',size='.18 .18 .175',
                  rgba='.15 .5 .95 1',contype='0',conaffinity='0')
    ET.SubElement(world,'geom',name='fallback_fixture',type='box',
                  pos=f'{config.fallback_goal[0]} {config.fallback_goal[1]} .175',size='.18 .18 .175',
                  rgba='.6 .35 .85 1',contype='0',conaffinity='0')
    for box in WALLS+GATES:
        ET.SubElement(world,'geom',name=box.name,type='box',
            pos=f'{box.xy[0]} {box.xy[1]} {box.height/2}',
            size=f'{box.size[0]/2} {box.size[1]/2} {box.height/2}',
            rgba='.85 .55 .12 1' if box in GATES else '.36 .42 .48 1')
    if show_lidar_range:
        add_range_overlay(asset, world, config.sensor_range, doors)
    base=world.find("body[@name='base_link']")
    ET.SubElement(base,'camera',name='head',pos='.34 0 .05',xyaxes='0 -1 0 .173648 0 .984808',fovy='110')
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    data=mujoco.MjData(model)
    data.qpos[:7]=[0.,0.,.32,1.,0.,0.,0.]
    for leg in ('FL','FR','RL','RR'):
        for joint,angle in (('hip',0.),('thigh',.8),('calf',-1.6)):
            jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,f'{leg}_{joint}_joint')
            data.qpos[model.jnt_qposadr[jid]]=angle
    set_doors(model,doors)
    mujoco.mj_forward(model,data)
    return model,data


def set_doors(model,doors):
    for box,opened in zip(GATES,doors):
        index=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,box.name)
        model.geom_pos[index,2]=-3. if opened else box.height/2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--doors',choices=('00','01','10','11'),default='00',help='middle/upper; 1=open, 0=closed')
    parser.add_argument('--save',type=Path)
    parser.add_argument('--camera',choices=('free','head'),default='free')
    parser.add_argument('--goal-xy',type=float,nargs=2,metavar=('X','Y'),
                        help='Static-preview override only; does not change the training task')
    parser.add_argument('--show-lidar-range',action='store_true',
                        help='Draw initial wall-clipped horizontal visibility within 2m')
    parser.add_argument('--width',type=int,default=1440)
    parser.add_argument('--height',type=int,default=1080)
    args=parser.parse_args()
    doors=[s=='1' for s in args.doors]
    model,data=build_scene(doors,goal_xy=args.goal_xy,show_lidar_range=args.show_lidar_range,
                           framebuffer_size=(args.width,args.height))
    camera=mujoco.MjvCamera()
    camera.lookat[:]=[3.,-.8,0.]
    camera.distance=17.
    camera.azimuth=90.
    camera.elevation=-68.
    if args.camera=='head':
        camera.type=mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,'head')
    if args.save:
        from PIL import Image
        with mujoco.Renderer(model,height=args.height,width=args.width) as renderer:
            option = mujoco.MjvOption()
            option.geomgroup[5] = args.show_lidar_range
            renderer.update_scene(data,camera=camera,scene_option=option)
            args.save.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(renderer.render()).save(args.save)
        print(args.save.resolve())
    else:
        from mujoco import viewer as mj_viewer
        def key_callback(key):
            if key in (49,50):
                doors[key-49]=not doors[key-49]
                set_doors(model,doors)
                mujoco.mj_forward(model,data)
                print('Doors (middle, upper), True=open:',doors,flush=True)
        with mj_viewer.launch_passive(model,data,key_callback=key_callback) as viewer:
            with viewer.lock():
                viewer.opt.geomgroup[5] = args.show_lidar_range
                for name in ('type','fixedcamid','distance','azimuth','elevation'): setattr(viewer.cam,name,getattr(camera,name))
                viewer.cam.lookat[:]=camera.lookat
            print('静态场景预览；绿色=基地，紫色=保底1分，蓝色=远点4分。实际运动环境使用 Genesis。')
            print('1/2 切换中门/上门，] 切狗头相机，Esc 返回自由视角。')
            while viewer.is_running():
                viewer.sync()
                time.sleep(.03)


if __name__=='__main__': main()
