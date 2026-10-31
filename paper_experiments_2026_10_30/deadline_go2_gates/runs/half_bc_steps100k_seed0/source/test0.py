"""MuJoCo static preview of the same gate geometry; actual locomotion is Genesis."""
import argparse
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from layout import Config, WALLS, GATES


def build_scene(doors=(False,False)):
    config=Config()
    asset_dir=Path(__file__).resolve().parent.parent/'deadline_go2_cargo/assets/go2'
    root=ET.parse(asset_dir/'go2.xml').getroot()
    root.find('compiler').set('meshdir',str(asset_dir/'assets'))
    visual=ET.SubElement(root,'visual')
    ET.SubElement(visual,'global',offwidth='1440',offheight='1080')
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
    ET.SubElement(world,'geom',name='loading_fixture',type='box',pos='6 0 .175',size='.18 .18 .175',
                  rgba='.15 .5 .95 1',contype='0',conaffinity='0')
    ET.SubElement(world,'geom',name='fallback_fixture',type='box',
                  pos=f'{config.fallback_goal[0]} {config.fallback_goal[1]} .175',size='.18 .18 .175',
                  rgba='.6 .35 .85 1',contype='0',conaffinity='0')
    for box in WALLS+GATES:
        ET.SubElement(world,'geom',name=box.name,type='box',
            pos=f'{box.xy[0]} {box.xy[1]} {box.height/2}',
            size=f'{box.size[0]/2} {box.size[1]/2} {box.height/2}',
            rgba='.85 .55 .12 1' if box in GATES else '.36 .42 .48 1')
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
    args=parser.parse_args()
    doors=[s=='1' for s in args.doors]
    model,data=build_scene(doors)
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
        with mujoco.Renderer(model,height=1080,width=1440) as renderer:
            renderer.update_scene(data,camera=camera)
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
                for name in ('type','fixedcamid','distance','azimuth','elevation'): setattr(viewer.cam,name,getattr(camera,name))
                viewer.cam.lookat[:]=camera.lookat
            print('静态场景预览；绿色=基地，紫色=保底1分，蓝色=远点4分。实际运动环境使用 Genesis。')
            print('1/2 切换中门/上门，] 切狗头相机，Esc 返回自由视角。')
            while viewer.is_running():
                viewer.sync()
                time.sleep(.03)


if __name__=='__main__': main()
