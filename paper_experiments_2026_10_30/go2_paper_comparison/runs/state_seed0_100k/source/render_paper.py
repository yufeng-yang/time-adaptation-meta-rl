"""Render the publication illustration; goal override is preview-only."""
import argparse
import itertools
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/time_adapt_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image
from layout import Config, GATES, WALLS
from test0 import build_scene, range_boundary


def project(scene, points, width, height):
    left, right = scene.camera
    position = (left.pos+right.pos)/2
    forward = (left.forward+right.forward)/2
    forward /= np.linalg.norm(forward)
    up = (left.up+right.up)/2
    up /= np.linalg.norm(up)
    horizontal = np.cross(forward, up)
    delta = np.asarray(points)-position
    scale = left.frustum_near/((left.frustum_top-left.frustum_bottom)/2)
    depth = delta@forward
    return np.c_[width/2+(delta@horizontal)*scale*height/(2*depth),
                 height/2-(delta@up)*scale*height/(2*depth)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'figures')
    parser.add_argument('--goal-xy', nargs=2, type=float, default=(6.,0.))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    width, height = 2400, 1800
    config = Config()
    model, data = build_scene((False,False), goal_xy=args.goal_xy,
                             show_lidar_range=True, framebuffer_size=(width,height))
    # Explicitly verify that the publication overlay has no contact effects.
    ids = [i for i in range(model.ngeom) if model.geom_group[i] == 5]
    assert ids and not model.geom_contype[ids].any() and not model.geom_conaffinity[ids].any()
    goal_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'loading_fixture')
    np.testing.assert_allclose(model.geom_pos[goal_id,:2], args.goal_xy)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [3.,-.8,.3]
    camera.distance, camera.azimuth, camera.elevation = 13.6, 0., -82.
    option = mujoco.MjvOption()
    option.geomgroup[5] = 1
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        pixels = renderer.render().copy()
        corners = []
        for box in WALLS+GATES:
            corners.extend((box.xy[0]+sx*box.size[0]/2,
                            box.xy[1]+sy*box.size[1]/2,z)
                           for sx,sy,z in itertools.product((-1,1),(-1,1),(0.,box.height)))
        bounds = project(renderer.scene,corners,width,height)
    margin = 36
    x0,y0 = np.maximum(0,np.floor(bounds.min(0)).astype(int)-margin)
    x1,y1 = np.minimum([width,height],np.ceil(bounds.max(0)).astype(int)+margin)
    # Mild eight-degree tilt from vertical, with upright framing and tight margins.
    pixels = np.ascontiguousarray(pixels[y0:y1,x0:x1])
    rendered_size = [width,height]
    height,width = pixels.shape[:2]
    clean = args.output/'environment4_lidar_range_landscape_clean.png'
    Image.fromarray(pixels).save(clean, dpi=(300,300))
    fig = plt.figure(figsize=(width/300,height/300), dpi=300)
    ax = fig.add_axes([0,0,1,1])
    ax.imshow(pixels, extent=(0,width,height,0))
    ax.set(xlim=(0,width),ylim=(height,0))
    ax.axis('off')
    for ext in ('png','pdf','svg'):
        fig.savefig(args.output/f'environment4_lidar_range_landscape.{ext}', dpi=300)
    plt.close(fig)
    origin = np.array([.30,0.])
    minimum_distances = [float(np.linalg.norm(np.maximum(
        np.abs(np.asarray(g.xy)-origin)-np.asarray(g.size)/2,0.))) for g in GATES]
    assert min(minimum_distances) > config.sensor_range
    metadata = dict(renderer='MuJoCo static scene; training physics is Genesis',
        resolution=[width,height],doors=[False,False],range_m=config.sensor_range,
        original_render_resolution=rendered_size,crop_xyxy=[int(x0),int(y0),int(x1),int(y1)],
        image_rotation='none',
        camera=dict(lookat=camera.lookat.tolist(),distance=camera.distance,
                    azimuth=camera.azimuth,elevation=camera.elevation),
        range_meaning='Initial horizontal first-wall visibility within 2m, inset 3.5cm from wall faces; schematic projection, not the full 3D voxel observation',
        range_clipped_at_walls=True,visible_boundary_xy=range_boundary(config.sensor_range).tolist(),
        range_center_xy=origin.tolist(),sensor_mount_offset=[.30,0.,.15],
        nearest_gate_surface_distances_m=minimum_distances,
        gate_figure_numbers={'upper_gate':1,'middle_gate':2},
        gate_array_order=['middle_gate','upper_gate'],
        both_gates_outside_initial_range=True,goal_preview_xy=list(args.goal_xy),
        goal_training_xy=list(config.goal),training_config_modified=False,
        policy_inputs_modified=False,labels='none',range_outline='dashed',range_fill='translucent cyan')
    (args.output/'environment4_lidar_range_landscape_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__ == '__main__':
    main()
