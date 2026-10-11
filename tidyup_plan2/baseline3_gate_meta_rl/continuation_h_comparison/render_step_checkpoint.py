"""Render a retained step-level policy checkpoint, defaulting to the 64.3w model.

Headless GIF output: six gate/deadline cases, each a complete three-episode trial.
Full map is for the viewer only; policy inputs and recurrent updates are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

from tidyup_plan2.baseline3_gate_meta_rl.continuation_h_comparison import train

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent/'release_0_64p5w'
DEFAULT_CHECKPOINT = RELEASE/'models/step_643351.pt'
TILE = 28
MAP_W,MAP_H = 12*TILE,8*TILE
PANEL_W,PANEL_H = MAP_W+146,MAP_H+104
PALETTE = ((245,245,245),(100,100,100),(40,180,70),(210,160,20),(250,220,80),(200,45,45))


def font(size):
    try:
        return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)
    except OSError:
        return ImageFont.load_default()


def semantic_image(observation,cell_size=22):
    """Draw the exact semantic channels fed to the actor, not raw floor colors."""
    cells = observation[:150].reshape(5,5,6).argmax(axis=-1)
    result = Image.new('RGB',(5*cell_size,5*cell_size),'white')
    draw = ImageDraw.Draw(result)
    for x in range(5):
        for y in range(5):
            draw.rectangle((x*cell_size,y*cell_size,(x+1)*cell_size-1,(y+1)*cell_size-1),
                           fill=PALETTE[int(cells[x,y])],outline=(210,210,210))
    direction = int(np.argmax(observation[150:154]))
    center = 2.5*cell_size
    vectors = ((1,0),(0,1),(-1,0),(0,-1))
    dx,dy = vectors[direction]
    draw.line((center,center,center+dx*cell_size*.4,center+dy*cell_size*.4),fill='white',width=3)
    return result


def panel(world,observation,hidden,episode_index,gate,deadline,reward_sum,action_label,status):
    result = Image.new('RGB',(PANEL_W,PANEL_H),(255,255,255))
    draw = ImageDraw.Draw(result)
    draw.text((8,6),f'{gate.upper()} | D={deadline} | Episode {episode_index}/3',font=font(17),fill=(25,25,25))
    draw.text((8,31),f'step {world.step_count}/{deadline} | return {reward_sum:+.2f} | {action_label}',font=font(13),fill=(50,50,50))
    full = Image.fromarray(world.get_frame(highlight=True,tile_size=TILE,agent_pov=False))
    result.paste(full,(8,56))
    x = MAP_W+18
    draw.text((x,60),'Policy input',font=font(13),fill=(35,35,35))
    draw.text((x,79),'local 5x5',font=font(12),fill=(60,60,60))
    result.paste(semantic_image(observation),(x,100))
    evidence = train.episode.visible_gate_state(observation)
    draw.text((x,218),f'Gate: {evidence or "unseen"}',font=font(12),fill=(35,35,35))
    draw.text((x,238),f'D-t: {deadline-world.step_count}',font=font(12),fill=(35,35,35))
    draw.text((x,258),f'||h||: {float(hidden.norm()):.2f}',font=font(12),fill=(35,35,35))
    draw.text((8,MAP_H+66),status,font=font(14),fill=(25,25,25))
    draw.rectangle((0,0,PANEL_W-1,PANEL_H-1),outline=(210,210,210))
    return result


@torch.no_grad()
def record_trial(model,args,gate,deadline,seed):
    env = train.episode.make_env(args)
    hidden = model.initial_hidden(torch.device('cpu'))
    records,frame_groups = [],[]
    try:
        for index in range(3):
            options = {'new_trial':index==0,'deadline':deadline}
            if index==0:
                options['gate_state'] = gate
            observation,_ = env.reset(seed=seed if index==0 else None,options=options)
            world = env.unwrapped
            if world.gate_state!=gate or world.deadline!=deadline or world.step_count!=0:
                raise AssertionError('Trial gate/deadline/budget changed')
            start_hidden = hidden.clone()
            positions = [list(map(int,world.agent_pos))]
            actions,rewards,evidence = [],[],[train.episode.visible_gate_state(observation)]
            hidden_norms = [float(hidden.norm())]
            frames = [panel(world,observation,hidden,index+1,gate,deadline,0.,'reset',
                            'New trial: h=0' if index==0 else 'New episode: h inherited')]
            done = False
            while not done:
                logits,_ = model(torch.as_tensor(observation,dtype=torch.float32),hidden)
                action = int(logits.argmax())
                nxt,reward,terminated,truncated,info = env.step(action)
                done = bool(terminated or truncated)
                # Same causal transition update as the trained collector, terminal included.
                hidden = model.update_transition(hidden,observation,action,float(reward),nxt,float(done))
                observation = nxt
                positions.append(list(map(int,world.agent_pos)))
                actions.append(action)
                rewards.append(float(reward))
                evidence.append(train.episode.visible_gate_state(observation))
                hidden_norms.append(float(hidden.norm()))
                route = train.episode.classify_route([tuple(p) for p in positions])
                status = ('GOAL reached' if info.get('success',False) else 'TIMEOUT') if done else 'Running'
                if done:
                    status += f' | route={route}'
                frames.append(panel(world,observation,hidden,index+1,gate,deadline,sum(rewards),
                                    ('left','right','forward')[action],status))
            if index>0 and not torch.equal(start_hidden,previous_terminal_hidden):
                raise AssertionError('Hidden state was reset at an episode boundary')
            previous_terminal_hidden = hidden.clone()
            records.append(dict(episode=index+1,success=bool(info.get('success',False)),steps=world.step_count,
                                route=route,**{'return':sum(rewards)},probed_gate=any(e is not None for e in evidence),
                                positions=positions,actions=actions,rewards=rewards,gate_observations=evidence,
                                hidden_norms=hidden_norms,initial_hidden=start_hidden.tolist(),terminal_hidden=hidden.tolist()))
            frame_groups.append(frames)
    finally:
        env.close()
    return dict(gate_state=gate,deadline=deadline,seed=seed,episodes=records),frame_groups


def compare_saved(records,checkpoint,steps,seed):
    release = checkpoint.parent.parent
    bundled = checkpoint.parent.name=='models' and (release/'curves.json').exists()
    if bundled:
        mode = checkpoint.stem.split('_')[0]
        reference_file = release/'evaluations'/f'{mode}_{steps}.json'
    else:
        reference_file = checkpoint.parent.parent/f'evaluation_{steps}.json'
    if not reference_file.exists():
        return {'checked':False,'reason':'No saved evaluation at exactly this checkpoint'}
    reference = json.loads(reference_file.read_text())
    if bundled:
        seeds = json.loads((release/'curves.json').read_text())['evaluation_seeds']
        if seed not in seeds:
            return {'checked':False,'reason':'Requested seed is outside saved evaluation seeds'}
        repeat = seeds.index(seed)
    else:
        config = json.loads((checkpoint.parent.parent/'config.json').read_text())
        repeat = seed-config['eval_seed']
        if not 0<=repeat<config['eval_trials_per_gate']:
            return {'checked':False,'reason':'Requested seed is outside saved evaluation seeds'}
    for trial in records:
        for actual in trial['episodes']:
            expected = next((r for r in reference['episodes'] if r['gate_state']==trial['gate_state']
                            and r['deadline']==trial['deadline'] and r['trial']==repeat and r['episode']==actual['episode']),None)
            if expected is None:
                return {'checked':False,'reason':'Requested case is not part of the saved evaluation'}
            for key in ('success','steps','route','probed_gate'):
                if expected[key]!=actual[key]:
                    raise AssertionError(f'Rendered execution differs from evaluation: {key}')
            if not np.isclose(expected['return'],actual['return'],atol=1e-6,rtol=0):
                raise AssertionError('Rendered reward differs from evaluation')
    return {'checked':True,'matches_saved_evaluation':True,'source':str(reference_file)}


def save_gif(path,frames,duration):
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=duration,loop=0,optimize=False)


def overview_frame(images,caption):
    columns = 3
    rows = (len(images)+columns-1)//columns
    canvas = Image.new('RGB',(columns*PANEL_W,rows*PANEL_H+60),(246,246,246))
    draw = ImageDraw.Draw(canvas)
    draw.text((10,7),caption,font=font(18),fill=(20,20,20))
    draw.text((10,33),'Full map: viewer only. Right: actual policy semantics. Gate, deadline and episode are labeled in each panel.',
              font=font(12),fill=(50,50,50))
    for index,im in enumerate(images):
        canvas.paste(im,((index%columns)*PANEL_W,(index//columns)*PANEL_H+60))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--seed',type=int,default=10000)
    parser.add_argument('--fps',type=float,default=5.)
    parser.add_argument('--gate',choices=('all','open','closed'),default='all')
    parser.add_argument('--deadline',choices=('all','15','17','19'),default='all')
    options = parser.parse_args()
    if not 0<options.fps<=30:
        parser.error('--fps must be between 0 and 30')
    torch.set_num_threads(1)
    checkpoint = options.checkpoint.resolve()
    value,args,model,_,steps,_,_ = train.load_runtime(checkpoint,'step')
    model.eval()
    output = (options.output_dir or RELEASE/'generated'/f'renders_step_{steps}').resolve()
    if output==checkpoint.parent or output==checkpoint.parent.parent:
        raise ValueError('Render output must not be a checkpoint/training directory')
    output.mkdir(parents=True,exist_ok=True)
    records,groups = [],[]
    gates = ('open','closed') if options.gate=='all' else (options.gate,)
    deadlines = (15,17,19) if options.deadline=='all' else (int(options.deadline),)
    for gate in gates:
        for deadline in deadlines:
            record,frames = record_trial(model,args,gate,deadline,options.seed)
            records.append(record)
            groups.append(frames)
    reference = compare_saved(records,checkpoint,steps,options.seed)
    metadata = dict(checkpoint=str(checkpoint),checkpoint_steps=steps,
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    deterministic=True,hidden_update='every transition, including terminal; inherited across episodes',
                    viewer_only='Full map and true gate labels are never added to policy inputs',
                    policy_semantics='Exact six-channel local input; danger floor is not a separate observation channel',
                    validation=reference,trials=records)
    duration = round(1000/options.fps)
    pause = max(1,round(options.fps*.8))
    for record,frames in zip(records,groups):
        flattened = []
        for episode in frames:
            flattened.extend(episode)
            flattened.extend([episode[-1]]*pause)
        name = f'{record["gate_state"]}_D{record["deadline"]}'
        save_gif(output/f'{name}_trial.gif',flattened,duration)
        frames[2][-1].save(output/f'{name}_episode3.png')
    combined = []
    for episode_index in range(3):
        longest = max(len(g[episode_index]) for g in groups)
        for time_index in range(longest):
            images = [g[episode_index][min(time_index,len(g[episode_index])-1)] for g in groups]
            combined.append(overview_frame(images,f'Step-level checkpoint {steps:,} steps ({steps/10000:.2f}w) | Episode {episode_index+1}/3'))
        combined.extend([combined[-1]]*pause)
        combined[-1].save(output/f'overview_episode{episode_index+1}.png')
    save_gif(output/'overview_trial.gif',combined,duration)
    (output/'trajectories.json').write_text(json.dumps(metadata,indent=2))
    print('Validation:',json.dumps(reference))
    for record in records:
        print(record['gate_state'],record['deadline'],[(e['route'],e['success'],e['steps'],round(e['return'],2)) for e in record['episodes']])
    print('Render outputs:',output)


if __name__=='__main__':
    main()
