#!/usr/bin/env python3
"""
Stage 1: Multi-ego sweep on 3 unseen Aichi-region maps.
For each map, sample N_EGO road-lanelets (deterministically by seed) as ego
starts. For each ego start, run all 6 experts (baseline + 5 SE-LoRA) and
record trajectory metrics. Saves per-(map, ego, expert) row to JSON.
"""
import os, sys, json, time, random
os.environ.setdefault('LD_LIBRARY_PATH', '/opt/autoware/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
sys.path.insert(0, '/opt/autoware/local/lib/python3.10/dist-packages')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner_ros')
sys.path.insert(0, 'third_party/Diffusion-Planner/preference_optimization')

import numpy as np
import torch
import lanelet2
from autoware_lanelet2_extension_python.projection import MGRSProjector

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.dimensions import (
    INPUT_T, MAX_NUM_AGENTS, MAX_NUM_NEIGHBORS, NUM_LINE_STRINGS,
    NUM_POLYGONS, NUM_SEGMENTS_IN_LANE, NUM_SEGMENTS_IN_ROUTE,
    POSE_DIM, POINTS_PER_LANELET, POINTS_PER_LINE_STRING, POINTS_PER_POLYGON,
    SEGMENT_POINT_DIM,
)
from diffusion_planner_ros.lanelet2_utils.lanelet_converter import (
    convert_lanelet, create_lane_tensor)
from train_reward_backprop import differentiable_dpm_solver_sample, NoiseScheduleVP

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ARGS = 'third_party/Diffusion-Planner/args.json'
BASE_PTH = 'third_party/Diffusion-Planner/best_model.pth'
SELORA_DIR = 'checkpoints/molora_v2_output'
N_EGO = 10
SEED = 42

UNSEEN = [
    ('aichi_hub_v2',
     'maps/unseen_extracted/知の拠点あいち/milee_service_knowledge_hub_aich.osm'),
    ('ishio_dai',
     'maps/unseen_extracted/石尾台地区/lanelet2_map.osm'),
    ('morikoro',
     'maps/unseen_extracted/モリコロ/モリコロ全体_20200413_LL2.osm'),
]
EXPERTS = ['baseline', 'shared_only', 'hongo', 'kashiwa', 'shalun', 'unagoya']


def load_model(expert: str, cfg):
    model = Diffusion_Planner(cfg).to(DEVICE).eval()
    if expert == 'baseline':
        ckpt = torch.load(BASE_PTH, map_location=DEVICE, weights_only=False)
        sd = ckpt.get('ema_state_dict') or ckpt['model']
    else:
        path = f'{SELORA_DIR}/merged_{expert}.pth'
        if not os.path.exists(path): return None
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        sd = ckpt.get('model', ckpt)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    enc_w = model.state_dict()['encoder.ego_encoder.channel_pre_project.fc1.weight']
    assert enc_w.norm().item() > 1.0, "load failed"
    return model


def make_map2bl(ego_x, ego_y, ego_yaw):
    c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
    T = np.eye(4)
    T[0, 0] = c; T[0, 1] = -s
    T[1, 0] = s; T[1, 1] =  c
    T[0, 3] = -(c * ego_x - s * ego_y)
    T[1, 3] = -(s * ego_x + c * ego_y)
    return T


def build_inputs(llmap, ego_xy, ego_yaw):
    cx, cy = ego_xy
    map2bl = make_map2bl(cx, cy, ego_yaw)
    lanes_t, lanes_sl, lanes_hsl = create_lane_tensor(
        list(llmap.lanelets.values()), map2bl, cx, cy, {},
        NUM_SEGMENTS_IN_LANE, DEVICE, do_sort=True)
    route_t, route_sl, route_hsl = create_lane_tensor(
        list(llmap.lanelets.values()), map2bl, cx, cy, {},
        NUM_SEGMENTS_IN_ROUTE, DEVICE, do_sort=True)
    B = 1
    inputs = dict(
        ego_agent_past   = torch.zeros(B, INPUT_T + 1, POSE_DIM, device=DEVICE),
        ego_current_state= torch.zeros(B, 10, device=DEVICE),
        neighbor_agents_past = torch.zeros(B, MAX_NUM_NEIGHBORS, INPUT_T + 1, 11, device=DEVICE),
        static_objects   = torch.zeros(B, 5, 10, device=DEVICE),
        lanes            = lanes_t,
        lanes_speed_limit= lanes_sl,
        lanes_has_speed_limit = lanes_hsl,
        route_lanes      = route_t,
        route_lanes_speed_limit = route_sl,
        route_lanes_has_speed_limit = route_hsl,
        polygons         = torch.zeros(B, NUM_POLYGONS, POINTS_PER_POLYGON, 2, device=DEVICE),
        line_strings     = torch.zeros(B, NUM_LINE_STRINGS, POINTS_PER_LINE_STRING, 2, device=DEVICE),
        goal_pose        = torch.tensor([[30.0, 0.0, 0.0]], device=DEVICE),
        ego_shape        = torch.tensor([[2.75, 4.34, 1.70]], device=DEVICE),
        turn_indicators  = torch.zeros(B, INPUT_T + 1, dtype=torch.float32, device=DEVICE),
    )
    inputs['ego_agent_past'] = heading_to_cos_sin(inputs['ego_agent_past'])
    inputs['goal_pose']      = heading_to_cos_sin(inputs['goal_pose'])
    return inputs


def pick_n_egos(llmap, n: int, seed: int):
    """Pick n road lanelets uniformly with deterministic seed."""
    road_lls = list(llmap.lanelets.values())
    rng = random.Random(seed)
    if len(road_lls) <= n:
        return road_lls
    return rng.sample(road_lls, n)


def metrics_for_traj(xy):
    """Return dict of (path_length_m, fwd_progress_m, max_lat_m, mean_step_m)."""
    if not np.all(np.isfinite(xy)):
        return {'finite': False}
    return {
        'finite': True,
        'path_len':       float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))),
        'fwd_progress':   float(xy[-1, 0] - xy[0, 0]),
        'max_lat':        float(np.max(np.abs(xy[:, 1]))),
        'final_dist':     float(np.linalg.norm(xy[-1])),
        'mean_step':      float(np.mean(np.linalg.norm(np.diff(xy, axis=0), axis=1))),
    }


def main():
    cfg = Config(ARGS)
    ns = NoiseScheduleVP()
    out = {'description': 'Stage 1: multi-ego sweep on 3 unseen ODDs',
           'n_ego_per_map': N_EGO, 'rows': []}

    proj = MGRSProjector(lanelet2.io.Origin(35.18, 137.09))

    print(f"[stage1] Loading {len(EXPERTS)} models...")
    models = {}
    for e in EXPERTS:
        m = load_model(e, cfg)
        if m is not None: models[e] = m
        print(f"  {e}: loaded")

    for tag, osm in UNSEEN:
        print(f"\n=== {tag} ===")
        try:
            llmap = convert_lanelet(osm)
        except Exception as e:
            print(f"  convert failed: {e}"); continue
        if not llmap.lanelets:
            print(f"  no road lanelets"); continue

        egos = pick_n_egos(llmap, N_EGO, seed=SEED)
        print(f"  {len(egos)} ego starts")

        for ego_idx, ego_ll in enumerate(egos):
            cl = ego_ll.centerline
            if cl.shape[0] < 2: continue
            cx, cy = float(cl[0, 0]), float(cl[0, 1])
            heading = float(np.arctan2(cl[1, 1] - cl[0, 1], cl[1, 0] - cl[0, 0]))
            inputs = build_inputs(llmap, (cx, cy), heading)
            inputs_norm = cfg.observation_normalizer({
                k: (v.clone() if isinstance(v, torch.Tensor) else v) for k,v in inputs.items()})
            n_lanes = int((inputs['lanes'].abs().sum(dim=(2,3)) > 0).sum().item())

            for expert, model in models.items():
                t0 = time.time()
                try:
                    with torch.no_grad():
                        ego_traj = differentiable_dpm_solver_sample(model, cfg, inputs_norm, ns, device=DEVICE)
                    xy = ego_traj[:, :2].cpu().numpy()
                    m = metrics_for_traj(xy)
                except Exception as e:
                    m = {'error': str(e)[:120]}
                row = {'map': tag, 'ego_idx': ego_idx, 'lanelet_id': ego_ll.id,
                       'ego_x': cx, 'ego_y': cy, 'yaw': heading,
                       'n_lanes_loaded': n_lanes, 'expert': expert,
                       'inference_s': time.time() - t0, **m}
                out['rows'].append(row)
            if (ego_idx+1) % 5 == 0:
                print(f"  ego {ego_idx+1}/{len(egos)} done")

    out_path = 'checkpoints/overnight_20260510/stage1_multiego.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\n[stage1] saved {out_path}  ({len(out['rows'])} rows)")


if __name__ == '__main__':
    main()
