"""LFS: Lane-Follow Selection.
Per scene + per expert, compute 'how close trajectory waypoints are to the nearest
route-lane centerline'. Pick the expert with lowest deviation. Model-agnostic, no
training. Predicted to beat best-fixed-Nagoya on at least some scenes."""
import os, sys, json, time
sys.path.insert(0, '/opt/autoware/local/lib/python3.10/dist-packages')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner_ros')
sys.path.insert(0, 'third_party/Diffusion-Planner/preference_optimization')
sys.path.insert(0, '.')

import numpy as np
import torch
from diffusion_planner.utils.config import Config
from diffusion_planner_ros.lanelet2_utils.lanelet_converter import convert_lanelet
from train_reward_backprop import differentiable_dpm_solver_sample, NoiseScheduleVP
from overnight_stage1_multiego import load_model, build_inputs, pick_n_egos, UNSEEN, N_EGO, SEED

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ARGS = 'third_party/Diffusion-Planner/args.json'
EXPERTS = ['shared_only','hongo','kashiwa','shalun','unagoya']

def lane_dist(traj_xy, route_lanes_np):
    """Mean nearest-route-lane distance of each waypoint.
    traj_xy: (T, 2). route_lanes_np: (25, 20, 33) — feature 0/1 are xy in ego frame."""
    valid_lanes = []
    for i in range(route_lanes_np.shape[0]):
        pts = route_lanes_np[i, :, :2]
        if np.abs(pts).sum() > 0.1: valid_lanes.append(pts)
    if not valid_lanes: return float('inf')
    all_pts = np.concatenate(valid_lanes, axis=0)  # (L, 2)
    # For each waypoint, find nearest point in all_pts
    dists = np.zeros(traj_xy.shape[0])
    for i, w in enumerate(traj_xy):
        d = np.linalg.norm(all_pts - w[None, :], axis=1)
        dists[i] = d.min()
    return float(dists.mean())

def main():
    cfg = Config(ARGS)
    ns = NoiseScheduleVP()
    print("Loading 5 experts...")
    models = {e: load_model(e, cfg) for e in EXPERTS}
    models = {e: m for e, m in models.items() if m is not None}

    s1 = json.load(open('checkpoints/overnight_20260510/stage1_multiego.json'))
    s1_idx = {(r['map'], r['ego_idx'], r['expert']): r for r in s1['rows']}

    out = {'description': 'LFS — pick expert by mean trajectory-to-route-lane distance',
           'rows': []}

    for tag, osm in UNSEEN:
        try: llmap = convert_lanelet(osm)
        except: continue
        if not llmap.lanelets: continue
        egos = pick_n_egos(llmap, N_EGO, seed=SEED)
        print(f"\n=== {tag} ===")
        for ego_idx, ego_ll in enumerate(egos):
            cl = ego_ll.centerline
            if cl.shape[0] < 2: continue
            cx, cy = float(cl[0,0]), float(cl[0,1])
            heading = float(np.arctan2(cl[1,1]-cl[0,1], cl[1,0]-cl[0,0]))
            inputs = build_inputs(llmap, (cx, cy), heading)
            inputs_norm = cfg.observation_normalizer({
                k: (v.clone() if isinstance(v, torch.Tensor) else v) for k,v in inputs.items()})
            route_np = inputs['route_lanes'].cpu().numpy()[0]

            scene = {'map': tag, 'ego_idx': ego_idx,
                     'lfs_score': {}, 'lfs_max_lat': {}, 'lfs_pick': None}
            for expert, model in models.items():
                torch.manual_seed(0)
                if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)
                with torch.no_grad():
                    ego_traj = differentiable_dpm_solver_sample(model, cfg, inputs_norm, ns, device=DEVICE)
                xy = ego_traj[:, :2].cpu().numpy()
                if not np.all(np.isfinite(xy)): continue
                lfs = lane_dist(xy, route_np)
                scene['lfs_score'][expert] = lfs
                # Look up max_lat from stage1
                r1 = s1_idx.get((tag, ego_idx, expert))
                if r1 and r1.get('max_lat') is not None:
                    scene['lfs_max_lat'][expert] = r1['max_lat']
            if scene['lfs_score']:
                scene['lfs_pick'] = min(scene['lfs_score'].items(), key=lambda kv: kv[1])[0]
            out['rows'].append(scene)

        from collections import Counter
        picks = Counter(r['lfs_pick'] for r in out['rows'] if r['map']==tag and r.get('lfs_pick'))
        print(f"  LFS pick distribution: {dict(picks)}")

    out_path = 'checkpoints/overnight_20260511/stage8_lfs.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\n=== Per-map summary ===")
    for tag in ['aichi_hub_v2','ishio_dai','morikoro']:
        rows = [r for r in out['rows'] if r['map']==tag]
        lfs_lats = [r['lfs_max_lat'].get(r['lfs_pick']) for r in rows
                    if r.get('lfs_pick') and r['lfs_pick'] in r.get('lfs_max_lat', {})]
        unagoya_lats = [r['lfs_max_lat'].get('unagoya') for r in rows
                        if 'unagoya' in r.get('lfs_max_lat', {})]
        unagoya_lats = [x for x in unagoya_lats if x is not None]
        lfs_lats = [x for x in lfs_lats if x is not None]
        print(f"\n  {tag}:  n={len(rows)}")
        if lfs_lats: print(f"    LFS lat:       {np.mean(lfs_lats):.1f} ± {np.std(lfs_lats,ddof=1) if len(lfs_lats)>1 else 0:.1f}")
        if unagoya_lats: print(f"    best-fixed unagoya: {np.mean(unagoya_lats):.1f} ± {np.std(unagoya_lats,ddof=1) if len(unagoya_lats)>1 else 0:.1f}")

main()
