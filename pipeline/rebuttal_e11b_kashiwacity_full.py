#!/usr/bin/env python3
"""E11 (rebuttal): clean switching-regime test on the Kashiwa-city DMP bus-route
map — an unseen region geographically ADJACENT to the kashiwa training ODD
(<1 km; hongo ~25 km, unagoya ~330 km, shalun ~2900 km), so the static
geographic rule selects KASHIWA (a strong expert), unlike the three Aichi maps
(ODD-D) and Shinjuku (ODD-A/weak). Ego lanelets inside the kashiwa training-map
bounding box (same MGRS frame) are excluded to keep scenes strictly unseen.
Question: is the nearest expert (kashiwa) the best fixed expert here?"""
import os, sys, json
os.environ.setdefault('LD_LIBRARY_PATH', '/opt/autoware/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
sys.path.insert(0, '/opt/autoware/local/lib/python3.10/dist-packages')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner_ros')
sys.path.insert(0, 'third_party/Diffusion-Planner/preference_optimization')
sys.path.insert(0, '.')

import numpy as np
import torch
import random
from collections import Counter
from diffusion_planner.utils.config import Config
from diffusion_planner.model.diffusion_utils.dpm_solver_pytorch import NoiseScheduleVP
from diffusion_planner_ros.lanelet2_utils.lanelet_converter import convert_lanelet
from train_reward_backprop import differentiable_dpm_solver_sample
from train_molora import set_active_expert, ODD_MAP_TO_IDX
from overnight_stage1_multiego import build_inputs, metrics_for_traj, ARGS
from rebuttal_e12_matched_baselines import build_model

OSM = 'outputs/rebuttal/kashiwa_city_maps/1.2.3-beta.osm'
TRAIN_OSM = '/autoware_map/kashiwa/lanelet2_map.osm'
EXPERTS = ['hongo', 'kashiwa', 'shalun', 'unagoya']
N_EGO = 30
DEVICE = torch.device('cuda')

cfg = Config(ARGS); ns = NoiseScheduleVP()
model = build_model(cfg)
model.eval()

# training-map bbox in the shared local frame
train_map = convert_lanelet(TRAIN_OSM)
tx, ty = [], []
for ll in train_map.lanelets.values():
    cl = ll.centerline
    tx.extend(cl[:, 0].tolist()); ty.extend(cl[:, 1].tolist())
bbox = (min(tx), max(tx), min(ty), max(ty))
print(f'train-map bbox x[{bbox[0]:.0f},{bbox[1]:.0f}] y[{bbox[2]:.0f},{bbox[3]:.0f}]')

llmap = convert_lanelet(OSM)
cands = []
n_excluded = 0
for ll in llmap.lanelets.values():
    cl = ll.centerline
    if cl.shape[0] < 2:
        continue
    cx, cy = float(cl[0, 0]), float(cl[0, 1])
    if bbox[0] - 50 <= cx <= bbox[1] + 50 and bbox[2] - 50 <= cy <= bbox[3] + 50:
        n_excluded += 1
        continue
    cands.append(ll)
print(f'{len(llmap.lanelets)} lanelets, {n_excluded} inside/near train bbox excluded, {len(cands)} candidates')

rng = random.Random(42)
egos = rng.sample(cands, min(N_EGO, len(cands)))

rows = []
for ego_idx, ego_ll in enumerate(egos):
    cl = ego_ll.centerline
    cx, cy = float(cl[0, 0]), float(cl[0, 1])
    heading = float(np.arctan2(cl[1, 1] - cl[0, 1], cl[1, 0] - cl[0, 0]))
    inputs = build_inputs(llmap, (cx, cy), heading)
    inputs_norm = cfg.observation_normalizer(
        {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()})
    row = dict(ego_idx=ego_idx, lanelet_id=ego_ll.id, x=cx, y=cy)
    for tag in EXPERTS + ['shared']:
        idx = ODD_MAP_TO_IDX[tag] if tag in ODD_MAP_TO_IDX else None
        set_active_expert(model, idx)
        try:
            with torch.no_grad():
                tj = differentiable_dpm_solver_sample(model, cfg, inputs_norm, ns, device=DEVICE)
            m = metrics_for_traj(tj[:, :2].cpu().numpy())
            row[tag] = m['max_lat'] if m.get('finite') else None
        except Exception:
            row[tag] = None
    rows.append(row)
    print(f"ego {ego_idx}: " + "  ".join(f"{t}={row[t]:.2f}" if row[t] is not None else f"{t}=NaN"
                                          for t in EXPERTS + ['shared']))

print('\n=== per-expert fleet mean ===')
stats = {}
for t in EXPERTS + ['shared']:
    v = [r[t] for r in rows if r[t] is not None]
    stats[t] = dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, n=len(v))
    print(f"{t:8s} {stats[t]['mean']:6.2f}±{stats[t]['std']:.2f}  n={stats[t]['n']}")

oracle_picks = [min(EXPERTS, key=lambda t: r[t] if r[t] is not None else 1e9) for r in rows]
best_fixed = min(EXPERTS, key=lambda t: stats[t]['mean'])
print('\noracle picks:', Counter(oracle_picks))
print(f'best fixed: {best_fixed}   static geographic rule picks: kashiwa')

json.dump(dict(description=__doc__, rows=rows, stats=stats,
               oracle_picks=oracle_picks, best_fixed=best_fixed,
               static_rule_pick='kashiwa', n_excluded_near_train=n_excluded),
          open('outputs/rebuttal/e11b_kashiwacity_full.json', 'w'), indent=1)
print('saved e11b_kashiwacity_full.json')
