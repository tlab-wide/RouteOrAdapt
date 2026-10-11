#!/usr/bin/env python3
"""E8 (rebuttal): expert-selection test on a held-out region whose geographically
nearest training ODD is NOT ODD-D — Nishishinjuku (central Tokyo).

Geographic distances: Shinjuku->hongo ~6 km, ->kashiwa ~27 km, ->unagoya ~260 km,
->shalun ~2100 km => the static geographic rule selects HONGO here.
Question (4EkA minor / iZGU W3+Q2): is the nearest expert also the best fixed
expert when the rule must switch away from ODD-D?

Caveat (stated in rebuttal): this region's data contributed to the SHARED module
during multi-domain training (it is the excluded 5th expert's ODD), so this tests
expert SELECTION only, not held-out adaptation. All four expert configs share the
same shared module, so the between-expert comparison is unaffected.
"""
import os, sys, json
os.environ.setdefault('LD_LIBRARY_PATH', '/opt/autoware/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
sys.path.insert(0, '/opt/autoware/local/lib/python3.10/dist-packages')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner_ros')
sys.path.insert(0, 'third_party/Diffusion-Planner/preference_optimization')
sys.path.insert(0, '.')

import numpy as np
import torch
from diffusion_planner.utils.config import Config
from diffusion_planner.model.diffusion_utils.dpm_solver_pytorch import NoiseScheduleVP
from diffusion_planner_ros.lanelet2_utils.lanelet_converter import convert_lanelet
from train_reward_backprop import differentiable_dpm_solver_sample
from train_molora import set_active_expert, ODD_MAP_TO_IDX
from overnight_stage1_multiego import build_inputs, pick_n_egos, metrics_for_traj, ARGS
from rebuttal_e12_matched_baselines import build_model

OSM = '/autoware_map/nishishinjuku_autoware_map/lanelet2_map.osm'
EXPERTS = ['hongo', 'kashiwa', 'shalun', 'unagoya']
N_EGO = 10
SEED = 42
DEVICE = torch.device('cuda')

cfg = Config(ARGS); ns = NoiseScheduleVP()
model = build_model(cfg)
model.eval()

llmap = convert_lanelet(OSM)
print(f'{len(llmap.lanelets)} lanelets loaded')
egos = pick_n_egos(llmap, N_EGO, seed=SEED)

rows = []
for ego_idx, ego_ll in enumerate(egos):
    cl = ego_ll.centerline
    if cl.shape[0] < 2:
        continue
    cx, cy = float(cl[0, 0]), float(cl[0, 1])
    heading = float(np.arctan2(cl[1, 1] - cl[0, 1], cl[1, 0] - cl[0, 0]))
    inputs = build_inputs(llmap, (cx, cy), heading)
    inputs_norm = cfg.observation_normalizer(
        {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()})
    row = dict(ego_idx=ego_idx, lanelet_id=ego_ll.id)
    for tag in EXPERTS + ['shared']:
        idx = ODD_MAP_TO_IDX[tag] if tag in ODD_MAP_TO_IDX else None
        set_active_expert(model, idx)
        try:
            with torch.no_grad():
                tj = differentiable_dpm_solver_sample(model, cfg, inputs_norm, ns, device=DEVICE)
            m = metrics_for_traj(tj[:, :2].cpu().numpy())
            row[tag] = m['max_lat'] if m.get('finite') else None
            row[f'{tag}_fwd'] = m.get('fwd_progress')
        except Exception as e:
            row[tag] = None
    rows.append(row)
    print(f"ego {ego_idx}: " + "  ".join(f"{t}={row[t]:.2f}" if row[t] is not None else f"{t}=NaN"
                                          for t in EXPERTS + ['shared']))

print('\n=== per-expert fleet mean (finite scenes) ===')
stats = {}
for t in EXPERTS + ['shared']:
    v = [r[t] for r in rows if r[t] is not None]
    stats[t] = dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)), n=len(v))
    print(f"{t:8s} {stats[t]['mean']:6.2f}±{stats[t]['std']:.2f}  n={stats[t]['n']}")

oracle_picks = [min(EXPERTS, key=lambda t: r[t] if r[t] is not None else 1e9) for r in rows]
from collections import Counter
print('\noracle picks:', Counter(oracle_picks))
best_fixed = min(EXPERTS, key=lambda t: stats[t]['mean'])
print(f'best fixed expert: {best_fixed}   static geographic rule picks: hongo')

json.dump(dict(description=__doc__, rows=rows, stats=stats,
               oracle_picks=oracle_picks, best_fixed=best_fixed,
               static_rule_pick='hongo'),
          open('outputs/rebuttal/e8_shinjuku_switching.json', 'w'), indent=1)
print('saved e8_shinjuku_switching.json')
