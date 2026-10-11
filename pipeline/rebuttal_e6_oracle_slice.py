#!/usr/bin/env python3
"""Oracle headroom on the tuning-eval slices (split80, train_seed in {42,43,44}):
roll out all four experts + shared on each seed's seed-7 eval pool, compute
static (unagoya), per-scene oracle, and map-mean fleet stats.
Answers co-author point 1: same-slice §3(a) numbers for the rebuttal table."""
import os, sys, json
os.environ.setdefault('LD_LIBRARY_PATH', '/opt/autoware/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
sys.path.insert(0, '/opt/autoware/local/lib/python3.10/dist-packages')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner_ros')
sys.path.insert(0, 'third_party/Diffusion-Planner/preference_optimization')
sys.path.insert(0, '.')

import numpy as np
import torch
from collections import defaultdict

from diffusion_planner.utils.config import Config
from diffusion_planner.model.diffusion_utils.dpm_solver_pytorch import NoiseScheduleVP
from train_molora import ODD_MAP_TO_IDX
from overnight_stage1_multiego import ARGS
from aichi_disjoint_split import disjoint_scene_split
from rebuttal_e12_matched_baselines import build_model, eval_aichi_per_scene

EXPERTS = ['hongo', 'kashiwa', 'shalun', 'unagoya']
cfg = Config(ARGS); ns = NoiseScheduleVP()
model = build_model(cfg)

out = {}
for seed in (42, 43, 44, 45, 46, 47, 48, 49):
    _, eval_scenes = disjoint_scene_split(n_train_per_map=80, n_eval_per_map=10,
                                          train_seed=seed, eval_seed=7)
    per_scene = defaultdict(dict)   # key -> {expert: max_lat, 'map': m}
    for tag in EXPERTS + ['shared']:
        idx = ODD_MAP_TO_IDX[tag] if tag in ODD_MAP_TO_IDX else None
        rows, agg, fleet = eval_aichi_per_scene(model, cfg, ns, eval_scenes, idx)
        for r in rows:
            k = (r['map'], r['lanelet_idx'])
            per_scene[k]['map'] = r['map']
            per_scene[k][tag] = r['max_lat'] if r.get('finite') else None
        print(f'seed {seed} {tag:8s} fleet(scene)={fleet:.2f}', flush=True)

    def mapmean(get):
        by = defaultdict(list)
        for k, v in per_scene.items():
            x = get(v)
            if x is not None:
                by[v['map']].append(x)
        return float(np.mean([np.mean(vs) for vs in by.values()]))

    res = dict(
        static=mapmean(lambda v: v.get('unagoya')),
        shared=mapmean(lambda v: v.get('shared')),
        oracle=mapmean(lambda v: (min(x for x in (v.get(e) for e in EXPERTS) if x is not None)
                                  if any(v.get(e) is not None for e in EXPERTS) else None)),
    )
    res['headroom'] = res['static'] - res['oracle']
    out[seed] = res
    print(f"seed {seed}: static={res['static']:.2f} oracle={res['oracle']:.2f} "
          f"headroom={res['headroom']:.2f} shared={res['shared']:.2f}", flush=True)

arr = lambda k: np.array([out[s][k] for s in (42, 43, 44, 45, 46, 47, 48, 49)])
summary = {k: dict(mean=float(arr(k).mean()), std=float(arr(k).std(ddof=1)))
           for k in ('static', 'shared', 'oracle', 'headroom')}
print('\nSUMMARY (map-mean, 3 slices):')
for k, v in summary.items():
    print(f"  {k:9s} {v['mean']:.2f}±{v['std']:.2f}")
json.dump(dict(per_seed={str(k): v for k, v in out.items()}, summary=summary),
          open('outputs/rebuttal/e6_oracle_slice.json', 'w'), indent=1)
