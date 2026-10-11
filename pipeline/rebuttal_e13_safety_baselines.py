#!/usr/bin/env python3
"""E13 (rebuttal, R1-W3): safety/comfort metrics for the matched baselines.
Re-evaluates the saved sp80 checkpoints (target_only, L2-SP, ft_best_expert,
joint_replay frng; seeds 42-44, each on its own tuning-eval slice) reporting
corridor-exit, jerk p95, lateral-acc p95, and forward progress alongside
max-lat — answers whether L2-SP's max-lat win survives safety metrics."""
import os, sys, json
os.environ.setdefault('LD_LIBRARY_PATH', '/opt/autoware/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
sys.path.insert(0, '/opt/autoware/local/lib/python3.10/dist-packages')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner_ros')
sys.path.insert(0, 'third_party/Diffusion-Planner/preference_optimization')
sys.path.insert(0, '.')
sys.path.insert(0, '.')

import numpy as np
import torch
from collections import defaultdict
from diffusion_planner.utils.config import Config
from diffusion_planner.model.diffusion_utils.dpm_solver_pytorch import NoiseScheduleVP
from train_reward_backprop import differentiable_dpm_solver_sample
from train_molora import set_active_expert, ODD_MAP_TO_IDX
from overnight_stage1_multiego import ARGS
from aichi_disjoint_split import disjoint_scene_split
from rebuttal_e12_matched_baselines import build_model
from jointreplay_aichi_safety import trajectory_metrics

DEVICE = torch.device('cuda')
OUT = 'outputs/rebuttal'
CONFIGS = {   # mode -> (checkpoint pattern, eval expert)
    'target_only':   ('e12_target_only_sp80_s{seed}_lora.pth', None),
    'l2sp':          ('e12_target_only_l2sp0.001_sp80_s{seed}_lora.pth', None),
    'ft_best_expert': ('e12_ft_best_expert_sp80_s{seed}_lora.pth', 'unagoya'),
    'joint_replay':  ('e12_joint_replay_sp80_frng_s{seed}_lora.pth', None),
    'static_rule':   (None, 'unagoya'),
}

cfg = Config(ARGS); ns = NoiseScheduleVP()
model = build_model(cfg)
model.eval()
orig_lora = {k: v.detach().clone() for k, v in model.state_dict().items()
             if 'shared_A' in k or 'shared_B' in k or 'expert_A' in k or 'expert_B' in k}

eval_cache = {}
def slice_for(seed):
    if seed not in eval_cache:
        _, ev = disjoint_scene_split(n_train_per_map=80, n_eval_per_map=10,
                                     train_seed=seed, eval_seed=7)
        eval_cache[seed] = ev
    return eval_cache[seed]

def eval_cfg(expert_tag):
    idx = ODD_MAP_TO_IDX[expert_tag] if expert_tag else None
    set_active_expert(model, idx)

results = {}
for name, (pat, expert) in CONFIGS.items():
    per_seed = []
    for seed in (42, 43, 44, 45, 46, 47, 48, 49):
        model.load_state_dict(orig_lora, strict=False)
        if pat:
            lora = torch.load(f'{OUT}/{pat.format(seed=seed)}', map_location=DEVICE, weights_only=False)
            model.load_state_dict(lora, strict=False)
        eval_cfg(expert)
        mets = defaultdict(list)
        for s in slice_for(seed):
            try:
                dn = cfg.observation_normalizer(
                    {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in s['inputs'].items()})
                with torch.no_grad():
                    tj = differentiable_dpm_solver_sample(model, cfg, dn, ns, device=DEVICE)
                xy = tj[:, :2].cpu().numpy()
                if not np.all(np.isfinite(xy)):
                    continue
                m = trajectory_metrics(xy)
                for k in ('corridor_exit_m', 'jerk_p95', 'lat_acc_p95'):
                    mets[k].append(m[k])
                mets['max_lat'].append(float(np.max(np.abs(xy[:, 1]))))
                mets['fwd_progress'].append(float(xy[-1, 0] - xy[0, 0]))
            except Exception:
                pass
        per_seed.append({k: float(np.mean(v)) for k, v in mets.items()})
        print(f'  {name} s{seed}: ' + '  '.join(f'{k}={np.mean(v):.2f}' for k, v in mets.items()))
    agg = {k: dict(mean=float(np.mean([r[k] for r in per_seed])),
                   std=float(np.std([r[k] for r in per_seed], ddof=1)))
           for k in per_seed[0]}
    results[name] = agg
    print(f'{name}: ' + '  '.join(f"{k}={v['mean']:.2f}±{v['std']:.2f}" for k, v in agg.items()))

model.load_state_dict(orig_lora, strict=False)
json.dump(dict(description=__doc__, results=results),
          open(f'{OUT}/e13_safety_baselines.json', 'w'), indent=1)
print('saved e13_safety_baselines.json')
