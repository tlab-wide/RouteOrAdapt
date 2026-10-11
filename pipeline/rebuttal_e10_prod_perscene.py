#!/usr/bin/env python3
"""E10 (rebuttal, feedback-v2 point 1): per-scene paired test of joint-replay
source-domain regression. eval_on_prod uses the LAST n_per_map=10 paths per ODD
(deterministic), so pre/post route-dev pairs per path are valid. 40 scenes x 3
JR seeds (frng checkpoints) = 120 paired deltas -> Wilcoxon with actual power,
replacing the underpowered per-ODD n=4 phrasing."""
import os, sys, json
os.environ.setdefault('LD_LIBRARY_PATH', '/opt/autoware/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
sys.path.insert(0, '/opt/autoware/local/lib/python3.10/dist-packages')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner_ros')
sys.path.insert(0, 'third_party/Diffusion-Planner/preference_optimization')
sys.path.insert(0, '.')

import numpy as np
import torch
from scipy import stats
from diffusion_planner.utils.config import Config
from train_molora import set_active_expert, ODD_MAP_TO_IDX
from overnight_stage12_park_repair import PROD_ODDS, generate_deterministic_trajectory
from overnight_stage1_multiego import ARGS
from utils import load_npz_data
from rebuttal_e12_matched_baselines import build_model

DEVICE = torch.device('cuda')
OUT_DIR = 'outputs/rebuttal'

cfg = Config(ARGS)
model = build_model(cfg)
model.eval()
orig_lora = {k: v.detach().clone() for k, v in model.state_dict().items()
             if 'shared_A' in k or 'shared_B' in k or 'expert_A' in k or 'expert_B' in k}

all_train = json.load(open('checkpoints/npz_multimap/path_list_train_v3.json'))
per_map = {m: [p for p in all_train if m in p.lower()] for m in PROD_ODDS}


def route_dev_per_scene(model):
    rows = {}
    for odd in PROD_ODDS:
        set_active_expert(model, ODD_MAP_TO_IDX.get(odd))
        for p in per_map[odd][-10:]:
            try:
                data = load_npz_data(p, DEVICE)
                with torch.no_grad():
                    ego = generate_deterministic_trajectory(model, cfg, data, device=DEVICE)
                xy = ego[:, :2]
                if not np.all(np.isfinite(xy)):
                    continue
                rl = data.get('route_lanes'); rd = 0.0
                if rl is not None:
                    rp = rl[0].cpu().numpy()[:, :, :2].reshape(-1, 2)
                    rp = rp[np.any(np.abs(rp) > 0.1, axis=1)]
                    if len(rp) > 2:
                        idx = np.linspace(0, xy.shape[0] - 1, 10, dtype=int)
                        rd = float(np.mean([np.min(np.linalg.norm(rp - xy[i], axis=1)) for i in idx]))
                rows[(odd, p)] = rd
            except Exception:
                pass
    return rows


print('[e10] PRE per-scene route-dev (base MoLoRA stack)...')
pre = route_dev_per_scene(model)
print(f'  {len(pre)} scenes')

deltas_all = []
per_seed = {}
for seed in (42, 43, 44, 45, 46, 47, 48, 49):
    sd = dict(orig_lora)
    lora = torch.load(f'{OUT_DIR}/e12_joint_replay_sp80_frng_s{seed}_lora.pth',
                      map_location=DEVICE, weights_only=False)
    model.load_state_dict({**sd, **lora}, strict=False)
    post = route_dev_per_scene(model)
    ks = sorted(set(pre) & set(post))
    d = np.array([post[k] - pre[k] for k in ks])
    per_seed[seed] = dict(n=len(d), mean=float(d.mean()),
                          n_worse=int((d > 1e-9).sum()), n_better=int((d < -1e-9).sum()))
    deltas_all.extend(d)
    print(f'  seed {seed}: n={len(d)} mean_delta={d.mean():+.3f} m  worse/better={per_seed[seed]["n_worse"]}/{per_seed[seed]["n_better"]}')

model.load_state_dict(orig_lora, strict=False)
d = np.array(deltas_all)
nz = d[np.abs(d) > 1e-9]
w = stats.wilcoxon(nz) if len(nz) else None
print(f'\nPOOLED: n={len(d)} paired (scene x seed), mean {d.mean():+.3f} m, median {np.median(d):+.3f} m')
print(f'nonzero n={len(nz)}, Wilcoxon vs 0: p={w.pvalue:.4f}' if w else 'all zero')

json.dump(dict(description=__doc__, n_pairs=len(d), mean_delta=float(d.mean()),
               median_delta=float(np.median(d)), n_nonzero=len(nz),
               wilcoxon_p=(float(w.pvalue) if w else None), per_seed=per_seed),
          open(f'{OUT_DIR}/e10_prod_perscene.json', 'w'), indent=1)
print('saved e10_prod_perscene.json')
