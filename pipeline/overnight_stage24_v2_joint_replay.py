#!/usr/bin/env python3
"""
Stage 24 v2: Joint Replay continual-learning baseline with DISJOINT train/eval.

Fixes v1's data leakage: v1 used pick_n_egos(seed=42) for both training and
evaluation. v2 uses seed=42 for training (mixed into prod-ODD scenes at 1:10)
and seed=7 for evaluation, drawn from disjoint lanelet subsets.

Output: stage24_v2_joint_replay.json.
"""
import os, sys, json, math, random, time
os.environ.setdefault('LD_LIBRARY_PATH', '/opt/autoware/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
sys.path.insert(0, '/opt/autoware/local/lib/python3.10/dist-packages')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner_ros')
sys.path.insert(0, 'third_party/Diffusion-Planner/preference_optimization')
sys.path.insert(0, '.')

import numpy as np
import torch
from torch import optim
from collections import defaultdict

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from diffusion_planner.model.diffusion_utils.dpm_solver_pytorch import NoiseScheduleVP
from train_reward_backprop import differentiable_dpm_solver_sample
from train_molora import (
    apply_molora, set_active_expert, odd_conditioned_reward,
    ODD_MAP_TO_IDX, N_EXPERTS,
)
from utils import load_npz_data
from overnight_stage12_park_repair import (
    perturb_park_inplace, eval_on_prod, PROD_ODDS, PARK_PROB,
)
from overnight_stage1_multiego import metrics_for_traj, ARGS, BASE_PTH
from aichi_disjoint_split import disjoint_scene_split

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = 'checkpoints/overnight_20260513'
MOLORA_FULL = 'checkpoints/molora_v2_output/molora_full.pth'

N_OPT_STEPS = 500
GRAD_ACCUM = 4
LR = 5e-5
N_PROD_PER_ODD = 50
SHARED_RANK = 4
EXPERT_RANK = 8


def main():
    print(f"[stage24_v2] Joint replay with DISJOINT train (seed=42) / eval (seed=7) Aichi split")
    cfg = Config(ARGS); ns = NoiseScheduleVP()

    train_scenes, eval_scenes = disjoint_scene_split(n_train_per_map=30, n_eval_per_map=10,
                                                      train_seed=42, eval_seed=7)
    print(f"  Aichi train: {len(train_scenes)}  Aichi eval: {len(eval_scenes)}  disjoint: yes")
    # Use the first 20 train scenes as the new-ODD pool (matching v1's 1:10 ratio)
    aichi_train_pool = train_scenes[:20]

    print("[stage24_v2] Loading MoLoRA backbone")
    model = Diffusion_Planner(cfg).to(DEVICE)
    base_ckpt = torch.load(BASE_PTH, map_location=DEVICE, weights_only=False)
    base_sd = base_ckpt.get('ema_state_dict') or base_ckpt['model']
    base_sd = {k.replace('module.', ''): v for k, v in base_sd.items()}
    model.load_state_dict(base_sd, strict=False)
    apply_molora(model.decoder.dit, n_experts=N_EXPERTS,
                 shared_rank=SHARED_RANK, expert_rank=EXPERT_RANK)
    mfull = torch.load(MOLORA_FULL, map_location=DEVICE, weights_only=False)
    msd = {k.replace('module.', ''): v for k, v in mfull['model'].items()}
    model.load_state_dict(msd, strict=False)

    # Unfreeze ALL LoRA params
    for p in model.parameters(): p.requires_grad_(False)
    n_train = 0
    for m in model.modules():
        if hasattr(m, 'shared_A') and hasattr(m, 'shared_B'):
            m.shared_A.weight.requires_grad_(True); m.shared_B.weight.requires_grad_(True)
            n_train += m.shared_A.weight.numel() + m.shared_B.weight.numel()
        if hasattr(m, 'expert_As'):
            for ea in m.expert_As:
                ea.weight.requires_grad_(True); n_train += ea.weight.numel()
            for eb in m.expert_Bs:
                eb.weight.requires_grad_(True); n_train += eb.weight.numel()
    print(f"  trainable params (shared+experts): {n_train:,}")
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)

    # Prod-ODD train pool
    all_train = json.load(open('checkpoints/npz_multimap/path_list_train_v3.json'))
    per_map_train = {m: [p for p in all_train if m in p.lower()] for m in PROD_ODDS}
    rng = random.Random(0)
    prod_pool = []
    for odd in PROD_ODDS:
        ps = per_map_train[odd][:]
        rng.shuffle(ps)
        for p in ps[:N_PROD_PER_ODD]:
            prod_pool.append(('prod', odd, p))
    aichi_pool = [('aichi', s['map'], s) for s in aichi_train_pool]
    pool = prod_pool + aichi_pool
    rng.shuffle(pool)
    print(f"  pool: {len(prod_pool)} prod + {len(aichi_pool)} aichi (ratio = {len(prod_pool)/max(1,len(aichi_pool)):.0f}:1)")

    # PRE eval on disjoint eval set
    def eval_aichi(model):
        model.eval()
        set_active_expert(model, None)
        by_map = defaultdict(list)
        n_skip = 0
        for s in eval_scenes:
            try:
                inputs_norm = cfg.observation_normalizer({k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in s['inputs'].items()})
                with torch.no_grad():
                    tj = differentiable_dpm_solver_sample(model, cfg, inputs_norm, ns, device=DEVICE)
                xy = tj[:, :2].cpu().numpy()
                m = metrics_for_traj(xy)
                if m.get('finite'):
                    by_map[s['map']].append(m['max_lat'])
                else:
                    n_skip += 1
                    if n_skip <= 3:
                        print(f'  [eval skip] {s["map"]}/idx={s.get("lanelet_idx")}: non-finite trajectory', flush=True)
            except Exception as e:
                n_skip += 1
                if n_skip <= 3:
                    print(f'  [eval skip] {s["map"]}/idx={s.get("lanelet_idx")}: {type(e).__name__}: {str(e)[:120]}', flush=True)
        if n_skip:
            print(f'  [eval] total skips: {n_skip}/{len(eval_scenes)}', flush=True)
        return {tag: {'mean': float(np.mean(v)) if v else None,
                      'std':  float(np.std(v, ddof=1)) if len(v)>1 else 0.0,
                      'n': len(v)} for tag, v in by_map.items()}

    print("\n[stage24_v2] PRE eval (frozen MoLoRA, shared-only)...")
    pre_aichi = eval_aichi(model)
    for tag, v in pre_aichi.items():
        if v['mean'] is not None: print(f"  [pre]  {tag}: {v['mean']:.2f}±{v['std']:.2f}  n={v['n']}")
    pre_prod = eval_on_prod(model, cfg, per_map_train, n_per_map=10)
    for odd, v in pre_prod.items():
        print(f"  [pre]  {odd}: progress={v['progress_mean']:.1f}  route_dev={v['route_dev_mean']:.2f}  n={v['n']}")

    # Train
    print(f"\n[stage24_v2] Training {N_OPT_STEPS} steps...")
    model.train()
    fi = step = 0; rewards = []; t0 = time.time()
    opt.zero_grad()
    rng2 = random.Random(1)
    while step < N_OPT_STEPS:
        rng2.shuffle(pool)
        for kind, tag, payload in pool:
            try:
                if kind == 'prod':
                    set_active_expert(model, ODD_MAP_TO_IDX.get(tag))
                    data = load_npz_data(payload, DEVICE)
                else:  # aichi
                    set_active_expert(model, None)
                    data = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in payload['inputs'].items()}
                dn = cfg.observation_normalizer({k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data.items()})
                ego_traj, deltas = differentiable_dpm_solver_sample(model, cfg, dn, ns, device=DEVICE, return_deltas=True)
                rl = data.get('route_lanes'); route_xy = None
                if rl is not None:
                    rp = rl[0].cpu().numpy()[:, :, :2].reshape(-1, 2)
                    rp = rp[np.any(np.abs(rp) > 0.1, axis=1)]
                    if len(rp) > 2: route_xy = torch.tensor(rp, dtype=torch.float32, device=DEVICE)
                nbf = data.get('neighbor_agents_future')
                nf = nbf[0].to(DEVICE) if nbf is not None else None
                reward_map_name = tag if kind == 'prod' else 'unagoya'
                reward = odd_conditioned_reward(ego_traj, route_xy, nf, deltas, reward_map_name)
                (-reward / GRAD_ACCUM).backward()
                rewards.append(float(reward.detach()))
                fi += 1
                if fi % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    opt.step(); opt.zero_grad()
                    step += 1
                    if step % 50 == 0 or step == 1:
                        print(f"  step {step}/{N_OPT_STEPS}  reward={np.mean(rewards[-200:]):.2f}  elapsed={(time.time()-t0)/60:.1f} min", flush=True)
                    if step >= N_OPT_STEPS: break
            except Exception: opt.zero_grad()

    print(f"[stage24_v2] Train done. {(time.time()-t0)/60:.1f} min")
    torch.save({'model': model.state_dict(), 'n_steps': N_OPT_STEPS,
                'mix_ratio': f"{len(aichi_pool)}/{len(prod_pool)}", 'eval_seed': 7},
               f'{OUT_DIR}/stage24_v2_joint_replay.pth')

    print("\n[stage24_v2] POST eval...")
    post_aichi = eval_aichi(model)
    for tag, v in post_aichi.items():
        if v['mean'] is not None:
            pre_v = pre_aichi.get(tag, {}).get('mean') or 0
            print(f"  [post] {tag}: {v['mean']:.2f}±{v['std']:.2f}  n={v['n']}  Δ={v['mean']-pre_v:+.2f}")
    set_active_expert(model, None)
    post_prod = eval_on_prod(model, cfg, per_map_train, n_per_map=10)
    for odd, v in post_prod.items():
        pre = pre_prod[odd]
        print(f"  [post] {odd}: progress={v['progress_mean']:.1f} (Δ{v['progress_mean']-pre['progress_mean']:+.1f}), route_dev={v['route_dev_mean']:.2f} (Δ{v['route_dev_mean']-pre['route_dev_mean']:+.2f})")

    forg_rd = sum(post_prod[o]['route_dev_mean'] - pre_prod[o]['route_dev_mean'] for o in PROD_ODDS)
    forg_pr = sum(post_prod[o]['progress_mean'] - pre_prod[o]['progress_mean'] for o in PROD_ODDS)

    out = {
        'description': 'Stage 24 v2: joint replay with DISJOINT train/eval Aichi split (seed=7)',
        'mix_ratio': f"{len(aichi_pool)} aichi : {len(prod_pool)} prod",
        'n_opt_steps': N_OPT_STEPS, 'n_train_params': n_train,
        'train_seed': 42, 'eval_seed': 7,
        'pre_aichi': pre_aichi, 'post_aichi': post_aichi,
        'pre_prod': pre_prod, 'post_prod': post_prod,
        'forgetting_sum_route_dev_m': forg_rd,
        'forgetting_sum_progress_m': forg_pr,
    }
    with open(f'{OUT_DIR}/stage24_v2_joint_replay.json', 'w') as f: json.dump(out, f, indent=2)
    print(f"\n[stage24_v2] saved {OUT_DIR}/stage24_v2_joint_replay.json")


if __name__ == '__main__':
    main()
