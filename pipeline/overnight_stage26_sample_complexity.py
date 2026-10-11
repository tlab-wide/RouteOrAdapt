#!/usr/bin/env python3
"""
Stage 26: Joint-replay sample-complexity sweep.

For n_aichi in {5, 10, 20, 40, 80}, train joint-replay (full MoLoRA stack,
500 steps, prod:aichi = 10:1 ratio) starting from molora_full, then evaluate
on the seed=7 disjoint Aichi eval pool AND on prod-ODD.

Produces curves of:
  - Aichi fleet-mean max_lat vs n_aichi
  - prod-ODD sum_route_dev forgetting vs n_aichi

Output: stage26_sample_complexity.json
"""
import os, sys, json, random, time
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

SWEEP_N_AICHI = [5, 10, 20, 40, 80]  # train pool sizes


def build_model(cfg, ns):
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
    return model, n_train


def eval_aichi(model, cfg, ns, eval_scenes):
    model.eval(); set_active_expert(model, None)
    by_map = defaultdict(list)
    for s in eval_scenes:
        inputs_norm = cfg.observation_normalizer({k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in s['inputs'].items()})
        try:
            with torch.no_grad():
                tj = differentiable_dpm_solver_sample(model, cfg, inputs_norm, ns, device=DEVICE)
            xy = tj[:, :2].cpu().numpy()
            m = metrics_for_traj(xy)
            if m.get('finite'): by_map[s['map']].append(m['max_lat'])
        except Exception: pass
    return {tag: {'mean': float(np.mean(v)) if v else None,
                  'std':  float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                  'n': len(v)} for tag, v in by_map.items()}


def fleet_mean(d):
    vals = [v['mean'] for v in d.values() if v.get('mean') is not None]
    return float(np.mean(vals)) if vals else None


def train_joint_replay(cfg, ns, n_aichi, train_scenes, prod_pool):
    """Train joint-replay with n_aichi Aichi train scenes mixed at 1:N_PROD_PER_ODD with prod."""
    model, n_params = build_model(cfg, ns)
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    rng = random.Random(0)

    aichi_pool = [('aichi', s['map'], s) for s in train_scenes[:n_aichi]]
    pool = prod_pool + aichi_pool
    rng.shuffle(pool)

    rng2 = random.Random(1)
    fi = step = 0
    rewards = []
    t0 = time.time()
    opt.zero_grad()
    model.train()
    while step < N_OPT_STEPS:
        rng2.shuffle(pool)
        for kind, tag, payload in pool:
            try:
                if kind == 'prod':
                    set_active_expert(model, ODD_MAP_TO_IDX.get(tag))
                    data = load_npz_data(payload, DEVICE)
                else:
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
                    if step >= N_OPT_STEPS: break
            except Exception:
                opt.zero_grad()
    final_reward = float(np.mean(rewards[-200:])) if rewards else None
    elapsed = (time.time() - t0) / 60
    return model, n_params, final_reward, elapsed


def main():
    print(f"[stage26] Joint-replay sample-complexity sweep: n_aichi ∈ {SWEEP_N_AICHI}")
    cfg = Config(ARGS); ns = NoiseScheduleVP()

    train_scenes, eval_scenes = disjoint_scene_split(n_train_per_map=80, n_eval_per_map=10,
                                                      train_seed=42, eval_seed=7)
    print(f"  available Aichi train scenes: {len(train_scenes)}  eval: {len(eval_scenes)}")
    # Cap n_aichi at available scenes
    sweep = [n for n in SWEEP_N_AICHI if n <= len(train_scenes)]
    if max(SWEEP_N_AICHI) > len(train_scenes):
        print(f"  capping sweep to {sweep} (only {len(train_scenes)} Aichi train scenes available)")

    # Prod-ODD pool
    all_train = json.load(open('checkpoints/npz_multimap/path_list_train_v3.json'))
    per_map_train = {m: [p for p in all_train if m in p.lower()] for m in PROD_ODDS}
    rng = random.Random(0)
    prod_pool = []
    for odd in PROD_ODDS:
        ps = per_map_train[odd][:]
        rng.shuffle(ps)
        for p in ps[:N_PROD_PER_ODD]:
            prod_pool.append(('prod', odd, p))
    print(f"  prod pool: {len(prod_pool)} ({N_PROD_PER_ODD} × {len(PROD_ODDS)})")

    # PRE eval (frozen MoLoRA) -- baseline
    print(f"\n[stage26] PRE eval (frozen MoLoRA)")
    pre_model, _ = build_model(cfg, ns)
    set_active_expert(pre_model, None)
    pre_aichi = eval_aichi(pre_model, cfg, ns, eval_scenes)
    pre_prod = eval_on_prod(pre_model, cfg, per_map_train, n_per_map=10)
    pre_fleet = fleet_mean(pre_aichi)
    pre_prod_route = sum(v['route_dev_mean'] for v in pre_prod.values())
    pre_prod_prog  = sum(v['progress_mean']  for v in pre_prod.values())
    print(f"  pre Aichi fleet-mean = {pre_fleet:.2f} m  (hub={pre_aichi.get('aichi_hub_v2',{}).get('mean')}, "
          f"ishio={pre_aichi.get('ishio_dai',{}).get('mean')}, morikoro={pre_aichi.get('morikoro',{}).get('mean')})")
    print(f"  pre prod sum_route_dev = {pre_prod_route:.2f}  sum_progress = {pre_prod_prog:.2f}")
    del pre_model; torch.cuda.empty_cache()

    out = {
        'description': 'Stage 26: joint-replay sample-complexity sweep over n_aichi',
        'sweep_n_aichi': sweep,
        'n_opt_steps': N_OPT_STEPS,
        'n_prod_per_odd': N_PROD_PER_ODD,
        'eval_seed': 7, 'train_seed': 42,
        'pre_aichi': pre_aichi, 'pre_prod': pre_prod,
        'pre_fleet_mean': pre_fleet,
        'pre_sum_route_dev': pre_prod_route, 'pre_sum_progress': pre_prod_prog,
        'sweep': [],
    }

    for n_aichi in sweep:
        print(f"\n[stage26] Training joint-replay n_aichi = {n_aichi}")
        model, n_params, final_reward, elapsed = train_joint_replay(cfg, ns, n_aichi, train_scenes, prod_pool)
        post_aichi = eval_aichi(model, cfg, ns, eval_scenes)
        set_active_expert(model, None)
        post_prod  = eval_on_prod(model, cfg, per_map_train, n_per_map=10)
        post_fleet = fleet_mean(post_aichi)
        post_prod_route = sum(v['route_dev_mean'] for v in post_prod.values())
        post_prod_prog  = sum(v['progress_mean']  for v in post_prod.values())
        delta_aichi = (post_fleet or 0) - (pre_fleet or 0)
        delta_route = post_prod_route - pre_prod_route
        delta_prog  = post_prod_prog  - pre_prod_prog
        print(f"  n_aichi={n_aichi}: Aichi fleet {post_fleet:.2f} (Δ {delta_aichi:+.2f})  "
              f"sum_route_dev Δ {delta_route:+.2f}  sum_progress Δ {delta_prog:+.2f}  "
              f"reward {final_reward:.1f}  elapsed {elapsed:.1f} min", flush=True)
        out['sweep'].append({
            'n_aichi':        n_aichi,
            'post_aichi':     post_aichi,
            'post_fleet_mean': post_fleet,
            'post_prod':      post_prod,
            'post_sum_route_dev': post_prod_route,
            'post_sum_progress':  post_prod_prog,
            'delta_fleet':         delta_aichi,
            'delta_sum_route_dev': delta_route,
            'delta_sum_progress':  delta_prog,
            'final_reward':        final_reward,
            'training_minutes':    elapsed,
        })
        del model; torch.cuda.empty_cache()

    with open(f'{OUT_DIR}/stage26_sample_complexity.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n[stage26] saved {OUT_DIR}/stage26_sample_complexity.json")


if __name__ == '__main__':
    main()
