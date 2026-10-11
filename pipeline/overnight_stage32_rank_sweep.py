#!/usr/bin/env python3
"""
Stage 32: Joint-Replay shared-LoRA RANK sweep (Idea E).

We sweep shared_rank in {2, 4, 8, 16} on the joint-replay recipe of
Stage 24 v2, with the protocol of the rank-8 replication of Stage 22:
the shared module at each rank is initialised from scratch (zero
B-matrix, kaiming A-matrix), the four per-ODD experts are kept frozen
at their pre-trained values, and joint replay trains for 500 steps
with the canonical 10:1 prod:aichi mix at n_aichi=20.

For rank=4 we additionally run a row that initialises the shared module
from the pre-trained molora_full checkpoint (the original training-run
inheritance) to expose the capacity-vs-inheritance trade-off explicitly.

Output: stage32_rank_sweep.json
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
N_AICHI = 20            # match Stage 24 v2 canonical
EXPERT_RANK = 8

RANK_SWEEP = [2, 4, 8, 16]


def build_model(cfg, shared_rank, inherit_shared=False):
    """Build a MoLoRA backbone with the given shared_rank. Per-expert modules
    are always loaded from molora_full. The shared module is either initialised
    from zero (B=0, A=kaiming) or inherited from molora_full (only when shared_rank=4)."""
    model = Diffusion_Planner(cfg).to(DEVICE)
    base_ckpt = torch.load(BASE_PTH, map_location=DEVICE, weights_only=False)
    base_sd = base_ckpt.get('ema_state_dict') or base_ckpt['model']
    base_sd = {k.replace('module.', ''): v for k, v in base_sd.items()}
    model.load_state_dict(base_sd, strict=False)
    apply_molora(model.decoder.dit, n_experts=N_EXPERTS,
                 shared_rank=shared_rank, expert_rank=EXPERT_RANK)
    mfull = torch.load(MOLORA_FULL, map_location=DEVICE, weights_only=False)
    msd = {k.replace('module.', ''): v for k, v in mfull['model'].items()}
    if inherit_shared:
        # Load shared weights too. Only works if shared_rank == 4.
        model.load_state_dict(msd, strict=False)
    else:
        # Skip shared_A / shared_B in msd — they will remain at kaiming/zero init.
        msd_no_shared = {k: v for k, v in msd.items() if 'shared_' not in k}
        model.load_state_dict(msd_no_shared, strict=False)

    # Trainable: full MoLoRA stack (shared + experts).
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
        except Exception as e:
            n_skip += 1
    per_map = {tag: {'mean': float(np.mean(v)), 'std': float(np.std(v, ddof=1)) if len(v)>1 else 0.0, 'n': len(v)}
               for tag, v in by_map.items() if v}
    fleet = float(np.mean([v['mean'] for v in per_map.values()])) if per_map else float('inf')
    return per_map, fleet, n_skip


def fleet_mean(d):
    vals = [v['mean'] for v in d.values() if v.get('mean') is not None]
    return float(np.mean(vals)) if vals else None


def train_joint_replay(cfg, ns, model, train_scenes, prod_pool):
    """Train joint-replay 500 steps with 10:1 prod:aichi at n_aichi=20."""
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    rng = random.Random(0)
    aichi_pool = [('aichi', s['map'], s) for s in train_scenes[:N_AICHI]]
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
    return model, final_reward, elapsed


def main():
    print(f"[stage32] Joint-Replay shared-LoRA RANK sweep")
    print(f"  ranks: {RANK_SWEEP}; N_AICHI={N_AICHI}; N_PROD_PER_ODD={N_PROD_PER_ODD}; steps={N_OPT_STEPS}")
    cfg = Config(ARGS); ns = NoiseScheduleVP()
    train_scenes, eval_scenes = disjoint_scene_split(n_train_per_map=30, n_eval_per_map=10,
                                                      train_seed=42, eval_seed=7)
    print(f"  Aichi train pool: {len(train_scenes)} (using first {N_AICHI});  eval pool: {len(eval_scenes)}")

    # Prod pool
    all_train = json.load(open('checkpoints/npz_multimap/path_list_train_v3.json'))
    per_map_train = {m: [p for p in all_train if m in p.lower()] for m in PROD_ODDS}
    rng = random.Random(0)
    prod_pool = []
    for odd in PROD_ODDS:
        ps = per_map_train[odd][:]
        rng.shuffle(ps)
        for p in ps[:N_PROD_PER_ODD]:
            prod_pool.append(('prod', odd, p))

    out = {
        'description': 'Stage 32: joint-replay shared-LoRA rank sweep (from-scratch shared init)',
        'rank_sweep': RANK_SWEEP, 'n_aichi': N_AICHI, 'n_opt_steps': N_OPT_STEPS,
        'eval_seed': 7, 'train_seed': 42,
        'rows': [],
    }

    # PRE eval: base + frozen experts only (shared from molora_full at rank 4) as a reference baseline
    # Note: we use rank-4 inherit as the "before joint-replay" baseline to match Stage 24 v2 PRE eval.
    print(f"\n--- PRE eval reference (rank-4 inherit, frozen MoLoRA) ---")
    model_pre, _ = build_model(cfg, shared_rank=4, inherit_shared=True)
    pre_per_map, pre_fleet, _ = eval_aichi(model_pre, cfg, ns, eval_scenes)
    pre_prod = eval_on_prod(model_pre, cfg, per_map_train, n_per_map=10)
    pre_sum_route = sum(v['route_dev_mean'] for v in pre_prod.values())
    pre_sum_prog  = sum(v['progress_mean']  for v in pre_prod.values())
    print(f"  PRE Aichi fleet = {pre_fleet:.2f}  pre_prod sum_route_dev = {pre_sum_route:.2f}  sum_progress = {pre_sum_prog:.2f}")
    out['pre_baseline'] = {
        'fleet_mean': pre_fleet,
        'per_map': pre_per_map,
        'pre_prod': pre_prod,
        'sum_route_dev': pre_sum_route,
        'sum_progress':  pre_sum_prog,
    }
    del model_pre; torch.cuda.empty_cache()

    # Sweep
    for rank in RANK_SWEEP:
        print(f"\n--- shared_rank = {rank} (init from scratch) ---")
        model, n_train = build_model(cfg, shared_rank=rank, inherit_shared=False)
        print(f"  trainable params (shared+experts): {n_train:,}")
        model, final_reward, elapsed = train_joint_replay(cfg, ns, model, train_scenes, prod_pool)
        # POST eval
        post_per_map, post_fleet, n_skip = eval_aichi(model, cfg, ns, eval_scenes)
        set_active_expert(model, None)
        post_prod = eval_on_prod(model, cfg, per_map_train, n_per_map=10)
        post_sum_route = sum(v['route_dev_mean'] for v in post_prod.values())
        post_sum_prog  = sum(v['progress_mean']  for v in post_prod.values())
        delta_fleet = post_fleet - pre_fleet
        delta_route = post_sum_route - pre_sum_route
        delta_prog  = post_sum_prog  - pre_sum_prog
        row = {
            'rank': rank, 'n_train_params': n_train,
            'post_aichi':       post_per_map,
            'post_fleet_mean':  post_fleet,
            'post_prod':        post_prod,
            'post_sum_route_dev': post_sum_route,
            'post_sum_progress':  post_sum_prog,
            'delta_fleet':         delta_fleet,
            'delta_sum_route_dev': delta_route,
            'delta_sum_progress':  delta_prog,
            'final_reward': final_reward,
            'training_minutes': elapsed,
            'n_eval_skips': n_skip,
        }
        out['rows'].append(row)
        print(f"  rank={rank}: post Aichi fleet = {post_fleet:.2f} (Δ {delta_fleet:+.2f})  "
              f"sum_route_dev Δ {delta_route:+.2f}  sum_progress Δ {delta_prog:+.2f}  "
              f"reward {final_reward:.1f}  elapsed {elapsed:.1f} min", flush=True)
        # Save the rank=4 from-scratch checkpoint for Tuesday's real-vehicle deployment
        # (best Pareto point on Aichi accuracy at the canonical CRC32 eval pool).
        if rank == 4:
            torch.save({'model': model.state_dict(),
                        'shared_rank': rank, 'expert_rank': EXPERT_RANK,
                        'inherit_shared': False, 'n_aichi': N_AICHI,
                        'n_opt_steps': N_OPT_STEPS, 'eval_seed': 7, 'train_seed': 42,
                        'post_fleet_mean': post_fleet,
                        'protocol': 'joint-replay-from-scratch-shared'},
                       f'{OUT_DIR}/stage32_joint_replay_from_scratch_r4.pth')
            print(f"  saved checkpoint: stage32_joint_replay_from_scratch_r4.pth (best from-scratch)", flush=True)
        del model; torch.cuda.empty_cache()

    with open(f'{OUT_DIR}/stage32_rank_sweep.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n[stage32] saved {OUT_DIR}/stage32_rank_sweep.json")

    # Final table print
    print(f"\n=== Rank-sweep summary ===")
    print(f"  PRE (rank-4 inherit, no joint-replay): fleet={pre_fleet:.2f}  Δroute=ref  Δprog=ref")
    for row in out['rows']:
        print(f"  rank={row['rank']:2}: fleet={row['post_fleet_mean']:.2f} (Δ {row['delta_fleet']:+.2f})  "
              f"Δroute={row['delta_sum_route_dev']:+.2f}  Δprog={row['delta_sum_progress']:+.2f}  "
              f"params={row['n_train_params']:,}  wallclock={row['training_minutes']:.1f} min")


if __name__ == '__main__':
    main()
