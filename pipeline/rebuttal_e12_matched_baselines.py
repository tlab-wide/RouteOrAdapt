#!/usr/bin/env python3
"""Rebuttal E1/E2: matched-target-data baselines vs joint replay.

Modes (all: same 20 Aichi train scenes, 500 opt steps, GRAD_ACCUM 4, LR 5e-5,
same LoRA ranks, disjoint seed-7 eval pool — identical to stage24_v2 protocol):

  target_only     E1: fine-tune shared+expert LoRA on the 20 target scenes ONLY
                      (no prod replay).  Answers Nkac Q1 / 4EkA W3.
  ft_best_expert  E2: fine-tune the static-rule expert (unagoya/ODD-D path:
                      shared + unagoya expert LoRA) on the 20 target scenes;
                      train AND eval with the unagoya expert active.
                      Answers Nkac Q4 / 4EkA W3.
  joint_replay    seed-matched rerun of stage24_v2 (20 aichi : 200 prod).

Per-scene eval rows are saved (fills the Pool-B per-scene gap for paired tests).

Usage: python3 rebuttal_e12_matched_baselines.py --mode target_only --seed 1
"""
import os, sys, json, random, time, argparse
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
from overnight_stage12_park_repair import eval_on_prod, PROD_ODDS
from overnight_stage1_multiego import metrics_for_traj, ARGS, BASE_PTH
from aichi_disjoint_split import disjoint_scene_split

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = 'outputs/rebuttal'
MOLORA_FULL = 'checkpoints/molora_v2_output/molora_full.pth'

N_OPT_STEPS = 500
GRAD_ACCUM = 4
LR = 5e-5
N_PROD_PER_ODD = 50
SHARED_RANK = 4
EXPERT_RANK = 8
UNA_IDX = ODD_MAP_TO_IDX['unagoya']


def build_model(cfg, scratch_shared=False):
    model = Diffusion_Planner(cfg).to(DEVICE)
    base_ckpt = torch.load(BASE_PTH, map_location=DEVICE, weights_only=False)
    base_sd = base_ckpt.get('ema_state_dict') or base_ckpt['model']
    base_sd = {k.replace('module.', ''): v for k, v in base_sd.items()}
    model.load_state_dict(base_sd, strict=False)
    apply_molora(model.decoder.dit, n_experts=N_EXPERTS,
                 shared_rank=SHARED_RANK, expert_rank=EXPERT_RANK)
    mfull = torch.load(MOLORA_FULL, map_location=DEVICE, weights_only=False)
    msd = {k.replace('module.', ''): v for k, v in mfull['model'].items()}
    if scratch_shared:
        # stage32 protocol: experts from molora_full, shared LoRA stays at
        # fresh kaiming/zero init (paper's deployed joint-replay recipe)
        msd = {k: v for k, v in msd.items() if 'shared_A' not in k and 'shared_B' not in k}
    model.load_state_dict(msd, strict=False)
    return model


def set_trainable(model, mode):
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for m in model.modules():
        if hasattr(m, 'shared_A') and hasattr(m, 'shared_B'):
            m.shared_A.weight.requires_grad_(True); m.shared_B.weight.requires_grad_(True)
            n += m.shared_A.weight.numel() + m.shared_B.weight.numel()
        if hasattr(m, 'expert_As'):
            if mode == 'ft_best_expert':
                idxs = [UNA_IDX]
            else:
                idxs = range(len(m.expert_As))
            for i in idxs:
                m.expert_As[i].weight.requires_grad_(True)
                m.expert_Bs[i].weight.requires_grad_(True)
                n += m.expert_As[i].weight.numel() + m.expert_Bs[i].weight.numel()
    return n


def eval_aichi_per_scene(model, cfg, ns, eval_scenes, expert_idx):
    model.eval()
    set_active_expert(model, expert_idx)
    rows, by_map = [], defaultdict(list)
    for s in eval_scenes:
        row = dict(map=s['map'], lanelet_idx=s.get('lanelet_idx'))
        try:
            inputs_norm = cfg.observation_normalizer(
                {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in s['inputs'].items()})
            with torch.no_grad():
                tj = differentiable_dpm_solver_sample(model, cfg, inputs_norm, ns, device=DEVICE)
            xy = tj[:, :2].cpu().numpy()
            m = metrics_for_traj(xy)
            row.update(max_lat=m.get('max_lat'), fwd_progress=m.get('fwd_progress'),
                       finite=bool(m.get('finite')))
            if m.get('finite'):
                by_map[s['map']].append(m['max_lat'])
        except Exception as e:
            row.update(error=f'{type(e).__name__}: {str(e)[:80]}', finite=False)
        rows.append(row)
    agg = {tag: {'mean': float(np.mean(v)) if v else None,
                 'std': float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                 'n': len(v)} for tag, v in by_map.items()}
    finite_lats = [r['max_lat'] for r in rows if r.get('finite')]
    fleet = float(np.mean(finite_lats)) if finite_lats else None
    return rows, agg, fleet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True,
                    choices=['target_only', 'ft_best_expert', 'joint_replay'])
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--scratch_shared', action='store_true',
                    help='stage32 protocol: re-init shared LoRA from scratch')
    ap.add_argument('--l2sp', type=float, default=0.0,
                    help='L2-SP coefficient anchoring trainable params to init')
    ap.add_argument('--ewc', type=float, default=0.0,
                    help='EWC coefficient; diagonal Fisher on prod data, anchor at init')
    ap.add_argument('--per', type=float, default=0.0,
                    help='prioritized replay: exponent alpha for '
                         'reward-deficit-weighted prod sampling (0 = uniform)')
    ap.add_argument('--fixed_rng', action='store_true',
                    help='stage26 rng convention: prod pool Random(0), order Random(1)')
    ap.add_argument('--split80', action='store_true',
                    help='stage26/tab:sample_complexity protocol: 80/10 split, '
                         'train_seed=args.seed (selects the 20 scenes), eval_seed=7')
    args = ap.parse_args()
    mode_tag = args.mode + ('_scratch' if args.scratch_shared else '')
    if args.l2sp > 0:
        mode_tag += f'_l2sp{args.l2sp:g}'
    if args.ewc > 0:
        mode_tag += f'_ewc{args.ewc:g}'
    if args.per > 0:
        mode_tag += f'_per{args.per:g}'
    if args.split80:
        mode_tag += '_sp80'
    if args.fixed_rng:
        mode_tag += '_frng'
    tag = f'{mode_tag}_s{args.seed}'
    print(f'[e12] mode={mode_tag} seed={args.seed}')

    cfg = Config(ARGS); ns = NoiseScheduleVP()
    if args.split80:
        train_scenes, eval_scenes = disjoint_scene_split(
            n_train_per_map=80, n_eval_per_map=10, train_seed=args.seed, eval_seed=7)
    else:
        train_scenes, eval_scenes = disjoint_scene_split(
            n_train_per_map=30, n_eval_per_map=10, train_seed=42, eval_seed=7)
    aichi_train_pool = train_scenes[:20]
    print(f'  aichi train {len(aichi_train_pool)} / eval {len(eval_scenes)}')

    model = build_model(cfg, scratch_shared=args.scratch_shared)
    n_train = set_trainable(model, args.mode)
    print(f'  trainable params: {n_train:,}')
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    anchor = {id(p): p.detach().clone() for p in model.parameters() if p.requires_grad} \
        if (args.l2sp > 0 or args.ewc > 0) else None
    fisher = None

    # deployment expert for aichi eval: static-rule path for ft_best_expert, shared for others
    eval_expert = UNA_IDX if args.mode == 'ft_best_expert' else None
    # expert active while training on aichi scenes
    train_expert = UNA_IDX if args.mode == 'ft_best_expert' else None

    # training pool
    rng = random.Random(0 if args.fixed_rng else args.seed)
    all_train = json.load(open('checkpoints/npz_multimap/path_list_train_v3.json'))
    per_map_train = {m: [p for p in all_train if m in p.lower()] for m in PROD_ODDS}
    pool = [('aichi', s['map'], s) for s in aichi_train_pool]
    if args.mode == 'joint_replay':
        for odd in PROD_ODDS:
            ps = per_map_train[odd][:]
            rng.shuffle(ps)
            for p in ps[:N_PROD_PER_ODD]:
                pool.append(('prod', odd, p))
    rng.shuffle(pool)
    n_prod = sum(1 for k, _, _ in pool if k == 'prod')
    print(f'  pool: {len(pool) - n_prod} aichi + {n_prod} prod')

    if args.ewc > 0:
        print('[e12] computing diagonal Fisher on prod data (25/ODD)...')
        trainables = [p for p in model.parameters() if p.requires_grad]
        fisher = {id(p): torch.zeros_like(p) for p in trainables}
        rng_f = random.Random(123)
        n_f = 0
        model.train()
        for odd in PROD_ODDS:
            ps = per_map_train[odd][:]
            rng_f.shuffle(ps)
            for path in ps[:25]:
                try:
                    set_active_expert(model, ODD_MAP_TO_IDX.get(odd))
                    data = load_npz_data(path, DEVICE)
                    dn = cfg.observation_normalizer(
                        {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data.items()})
                    ego_traj, deltas = differentiable_dpm_solver_sample(
                        model, cfg, dn, ns, device=DEVICE, return_deltas=True)
                    rl = data.get('route_lanes'); route_xy = None
                    if rl is not None:
                        rp = rl[0].cpu().numpy()[:, :, :2].reshape(-1, 2)
                        rp = rp[np.any(np.abs(rp) > 0.1, axis=1)]
                        if len(rp) > 2:
                            route_xy = torch.tensor(rp, dtype=torch.float32, device=DEVICE)
                    nbf = data.get('neighbor_agents_future')
                    nf = nbf[0].to(DEVICE) if nbf is not None else None
                    reward = odd_conditioned_reward(ego_traj, route_xy, nf, deltas, odd)
                    model.zero_grad()
                    (-reward).backward()
                    for pmt in trainables:
                        if pmt.grad is not None:
                            fisher[id(pmt)] += pmt.grad.detach() ** 2
                    n_f += 1
                except Exception:
                    model.zero_grad()
        model.zero_grad()
        for k in fisher:
            fisher[k] /= max(1, n_f)
        print(f'  fisher from {n_f} prod samples')

    print('[e12] PRE eval...')
    pre_rows, pre_agg, pre_fleet = eval_aichi_per_scene(model, cfg, ns, eval_scenes, eval_expert)
    print(f'  [pre] fleet={pre_fleet:.2f}  ' +
          '  '.join(f'{t}:{v["mean"]:.2f}' for t, v in pre_agg.items() if v['mean'] is not None))
    set_active_expert(model, None)
    pre_prod = eval_on_prod(model, cfg, per_map_train, n_per_map=10)

    print(f'[e12] training {N_OPT_STEPS} steps...')
    model.train()
    fi = step = 0; rewards = []; t0 = time.time()
    opt.zero_grad()
    rng2 = random.Random(1 if args.fixed_rng else args.seed + 1000)
    prod_items = [x for x in pool if x[0] == 'prod']
    aichi_items = [x for x in pool if x[0] == 'aichi']
    per_pri = [1.0] * len(prod_items) if (args.per > 0 and prod_items) else None
    reward_hi = 0.0
    while step < N_OPT_STEPS:
        if per_pri is not None:
            w = [(pr + 1e-2) ** args.per for pr in per_pri]
            picks = rng2.choices(range(len(prod_items)), weights=w, k=len(prod_items))
            epoch = [(k, t, pl, None) for k, t, pl in aichi_items] + \
                    [prod_items[j] + (j,) for j in picks]
        else:
            epoch = [(k, t, pl, None) for k, t, pl in pool]
        rng2.shuffle(epoch)
        for kind, mtag, payload, pidx in epoch:
            try:
                if kind == 'prod':
                    set_active_expert(model, ODD_MAP_TO_IDX.get(mtag))
                    data = load_npz_data(payload, DEVICE)
                else:
                    set_active_expert(model, train_expert)
                    data = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                            for k, v in payload['inputs'].items()}
                dn = cfg.observation_normalizer(
                    {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data.items()})
                ego_traj, deltas = differentiable_dpm_solver_sample(
                    model, cfg, dn, ns, device=DEVICE, return_deltas=True)
                rl = data.get('route_lanes'); route_xy = None
                if rl is not None:
                    rp = rl[0].cpu().numpy()[:, :, :2].reshape(-1, 2)
                    rp = rp[np.any(np.abs(rp) > 0.1, axis=1)]
                    if len(rp) > 2:
                        route_xy = torch.tensor(rp, dtype=torch.float32, device=DEVICE)
                nbf = data.get('neighbor_agents_future')
                nf = nbf[0].to(DEVICE) if nbf is not None else None
                reward_map_name = mtag if kind == 'prod' else 'unagoya'
                reward = odd_conditioned_reward(ego_traj, route_xy, nf, deltas, reward_map_name)
                loss = -reward
                if anchor is not None:
                    if fisher is not None:
                        sp = sum((fisher[id(p)] * (p - anchor[id(p)]) ** 2).sum()
                                 for p in model.parameters() if p.requires_grad)
                        loss = loss + args.ewc * sp
                    else:
                        sp = sum(((p - anchor[id(p)]) ** 2).sum()
                                 for p in model.parameters() if p.requires_grad)
                        loss = loss + args.l2sp * sp
                (loss / GRAD_ACCUM).backward()
                rewards.append(float(reward.detach()))
                if per_pri is not None and pidx is not None:
                    r = float(reward.detach())
                    reward_hi = max(reward_hi, r)
                    per_pri[pidx] = max(reward_hi - r, 0.0)
                fi += 1
                if fi % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                    opt.step(); opt.zero_grad()
                    step += 1
                    if step % 100 == 0 or step == 1:
                        print(f'  step {step}/{N_OPT_STEPS}  reward={np.mean(rewards[-200:]):.2f}  '
                              f'{(time.time()-t0)/60:.1f} min', flush=True)
                    if step >= N_OPT_STEPS:
                        break
            except Exception:
                opt.zero_grad()
    print(f'[e12] train done {(time.time()-t0)/60:.1f} min')

    print('[e12] POST eval...')
    post_rows, post_agg, post_fleet = eval_aichi_per_scene(model, cfg, ns, eval_scenes, eval_expert)
    print(f'  [post] fleet={post_fleet:.2f}  ' +
          '  '.join(f'{t}:{v["mean"]:.2f}' for t, v in post_agg.items() if v['mean'] is not None))
    set_active_expert(model, None)
    post_prod = eval_on_prod(model, cfg, per_map_train, n_per_map=10)
    forg_rd = sum(post_prod[o]['route_dev_mean'] - pre_prod[o]['route_dev_mean'] for o in PROD_ODDS)
    forg_pr = sum(post_prod[o]['progress_mean'] - pre_prod[o]['progress_mean'] for o in PROD_ODDS)
    print(f'  forgetting: route_dev {forg_rd:+.2f} m, progress {forg_pr:+.1f} m')

    lora_sd = {k: v.cpu() for k, v in model.state_dict().items()
               if 'shared_A' in k or 'shared_B' in k or 'expert_A' in k or 'expert_B' in k}
    torch.save(lora_sd, f'{OUT_DIR}/e12_{tag}_lora.pth')

    out = dict(description=__doc__, mode=mode_tag, seed=args.seed,
               n_opt_steps=N_OPT_STEPS, lr=LR, grad_accum=GRAD_ACCUM,
               n_train_params=n_train, eval_expert=('unagoya' if eval_expert is not None else 'shared'),
               mix=f'{len(pool) - n_prod} aichi : {n_prod} prod',
               pre_fleet=pre_fleet, post_fleet=post_fleet,
               pre_aichi=pre_agg, post_aichi=post_agg,
               pre_prod=pre_prod, post_prod=post_prod,
               forgetting_sum_route_dev_m=forg_rd, forgetting_sum_progress_m=forg_pr,
               pre_rows=pre_rows, post_rows=post_rows,
               train_minutes=(time.time() - t0) / 60)
    with open(f'{OUT_DIR}/e12_{tag}.json', 'w') as f:
        json.dump(out, f, indent=1)
    print(f'[e12] saved {OUT_DIR}/e12_{tag}.json')


if __name__ == '__main__':
    main()
