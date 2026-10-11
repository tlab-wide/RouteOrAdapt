"""TENT-fullblood: maximum-effort TENT replication.

Goes beyond stage29 (lightweight TENT-LN-K4) and tent_strong (1-pass with
prod warmup). This script implements:

  - **K = 32 samples** per scene (sample-variance entropy surrogate)
  - **Adam lr = 1e-3** (Wang 2021 default)
  - **All LayerNorm/GroupNorm affines trainable**
  - **No prod warmup**: vanilla TENT directly on the OOD stream
  - **10 epochs through the 24-scene Aichi stream** with online state
    persistence (240 adapt steps total)
  - **Eval after every epoch** to track convergence
  - **SAR-style sample selection** as a bonus stronger variant: only
    backprop on the half of scenes with LOWEST K-sample variance
    (entropy-equivalent "high-confidence" filtering, Niu 2023)

If even this full-blood TENT fails to beat the static geographic-prior
baseline (~13.74 m fleet-mean on best-fixed ODD-D), the reviewer's
challenge that "TENT failed because we under-powered the recipe" is
settled. If it wins, the paper baseline number needs updating.

Output: outputs/tent_fullblood.json
"""
import os, sys, json, random, time
os.environ.setdefault('LD_LIBRARY_PATH', '/opt/autoware/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
sys.path.insert(0, '/opt/autoware/local/lib/python3.10/dist-packages')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner_ros')
sys.path.insert(0, 'third_party/Diffusion-Planner/preference_optimization')
sys.path.insert(0, '.')

import numpy as np
import torch, torch.nn as nn
from torch import optim
from collections import defaultdict

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from diffusion_planner_ros.lanelet2_utils.lanelet_converter import convert_lanelet
from diffusion_planner.model.diffusion_utils.dpm_solver_pytorch import NoiseScheduleVP

from overnight_stage1_multiego import (
    build_inputs, pick_n_egos, UNSEEN, N_EGO, SEED, ARGS, BASE_PTH, metrics_for_traj,
)
from train_molora import apply_molora, set_active_expert, N_EXPERTS
from train_reward_backprop import differentiable_dpm_solver_sample

DEVICE = torch.device('cuda')
OUT = 'outputs/tent_fullblood.json'
MOLORA_FULL = 'checkpoints/molora_v2_output/molora_full.pth'
SHARED_RANK = 4
EXPERT_RANK = 8

K_SAMPLES = 32
LR_LN = 1e-3
N_EPOCHS = 10
SAR_KEEP_FRAC = 0.5  # only backprop on bottom-50% variance scenes


def build_molora_model(cfg):
    model = Diffusion_Planner(cfg).to(DEVICE)
    base = torch.load(BASE_PTH, map_location=DEVICE, weights_only=False)
    bsd = base.get('ema_state_dict') or base['model']
    bsd = {k.replace('module.', ''): v for k, v in bsd.items()}
    model.load_state_dict(bsd, strict=False)
    apply_molora(model.decoder.dit, n_experts=N_EXPERTS,
                 shared_rank=SHARED_RANK, expert_rank=EXPERT_RANK)
    mfull = torch.load(MOLORA_FULL, map_location=DEVICE, weights_only=False)
    msd = {k.replace('module.', ''): v for k, v in mfull['model'].items()}
    model.load_state_dict(msd, strict=False)
    return model


def make_only_ln_trainable(model):
    for p in model.parameters():
        p.requires_grad_(False)
    n_train = 0; n_modules = 0
    for m in model.modules():
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            n_modules += 1
            if m.weight is not None:
                m.weight.requires_grad_(True); n_train += m.weight.numel()
            if m.bias is not None:
                m.bias.requires_grad_(True); n_train += m.bias.numel()
    return n_train, n_modules


def k_sample_variance(model, cfg, inputs, ns, K=K_SAMPLES):
    trajs = []
    for k in range(K):
        torch.manual_seed(int(7919 * k + 1))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(7919 * k + 1))
        inputs_norm = cfg.observation_normalizer({k_: (v.clone() if isinstance(v, torch.Tensor) else v) for k_, v in inputs.items()})
        tj = differentiable_dpm_solver_sample(model, cfg, inputs_norm, ns, device=DEVICE)
        trajs.append(tj[:, :2])
    stacked = torch.stack(trajs, dim=0)
    var_per_step = stacked.var(dim=0, unbiased=False).sum(dim=-1)
    return var_per_step.mean()


def rollout_max_lat(model, cfg, inputs, ns):
    try:
        inputs_norm = cfg.observation_normalizer({k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()})
        with torch.no_grad():
            tj = differentiable_dpm_solver_sample(model, cfg, inputs_norm, ns, device=DEVICE)
        xy = tj[:, :2].cpu().numpy()
        m = metrics_for_traj(xy)
        if m.get('finite'): return m['max_lat']
    except Exception:
        return None
    return None


def build_aichi_scenes(cfg):
    scenes = []
    for tag, osm in UNSEEN:
        try:
            llmap = convert_lanelet(osm)
        except Exception:
            continue
        egos = pick_n_egos(llmap, N_EGO, seed=SEED)
        for ego_idx, ego_ll in enumerate(egos):
            cl = ego_ll.centerline
            if cl.shape[0] < 2: continue
            cx, cy = float(cl[0, 0]), float(cl[0, 1])
            heading = float(np.arctan2(cl[1, 1]-cl[0, 1], cl[1, 0]-cl[0, 0]))
            try:
                inputs = build_inputs(llmap, (cx, cy), heading)
                scenes.append(dict(map=tag, ego_idx=ego_idx, inputs=inputs))
            except Exception:
                pass
    return scenes


def eval_per_map(model, cfg, ns, scenes):
    model.eval()
    by_map = defaultdict(list)
    for s in scenes:
        lat = rollout_max_lat(model, cfg, s['inputs'], ns)
        if lat is not None: by_map[s['map']].append(lat)
    summary = {m: {'mean': float(np.mean(v)), 'std': float(np.std(v, ddof=1)) if len(v)>1 else 0.0, 'n': len(v)}
               for m, v in by_map.items()}
    fleet = float(np.mean([v['mean'] for v in summary.values()])) if summary else None
    return summary, fleet


def main():
    cfg = Config(ARGS); ns = NoiseScheduleVP()
    print(f"[tent-fullblood] K={K_SAMPLES} lr={LR_LN} epochs={N_EPOCHS} optim=Adam SAR_keep_frac={SAR_KEEP_FRAC}")

    # Setup
    aichi_scenes = build_aichi_scenes(cfg)
    print(f"  Aichi pool: {len(aichi_scenes)} scenes")

    # ── Variant 1: Vanilla TENT-fullblood (all scenes back-propagated) ──
    print(f"\n══════════════════════════════════════════")
    print(f"  VARIANT 1: Vanilla TENT-fullblood (Wang 2021)")
    print(f"══════════════════════════════════════════")
    model = build_molora_model(cfg); set_active_expert(model, None)
    n_train, n_mod = make_only_ln_trainable(model)
    print(f"  trainable: {n_train:,} params across {n_mod} LN/GN modules")
    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR_LN)

    # Pre-eval
    pre_sum, pre_fleet = eval_per_map(model, cfg, ns, aichi_scenes)
    print(f"  pre fleet: {pre_fleet:.2f} m")

    epoch_history_vanilla = []
    rng = random.Random(2026)
    for ep in range(N_EPOCHS):
        model.train()
        order = list(aichi_scenes); rng.shuffle(order)
        ep_losses = []; t0 = time.time()
        for s in order:
            try:
                opt.zero_grad()
                loss = k_sample_variance(model, cfg, s['inputs'], ns, K=K_SAMPLES)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p2 for p2 in model.parameters() if p2.requires_grad], 1.0)
                opt.step()
                ep_losses.append(float(loss.detach()))
            except Exception as e:
                continue
        post_sum, post_fleet = eval_per_map(model, cfg, ns, aichi_scenes)
        delta = post_fleet - pre_fleet if pre_fleet else None
        ep_loss_mean = float(np.mean(ep_losses)) if ep_losses else None
        epoch_history_vanilla.append(dict(epoch=ep+1, loss=ep_loss_mean,
                                           per_map=post_sum, fleet=post_fleet,
                                           delta_vs_pre=delta))
        print(f"  ep {ep+1}/{N_EPOCHS}  K-var={ep_loss_mean:.4f}  fleet={post_fleet:.2f}  Δ={delta:+.2f} m  ({(time.time()-t0):.0f}s)", flush=True)

    # ── Variant 2: SAR-style sample selection (Niu 2023 stronger variant) ──
    print(f"\n══════════════════════════════════════════")
    print(f"  VARIANT 2: SAR-style sample-selective TENT (Niu 2023)")
    print(f"══════════════════════════════════════════")
    model_sar = build_molora_model(cfg); set_active_expert(model_sar, None)
    n_train, n_mod = make_only_ln_trainable(model_sar)
    opt_sar = optim.Adam([p for p in model_sar.parameters() if p.requires_grad], lr=LR_LN)
    epoch_history_sar = []
    rng2 = random.Random(2026)
    for ep in range(N_EPOCHS):
        model_sar.train()
        order = list(aichi_scenes); rng2.shuffle(order)
        # First pass: compute K-var for every scene without grad to rank
        per_scene_var = []
        with torch.no_grad():
            for s in order:
                try:
                    v = k_sample_variance(model_sar, cfg, s['inputs'], ns, K=K_SAMPLES)
                    per_scene_var.append(float(v))
                except Exception:
                    per_scene_var.append(float('inf'))
        # Keep bottom SAR_KEEP_FRAC by variance (most "confident" scenes per Niu 2023)
        thresh = np.quantile(per_scene_var, SAR_KEEP_FRAC)
        keep_mask = [v <= thresh for v in per_scene_var]
        # Second pass: backprop only on kept scenes
        ep_losses = []; t0 = time.time(); n_used = 0
        for s, keep in zip(order, keep_mask):
            if not keep: continue
            try:
                opt_sar.zero_grad()
                loss = k_sample_variance(model_sar, cfg, s['inputs'], ns, K=K_SAMPLES)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p2 for p2 in model_sar.parameters() if p2.requires_grad], 1.0)
                opt_sar.step()
                ep_losses.append(float(loss.detach()))
                n_used += 1
            except Exception:
                continue
        post_sum_sar, post_fleet_sar = eval_per_map(model_sar, cfg, ns, aichi_scenes)
        delta = post_fleet_sar - pre_fleet if pre_fleet else None
        ep_loss_mean = float(np.mean(ep_losses)) if ep_losses else None
        epoch_history_sar.append(dict(epoch=ep+1, n_used=n_used, loss=ep_loss_mean,
                                       per_map=post_sum_sar, fleet=post_fleet_sar,
                                       delta_vs_pre=delta))
        print(f"  ep {ep+1}/{N_EPOCHS}  used={n_used}/{len(order)}  K-var={ep_loss_mean:.4f}  fleet={post_fleet_sar:.2f}  Δ={delta:+.2f} m  ({(time.time()-t0):.0f}s)", flush=True)

    # Summary
    static_geographic_prior = 13.74  # paper Table tab:routing best-fixed ODD-D
    joint_replay_headline = 4.26  # paper headline
    out = dict(
        description='TENT-fullblood: vanilla + SAR variants, 10 epochs online',
        config=dict(K_samples=K_SAMPLES, lr_ln=LR_LN, optimizer='Adam',
                    n_epochs=N_EPOCHS, sar_keep_frac=SAR_KEEP_FRAC),
        n_aichi=len(aichi_scenes), n_train=n_train, n_ln_modules=n_mod,
        pre_aichi=pre_sum, pre_fleet=pre_fleet,
        vanilla=dict(epoch_history=epoch_history_vanilla,
                     final_fleet=epoch_history_vanilla[-1]['fleet'] if epoch_history_vanilla else None,
                     best_fleet=min((h['fleet'] for h in epoch_history_vanilla if h['fleet']), default=None),
                     best_epoch=int(np.argmin([h['fleet'] for h in epoch_history_vanilla]) + 1) if epoch_history_vanilla else None),
        sar=dict(epoch_history=epoch_history_sar,
                 final_fleet=epoch_history_sar[-1]['fleet'] if epoch_history_sar else None,
                 best_fleet=min((h['fleet'] for h in epoch_history_sar if h['fleet']), default=None),
                 best_epoch=int(np.argmin([h['fleet'] for h in epoch_history_sar]) + 1) if epoch_history_sar else None),
        reference=dict(static_geographic_prior=static_geographic_prior,
                       joint_replay_headline=joint_replay_headline),
    )
    json.dump(out, open(OUT, 'w'), indent=2, default=float)
    print(f"\n=== TENT-fullblood SUMMARY ===")
    print(f"  pre-TENT fleet (frozen): {pre_fleet:.2f} m")
    print(f"  vanilla best  fleet: {out['vanilla']['best_fleet']:.2f} m  (ep {out['vanilla']['best_epoch']})")
    print(f"  vanilla final fleet: {out['vanilla']['final_fleet']:.2f} m")
    print(f"  SAR best  fleet:     {out['sar']['best_fleet']:.2f} m  (ep {out['sar']['best_epoch']})")
    print(f"  SAR final fleet:     {out['sar']['final_fleet']:.2f} m")
    print(f"  ── reference baselines ──")
    print(f"  static geographic prior:  {static_geographic_prior:.2f} m")
    print(f"  joint replay (headline):  {joint_replay_headline:.2f} m")
    print(f"\nSaved {OUT}")


if __name__ == '__main__':
    main()
