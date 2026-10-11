"""Forensic diagnostics for the TENT/hypernet results — verify trajectories
are NOT degenerate (stationary, fixed, or route-center-collapsed).

For each model state, run on all 24 Aichi scenes and dump:
  - full ego trajectory (80 × 3) per scene
  - forward progress (|xy[T] - xy[0]|), mean speed
  - K-sample variance at deployment time (re-sample K=8 traj per scene)
  - cross-scene trajectory similarity (pairwise L2 between scenes; a
    collapsed model produces near-identical trajectories across all inputs,
    yielding tiny pairwise L2)

Six model states to compare:
  (1) Frozen shared-only MoLoRA (baseline)
  (2) Best-fixed ODD-D (geographic prior)
  (3) vMF soft mixture τ=200
  (4) Hypernet-LN trained (no_jitter, h=64)
  (5) Hypernet + 1-epoch TENT polish
  (6) Best-fixed-D + 1-epoch TENT polish

Output: outputs/forensics.json
"""
import os, sys, json, math, random, time
os.environ.setdefault('LD_LIBRARY_PATH', '/opt/autoware/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
sys.path.insert(0, '/opt/autoware/local/lib/python3.10/dist-packages')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner_ros')
sys.path.insert(0, 'third_party/Diffusion-Planner/preference_optimization')
sys.path.insert(0, '.')
sys.path.insert(0, '.')

import numpy as np
import torch, torch.nn as nn
from torch import optim
from collections import defaultdict

from diffusion_planner.utils.config import Config
from diffusion_planner.model.diffusion_utils.dpm_solver_pytorch import NoiseScheduleVP
from train_molora import set_active_expert, MoLoRALinear
from train_reward_backprop import differentiable_dpm_solver_sample
from overnight_stage1_multiego import ARGS, metrics_for_traj

from tent_fullblood import build_molora_model, make_only_ln_trainable, k_sample_variance, build_aichi_scenes, LR_LN, K_SAMPLES
from hypernet_ln import (
    HyperLNNet, install_ln_hooks, latlon_to_ecef_unit,
    ODD_LATLON, AICHI_LATLON,
)
from tent_init_variants import setup_vmf_mixture

DEVICE = torch.device('cuda')
OUT = 'outputs/forensics.json'

K_DEPLOY = 8  # K samples at deployment time (for K-var measurement)


@torch.no_grad()
def sample_trajectories(model, cfg, inputs, ns, n_samples=1, base_seed=0):
    """Return list of (80, 3) trajectory arrays from n_samples noise seeds."""
    out = []
    for k in range(n_samples):
        torch.manual_seed(int(base_seed * 100003 + k * 7919 + 1))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(base_seed * 100003 + k * 7919 + 1))
        inputs_norm = cfg.observation_normalizer({k_: (v.clone() if isinstance(v, torch.Tensor) else v) for k_, v in inputs.items()})
        try:
            tj = differentiable_dpm_solver_sample(model, cfg, inputs_norm, ns, device=DEVICE)
            out.append(tj[:, :3].cpu().numpy())
        except Exception as e:
            out.append(None)
    return out


def forensic_eval(model, cfg, ns, scenes, label, hyper=None, ell_per_map=None):
    """Run all scenes, collect trajectories + diagnostics."""
    print(f"  forensic eval: {label} on {len(scenes)} scenes")
    rows = []
    for i, s in enumerate(scenes):
        if hyper is not None and ell_per_map is not None:
            ell = ell_per_map.get(s['map'])
            if ell is None: continue
            with torch.no_grad():
                hyper.install(torch.tensor(ell, device=DEVICE).unsqueeze(0))
        # Sample K trajectories for variance
        ks = sample_trajectories(model, cfg, s['inputs'], ns, n_samples=K_DEPLOY, base_seed=i)
        ks = [t for t in ks if t is not None]
        if hyper is not None: hyper.clear()
        if not ks: continue
        stacked = np.stack(ks, axis=0)  # (K, 80, 3)
        mean_traj = stacked.mean(axis=0)  # (80, 3)
        # Forward progress: dist from frame 0 to frame 79
        progress = float(np.linalg.norm(mean_traj[-1, :2] - mean_traj[0, :2]))
        # Mean speed (8s horizon): progress / 8
        mean_speed = progress / 8.0
        # K-sample variance at deployment time (across K samples)
        per_step_var = stacked[:, :, :2].var(axis=0).sum(axis=-1)  # (80,)
        deploy_k_var = float(per_step_var.mean())
        # Max lateral offset of mean trajectory
        m = metrics_for_traj(mean_traj[:, :2])
        max_lat = float(m['max_lat']) if m.get('finite') else None
        # Heading change (does the trajectory turn?)
        if len(mean_traj) > 1:
            dx = mean_traj[1:, 0] - mean_traj[:-1, 0]
            dy = mean_traj[1:, 1] - mean_traj[:-1, 1]
            headings = np.arctan2(dy, dx)
            heading_range = float(headings.max() - headings.min())
        else:
            heading_range = 0.0
        rows.append(dict(
            scene_idx=i, map=s['map'], ego_idx=s.get('ego_idx', -1),
            max_lat=max_lat, progress=progress, mean_speed=mean_speed,
            deploy_k_var=deploy_k_var, heading_range_rad=heading_range,
            mean_traj_xy=mean_traj[:, :2].tolist(),  # save full trajectory
        ))
    # Pairwise cross-scene similarity (collapse detector)
    trajs = np.array([r['mean_traj_xy'] for r in rows])  # (N, 80, 2)
    # Centre each trajectory at its own starting point so we compare shapes
    centred = trajs - trajs[:, :1, :]
    n = len(centred)
    pairwise_l2 = []
    for i in range(n):
        for j in range(i+1, n):
            d = float(np.linalg.norm(centred[i] - centred[j]))
            pairwise_l2.append(d)
    if pairwise_l2:
        pair_mean = float(np.mean(pairwise_l2))
        pair_min = float(np.min(pairwise_l2))
    else:
        pair_mean = pair_min = None
    summary = dict(
        label=label,
        n_scenes=len(rows),
        max_lat_mean=float(np.mean([r['max_lat'] for r in rows if r['max_lat'] is not None])),
        max_lat_std=float(np.std([r['max_lat'] for r in rows if r['max_lat'] is not None], ddof=1)),
        progress_mean=float(np.mean([r['progress'] for r in rows])),
        progress_min=float(np.min([r['progress'] for r in rows])),
        progress_max=float(np.max([r['progress'] for r in rows])),
        mean_speed_mean=float(np.mean([r['mean_speed'] for r in rows])),
        deploy_k_var_mean=float(np.mean([r['deploy_k_var'] for r in rows])),
        deploy_k_var_median=float(np.median([r['deploy_k_var'] for r in rows])),
        heading_range_mean=float(np.mean([r['heading_range_rad'] for r in rows])),
        cross_scene_pairwise_l2_mean=pair_mean,
        cross_scene_pairwise_l2_min=pair_min,
        rows=rows,
    )
    print(f"    max_lat={summary['max_lat_mean']:.2f}±{summary['max_lat_std']:.2f}  "
          f"progress={summary['progress_mean']:.1f}m (min {summary['progress_min']:.1f}, max {summary['progress_max']:.1f})  "
          f"deploy_K-var={summary['deploy_k_var_mean']:.3f}  "
          f"cross-scene_L2={pair_mean:.1f}", flush=True)
    return summary


def main():
    cfg = Config(ARGS); ns = NoiseScheduleVP()
    aichi_scenes = build_aichi_scenes(cfg)
    aichi_ell = {tag: latlon_to_ecef_unit(*ll) for tag, ll in AICHI_LATLON.items()}
    print(f"[forensics] Aichi scenes: {len(aichi_scenes)}")

    results = []

    # ── 1: Frozen shared-only ──────────────────────────────────────────
    model = build_molora_model(cfg)
    set_active_expert(model, None); model.eval()
    results.append(forensic_eval(model, cfg, ns, aichi_scenes, label='1_frozen_shared'))
    del model; torch.cuda.empty_cache()

    # ── 2: Best-fixed ODD-D ─────────────────────────────────────────────
    model = build_molora_model(cfg)
    set_active_expert(model, 3); model.eval()
    results.append(forensic_eval(model, cfg, ns, aichi_scenes, label='2_best_fixed_ODD-D'))
    del model; torch.cuda.empty_cache()

    # ── 3: vMF soft mixture τ=200 ──────────────────────────────────────
    model = build_molora_model(cfg)
    setup_vmf_mixture(model, tau_km=200.0); model.eval()
    results.append(forensic_eval(model, cfg, ns, aichi_scenes, label='3_vmf_softmix_tau200'))
    del model; torch.cuda.empty_cache()

    # ── 4: Hypernet-LN (load trained hypernet) ──────────────────────────
    if os.path.exists('outputs/hypernet_ln.pth'):
        model = build_molora_model(cfg)
        install_ln_hooks(model)
        hyper = HyperLNNet(model, ell_dim=3, hidden=64, out_scale=0.1).to(DEVICE)
        sd = torch.load('outputs/hypernet_ln.pth', map_location=DEVICE)
        hyper.load_state_dict(sd['hypernet'])
        set_active_expert(model, None); model.eval()
        results.append(forensic_eval(model, cfg, ns, aichi_scenes, label='4_hypernet_ln',
                                      hyper=hyper, ell_per_map=aichi_ell))
        del model, hyper; torch.cuda.empty_cache()

    # ── 5: Hypernet + 1-epoch TENT polish ───────────────────────────────
    model = build_molora_model(cfg)
    install_ln_hooks(model)
    hyper = HyperLNNet(model, ell_dim=3, hidden=64, out_scale=0.1).to(DEVICE)
    if os.path.exists('outputs/hypernet_ln.pth'):
        hyper.load_state_dict(torch.load('outputs/hypernet_ln.pth', map_location=DEVICE)['hypernet'])
    set_active_expert(model, None)
    n_train, n_mod = make_only_ln_trainable(model)
    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR_LN)
    model.train()
    rng = random.Random(2026); order = list(aichi_scenes); rng.shuffle(order)
    for s in order:
        try:
            ell = aichi_ell.get(s['map'])
            if ell is None: continue
            with torch.no_grad():
                hyper.install(torch.tensor(ell, device=DEVICE).unsqueeze(0))
            opt.zero_grad()
            loss = k_sample_variance(model, cfg, s['inputs'], ns, K=16)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            hyper.clear()
        except Exception: continue
    model.eval()
    results.append(forensic_eval(model, cfg, ns, aichi_scenes, label='5_hypernet_plus_tent_polish',
                                  hyper=hyper, ell_per_map=aichi_ell))
    del model, hyper; torch.cuda.empty_cache()

    # ── 6: Best-fixed ODD-D + 1-epoch TENT polish ──────────────────────
    model = build_molora_model(cfg)
    set_active_expert(model, 3)
    n_train, n_mod = make_only_ln_trainable(model)
    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR_LN)
    model.train()
    rng = random.Random(2026); order = list(aichi_scenes); rng.shuffle(order)
    for s in order:
        try:
            opt.zero_grad()
            loss = k_sample_variance(model, cfg, s['inputs'], ns, K=16)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
        except Exception: continue
    model.eval()
    results.append(forensic_eval(model, cfg, ns, aichi_scenes, label='6_best_fixed_plus_tent_polish'))
    del model; torch.cuda.empty_cache()

    out = dict(
        description='Forensic diagnostics: forward progress, K-var, cross-scene diversity',
        K_deploy=K_DEPLOY,
        n_scenes=len(aichi_scenes),
        results=results,
    )
    # Save without the full per-scene trajectory list to keep JSON small
    out_lite = dict(out)
    out_lite['results'] = [
        {k: v for k, v in r.items() if k != 'rows'}
        for r in results
    ]
    json.dump(out_lite, open(OUT, 'w'), indent=2, default=float)
    json.dump(out, open(OUT.replace('.json', '_full.json'), 'w'), indent=2, default=float)
    print(f"\n=== FORENSICS SUMMARY ===")
    print(f"{'label':<35s} {'lat_mean':>9s} {'progress':>10s} {'speed':>7s} {'K_var':>8s} {'cross_L2':>9s}")
    for r in results:
        print(f"{r['label']:<35s} {r['max_lat_mean']:9.2f} {r['progress_mean']:10.1f} {r['mean_speed_mean']:7.2f} {r['deploy_k_var_mean']:8.3f} {r['cross_scene_pairwise_l2_mean']:9.1f}")
    print(f"\nSaved {OUT}")


if __name__ == '__main__':
    main()
