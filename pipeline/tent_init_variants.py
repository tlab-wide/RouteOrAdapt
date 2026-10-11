"""TENT from three different starting states (Block 3 of overnight chain).

Tests whether context-aware initialisation helps TENT avoid the variance-collapse
mode. Three starting states:
  (a) shared-only (no per-ODD expert) — vanilla TENT-fullblood result
  (b) best-fixed ODD-D (static geographic prior pick) — start from in-region expert
  (c) vMF soft mixture τ=200km — start from geographic posterior mean of weights

For each, run 1 epoch of TENT-fullblood (K=32, Adam lr=1e-3, online, one pass).
Compare final fleet against:
  - the corresponding starting state itself (no TENT)
  - vanilla TENT-fullblood (4.11 m @ ep 1)
  - hypernet-LN (~6.78 m)

Output: outputs/tent_init_variants.json
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
from train_molora import set_active_expert
from overnight_stage1_multiego import ARGS

from tent_fullblood import (
    build_molora_model, make_only_ln_trainable, k_sample_variance,
    rollout_max_lat, build_aichi_scenes, eval_per_map,
    K_SAMPLES, LR_LN,
)
from hypernet_ln import latlon_to_ecef_unit, ODD_LATLON, AICHI_LATLON

DEVICE = torch.device('cuda')
OUT = 'outputs/tent_init_variants.json'
PROD_ODDS = ['hongo', 'kashiwa', 'shalun', 'unagoya']


def setup_vmf_mixture(model, tau_km=200.0):
    """Set the model's per-ODD LoRA weights to their vMF-prior-weighted mixture.
    For each Aichi map, we compute alpha and blend. But the model only has one
    active state at a time. Strategy: install the mixture by averaging W_d weighted
    by alpha at the Aichi region centroid."""
    # Compute alpha(ell_Aichi_centre) as the soft mixture
    # Use M3 (open park) as the canonical Aichi target for this demonstration
    ell = latlon_to_ecef_unit(*AICHI_LATLON['morikoro'])
    R = 6371.0
    dists = []
    for odd in PROD_ODDS:
        mu = latlon_to_ecef_unit(*ODD_LATLON[odd])
        # great-circle distance in km
        cos_angle = float(np.clip(np.dot(mu, ell), -1, 1))
        d = R * math.acos(cos_angle)
        dists.append(d)
    z = np.array(dists) / tau_km
    z = -z; z -= z.max()
    alpha = np.exp(z) / np.exp(z).sum()
    print(f"  vMF mix at τ={tau_km} for Aichi-M3: α={dict(zip(PROD_ODDS, alpha.round(3).tolist()))}")
    # Set model LoRA weights to the alpha-mixed combination
    # Each per-expert LoRA module has expert_A[idx], expert_B[idx]; we set the shared
    # forward to call set_active_expert and combine. Simpler: replace per-expert A/B
    # with a single mixed A/B.
    # For MoLoRA: shared LoRA is in m.shared_A, per-expert in m.expert_As[idx]
    from train_molora import MoLoRALinear
    for nm, m in model.named_modules():
        if isinstance(m, MoLoRALinear):
            # Mix per-expert
            mixed_A = sum(alpha[i] * m.expert_As[i].weight.data for i in range(len(alpha)))
            mixed_B = sum(alpha[i] * m.expert_Bs[i].weight.data for i in range(len(alpha)))
            # Overwrite expert 0 and force active=0
            m.expert_As[0].weight.data.copy_(mixed_A)
            m.expert_Bs[0].weight.data.copy_(mixed_B)
    set_active_expert(model, 0)
    return alpha


def run_init_variant(label, init_fn, cfg, ns, n_epochs=1, K=16):
    print(f"\n══════════════════════════════════════════")
    print(f" Init variant: {label}, K={K}, epochs={n_epochs}")
    print(f"══════════════════════════════════════════")
    model = build_molora_model(cfg)
    init_fn(model)
    aichi_scenes = build_aichi_scenes(cfg)

    # Pre-TENT eval
    pre_sum, pre_fleet = eval_per_map(model, cfg, ns, aichi_scenes)
    print(f"  pre-TENT fleet: {pre_fleet:.2f} m")

    n_train, n_mod = make_only_ln_trainable(model)
    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR_LN)
    epoch_history = []
    rng = random.Random(2026)
    for ep in range(n_epochs):
        model.train()
        order = list(aichi_scenes); rng.shuffle(order)
        ep_losses = []
        for s in order:
            try:
                opt.zero_grad()
                loss = k_sample_variance(model, cfg, s['inputs'], ns, K=K)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                ep_losses.append(float(loss.detach()))
            except Exception: continue
        post_sum, post_fleet = eval_per_map(model, cfg, ns, aichi_scenes)
        ep_loss_mean = float(np.mean(ep_losses)) if ep_losses else None
        delta = post_fleet - pre_fleet if pre_fleet else None
        epoch_history.append(dict(epoch=ep+1, loss=ep_loss_mean,
                                    per_map=post_sum, fleet=post_fleet,
                                    delta_vs_pre=delta))
        print(f"  ep {ep+1}: K-var={ep_loss_mean:.4f}  fleet={post_fleet:.2f}  Δ={delta:+.2f}")

    res = dict(label=label, pre_aichi=pre_sum, pre_fleet=pre_fleet,
                epoch_history=epoch_history,
                final_fleet=epoch_history[-1]['fleet'] if epoch_history else None)
    del model; torch.cuda.empty_cache()
    return res


def main():
    cfg = Config(ARGS); ns = NoiseScheduleVP()
    results = []

    # (a) shared-only init (same as vanilla TENT-fullblood)
    def init_shared(model): set_active_expert(model, None)
    results.append(run_init_variant('shared_only', init_shared, cfg, ns, n_epochs=1, K=16))

    # (b) best-fixed ODD-D (geographic prior pick = unagoya, index 3)
    def init_best_fixed(model): set_active_expert(model, 3)
    results.append(run_init_variant('best_fixed_ODD-D', init_best_fixed, cfg, ns, n_epochs=1, K=16))

    # (c) vMF soft mixture τ=200
    def init_vmf(model): setup_vmf_mixture(model, tau_km=200.0)
    results.append(run_init_variant('vmf_softmix_tau200', init_vmf, cfg, ns, n_epochs=1, K=16))

    out = dict(
        description='TENT from 3 starting states (Context-Aware TTA test)',
        config=dict(K=16, n_epochs=1, optimizer='Adam', lr=LR_LN),
        variants=results,
        references=dict(shared_only_frozen=22.76, tent_fullblood_vanilla_ep1=4.11,
                        best_fixed_D=13.74, vmf_softmix=15.85, hypernet_ln=6.78,
                        joint_replay=4.26),
    )
    json.dump(out, open(OUT, 'w'), indent=2, default=float)
    print(f"\n\n=== TENT INIT VARIANT SUMMARY ===")
    for r in results:
        print(f"  {r['label']:<25s}  pre={r['pre_fleet']:.2f} → post(ep1)={r['final_fleet']:.2f}  Δ={r['final_fleet']-r['pre_fleet']:+.2f}")
    print(f"\nSaved {OUT}")


if __name__ == '__main__':
    main()
