"""Hypernetwork-conditioned LayerNorm for geographic OOD deployment.

Implements Gemini's "Route A": train a small MLP that maps the deployment
location ell ∈ S² (ECEF unit vector) to an additive offset for every
LayerNorm affine in the decoder. At test time on a held-out Aichi map,
the hypernetwork produces a location-conditioned LN configuration without
any further training.

This is a true context-aware adaptation: env prior is injected at the model
state level, not at the feature level. Compares directly against:
  - vanilla MoLoRA shared-only (no per-expert, no env)
  - best-fixed expert (static geographic prior)
  - vMF soft mixture (existing paper baseline)
  - vanilla TENT-fullblood (no env)

Output: outputs/hypernet_ln_results.json
"""
import os, sys, json, math, random, time
os.environ.setdefault('LD_LIBRARY_PATH', '/opt/autoware/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
sys.path.insert(0, '/opt/autoware/local/lib/python3.10/dist-packages')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner_ros')
sys.path.insert(0, 'third_party/Diffusion-Planner/preference_optimization')
sys.path.insert(0, '.')

import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
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
from utils import load_npz_data

DEVICE = torch.device('cuda')
OUT = 'outputs/hypernet_ln_results.json'
MOLORA_FULL = 'checkpoints/molora_v2_output/molora_full.pth'

# Lat/lon for the 4 prod ODD centres and the Aichi target region
ODD_LATLON = {
    'hongo':   (35.713, 139.762),
    'kashiwa': (35.862, 139.969),
    'shalun':  (22.971, 120.272),
    'unagoya': (35.183, 136.907),
}
AICHI_LATLON = {
    'aichi_hub_v2': (35.105, 137.005),  # knowledge hub
    'ishio_dai':    (34.880, 137.230),  # suburban district
    'morikoro':     (35.190, 137.105),  # open park
}
ELL_JITTER_DEG = 0.1  # ~11 km spherical noise; smaller than min inter-ODD distance


def latlon_to_ecef_unit(lat_deg, lon_deg):
    lat = math.radians(lat_deg); lon = math.radians(lon_deg)
    return np.array([math.cos(lat) * math.cos(lon),
                     math.cos(lat) * math.sin(lon),
                     math.sin(lat)], dtype=np.float32)


def build_molora_model(cfg):
    model = Diffusion_Planner(cfg).to(DEVICE)
    base = torch.load(BASE_PTH, map_location=DEVICE, weights_only=False)
    bsd = base.get('ema_state_dict') or base['model']
    bsd = {k.replace('module.', ''): v for k, v in bsd.items()}
    model.load_state_dict(bsd, strict=False)
    apply_molora(model.decoder.dit, n_experts=N_EXPERTS, shared_rank=4, expert_rank=8)
    mfull = torch.load(MOLORA_FULL, map_location=DEVICE, weights_only=False)
    msd = {k.replace('module.', ''): v for k, v in mfull['model'].items()}
    model.load_state_dict(msd, strict=False)
    return model


class HyperLNNet(nn.Module):
    """Maps ell ∈ R^3 -> per-LN (Δγ, Δβ) offsets for every LayerNorm in target_model."""

    def __init__(self, target_model, ell_dim=3, hidden=64, out_scale=0.1):
        super().__init__()
        self.ln_specs = []   # list of (module, w_start, w_end, b_end)
        cur = 0
        # Avoid registering target_model as submodule (we don't want to train it)
        object.__setattr__(self, '_target', target_model)
        for nm, m in target_model.named_modules():
            if isinstance(m, nn.LayerNorm) and m.weight is not None:
                w_size = m.weight.numel()
                b_size = m.bias.numel() if m.bias is not None else 0
                self.ln_specs.append((m, cur, cur + w_size, cur + w_size + b_size))
                cur += w_size + b_size
        self.total_dim = cur
        print(f"  HyperLN: {len(self.ln_specs)} LN modules, total_dim={cur:,}")
        self.mlp = nn.Sequential(
            nn.Linear(ell_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden * 2), nn.SiLU(),
            nn.Linear(hidden * 2, cur),
        )
        # Zero-init the output layer so initial offsets are zero
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.out_scale = out_scale

    def install(self, ell):
        """Compute offsets from ell (B, 3); store on each LN module for the hook to use."""
        offsets = self.mlp(ell) * self.out_scale   # (B, total_dim)
        if offsets.dim() == 2:
            offsets = offsets[0]
        for m, ws, we, be in self.ln_specs:
            dgamma = offsets[ws:we].view_as(m.weight)
            dbeta = offsets[we:be].view_as(m.bias) if be > we else None
            m._dgamma = dgamma
            m._dbeta = dbeta

    def clear(self):
        for m, *_ in self.ln_specs:
            if hasattr(m, '_dgamma'): del m._dgamma
            if hasattr(m, '_dbeta'):  del m._dbeta


def install_ln_hooks(target_model):
    """Forward hook on every LN: if _dgamma is present, recompute LN with modified affines."""
    def make_hook():
        def hook(module, inputs, output):
            if hasattr(module, '_dgamma') and module._dgamma is not None:
                x = inputs[0]
                w = module.weight + module._dgamma
                b = module.bias + (module._dbeta if module._dbeta is not None else 0)
                return F.layer_norm(x, module.normalized_shape, w, b, module.eps)
            return output
        return hook
    handles = []
    for nm, m in target_model.named_modules():
        if isinstance(m, nn.LayerNorm) and m.weight is not None:
            handles.append(m.register_forward_hook(make_hook()))
    return handles


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


def eval_per_map(model, cfg, ns, scenes, hyper=None, ell_per_map=None):
    model.eval()
    by_map = defaultdict(list)
    for s in scenes:
        if hyper is not None and ell_per_map is not None:
            ell = ell_per_map.get(s['map'])
            if ell is None: continue
            with torch.no_grad():
                hyper.install(torch.tensor(ell, device=DEVICE).unsqueeze(0))
        lat = rollout_max_lat(model, cfg, s['inputs'], ns)
        if hyper is not None: hyper.clear()
        if lat is not None: by_map[s['map']].append(lat)
    summary = {m: {'mean': float(np.mean(v)), 'std': float(np.std(v, ddof=1)) if len(v)>1 else 0.0, 'n': len(v)}
               for m, v in by_map.items()}
    fleet = float(np.mean([v['mean'] for v in summary.values()])) if summary else None
    return summary, fleet


def jitter_ell(ell_np, sigma_deg=ELL_JITTER_DEG, rng=None):
    """Spherical jitter of the unit-vector ell by ~sigma_deg degrees."""
    if rng is None: rng = np.random
    # Add small random perturbation, then renormalize
    perturb = rng.normal(0, math.radians(sigma_deg), size=3).astype(np.float32)
    new = ell_np + perturb
    return new / (np.linalg.norm(new) + 1e-9)


def train_hypernet(model, cfg, ns, hyper, train_paths_by_odd,
                    n_steps=500, grad_accum=4, lr=5e-5, wd=1e-3, seed=0):
    """Train hyper on prod ODDs to predict good LN offsets that minimise
    the diffusion training loss (using recorded ego trajectories as gt)."""
    from train_molora import set_active_expert
    set_active_expert(model, None)        # shared-only forward (hyper modulates LN)
    for p in model.parameters(): p.requires_grad_(False)
    opt = optim.AdamW(hyper.parameters(), lr=lr, weight_decay=wd)
    print(f"  training hypernet: {sum(p.numel() for p in hyper.parameters()):,} params, {n_steps} steps")

    # Build (scene_path, ell, odd) pool
    pool = []
    for odd, paths in train_paths_by_odd.items():
        ell = latlon_to_ecef_unit(*ODD_LATLON[odd])
        for p in paths[:200]:
            pool.append((p, ell, odd))
    print(f"  training pool: {len(pool)} (ODD, scene) pairs")

    rng = random.Random(seed); rng_np = np.random.default_rng(seed)
    rng.shuffle(pool)
    losses = []; t0 = time.time()
    forward_i = 0; opt_step = 0
    opt.zero_grad()
    hyper.train(); model.eval()
    while opt_step < n_steps:
        path, ell, odd = pool[forward_i % len(pool)]
        forward_i += 1
        try:
            data = load_npz_data(path, DEVICE)
            # jitter ell
            ell_j = jitter_ell(ell, rng=rng_np)
            ell_t = torch.tensor(ell_j, device=DEVICE).float().unsqueeze(0)
            hyper.install(ell_t)
            # forward diffusion sampling (full unroll for trajectory + GT comparison)
            inputs_norm = cfg.observation_normalizer({k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data.items()})
            ego_traj = differentiable_dpm_solver_sample(model, cfg, inputs_norm, ns, device=DEVICE)
            # Loss: MSE between predicted ego and recorded ego_agent_future
            gt = data.get('ego_agent_future')
            if gt is None: hyper.clear(); continue
            gt = gt[0] if gt.ndim == 3 else gt  # (T, 3) -> per scene
            T = min(ego_traj.shape[0], gt.shape[0])
            loss = F.mse_loss(ego_traj[:T, :2], gt[:T, :2].to(DEVICE).float())
            (loss / grad_accum).backward()
            losses.append(float(loss))
            hyper.clear()
            if forward_i % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(hyper.parameters(), 1.0)
                opt.step(); opt.zero_grad()
                opt_step += 1
                if opt_step % 50 == 0 or opt_step == 1:
                    et = time.time() - t0
                    print(f"  step {opt_step}/{n_steps}  loss={np.mean(losses[-100:]):.4f}  elapsed={et:.0f}s", flush=True)
        except Exception as e:
            opt.zero_grad()
            if 'CUDA' in str(e): raise
            continue
    print(f"  training done. final loss(mean last 100)={np.mean(losses[-100:]):.4f}  elapsed={(time.time()-t0)/60:.1f}min")
    return losses


def main():
    cfg = Config(ARGS); ns = NoiseScheduleVP()
    print(f"[hypernet] Loading MoLoRA model...")
    model = build_molora_model(cfg)

    print(f"[hypernet] Installing LN hooks...")
    install_ln_hooks(model)
    hyper = HyperLNNet(model, ell_dim=3, hidden=64, out_scale=0.1).to(DEVICE)

    # Eval pool: 24 Aichi scenes
    aichi_scenes = build_aichi_scenes(cfg)
    print(f"[hypernet] Aichi eval pool: {len(aichi_scenes)} scenes")

    # Aichi ell per map
    aichi_ell = {tag: latlon_to_ecef_unit(*ll) for tag, ll in AICHI_LATLON.items()}

    # === PRE-train baseline: hypernet at zero output (LN unchanged) ===
    from train_molora import set_active_expert
    set_active_expert(model, None)
    pre_sum, pre_fleet = eval_per_map(model, cfg, ns, aichi_scenes, hyper=hyper, ell_per_map=aichi_ell)
    print(f"\nPre-training hypernet (zero offsets, == shared-only baseline):")
    for tag, v in pre_sum.items():
        print(f"  {tag}: {v['mean']:.2f}±{v['std']:.2f}  n={v['n']}")
    print(f"  fleet: {pre_fleet:.2f} m")

    # === Train hypernet on 4 prod ODDs ===
    train_paths = json.load(open('checkpoints/npz_multimap/path_list_train_v3.json'))
    per_odd = {m: [p for p in train_paths if m in p.lower()] for m in ODD_LATLON.keys()}
    for odd, paths in per_odd.items():
        print(f"  {odd}: {len(paths)} train scenes")

    losses = train_hypernet(model, cfg, ns, hyper, per_odd,
                             n_steps=500, grad_accum=4, lr=5e-5, wd=1e-3, seed=42)

    # === POST eval on Aichi (with trained hypernet) ===
    set_active_expert(model, None)
    post_sum, post_fleet = eval_per_map(model, cfg, ns, aichi_scenes, hyper=hyper, ell_per_map=aichi_ell)
    print(f"\nPost-training hypernet (Aichi ell injected):")
    for tag, v in post_sum.items():
        print(f"  {tag}: {v['mean']:.2f}±{v['std']:.2f}  n={v['n']}")
    print(f"  fleet: {post_fleet:.2f} m")

    out = dict(
        description='HyperLN: hypernetwork-conditioned LayerNorm from geographic prior (Gemini Route A)',
        config=dict(ell_dim=3, hidden=64, out_scale=0.1,
                    n_train_steps=500, grad_accum=4, lr=5e-5, wd=1e-3,
                    jitter_deg=ELL_JITTER_DEG),
        n_hypernet_params=int(sum(p.numel() for p in hyper.parameters())),
        n_aichi_eval=len(aichi_scenes),
        pre_aichi=pre_sum, pre_fleet=pre_fleet,
        post_aichi=post_sum, post_fleet=post_fleet,
        train_loss_first=float(np.mean(losses[:20])) if losses else None,
        train_loss_last=float(np.mean(losses[-100:])) if losses else None,
        # Reference baselines from main paper
        references=dict(shared_only=22.76, best_fixed_D=13.74,
                        vmf_softmix=15.85, joint_replay=4.26,
                        tent_fullblood_ep1=4.11),
        # Per-ODD lat/lon (anon: only used for ell vector computation)
        odd_latlon_keys=list(ODD_LATLON.keys()),
        aichi_latlon_keys=list(AICHI_LATLON.keys()),
    )
    torch.save({'hypernet': hyper.state_dict()}, 'outputs/hypernet_ln.pth')
    json.dump(out, open(OUT, 'w'), indent=2, default=float)
    print(f"\nSaved {OUT}")
    print(f"\n=== SUMMARY ===")
    print(f"  Pre-train hypernet (zero offsets):  fleet = {pre_fleet:.2f} m")
    print(f"  Post-train hypernet (Aichi ell):    fleet = {post_fleet:.2f} m  (Δ = {post_fleet - pre_fleet:+.2f})")
    print(f"  Reference: shared-only baseline:    {22.76:.2f} m")
    print(f"  Reference: best-fixed ODD-D:        {13.74:.2f} m")
    print(f"  Reference: vMF mix τ=200:           {15.85:.2f} m")
    print(f"  Reference: joint replay:            {4.26:.2f} m")


if __name__ == '__main__':
    main()
