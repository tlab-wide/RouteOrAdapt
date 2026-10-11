#!/usr/bin/env python3
"""
Stage 12 (Idea 4): Park-Aware Backbone Repair via Lane-Topology Augmentation.

Fine-tune ONLY the shared LoRA of the SE-LoRA MoLoRA stack on 4 prod-ODD npz
scenes that have been augmented with park-like geometric perturbations:
  - per-lanelet random rotation U(-45°, 45°)
  - per-lanelet cubic Bezier curvature warp (±3 m peaks)
  - drop the left-or-right boundary so the model cannot anchor on opposing-lane symmetry

Per-expert LoRAs are frozen; only shared_A, shared_B receive gradients.

Training: 500 optimizer steps, lr=5e-5, grad_accum=4 (2000 forward passes),
shared-only forward (set_active_expert(None)). The shared LoRA must learn to
plan in low-orientation-consistency, curving environments without relying on
opposite-lane structure or canonical heading.

Outputs:
  checkpoints/overnight_20260513/stage12_shared_park.pth
  checkpoints/overnight_20260513/stage12_train_log.json
  checkpoints/overnight_20260513/stage12_eval.json
"""
import os, sys, json, random, time, math
os.environ.setdefault('LD_LIBRARY_PATH', '/opt/autoware/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
sys.path.insert(0, '/opt/autoware/local/lib/python3.10/dist-packages')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner_ros')
sys.path.insert(0, 'third_party/Diffusion-Planner/preference_optimization')
sys.path.insert(0, '.')

import numpy as np
import torch
from torch import optim

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from diffusion_planner.model.diffusion_utils.dpm_solver_pytorch import NoiseScheduleVP
from diffusion_planner_ros.lanelet2_utils.lanelet_converter import convert_lanelet

from utils import load_npz_data, generate_deterministic_trajectory
from train_reward_backprop import differentiable_dpm_solver_sample
from train_molora import (
    apply_molora, set_active_expert, freeze_non_molora,
    odd_conditioned_reward, extract_map_name, ODD_MAP_TO_IDX, ODD_MAPS,
    N_EXPERTS,
)
from overnight_stage1_multiego import (
    build_inputs, pick_n_egos, metrics_for_traj, UNSEEN, N_EGO, SEED, ARGS,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = 'checkpoints/overnight_20260513'
os.makedirs(OUT_DIR, exist_ok=True)

MOLORA_FULL = 'checkpoints/molora_v2_output/molora_full.pth'
BASE_PTH    = 'third_party/Diffusion-Planner/best_model.pth'
PROD_ODDS   = ['hongo', 'kashiwa', 'shalun', 'unagoya']

# Training hyperparameters
N_OPT_STEPS = 500
GRAD_ACCUM  = 4
LR          = 5e-5
PARK_PROB   = 0.5
ROT_MAX_RAD = math.pi / 4
BEZIER_AMP  = 3.0     # m
SHARED_RANK = 4
EXPERT_RANK = 8


# ── Park-topology augmentation ────────────────────────────────────────────
def perturb_park_inplace(data, prob=PARK_PROB, rng=None):
    """Apply per-lanelet rotation + Bezier curvature warp + boundary drop.
    Modifies data['lanes'] and data['route_lanes'] in place."""
    if rng is None: rng = random
    for key in ['lanes', 'route_lanes']:
        if key not in data: continue
        t = data[key]
        if t.ndim != 4: continue
        arr = t[0].detach().cpu().numpy()  # (N, P, 33)
        N, P, D = arr.shape
        if D < 8: continue
        mask = (np.abs(arr[:, :, :2]).sum(axis=(1, 2)) > 0.1)
        if not mask.any(): continue

        for n in np.where(mask)[0]:
            if rng.random() > prob: continue
            lane = arr[n]
            anchor = lane[0, :2].copy()

            theta = rng.uniform(-ROT_MAX_RAD, ROT_MAX_RAD)
            c, s = math.cos(theta), math.sin(theta)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)

            pts = (lane[:, :2] - anchor) @ R.T + anchor
            lb  = (lane[:, 4:6] - anchor) @ R.T + anchor
            rb  = (lane[:, 6:8] - anchor) @ R.T + anchor

            h1 = rng.uniform(-BEZIER_AMP, BEZIER_AMP)
            h2 = rng.uniform(-BEZIER_AMP, BEZIER_AMP)
            ts = np.linspace(0, 1, P).astype(np.float32)
            bezier = 3*(1-ts)**2 * ts * h1 + 3*(1-ts) * ts**2 * h2

            tangent = pts[1] - pts[0] if P > 1 else np.array([1.0, 0.0], dtype=np.float32)
            tn = float(np.linalg.norm(tangent)) + 1e-6
            perp = np.array([-tangent[1], tangent[0]], dtype=np.float32) / tn
            off = bezier[:, None] * perp[None, :]

            pts += off; lb += off; rb += off

            if rng.random() < 0.5: lb[:] = 0.0
            else:                   rb[:] = 0.0

            lane[:, :2]  = pts
            lane[:, 4:6] = lb
            lane[:, 6:8] = rb
            if P > 1:
                lane[:-1, 2] = pts[1:, 0] - pts[:-1, 0]
                lane[:-1, 3] = pts[1:, 1] - pts[:-1, 1]
                lane[-1, 2:4] = lane[-2, 2:4]
            arr[n] = lane

        t[0] = torch.from_numpy(arr).to(t.device).to(t.dtype)


# ── Build training-scene iterator ─────────────────────────────────────────
def build_train_paths(per_map_train, n_per_map=200, seed=42):
    rng = random.Random(seed)
    paths = []
    for m in PROD_ODDS:
        ps = per_map_train.get(m, [])[:]
        rng.shuffle(ps)
        paths.extend([(m, p) for p in ps[:n_per_map]])
    rng.shuffle(paths)
    return paths


# ── Load MoLoRA model with the trained checkpoint ─────────────────────────
def load_molora_model(cfg):
    """Build a Diffusion_Planner with MoLoRA applied, then load molora_full.pth."""
    model = Diffusion_Planner(cfg).to(DEVICE)
    # First load base weights so non-LoRA layers are populated
    base_ckpt = torch.load(BASE_PTH, map_location=DEVICE, weights_only=False)
    base_sd = base_ckpt.get('ema_state_dict') or base_ckpt['model']
    base_sd = {k.replace('module.', ''): v for k, v in base_sd.items()}
    model.load_state_dict(base_sd, strict=False)
    # Apply MoLoRA to DiT layers
    apply_molora(model.decoder.dit, n_experts=N_EXPERTS,
                 shared_rank=SHARED_RANK, expert_rank=EXPERT_RANK)
    # Load trained MoLoRA weights
    mfull = torch.load(MOLORA_FULL, map_location=DEVICE, weights_only=False)
    msd = mfull['model']
    msd = {k.replace('module.', ''): v for k, v in msd.items()}
    missing, unexpected = model.load_state_dict(msd, strict=False)
    n_shared = sum(1 for k in msd if 'shared_' in k)
    n_expert = sum(1 for k in msd if 'expert_' in k)
    print(f"  loaded molora_full.pth (shared keys={n_shared}, expert keys={n_expert})")
    print(f"  missing={len(missing)}, unexpected={len(unexpected)}")
    return model


def freeze_to_shared_only(model):
    """Freeze base + per-expert LoRAs; only shared_A/B are trainable."""
    for p in model.parameters():
        p.requires_grad_(False)
    n_train = 0
    for m in model.modules():
        if hasattr(m, 'shared_A') and hasattr(m, 'shared_B'):
            m.shared_A.weight.requires_grad_(True)
            m.shared_B.weight.requires_grad_(True)
            n_train += m.shared_A.weight.numel() + m.shared_B.weight.numel()
    return n_train


def eval_on_aichi(model, cfg, ns):
    """Evaluate the (currently active expert config of) model on 3 Aichi maps.
    Returns {map: {ego_idx: max_lat}}."""
    out = {}
    for tag, osm in UNSEEN:
        out[tag] = {}
        try:
            llmap = convert_lanelet(osm)
        except Exception as e:
            print(f"  eval: convert {tag} failed: {e}")
            continue
        egos = pick_n_egos(llmap, N_EGO, seed=SEED)
        for ego_idx, ego_ll in enumerate(egos):
            cl = ego_ll.centerline
            if cl.shape[0] < 2: continue
            cx, cy = float(cl[0, 0]), float(cl[0, 1])
            heading = float(np.arctan2(cl[1, 1]-cl[0, 1], cl[1, 0]-cl[0, 0]))
            inputs = build_inputs(llmap, (cx, cy), heading)
            inputs_norm = cfg.observation_normalizer({
                k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()})
            try:
                with torch.no_grad():
                    ego_traj = differentiable_dpm_solver_sample(model, cfg, inputs_norm, ns, device=DEVICE)
                xy = ego_traj[:, :2].cpu().numpy()
                m = metrics_for_traj(xy)
                if m.get('finite'):
                    out[tag][ego_idx] = m['max_lat']
            except Exception as e:
                pass
    return out


def eval_on_prod(model, cfg, per_map_train, n_per_map=10):
    """Quick regression eval on prod-ODD held-out scenes (NO perturbation).
    Returns {odd: {progress_mean, route_dev_mean}}.
    Activates the correct per-ODD expert during forward."""
    out = {}
    for odd in PROD_ODDS:
        paths = per_map_train.get(odd, [])[-n_per_map:]
        expert_idx = ODD_MAP_TO_IDX.get(odd)
        set_active_expert(model, expert_idx)
        progs, devs = [], []
        for p in paths:
            try:
                data = load_npz_data(p, DEVICE)
                ego = generate_deterministic_trajectory(model, cfg, data, device=DEVICE)
                xy = ego[:, :2]
                if not np.all(np.isfinite(xy)): continue
                prog = float(np.linalg.norm(xy[-1] - xy[0]))
                rl = data.get('route_lanes')
                rd = 0.0
                if rl is not None:
                    rl_np = rl[0].cpu().numpy()
                    rp = rl_np[:, :, :2].reshape(-1, 2)
                    rp = rp[np.any(np.abs(rp) > 0.1, axis=1)]
                    if len(rp) > 2:
                        idx = np.linspace(0, xy.shape[0]-1, 10, dtype=int)
                        rd = float(np.mean([np.min(np.linalg.norm(rp - xy[i], axis=1)) for i in idx]))
                progs.append(prog); devs.append(rd)
            except Exception:
                pass
        out[odd] = {'n': len(progs),
                    'progress_mean': float(np.mean(progs)) if progs else 0,
                    'route_dev_mean': float(np.mean(devs)) if devs else 0}
    set_active_expert(model, None)
    return out


def main():
    print(f"[stage12] Park-Aware Backbone Repair  N_STEPS={N_OPT_STEPS}  lr={LR}  park_prob={PARK_PROB}")
    cfg = Config(ARGS)
    ns = NoiseScheduleVP()

    print("[stage12] Loading MoLoRA model...")
    model = load_molora_model(cfg)
    n_train_params = freeze_to_shared_only(model)
    print(f"  trainable params (shared only): {n_train_params:,}")

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR, weight_decay=0.01)

    # Load training paths (per-ODD grouped)
    print("[stage12] Loading training path list...")
    train_paths = json.load(open('checkpoints/npz_multimap/path_list_train_v3.json'))
    per_map_train = {m: [p for p in train_paths if m in p.lower()] for m in PROD_ODDS}
    for m, ps in per_map_train.items():
        print(f"  {m}: {len(ps)} scenes")

    train_order = build_train_paths(per_map_train, n_per_map=200, seed=42)
    print(f"  total training scenes for this run: {len(train_order)}")

    # ── Pre-training baseline eval (shared-only on Aichi) ──────────────────
    print("\n[stage12] Pre-training baseline eval (shared-only on Aichi)...")
    set_active_expert(model, None)
    model.eval()
    pre_eval = eval_on_aichi(model, cfg, ns)
    pre_summary = {}
    for tag, sc in pre_eval.items():
        lats = list(sc.values())
        pre_summary[tag] = {'mean': float(np.mean(lats)) if lats else None,
                            'std': float(np.std(lats, ddof=1)) if len(lats) > 1 else 0.0,
                            'n': len(lats)}
        print(f"  [pre]  {tag}: lat={pre_summary[tag]['mean']:.2f}±{pre_summary[tag]['std']:.2f}  n={pre_summary[tag]['n']}")

    print("\n[stage12] Pre-training prod-ODD regression check...")
    pre_prod = eval_on_prod(model, cfg, per_map_train, n_per_map=10)
    for odd, v in pre_prod.items():
        print(f"  [pre]  {odd}: progress={v['progress_mean']:.1f}m, route_dev={v['route_dev_mean']:.2f}m, n={v['n']}")

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"\n[stage12] Training {N_OPT_STEPS} optimizer steps "
          f"({N_OPT_STEPS * GRAD_ACCUM} forward passes)")
    set_active_expert(model, None)  # SHARED-ONLY forward during training
    model.train()
    train_log = {'steps': [], 'pre_aichi': pre_summary, 'pre_prod': pre_prod}

    rng = random.Random(0)
    forward_i = 0
    opt_step = 0
    epoch_rewards = []
    t0 = time.time()
    optimizer.zero_grad()

    while opt_step < N_OPT_STEPS:
        idx = forward_i % len(train_order)
        map_name, npz_path = train_order[idx]
        forward_i += 1
        try:
            data = load_npz_data(npz_path, DEVICE)
            perturb_park_inplace(data, prob=PARK_PROB, rng=rng)
            data_norm = cfg.observation_normalizer({
                k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data.items()})
            ego_traj, deltas = differentiable_dpm_solver_sample(
                model, cfg, data_norm, ns, device=DEVICE, return_deltas=True)

            rl = data.get('route_lanes')
            route_xy = None
            if rl is not None:
                rl_np = rl[0].cpu().numpy()
                rp = rl_np[:, :, :2].reshape(-1, 2)
                rp = rp[np.any(np.abs(rp) > 0.1, axis=1)]
                if len(rp) > 2:
                    route_xy = torch.tensor(rp, dtype=torch.float32, device=DEVICE)
            nb_fut = data.get('neighbor_agents_future')
            neighbor_future = nb_fut[0].to(DEVICE) if nb_fut is not None else None

            reward = odd_conditioned_reward(ego_traj, route_xy, neighbor_future, deltas, map_name)
            loss = -reward / GRAD_ACCUM
            loss.backward()
            epoch_rewards.append(float(reward))

            if forward_i % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()
                opt_step += 1
                if opt_step % 25 == 0 or opt_step == 1:
                    mean_r = float(np.mean(epoch_rewards[-25*GRAD_ACCUM:]))
                    et = time.time() - t0
                    print(f"  step {opt_step:>3}/{N_OPT_STEPS}  "
                          f"mean_reward(recent)={mean_r:>6.2f}  elapsed={et/60:.1f} min")
                    train_log['steps'].append({'step': opt_step, 'mean_reward': mean_r,
                                                'elapsed_s': et})
        except Exception as e:
            print(f"  skip step (forward {forward_i}): {type(e).__name__}: {str(e)[:160]}")
            optimizer.zero_grad()

    print(f"\n[stage12] Training done. Total elapsed: {(time.time()-t0)/60:.1f} min")

    # Save shared-only state dict
    shared_sd = {k: v for k, v in model.state_dict().items() if 'shared_' in k}
    torch.save({'shared_state_dict': shared_sd,
                'n_steps': N_OPT_STEPS, 'lr': LR, 'park_prob': PARK_PROB,
                'shared_rank': SHARED_RANK},
               f'{OUT_DIR}/stage12_shared_park.pth')
    print(f"[stage12] saved {OUT_DIR}/stage12_shared_park.pth")

    # ── Post-training eval ─────────────────────────────────────────────────
    print("\n[stage12] Post-training eval (shared-only on Aichi)...")
    set_active_expert(model, None)
    model.eval()
    post_eval = eval_on_aichi(model, cfg, ns)
    post_summary = {}
    for tag, sc in post_eval.items():
        lats = list(sc.values())
        post_summary[tag] = {'mean': float(np.mean(lats)) if lats else None,
                             'std': float(np.std(lats, ddof=1)) if len(lats) > 1 else 0.0,
                             'n': len(lats)}
        pre = train_log['pre_aichi'][tag]
        delta = post_summary[tag]['mean'] - pre['mean'] if pre['mean'] else 0
        print(f"  [post] {tag}: lat={post_summary[tag]['mean']:.2f}±"
              f"{post_summary[tag]['std']:.2f}  n={post_summary[tag]['n']}  "
              f"Δ vs pre = {delta:+.2f}")

    print("\n[stage12] Post-training prod-ODD regression check...")
    post_prod = eval_on_prod(model, cfg, per_map_train, n_per_map=10)
    for odd, v in post_prod.items():
        pre = train_log['pre_prod'][odd]
        prog_d = v['progress_mean'] - pre['progress_mean']
        rd_d = v['route_dev_mean'] - pre['route_dev_mean']
        print(f"  [post] {odd}: progress={v['progress_mean']:.1f}m (Δ{prog_d:+.1f}), "
              f"route_dev={v['route_dev_mean']:.2f}m (Δ{rd_d:+.2f})")

    out = {
        'description': 'Stage 12 (Idea 4): Park-Aware Backbone Repair via shared-LoRA fine-tune on park-perturbed prod-ODD scenes',
        'n_opt_steps': N_OPT_STEPS, 'grad_accum': GRAD_ACCUM, 'lr': LR,
        'park_prob': PARK_PROB, 'rot_max_rad': ROT_MAX_RAD, 'bezier_amp_m': BEZIER_AMP,
        'pre_aichi': pre_summary,
        'post_aichi': post_summary,
        'pre_prod': pre_prod,
        'post_prod': post_prod,
        'train_log': train_log,
    }
    with open(f'{OUT_DIR}/stage12_eval.json', 'w') as f: json.dump(out, f, indent=2)
    print(f"\n[stage12] saved {OUT_DIR}/stage12_eval.json")


if __name__ == '__main__':
    main()
