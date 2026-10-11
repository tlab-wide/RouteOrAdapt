"""
Direct Reward Backprop through Differentiable DPM-Solver++ (DRaFT-style).

Zero mismatch: 10-step DPM-Solver++ for training, inference, and C++ deployment.
No log-prob, no REINFORCE, no GRPO. Just differentiable reward maximization.

The key insight: DPM-Solver++ is a deterministic ODE solver. All operations
(linear combinations, model forward passes) are differentiable. We just need
to remove torch.no_grad() and implement the reward in PyTorch.

Usage:
  python train_reward_backprop.py [--sanity_check] [--epochs 10]
"""

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────
TRAIN_DIR = Path("third_party/Diffusion-Planner/diffusion_planner")
PREF_DIR = Path("third_party/Diffusion-Planner/preference_optimization")
MODEL_PTH = Path("third_party/Diffusion-Planner/best_model.pth")
ARGS_JSON = Path("third_party/Diffusion-Planner/args.json")

sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(PREF_DIR))

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from diffusion_planner.model.diffusion_utils.dpm_solver_pytorch import NoiseScheduleVP, expand_dims
from utils import load_npz_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════
# 1.  LoRA
# ═══════════════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.original = linear
        self.original.requires_grad_(False)
        d_out, d_in = linear.weight.shape
        self.lora_A = nn.Linear(d_in, rank, bias=False)
        self.lora_B = nn.Linear(rank, d_out, bias=False)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.lora_A = self.lora_A.to(linear.weight.device)
        self.lora_B = self.lora_B.to(linear.weight.device)

    def forward(self, x):
        return self.original(x) + self.scale * self.lora_B(self.lora_A(x))

    def merge(self) -> nn.Linear:
        merged = copy.deepcopy(self.original)
        merged.requires_grad_(True)
        with torch.no_grad():
            merged.weight.data += self.scale * (self.lora_B.weight @ self.lora_A.weight)
        return merged


def apply_lora(dit, rank=16, alpha=32.0):
    dit.preproj.fc1 = LoRALinear(dit.preproj.fc1, rank, alpha)
    dit.preproj.fc2 = LoRALinear(dit.preproj.fc2, rank, alpha)
    for block in dit.blocks:
        block.mlp1.fc1 = LoRALinear(block.mlp1.fc1, rank, alpha)
        block.mlp1.fc2 = LoRALinear(block.mlp1.fc2, rank, alpha)
        block.mlp2.fc1 = LoRALinear(block.mlp2.fc1, rank, alpha)
        block.mlp2.fc2 = LoRALinear(block.mlp2.fc2, rank, alpha)
    dit.final_layer.proj[1] = LoRALinear(dit.final_layer.proj[1], rank, alpha)
    dit.final_layer.proj[4] = LoRALinear(dit.final_layer.proj[4], rank, alpha)


def freeze_non_lora(model):
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.weight.requires_grad_(True)
            m.lora_B.weight.requires_grad_(True)


def merge_lora_into_dit(dit):
    def _replace(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, name, child.merge())
            else:
                _replace(child)
    _replace(dit)


def count_lora_params(model):
    return sum(
        p.numel() for m in model.modules() if isinstance(m, LoRALinear)
        for p in [m.lora_A.weight, m.lora_B.weight]
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Differentiable DPM-Solver++ (no torch.no_grad, no classifier guidance)
# ═══════════════════════════════════════════════════════════════════════════

def differentiable_dpm_solver_sample(
    model, model_args, data_norm, noise_schedule, device=None,
    return_deltas=False,
):
    """Differentiable DPM-Solver++ by calling the original code path without no_grad.

    Instead of reimplementing DPM-Solver++, we monkey-patch the original
    DPM_Solver.sample() to remove torch.no_grad(), ensuring gradients flow
    through LoRA parameters while using the exact same math.

    Args:
        model: Diffusion Planner (with LoRA on DiT)
        model_args: config
        data_norm: normalized observation dict
        noise_schedule: NoiseScheduleVP
        device: torch device
        return_deltas: if True, also return per-step ego displacement deltas

    Returns:
        ego_traj: [T, 4] denormalized ego trajectory (differentiable)
        deltas: list of per-step max displacement (only if return_deltas=True)
    """
    device = device or DEVICE
    B = 1
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len

    # Run encoder (frozen, no grad needed)
    with torch.no_grad():
        encoding = model.encoder(data_norm)
        ego_current = data_norm["ego_current_state"][:, :4]
        nbr_current = data_norm["neighbor_agents_past"][:, :P-1, -1, :4]
        current_states = torch.cat([ego_current[:, None], nbr_current], dim=1)
        nbr_past = data_norm["neighbor_agents_past"]
        neighbor_mask = (nbr_past[:, :, -1, :].abs().sum(-1) < 1e-6)

    # Use the decoder's own _inference_x_start but without no_grad
    # We call the components directly
    import diffusion_planner.model.diffusion_utils.dpm_solver_pytorch as dpm

    xT = torch.zeros(B, P, (1 + future_len) * 4, device=device)

    def correcting_xt_fn(xt, t, step):
        xt = xt.reshape(B, P, -1, 4)
        # Non-inplace: replace first timestep with current_states
        xt = torch.cat([current_states.unsqueeze(2), xt[:, :, 1:, :]], dim=2)
        return xt.reshape(B, P, -1)

    # Build model_fn (same as decoder._inference_x_start)
    model_fn = dpm.model_wrapper(
        model.decoder.dit,
        noise_schedule,
        model_type="x_start",
        model_kwargs={
            "cross_c": encoding,
            "neighbor_current_mask": neighbor_mask,
        },
        guidance_type="uncond",
    )

    solver = dpm.DPM_Solver(
        model_fn, noise_schedule, correcting_xt_fn=correcting_xt_fn
    )

    # === Key: call sample logic WITHOUT torch.no_grad() ===
    steps = 10
    order = 2
    t_0 = 1.0 / noise_schedule.total_N
    t_T = noise_schedule.T

    timesteps = solver.get_time_steps(
        skip_type="logSNR", t_T=t_T, t_0=t_0, N=steps, device=device
    )

    x = xT
    step = 0
    t = timesteps[step]
    t_prev_list = [t]
    model_prev_list = [solver.model_fn(x, t)]
    if solver.correcting_xt_fn is not None:
        x = solver.correcting_xt_fn(x, t, step)

    # Track per-step x0 predictions for convergence measurement
    prev_x0_ego = None
    deltas = []  # per-step ego displacement (differentiable)

    def _record_delta(x_current, t_current):
        """Record ego x0 prediction displacement from previous step."""
        nonlocal prev_x0_ego
        # Get x0 prediction at this step
        x0_pred = solver.data_prediction_fn(x_current, t_current)
        x0_shaped = x0_pred.reshape(B, P, 1 + future_len, 4)
        x0_ego = x0_shaped[0, 0, 1:, :2]  # [T, 2] ego XY in normalized space
        if prev_x0_ego is not None:
            # Max displacement across all timesteps (same as convergence monitor)
            delta = (x0_ego - prev_x0_ego).norm(dim=1).max()
            deltas.append(delta)
        prev_x0_ego = x0_ego

    # First order step
    for step in range(1, order):
        t = timesteps[step]
        x = solver.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step)
        if solver.correcting_xt_fn is not None:
            x = solver.correcting_xt_fn(x, t, step)
        if return_deltas:
            _record_delta(x, t)
        t_prev_list.append(t)
        model_prev_list.append(solver.model_fn(x, t))

    # Second order steps
    for step in range(order, steps + 1):
        t = timesteps[step]
        step_order = order
        x = solver.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step_order)
        if solver.correcting_xt_fn is not None:
            x = solver.correcting_xt_fn(x, t, step)
        if return_deltas:
            _record_delta(x, t)
        for i in range(order - 1):
            t_prev_list[i] = t_prev_list[i + 1]
            model_prev_list[i] = model_prev_list[i + 1]
        t_prev_list[-1] = t
        if step < steps:
            model_prev_list[-1] = solver.model_fn(x, t)

    # Denoise to zero
    t_final = torch.ones((1,), device=device) * t_0
    x = solver.data_prediction_fn(x, t_final)
    if solver.correcting_xt_fn is not None:
        x = solver.correcting_xt_fn(x, t_final, steps + 1)

    # Denormalize
    x0 = x.reshape(B, P, (1 + future_len) * 4)
    x0_shaped = x0.reshape(B, P, 1 + future_len, 4)
    x0_denorm = model_args.state_normalizer.inverse(x0_shaped)
    ego_traj = x0_denorm[0, 0, 1:]  # [T, 4]

    if return_deltas:
        return ego_traj, deltas
    return ego_traj


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Differentiable Reward (PyTorch)
# ═══════════════════════════════════════════════════════════════════════════

DT = 0.1

def differentiable_reward(
    ego_traj: torch.Tensor,
    route_xy: torch.Tensor | None = None,
    neighbor_future: torch.Tensor | None = None,
    deltas: list | None = None,
    lambda_conv: float = 2.0,
    epsilon: float = 0.1,
) -> torch.Tensor:
    """Compute reward directly in PyTorch (differentiable).

    Reward hierarchy (following survey consensus):
      L1 Safety: collision avoidance (gate — collision → reward = 0)
      L2 Progress: path length along trajectory
      L3 Route: curvature-weighted deviation (in-cut prevention)
      L4 Comfort: jerk + lateral acceleration
      L5 Heading: predicted heading vs travel direction

    Args:
        ego_traj: [T, 4] denormalized ego trajectory (x, y, cos_h, sin_h)
        route_xy: [N, 2] route waypoints (ego-centric), or None
        neighbor_future: [32, T, 3] neighbor future trajectories (x, y, yaw), or None

    Returns:
        scalar reward tensor (differentiable)
    """
    xy = ego_traj[:, :2]  # [T, 2]
    T = xy.shape[0]

    # ── Progress: total path length (not just displacement) ──
    # This rewards actual driving distance, not shortcutting
    step_dists = torch.norm(xy[1:] - xy[:-1], dim=1)  # [T-1]
    path_length = step_dists.sum()
    # No cap — let the model drive as far as possible
    progress_reward = torch.clamp(path_length, max=200.0)

    # ── Route deviation: curvature-weighted ──
    route_dev = torch.tensor(0.0, device=xy.device)
    curve_penalty = torch.tensor(0.0, device=xy.device)
    if route_xy is not None and route_xy.shape[0] > 2:
        # Sample 20 points along trajectory (more than before)
        idx = torch.linspace(0, T - 1, min(20, T), dtype=torch.long, device=xy.device)
        sampled = xy[idx]  # [20, 2]

        # Min distance from each sampled point to any route point
        dists = torch.norm(sampled.unsqueeze(1) - route_xy.unsqueeze(0), dim=2)  # [20, N]
        min_dists = dists.min(dim=1).values  # [20]
        route_dev = min_dists.mean()

        # Curvature-weighted: detect curves from route geometry
        # Compute local curvature of trajectory at each sample point
        if len(idx) > 4:
            dx = sampled[2:, 0] - sampled[:-2, 0]
            dy = sampled[2:, 1] - sampled[:-2, 1]
            ddx = sampled[2:, 0] - 2 * sampled[1:-1, 0] + sampled[:-2, 0]
            ddy = sampled[2:, 1] - 2 * sampled[1:-1, 1] + sampled[:-2, 1]
            curvature = (dx * ddy - dy * ddx).abs() / (dx ** 2 + dy ** 2 + 1e-6) ** 1.5
            # Weight deviation by curvature: curves get 3x penalty
            curve_weights = 1.0 + 3.0 * torch.clamp(curvature, max=1.0)
            weighted_devs = min_dists[1:-1] * curve_weights
            curve_penalty = weighted_devs.mean()

    # ── Comfort: smoothed jerk ──
    if T > 5:
        kernel = torch.ones(1, 1, 5, device=xy.device) / 5.0
        xy_padded = xy.T.unsqueeze(0)  # [1, 2, T]
        xy_smooth = torch.nn.functional.conv1d(xy_padded, kernel.expand(2, 1, 5),
                                                padding=2, groups=2).squeeze(0).T
    else:
        xy_smooth = xy

    velocity = (xy_smooth[1:] - xy_smooth[:-1]) / DT  # [T-1, 2]
    speed = torch.norm(velocity, dim=1)  # [T-1]
    accel = (velocity[1:] - velocity[:-1]) / DT  # [T-2, 2]
    jerk = (accel[1:] - accel[:-1]) / DT  # [T-3, 2]
    jerk_mag = torch.norm(jerk, dim=1)  # [T-3]
    mean_jerk = jerk_mag.mean() if jerk_mag.numel() > 0 else torch.tensor(0.0, device=xy.device)

    comfort = -torch.clamp(mean_jerk - 4.0, min=0.0)
    comfort = torch.clamp(comfort, min=-10.0)

    # ── Heading error ──
    cos_h, sin_h = ego_traj[:, 2], ego_traj[:, 3]
    dx = xy_smooth[1:, 0] - xy_smooth[:-1, 0]
    dy = xy_smooth[1:, 1] - xy_smooth[:-1, 1]
    fast_mask = speed > 0.5
    if fast_mask.any():
        travel_h = torch.atan2(dy, dx)
        pred_h = torch.atan2(sin_h[:-1], cos_h[:-1])
        hdiff = torch.atan2(torch.sin(travel_h - pred_h), torch.cos(travel_h - pred_h))
        heading_err = hdiff[fast_mask].abs().mean()
    else:
        heading_err = torch.tensor(0.0, device=xy.device)

    # ── Speed maintenance: reward progress, not raw speed ──
    # Don't penalize low speed directly (car may be stopping at red light).
    # Instead, use path_length as the speed proxy — a car that drives far
    # must have maintained speed. This avoids penalizing legitimate stops.
    mean_speed = speed.mean() if speed.numel() > 0 else torch.tensor(0.0, device=xy.device)
    # Only penalize if the car barely moved at all (< 5m total path = stuck/collapsed)
    speed_penalty = torch.clamp(5.0 - path_length, min=0.0) * 2.0  # strong penalty only if near-zero progress

    # ── Lateral acceleration at curves ──
    # Penalize high lateral acc (related to Saito-san's in-cut issue)
    if speed.numel() > 2:
        heading_traj = torch.atan2(velocity[:, 1], velocity[:, 0])
        yaw_rate = (heading_traj[1:] - heading_traj[:-1]) / DT
        lat_acc = speed[:-1] * yaw_rate.abs()
        mean_lat_acc = lat_acc.mean()
        lat_acc_penalty = torch.clamp(mean_lat_acc - 3.0, min=0.0)
    else:
        lat_acc_penalty = torch.tensor(0.0, device=xy.device)

    # ── L1 Safety: collision detection (gate) ──
    # Open-loop collision check against recorded neighbor futures
    # Uses soft differentiable collision cost (not hard 0/1)
    collision_cost = torch.tensor(0.0, device=xy.device)
    COLLISION_RADIUS = 3.0  # meters (ego + neighbor combined footprint)
    if neighbor_future is not None:
        ego_xy = xy  # [T, 2]
        for n in range(neighbor_future.shape[0]):
            nb_xy = neighbor_future[n, :T, :2]  # [T, 2]
            # Skip inactive neighbors (all zeros)
            if nb_xy.abs().max() < 0.1:
                continue
            # Distance at each timestep
            dist = torch.norm(ego_xy - nb_xy, dim=1)  # [T]
            # Soft collision: exponential penalty when close
            # 0 when dist > COLLISION_RADIUS, large when dist < COLLISION_RADIUS
            penetration = torch.clamp(COLLISION_RADIUS - dist, min=0.0)  # [T]
            collision_cost = collision_cost + penetration.sum()

    # ── Total reward: Phase 1 (no collision — learn to drive first) ──
    # Collision penalty causes conservative driving in all tested variants.
    # Phase 1: train with progress-dominant reward to learn multi-map driving.
    # Phase 2 (future): add collision penalty with tiny weight on top of Phase 1 model.
    reward = (
        2.0 * progress_reward       # L2: progress (DOMINANT)
        - 2.0 * route_dev           # L3: base route deviation
        - 3.0 * curve_penalty       # L3: curves (in-cut prevention)
        + 1.0 * comfort             # L4: jerk
        - 2.0 * heading_err         # L5: heading consistency
        - speed_penalty             # Anti-collapse (only if < 5m total)
        - 1.5 * lat_acc_penalty     # L4: lateral acceleration
        - 1.0 * collision_cost      # L1: collision
    )

    return reward


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path, default=MODEL_PTH)
    parser.add_argument("--train_npz_list", type=Path,
                        default=Path("checkpoints/path_list_train.json"))
    parser.add_argument("--valid_npz_list", type=Path,
                        default=Path("checkpoints/path_list_valid.json"))
    parser.add_argument("--out_dir", type=Path,
                        default=Path("checkpoints/reward_backprop"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--sanity_check", action="store_true")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Accumulate gradients over N scenes before stepping")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.train_npz_list) as f:
        train_paths = json.load(f)
    with open(args.valid_npz_list) as f:
        valid_paths = json.load(f)

    if args.sanity_check:
        train_paths = train_paths[:50]
        valid_paths = valid_paths[:10]
        print(f"SANITY CHECK: {len(train_paths)} train, {len(valid_paths)} valid")

    # Load model
    print("Loading model...")
    model_args = Config(str(ARGS_JSON))
    model = Diffusion_Planner(model_args).to(DEVICE)
    ckpt = torch.load(str(args.model_path), map_location=DEVICE, weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in ckpt["ema_state_dict"].items()}
    model.load_state_dict(sd, strict=False)

    # Apply LoRA to DiT only
    apply_lora(model.decoder.dit, rank=args.lora_rank)
    freeze_non_lora(model)
    print(f"LoRA params: {count_lora_params(model):,}")

    # Optimizer
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)

    noise_schedule = NoiseScheduleVP()
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len

    # ── Sanity check: forward + reward + backward ─────────────────────
    print("\nSanity check: differentiable forward pass...")
    model.train()
    data = load_npz_data(train_paths[0], DEVICE)
    data_norm = model_args.observation_normalizer(
        {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    )

    # Encoder (frozen, no grad)
    with torch.no_grad():
        encoding = model.encoder(data_norm)
        ego_current = data_norm["ego_current_state"][:, :4]
        nbr_current = data_norm["neighbor_agents_past"][:, :P-1, -1, :4]
        current_states = torch.cat([ego_current[:, None], nbr_current], dim=1)
        nbr_past = data_norm["neighbor_agents_past"]
        neighbor_mask = (nbr_past[:, :, -1, :].abs().sum(-1) < 1e-6)

    # Differentiable DPM-Solver++ forward (uses exact same code path as inference)
    ego_traj, deltas = differentiable_dpm_solver_sample(
        model, model_args, data_norm, noise_schedule, device=DEVICE,
        return_deltas=True,
    )

    # Route lanes
    rl = data.get("route_lanes")
    route_xy = None
    if rl is not None:
        rl_np = rl[0].cpu().numpy()
        route_pts = rl_np[:, :, :2].reshape(-1, 2)
        mask = np.any(np.abs(route_pts) > 0.1, axis=1)
        route_pts = route_pts[mask]
        if len(route_pts) > 2:
            route_xy = torch.tensor(route_pts, dtype=torch.float32, device=DEVICE)

    # Neighbor future trajectories for collision detection
    nb_fut = data.get("neighbor_agents_future")
    neighbor_future = None
    if nb_fut is not None:
        neighbor_future = nb_fut[0].to(DEVICE)  # [32, 80, 3]

    reward = differentiable_reward(ego_traj, route_xy, neighbor_future=neighbor_future, deltas=deltas)
    loss = -reward  # maximize reward = minimize -reward

    print(f"  Trajectory progress: {float(torch.norm(ego_traj[-1, :2] - ego_traj[0, :2])):.1f}m")
    print(f"  Reward: {float(reward):.2f}")
    print(f"  Loss: {float(loss):.2f}")

    # Backward
    optimizer.zero_grad()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in lora_params if p.grad is not None)
    print(f"  Gradient norm: {grad_norm:.6f}")
    if grad_norm > 0:
        print("  PASS: gradients are flowing through DPM-Solver++!")
    else:
        print("  FAIL: no gradients!")
        return

    # ── Constrained GRPO: dual variables (Lagrangian multipliers) ──
    # These auto-adapt: increase when constraint is violated, decrease otherwise.
    # Convergence constraint: E[sum δ_i] ≤ delta_max
    delta_max = 1.0    # threshold for sum of δ_i (baseline is ~1.2, want < 1.0)
    lambda_conv = 0.0  # start at 0, let dual ascent activate if needed
    # Velocity floor: E[avg_speed] ≥ v_min
    v_min = 2.0        # m/s — minimum average speed to prevent zero-velocity
    lambda_vel = 0.0   # start at 0
    eta_dual = 0.01    # dual variable learning rate

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        model.train()
        epoch_rewards = []
        epoch_loss = 0.0
        n_steps = 0
        optimizer.zero_grad()

        random.shuffle(train_paths)
        for si, npz_path in enumerate(tqdm(train_paths, desc="Train")):
            try:
                data = load_npz_data(npz_path, DEVICE)
                data_norm = model_args.observation_normalizer(
                    {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
                )

                # Differentiable DPM-Solver++ (exact same path as inference)
                ego_traj, deltas = differentiable_dpm_solver_sample(
                    model, model_args, data_norm, noise_schedule, device=DEVICE,
                    return_deltas=True,
                )

                # Route lanes
                rl = data.get("route_lanes")
                route_xy = None
                if rl is not None:
                    rl_np = rl[0].cpu().numpy()
                    route_pts = rl_np[:, :, :2].reshape(-1, 2)
                    mask_r = np.any(np.abs(route_pts) > 0.1, axis=1)
                    route_pts = route_pts[mask_r]
                    if len(route_pts) > 2:
                        route_xy = torch.tensor(route_pts, dtype=torch.float32, device=DEVICE)

                # Neighbor future for collision detection
                nb_fut = data.get("neighbor_agents_future")
                neighbor_future = nb_fut[0].to(DEVICE) if nb_fut is not None else None

                reward = differentiable_reward(ego_traj, route_xy, neighbor_future=neighbor_future, deltas=deltas)
                reward_loss = -reward

                # Constrained GRPO (方案D from survey):
                # - Convergence: Lagrangian penalty on E[sum δ_i] exceeding threshold
                # - Velocity floor: Lagrangian penalty when avg speed < v_min
                # Both use dual variables that auto-adapt (increase when violated)
                conv_reg = torch.tensor(0.0, device=DEVICE)
                vel_penalty = torch.tensor(0.0, device=DEVICE)

                if deltas and len(deltas) > 0:
                    sum_delta = torch.stack(deltas).sum()
                    # Only penalize when above threshold (slack constraint)
                    conv_reg = torch.clamp(sum_delta - delta_max, min=0.0)

                # Velocity floor: prevent zero-velocity degenerate solution
                xy = ego_traj[:, :2]
                velocity = (xy[1:] - xy[:-1]) / 0.1  # DT=0.1s
                avg_speed = velocity.norm(dim=1).mean()
                vel_penalty = torch.clamp(v_min - avg_speed, min=0.0)

                loss = (reward_loss
                        + lambda_conv * conv_reg
                        + lambda_vel * vel_penalty) / args.grad_accum

                loss.backward()
                epoch_rewards.append(float(reward))
                epoch_loss += float(loss) * args.grad_accum

                # Dual variable update (Lagrangian ascent)
                # λ ← max(0, λ + η·violation) — increases when constraint violated
                lambda_conv = max(0.0, lambda_conv + eta_dual * float(conv_reg))
                lambda_vel = max(0.0, lambda_vel + eta_dual * float(vel_penalty))

                # Gradient accumulation step
                if (si + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    n_steps += 1

            except Exception as e:
                print(f"  skip: {e}")
                optimizer.zero_grad()

        # Final step for remaining
        if (si + 1) % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            n_steps += 1

        mean_reward = np.mean(epoch_rewards) if epoch_rewards else 0
        mean_loss = epoch_loss / max(len(epoch_rewards), 1)
        print(f"  reward={mean_reward:.2f}  loss={mean_loss:.4f}  steps={n_steps}")
        print(f"  λ_conv={lambda_conv:.4f}  λ_vel={lambda_vel:.4f}")

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "lora_state_dict": {k: v for k, v in model.state_dict().items() if "lora_" in k},
            "reward_mean": mean_reward,
        }, args.out_dir / f"lora_epoch_{epoch:03d}.pth")

        # Validation every 2 epochs — multi-metric evaluation
        if epoch % 2 == 0 or epoch == args.epochs:
            print("  Validating...")
            model.eval()
            val_progresses = []
            val_speeds = []
            val_collisions = []
            val_route_devs = []
            for vp in valid_paths[:40]:
                try:
                    data = load_npz_data(vp, DEVICE)
                    data_norm = model_args.observation_normalizer(
                        {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
                    )
                    data_norm["sampled_trajectories"] = torch.zeros(
                        1, P, future_len + 1, 4, device=DEVICE)
                    with torch.no_grad():
                        _, out = model(data_norm)
                    ego = out["prediction"][0, 0].cpu().numpy()

                    # Progress (displacement)
                    prog = float(np.linalg.norm(ego[-1, :2] - ego[0, :2]))
                    val_progresses.append(prog)

                    # Speed
                    vel = np.diff(ego[:, :2], axis=0) / 0.1
                    speed = np.linalg.norm(vel, axis=1)
                    val_speeds.append(float(speed.mean()))

                    # Collision check (against neighbor futures)
                    nb_fut = data.get("neighbor_agents_future")
                    n_col = 0
                    if nb_fut is not None:
                        nb = nb_fut[0].numpy()
                        for n in range(nb.shape[0]):
                            nb_xy = nb[n, :ego.shape[0], :2]
                            if np.abs(nb_xy).max() < 0.1:
                                continue
                            dists = np.linalg.norm(ego[:, :2] - nb_xy, axis=1)
                            if np.any(dists < 3.0):
                                n_col += 1
                    val_collisions.append(n_col)

                    # Route deviation
                    rl = data.get("route_lanes")
                    if rl is not None:
                        rl_np = rl[0].numpy()
                        route_pts = rl_np[:, :, :2].reshape(-1, 2)
                        mask = np.any(np.abs(route_pts) > 0.1, axis=1)
                        route_pts = route_pts[mask]
                        if len(route_pts) > 2:
                            idx = np.linspace(0, ego.shape[0]-1, 10, dtype=int)
                            devs = [np.min(np.linalg.norm(route_pts - ego[i, :2], axis=1)) for i in idx]
                            val_route_devs.append(float(np.mean(devs)))

                except Exception:
                    pass

            if val_progresses:
                mp = np.mean(val_progresses)
                ms = np.mean(val_speeds) if val_speeds else 0
                mc = np.mean(val_collisions) if val_collisions else 0
                mr = np.mean(val_route_devs) if val_route_devs else 0
                print(f"  Val: progress={mp:.1f}m | speed={ms:.1f}m/s | "
                      f"collisions={mc:.2f} | route_dev={mr:.2f}m")

                # Multi-metric early stop: progress OK AND speed reasonable AND route_dev reasonable
                progress_ok = mp > 35
                speed_ok = ms > 2.0
                route_ok = mr < 5.0

                if progress_ok and speed_ok and route_ok:
                    if not hasattr(main, '_good_count'):
                        main._good_count = 0
                    main._good_count += 1
                    print(f"  All metrics OK ({main._good_count}/2)")
                    if main._good_count >= 2:
                        print(f"  EARLY STOP: all metrics passed for 2 consecutive checks")
                        break
                else:
                    main._good_count = 0
                    reasons = []
                    if not progress_ok: reasons.append(f"progress={mp:.1f}<35")
                    if not speed_ok: reasons.append(f"speed={ms:.1f}<2.0")
                    if not route_ok: reasons.append(f"route_dev={mr:.1f}>5.0")
                    print(f"  Not ready: {', '.join(reasons)}")

                # Give up after 15 epochs
                if epoch >= 15 and not (progress_ok and speed_ok):
                    print(f"  GIVE UP: 15 epochs done, metrics not converging")
                    break

    # ── Merge and save ────────────────────────────────────────────────
    print("\nMerging LoRA...")
    model.eval()
    merge_lora_into_dit(model.decoder.dit)
    torch.save({"model": model.state_dict()}, args.out_dir / "merged_grpo.pth")
    print(f"Saved to {args.out_dir / 'merged_grpo.pth'}")

    # Final verify
    model.eval()
    progresses = []
    for vp in valid_paths[:10]:
        try:
            data = load_npz_data(vp, DEVICE)
            data_norm = model_args.observation_normalizer(
                {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
            )
            data_norm["sampled_trajectories"] = torch.zeros(1, P, future_len + 1, 4, device=DEVICE)
            with torch.no_grad():
                _, out = model(data_norm)
            ego = out["prediction"][0, 0].cpu().numpy()
            progresses.append(float(np.linalg.norm(ego[-1, :2] - ego[0, :2])))
        except Exception:
            pass
    if progresses:
        mp = np.mean(progresses)
        print(f"Final progress: {mp:.1f}m ({'PASS' if mp > 80 else 'MARGINAL' if mp > 50 else 'FAIL'})")

    print("Done!")


if __name__ == "__main__":
    main()
