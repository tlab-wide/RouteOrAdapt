"""
MoLoRA: Mixture-of-LoRA Experts for ODD-Aware Diffusion Planning.

Architecture:
  Base Diffusion Planner (frozen)
    + Shared LoRA (rank=4)    -- cross-ODD general improvement
    + Expert_0 LoRA (rank=8)  -- hongo (dense urban, high curvature)
    + Expert_1 LoRA (rank=8)  -- kashiwa (suburban, long straight)
    + Expert_2 LoRA (rank=8)  -- shalun (test track, low speed)
    + Expert_3 LoRA (rank=8)  -- unagoya (mixed urban/suburban)

Forward: output = original(x) + shared(x) + expert_i(x)
ODD routing: deterministic based on map name in npz path.
ODD-conditioned reward: per-ODD reward weights (ISO 34503 ODD taxonomy).

Usage:
  python train_molora.py --epochs 10 --lr 2e-6 \
    --train_npz_list .../path_list_train_v2.json \
    --valid_npz_list .../path_list_valid_v2.json \
    --out_dir .../molora_output
"""

import argparse
import copy
import json
import math
import random
import re
import sys
from collections import defaultdict
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
from diffusion_planner.model.diffusion_utils.dpm_solver_pytorch import NoiseScheduleVP
from utils import load_npz_data

# Reuse differentiable DPM-Solver++ and base reward from original
from train_reward_backprop import (
    differentiable_dpm_solver_sample,
    differentiable_reward,
    merge_lora_into_dit,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════════════════
# ODD Configuration (ISO 34503 taxonomy)
# ═══════════════════════════════════════════════════════════════════════════

ODD_MAPS = ["hongo", "kashiwa", "shalun", "unagoya", "nishishinjuku"]
ODD_MAP_TO_IDX = {m: i for i, m in enumerate(ODD_MAPS)}
N_EXPERTS = len(ODD_MAPS)

# ODD-conditioned reward weights: adjusted per map characteristics
# Based on Autoware E2E Report p34: "metrics should evaluate safety of
# observed behaviour" and ISO 34503 ODD taxonomy
ODD_REWARD_WEIGHTS = {
    "hongo": {
        # Dense urban: many curves, intersections → emphasis on route/heading
        "progress": 1.5, "route_dev": 3.0, "curve": 4.0,
        "comfort": 1.0, "heading": 3.0, "speed_pen": 1.0,
        "lat_acc": 2.0, "collision": 1.5,
        "rss": 2.0, "rss_route": 0.5,   # high traffic → strong RSS
    },
    "kashiwa": {
        # Suburban: long roads, higher speed → emphasis on progress/speed
        "progress": 3.0, "route_dev": 1.5, "curve": 2.0,
        "comfort": 1.0, "heading": 1.5, "speed_pen": 1.5,
        "lat_acc": 1.0, "collision": 1.0,
        "rss": 1.5, "rss_route": 0.3,   # higher speed → braking dist matters more
    },
    "shalun": {
        # Test track: tight layout, low speed → emphasis on route/comfort
        "progress": 2.0, "route_dev": 2.5, "curve": 3.5,
        "comfort": 2.0, "heading": 2.0, "speed_pen": 0.5,
        "lat_acc": 2.0, "collision": 1.0,
        "rss": 1.0, "rss_route": 1.0,   # tight track → route RSS more important
    },
    "unagoya": {
        # Mixed urban/suburban: balanced
        "progress": 2.5, "route_dev": 2.0, "curve": 2.5,
        "comfort": 1.5, "heading": 2.0, "speed_pen": 1.0,
        "lat_acc": 1.5, "collision": 1.0,
        "rss": 1.5, "rss_route": 0.5,
    },
    "nishishinjuku": {
        # Dense urban with complex intersections, high traffic density
        "progress": 1.5, "route_dev": 3.0, "curve": 3.5,
        "comfort": 1.0, "heading": 3.0, "speed_pen": 1.0,
        "lat_acc": 2.0, "collision": 2.0,
        "rss": 2.5, "rss_route": 0.5,   # highest traffic → strongest RSS
    },
}

# Default (fallback for unknown maps)
ODD_REWARD_DEFAULT = {
    "progress": 2.0, "route_dev": 2.0, "curve": 3.0,
    "comfort": 1.0, "heading": 2.0, "speed_pen": 1.0,
    "lat_acc": 1.5, "collision": 1.0,
    "rss": 1.5, "rss_route": 0.5,
}


# ═══════════════════════════════════════════════════════════════════════════
# 1. MoLoRA: Shared LoRA + Per-Expert LoRA
# ═══════════════════════════════════════════════════════════════════════════

class MoLoRALinear(nn.Module):
    """Mixture-of-LoRA: shared adapter + N expert adapters.

    forward(x, expert_idx=None):
      y = original(x) + shared_lora(x)
      if expert_idx is not None:
        y += expert_loras[expert_idx](x)
      return y
    """

    def __init__(self, linear: nn.Linear, n_experts: int = 4,
                 shared_rank: int = 4, expert_rank: int = 8,
                 alpha: float = 32.0):
        super().__init__()
        self.original = linear
        self.original.requires_grad_(False)
        d_out, d_in = linear.weight.shape
        self.n_experts = n_experts

        # Shared LoRA (always active)
        self.shared_scale = alpha / shared_rank
        self.shared_A = nn.Linear(d_in, shared_rank, bias=False)
        self.shared_B = nn.Linear(shared_rank, d_out, bias=False)
        nn.init.kaiming_uniform_(self.shared_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.shared_B.weight)

        # Per-expert LoRAs
        self.expert_scale = alpha / expert_rank
        self.expert_As = nn.ModuleList([
            nn.Linear(d_in, expert_rank, bias=False) for _ in range(n_experts)
        ])
        self.expert_Bs = nn.ModuleList([
            nn.Linear(expert_rank, d_out, bias=False) for _ in range(n_experts)
        ])
        for i in range(n_experts):
            nn.init.kaiming_uniform_(self.expert_As[i].weight, a=math.sqrt(5))
            nn.init.zeros_(self.expert_Bs[i].weight)

        # Move to same device as original
        dev = linear.weight.device
        self.shared_A = self.shared_A.to(dev)
        self.shared_B = self.shared_B.to(dev)
        self.expert_As = self.expert_As.to(dev)
        self.expert_Bs = self.expert_Bs.to(dev)

        # Active expert index (set before forward pass)
        self._active_expert: int | None = None

    def set_expert(self, idx: int | None):
        """Set which expert to activate. None = shared only."""
        self._active_expert = idx

    def forward(self, x):
        y = self.original(x) + self.shared_scale * self.shared_B(self.shared_A(x))
        if self._active_expert is not None:
            i = self._active_expert
            y = y + self.expert_scale * self.expert_Bs[i](self.expert_As[i](x))
        return y

    def merge(self, expert_idx: int | None = None) -> nn.Linear:
        """Merge shared + specific expert into a single Linear."""
        merged = copy.deepcopy(self.original)
        merged.requires_grad_(True)
        with torch.no_grad():
            # Add shared
            merged.weight.data += self.shared_scale * (
                self.shared_B.weight @ self.shared_A.weight)
            # Add expert
            if expert_idx is not None:
                merged.weight.data += self.expert_scale * (
                    self.expert_Bs[expert_idx].weight @ self.expert_As[expert_idx].weight)
        return merged


def apply_molora(dit, n_experts=4, shared_rank=4, expert_rank=8, alpha=32.0):
    """Replace Linear layers in DiT with MoLoRA layers."""
    dit.preproj.fc1 = MoLoRALinear(dit.preproj.fc1, n_experts, shared_rank, expert_rank, alpha)
    dit.preproj.fc2 = MoLoRALinear(dit.preproj.fc2, n_experts, shared_rank, expert_rank, alpha)
    for block in dit.blocks:
        block.mlp1.fc1 = MoLoRALinear(block.mlp1.fc1, n_experts, shared_rank, expert_rank, alpha)
        block.mlp1.fc2 = MoLoRALinear(block.mlp1.fc2, n_experts, shared_rank, expert_rank, alpha)
        block.mlp2.fc1 = MoLoRALinear(block.mlp2.fc1, n_experts, shared_rank, expert_rank, alpha)
        block.mlp2.fc2 = MoLoRALinear(block.mlp2.fc2, n_experts, shared_rank, expert_rank, alpha)
    dit.final_layer.proj[1] = MoLoRALinear(dit.final_layer.proj[1], n_experts, shared_rank, expert_rank, alpha)
    dit.final_layer.proj[4] = MoLoRALinear(dit.final_layer.proj[4], n_experts, shared_rank, expert_rank, alpha)


def set_active_expert(model, expert_idx: int | None):
    """Set active expert across all MoLoRA layers."""
    for m in model.modules():
        if isinstance(m, MoLoRALinear):
            m.set_expert(expert_idx)


def freeze_non_molora(model):
    """Freeze everything except MoLoRA parameters."""
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, MoLoRALinear):
            m.shared_A.weight.requires_grad_(True)
            m.shared_B.weight.requires_grad_(True)
            for ea in m.expert_As:
                ea.weight.requires_grad_(True)
            for eb in m.expert_Bs:
                eb.weight.requires_grad_(True)


def count_molora_params(model):
    """Count total MoLoRA parameters."""
    total = 0
    for m in model.modules():
        if isinstance(m, MoLoRALinear):
            total += m.shared_A.weight.numel() + m.shared_B.weight.numel()
            for ea in m.expert_As:
                total += ea.weight.numel()
            for eb in m.expert_Bs:
                total += eb.weight.numel()
    return total


def merge_molora_into_dit(dit, expert_idx: int | None = None):
    """Merge shared + expert LoRA into base weights."""
    def _replace(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, MoLoRALinear):
                setattr(parent, name, child.merge(expert_idx))
            else:
                _replace(child)
    _replace(dit)


# ═══════════════════════════════════════════════════════════════════════════
# 2. ODD-Conditioned Reward
# ═══════════════════════════════════════════════════════════════════════════

DT = 0.1

def odd_conditioned_reward(
    ego_traj: torch.Tensor,
    route_xy: torch.Tensor | None,
    neighbor_future: torch.Tensor | None,
    deltas: list | None,
    map_name: str,
) -> torch.Tensor:
    """Compute reward with ODD-specific weights.

    Same reward components as differentiable_reward() but scaled per-ODD.
    """
    w = ODD_REWARD_WEIGHTS.get(map_name, ODD_REWARD_DEFAULT)

    xy = ego_traj[:, :2]
    T = xy.shape[0]
    cos_h, sin_h = ego_traj[:, 2], ego_traj[:, 3]

    # Progress
    step_dists = torch.norm(xy[1:] - xy[:-1], dim=1)
    path_length = step_dists.sum()
    progress_reward = torch.clamp(path_length, max=200.0)

    # Route deviation (curvature-weighted)
    route_dev = torch.tensor(0.0, device=xy.device)
    curve_penalty = torch.tensor(0.0, device=xy.device)
    if route_xy is not None and route_xy.shape[0] > 2:
        idx = torch.linspace(0, T - 1, min(20, T), dtype=torch.long, device=xy.device)
        sampled = xy[idx]
        dists = torch.norm(sampled.unsqueeze(1) - route_xy.unsqueeze(0), dim=2)
        min_dists = dists.min(dim=1).values
        route_dev = min_dists.mean()

        # Curvature-weighted penalty
        route_dx = route_xy[1:] - route_xy[:-1]
        route_speeds = torch.norm(route_dx, dim=1)
        if route_speeds.shape[0] > 1:
            route_headings = torch.atan2(route_dx[:, 1], route_dx[:, 0])
            route_curvature = (route_headings[1:] - route_headings[:-1]).abs()
            high_curv_mask = route_curvature > 0.05
            if high_curv_mask.any():
                high_curv_indices = high_curv_mask.nonzero(as_tuple=True)[0]
                if len(high_curv_indices) > 0:
                    curv_sample_idx = high_curv_indices[
                        torch.linspace(0, len(high_curv_indices)-1, min(10, len(high_curv_indices)),
                                       dtype=torch.long, device=xy.device)]
                    curv_route_pts = route_xy[curv_sample_idx]
                    curv_dists = torch.norm(
                        xy.unsqueeze(1) - curv_route_pts.unsqueeze(0), dim=2)
                    curv_min = curv_dists.min(dim=0).values
                    curve_penalty = curv_min.mean()

    # Comfort (jerk)
    velocity = (xy[1:] - xy[:-1]) / DT
    speed = velocity.norm(dim=1)
    accel = (velocity[1:] - velocity[:-1]) / DT
    jerk = (accel[1:] - accel[:-1]) / DT
    jerk_norm = jerk.norm(dim=1)
    comfort = -jerk_norm.mean() * 0.001

    # Heading error
    dx = xy[1:, 0] - xy[:-1, 0]
    dy = xy[1:, 1] - xy[:-1, 1]
    fast_mask = speed > 0.5
    heading_err = torch.tensor(0.0, device=xy.device)
    if fast_mask.any():
        travel_h = torch.atan2(dy, dx)
        pred_h = torch.atan2(sin_h[:-1], cos_h[:-1])
        hdiff = torch.atan2(torch.sin(travel_h - pred_h), torch.cos(travel_h - pred_h))
        heading_err = hdiff[fast_mask].abs().mean()

    # Speed penalty (only if car barely moved)
    speed_penalty = torch.clamp(5.0 - path_length, min=0.0) * 2.0

    # Lateral acceleration
    lat_acc_penalty = torch.tensor(0.0, device=xy.device)
    if speed.numel() > 2:
        heading_traj = torch.atan2(velocity[:, 1], velocity[:, 0])
        yaw_rate = (heading_traj[1:] - heading_traj[:-1]) / DT
        lat_acc = speed[:-1] * yaw_rate.abs()
        lat_acc_penalty = torch.clamp(lat_acc.mean() - 3.0, min=0.0)

    # Collision
    collision_cost = torch.tensor(0.0, device=xy.device)
    COLLISION_RADIUS = 3.0
    if neighbor_future is not None:
        for n in range(neighbor_future.shape[0]):
            nb_xy = neighbor_future[n, :T, :2]
            if nb_xy.abs().max() < 0.1:
                continue
            dist = torch.norm(xy - nb_xy, dim=1)
            penetration = torch.clamp(COLLISION_RADIUS - dist, min=0.0)
            collision_cost = collision_cost + penetration.sum()

    # ── Reachable Safe State (RSS) ──────────────────────────────────
    # Autoware E2E Report p35: "maintain the system in a state where
    # there is always an action available that can be performed safely
    # (i.e. minimum risk manoeuvre or emergency stop)"
    #
    # At each timestep, verify: braking_distance < distance_to_nearest_obstacle
    # If violated, the ego is in an unsafe state where emergency stop
    # cannot prevent collision — penalize this.
    #
    # braking_dist = v² / (2 * a_max)
    # a_max = 5.0 m/s² (comfortable emergency braking, ISO 22179)
    A_MAX_BRAKE = 5.0
    rss_penalty = torch.tensor(0.0, device=xy.device)

    if neighbor_future is not None and speed.numel() > 0:
        # Compute braking distance at each timestep: d = v² / (2a)
        braking_dist = speed ** 2 / (2.0 * A_MAX_BRAKE)  # [T-1]

        # Find min distance to any active neighbor at each timestep
        # Use T-1 timesteps (aligned with speed)
        min_nb_dist = torch.full((T - 1,), 1e6, device=xy.device)
        ego_xy_steps = xy[:-1]  # [T-1, 2]

        for n in range(neighbor_future.shape[0]):
            nb_xy = neighbor_future[n, :T-1, :2]
            if nb_xy.abs().max() < 0.1:
                continue
            nb_dist = torch.norm(ego_xy_steps - nb_xy, dim=1)  # [T-1]
            min_nb_dist = torch.minimum(min_nb_dist, nb_dist)

        # RSS violation: braking_dist > available_dist - safety_margin
        SAFETY_MARGIN = 2.0  # meters (vehicle length buffer)
        available_dist = min_nb_dist - SAFETY_MARGIN
        rss_violation = torch.clamp(braking_dist - available_dist, min=0.0)

        # Only penalize where neighbors are actually present (dist < 100m)
        active_mask = min_nb_dist < 100.0
        if active_mask.any():
            rss_penalty = rss_violation[active_mask].mean()

    # Also check RSS against route boundary (drivable area proxy)
    # If route_xy available, ensure ego can stop before leaving route corridor
    rss_route_penalty = torch.tensor(0.0, device=xy.device)
    if route_xy is not None and route_xy.shape[0] > 2 and speed.numel() > 0:
        # Distance from ego to nearest route point (proxy for road boundary)
        ROUTE_CORRIDOR = 5.0  # meters — max acceptable lateral deviation
        ego_route_dists = torch.norm(
            xy[:-1].unsqueeze(1) - route_xy.unsqueeze(0), dim=2)  # [T-1, N]
        min_route_dist = ego_route_dists.min(dim=1).values  # [T-1]

        # If going fast and far from route → can't stop before leaving corridor
        braking_dist = speed ** 2 / (2.0 * A_MAX_BRAKE)
        lateral_margin = ROUTE_CORRIDOR - min_route_dist  # remaining margin
        rss_route_violation = torch.clamp(
            braking_dist * 0.1 - lateral_margin, min=0.0)  # 0.1: lateral fraction
        if (min_route_dist > 1.0).any():
            rss_route_penalty = rss_route_violation[min_route_dist > 1.0].mean()

    # ODD-weighted reward
    reward = (
        w["progress"] * progress_reward
        - w["route_dev"] * route_dev
        - w["curve"] * curve_penalty
        + w["comfort"] * comfort
        - w["heading"] * heading_err
        - w["speed_pen"] * speed_penalty
        - w["lat_acc"] * lat_acc_penalty
        - w["collision"] * collision_cost
        - w.get("rss", 1.5) * rss_penalty           # L0: reachable safe state (obstacle)
        - w.get("rss_route", 0.5) * rss_route_penalty  # L0: reachable safe state (route)
    )

    return reward


# ═══════════════════════════════════════════════════════════════════════════
# 3. Utility: extract map name from npz path
# ═══════════════════════════════════════════════════════════════════════════

def extract_map_name(npz_path: str) -> str:
    """Extract map name from npz file path."""
    path_lower = npz_path.lower()
    for m in ODD_MAPS:
        if m in path_lower:
            return m
    return "unknown"


def stratified_sample(per_map_paths: dict, batch_size: int) -> list:
    """ODD-stratified sampling: each batch has balanced map representation."""
    maps = list(per_map_paths.keys())
    per_map = max(1, batch_size // len(maps))
    batch = []
    for m in maps:
        if per_map_paths[m]:
            batch.extend(random.sample(per_map_paths[m],
                         min(per_map, len(per_map_paths[m]))))
    random.shuffle(batch)
    return batch


# ═══════════════════════════════════════════════════════════════════════════
# 4. Main Training
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path, default=MODEL_PTH)
    parser.add_argument("--train_npz_list", type=Path,
                        default=Path("checkpoints/npz_multimap/path_list_train_v2.json"))
    parser.add_argument("--valid_npz_list", type=Path,
                        default=Path("checkpoints/npz_multimap/path_list_valid_v2.json"))
    parser.add_argument("--out_dir", type=Path,
                        default=Path("checkpoints/molora_output"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--shared_rank", type=int, default=4)
    parser.add_argument("--expert_rank", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--sanity_check", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.train_npz_list) as f:
        train_paths = json.load(f)
    with open(args.valid_npz_list) as f:
        valid_paths = json.load(f)

    # Group paths by map (ODD-stratified sampling)
    per_map_train = defaultdict(list)
    for p in train_paths:
        m = extract_map_name(p)
        per_map_train[m].append(p)

    print(f"Training data by ODD:")
    for m, paths in sorted(per_map_train.items()):
        print(f"  {m}: {len(paths)} scenes")

    if args.sanity_check:
        for m in per_map_train:
            per_map_train[m] = per_map_train[m][:20]
        valid_paths = valid_paths[:10]
        print("SANITY CHECK mode")

    # Load model
    print("\nLoading model...")
    model_args = Config(str(ARGS_JSON))
    model = Diffusion_Planner(model_args).to(DEVICE)
    ckpt = torch.load(str(args.model_path), map_location=DEVICE, weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in ckpt["ema_state_dict"].items()}
    model.load_state_dict(sd, strict=False)

    # Apply MoLoRA
    apply_molora(model.decoder.dit, n_experts=N_EXPERTS,
                 shared_rank=args.shared_rank, expert_rank=args.expert_rank)
    freeze_non_molora(model)
    n_params = count_molora_params(model)
    print(f"MoLoRA params: {n_params:,} (shared_rank={args.shared_rank}, expert_rank={args.expert_rank}, n_experts={N_EXPERTS})")

    # Optimizer: single optimizer for all params (shared + all experts)
    molora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(molora_params, lr=args.lr, weight_decay=0.01)

    noise_schedule = NoiseScheduleVP()
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len

    # ── Sanity check ──
    print("\nSanity check: MoLoRA forward + ODD reward + backward...")
    model.train()
    test_path = train_paths[0]
    test_map = extract_map_name(test_path)
    set_active_expert(model, ODD_MAP_TO_IDX.get(test_map, 0))

    data = load_npz_data(test_path, DEVICE)
    data_norm = model_args.observation_normalizer(
        {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    )
    ego_traj, deltas = differentiable_dpm_solver_sample(
        model, model_args, data_norm, noise_schedule, device=DEVICE, return_deltas=True)

    rl = data.get("route_lanes")
    route_xy = None
    if rl is not None:
        rl_np = rl[0].cpu().numpy()
        route_pts = rl_np[:, :, :2].reshape(-1, 2)
        mask = np.any(np.abs(route_pts) > 0.1, axis=1)
        route_pts = route_pts[mask]
        if len(route_pts) > 2:
            route_xy = torch.tensor(route_pts, dtype=torch.float32, device=DEVICE)

    nb_fut = data.get("neighbor_agents_future")
    neighbor_future = nb_fut[0].to(DEVICE) if nb_fut is not None else None

    reward = odd_conditioned_reward(ego_traj, route_xy, neighbor_future, deltas, test_map)
    loss = -reward

    optimizer.zero_grad()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in molora_params if p.grad is not None)
    print(f"  Map: {test_map}, Expert: {ODD_MAP_TO_IDX.get(test_map, 0)}")
    print(f"  Reward: {float(reward):.2f}, Grad norm: {grad_norm:.6f}")
    if grad_norm > 0:
        print("  PASS: gradients flow through MoLoRA + DPM-Solver++!")
    else:
        print("  FAIL: no gradients!")
        return

    # ── Training loop (ODD-stratified) ──
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        model.train()
        epoch_rewards = []
        per_map_rewards = defaultdict(list)
        epoch_loss = 0.0
        n_steps = 0
        optimizer.zero_grad()

        # ODD-stratified: interleave maps so each grad_accum batch is balanced
        # Build epoch order: cycle through maps round-robin
        maps_available = [m for m in ODD_MAPS if per_map_train.get(m)]
        per_map_shuffled = {}
        for m in maps_available:
            paths = per_map_train[m][:]
            random.shuffle(paths)
            per_map_shuffled[m] = paths

        # Round-robin: take one from each map in sequence
        max_per_map = max(len(v) for v in per_map_shuffled.values())
        epoch_order = []
        for i in range(max_per_map):
            for m in maps_available:
                if i < len(per_map_shuffled[m]):
                    epoch_order.append((m, per_map_shuffled[m][i]))

        for si, (map_name, npz_path) in enumerate(tqdm(epoch_order, desc="Train")):
            try:
                # Activate the correct expert
                expert_idx = ODD_MAP_TO_IDX.get(map_name)
                set_active_expert(model, expert_idx)

                data = load_npz_data(npz_path, DEVICE)
                data_norm = model_args.observation_normalizer(
                    {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
                )

                ego_traj, deltas = differentiable_dpm_solver_sample(
                    model, model_args, data_norm, noise_schedule, device=DEVICE,
                    return_deltas=True)

                rl = data.get("route_lanes")
                route_xy = None
                if rl is not None:
                    rl_np = rl[0].cpu().numpy()
                    route_pts = rl_np[:, :, :2].reshape(-1, 2)
                    mask_r = np.any(np.abs(route_pts) > 0.1, axis=1)
                    route_pts = route_pts[mask_r]
                    if len(route_pts) > 2:
                        route_xy = torch.tensor(route_pts, dtype=torch.float32, device=DEVICE)

                nb_fut = data.get("neighbor_agents_future")
                neighbor_future = nb_fut[0].to(DEVICE) if nb_fut is not None else None

                # ODD-conditioned reward
                reward = odd_conditioned_reward(ego_traj, route_xy, neighbor_future, deltas, map_name)
                loss = -reward / args.grad_accum

                loss.backward()
                epoch_rewards.append(float(reward))
                per_map_rewards[map_name].append(float(reward))
                epoch_loss += float(loss) * args.grad_accum

                if (si + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(molora_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    n_steps += 1

            except Exception as e:
                print(f"  skip ({map_name}): {e}")
                optimizer.zero_grad()

        # Final step
        if (si + 1) % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(molora_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            n_steps += 1

        mean_reward = np.mean(epoch_rewards) if epoch_rewards else 0
        mean_loss = epoch_loss / max(len(epoch_rewards), 1)
        print(f"  Overall: reward={mean_reward:.2f}  loss={mean_loss:.4f}  steps={n_steps}")
        for m in ODD_MAPS:
            mr = per_map_rewards.get(m, [])
            if mr:
                print(f"  {m}: reward={np.mean(mr):.2f} (n={len(mr)})")

        # Save checkpoint (all shared + expert weights)
        torch.save({
            "epoch": epoch,
            "molora_state_dict": {k: v for k, v in model.state_dict().items()
                                  if "shared_" in k or "expert_" in k},
            "reward_mean": mean_reward,
            "per_map_rewards": {m: np.mean(v) if v else 0 for m, v in per_map_rewards.items()},
        }, args.out_dir / f"molora_epoch_{epoch:03d}.pth")

        # Validation every 2 epochs
        if epoch % 2 == 0 or epoch == args.epochs:
            print("  Validating...")
            model.eval()
            val_results = defaultdict(lambda: defaultdict(list))

            for vp in valid_paths[:60]:
                try:
                    vm = extract_map_name(vp)
                    expert_idx = ODD_MAP_TO_IDX.get(vm)
                    set_active_expert(model, expert_idx)

                    data = load_npz_data(vp, DEVICE)
                    data_norm = model_args.observation_normalizer(
                        {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
                    )
                    data_norm["sampled_trajectories"] = torch.zeros(
                        1, P, future_len + 1, 4, device=DEVICE)
                    with torch.no_grad():
                        _, out = model(data_norm)
                    ego = out["prediction"][0, 0].cpu().numpy()

                    prog = float(np.linalg.norm(ego[-1, :2] - ego[0, :2]))
                    vel = np.diff(ego[:, :2], axis=0) / 0.1
                    spd = np.linalg.norm(vel, axis=1).mean()

                    nb_fut = data.get("neighbor_agents_future")
                    n_col = 0
                    if nb_fut is not None:
                        nb = nb_fut[0].numpy()
                        for n in range(nb.shape[0]):
                            nb_xy = nb[n, :ego.shape[0], :2]
                            if np.abs(nb_xy).max() < 0.1:
                                continue
                            if np.any(np.linalg.norm(ego[:, :2] - nb_xy, axis=1) < 3.0):
                                n_col += 1

                    rl = data.get("route_lanes")
                    rd = 0.0
                    if rl is not None:
                        rl_np = rl[0].numpy()
                        rp = rl_np[:, :, :2].reshape(-1, 2)
                        mask = np.any(np.abs(rp) > 0.1, axis=1)
                        rp = rp[mask]
                        if len(rp) > 2:
                            idx = np.linspace(0, ego.shape[0]-1, 10, dtype=int)
                            devs = [np.min(np.linalg.norm(rp - ego[i, :2], axis=1)) for i in idx]
                            rd = float(np.mean(devs))

                    val_results[vm]["progress"].append(prog)
                    val_results[vm]["speed"].append(spd)
                    val_results[vm]["collisions"].append(n_col)
                    val_results[vm]["route_dev"].append(rd)

                except Exception:
                    pass

            # Print per-map validation
            all_prog, all_spd, all_col, all_rd = [], [], [], []
            for m in ODD_MAPS:
                r = val_results.get(m, {})
                if r:
                    mp = np.mean(r["progress"])
                    ms = np.mean(r["speed"])
                    mc = np.mean(r["collisions"])
                    mr = np.mean(r["route_dev"])
                    print(f"  Val {m}: progress={mp:.1f}m | speed={ms:.1f}m/s | "
                          f"collisions={mc:.2f} | route_dev={mr:.2f}m")
                    all_prog.extend(r["progress"])
                    all_spd.extend(r["speed"])
                    all_col.extend(r["collisions"])
                    all_rd.extend(r["route_dev"])

            if all_prog:
                print(f"  Val ALL: progress={np.mean(all_prog):.1f}m | speed={np.mean(all_spd):.1f}m/s | "
                      f"collisions={np.mean(all_col):.2f} | route_dev={np.mean(all_rd):.2f}m")

                # Early stop: per-map check
                maps_ok = 0
                for m in ODD_MAPS:
                    r = val_results.get(m, {})
                    if r and np.mean(r["progress"]) > 30 and np.mean(r["speed"]) > 1.5:
                        maps_ok += 1
                if maps_ok >= 3:
                    if not hasattr(main, '_good_count'):
                        main._good_count = 0
                    main._good_count += 1
                    print(f"  {maps_ok}/4 maps OK ({main._good_count}/2 consecutive)")
                    if main._good_count >= 2:
                        print(f"  EARLY STOP: majority of ODDs passing")
                        break
                else:
                    main._good_count = 0

    # ── Merge and save per-expert models ──
    print("\n" + "="*60)
    print("Merging MoLoRA experts...")
    print("="*60)

    # Save the full MoLoRA state first
    torch.save({
        "model": model.state_dict(),
    }, args.out_dir / "molora_full.pth")

    # Create per-expert merged models (for ONNX export)
    for expert_idx, map_name in enumerate(ODD_MAPS):
        print(f"\nMerging expert {expert_idx} ({map_name})...")
        # Reload base model
        expert_model = Diffusion_Planner(model_args).to("cpu")
        expert_model.load_state_dict(sd, strict=False)
        apply_molora(expert_model.decoder.dit, n_experts=N_EXPERTS,
                     shared_rank=args.shared_rank, expert_rank=args.expert_rank)
        # Load trained MoLoRA weights
        expert_model.load_state_dict(model.state_dict(), strict=False)
        # Merge shared + this expert
        merge_molora_into_dit(expert_model.decoder.dit, expert_idx=expert_idx)
        out_path = args.out_dir / f"merged_{map_name}.pth"
        torch.save({"model": expert_model.state_dict()}, out_path)
        print(f"  Saved {out_path}")

    # Also save shared-only (no expert) as baseline comparison
    print("\nMerging shared-only (no expert)...")
    shared_model = Diffusion_Planner(model_args).to("cpu")
    shared_model.load_state_dict(sd, strict=False)
    apply_molora(shared_model.decoder.dit, n_experts=N_EXPERTS,
                 shared_rank=args.shared_rank, expert_rank=args.expert_rank)
    shared_model.load_state_dict(model.state_dict(), strict=False)
    merge_molora_into_dit(shared_model.decoder.dit, expert_idx=None)
    torch.save({"model": shared_model.state_dict()}, args.out_dir / "merged_shared_only.pth")
    print(f"  Saved merged_shared_only.pth")

    # Final validation summary
    print(f"\nDone! Models saved to {args.out_dir}/")
    print(f"  molora_full.pth          -- full MoLoRA state (for resume)")
    print(f"  merged_shared_only.pth   -- shared LoRA only (ablation)")
    for m in ODD_MAPS:
        print(f"  merged_{m}.pth       -- shared + expert_{m}")


if __name__ == "__main__":
    main()
