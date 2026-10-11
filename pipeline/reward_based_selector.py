"""Reward-based Selector baseline (reviewer Q3 / G1).

The eighth routing scheme tested in §sec:routing trains a soft mixture gate
directly on realised lateral offset on prod-ODD scenes; reviewer G1 asks
for a no-training Selector that scores K=4 candidate trajectories using the
planner's own 10-term reward function at inference and picks the best.

Setup:
  - For each of 24 Aichi scenes, run inference with each of 4 expert merged
    state dictionaries (merged_hongo, merged_kashiwa, merged_shalun,
    merged_unagoya) → 4 candidate trajectories.
  - Compute the 10-term reward (paper §subsec:reward) on each candidate using
    the route_lanes + neighbor_agents_past from the scene's input dict.
  - Pick argmax-reward expert.
  - Report: routing accuracy vs oracle, induced max_lat under the picks.

Output: outputs/reward_based_selector.json
"""
import os, sys, json, time
os.environ.setdefault('LD_LIBRARY_PATH', '/opt/autoware/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
sys.path.insert(0, '/opt/autoware/local/lib/python3.10/dist-packages')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner_ros')
sys.path.insert(0, 'third_party/Diffusion-Planner/preference_optimization')
sys.path.insert(0, '.')

import numpy as np
import torch
from collections import defaultdict

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from diffusion_planner.model.diffusion_utils.dpm_solver_pytorch import NoiseScheduleVP
from diffusion_planner_ros.lanelet2_utils.lanelet_converter import convert_lanelet
from train_molora import odd_conditioned_reward
from train_reward_backprop import differentiable_dpm_solver_sample
from overnight_stage1_multiego import (build_inputs, pick_n_egos, UNSEEN, N_EGO, SEED,
                                         ARGS, BASE_PTH)

DEVICE = torch.device('cuda')
EXPERTS = ['hongo', 'kashiwa', 'shalun', 'unagoya']
SELORA_DIR = 'checkpoints/molora_v2_output'
S1 = 'checkpoints/overnight_20260510/stage1_multiego.json'
OUT = 'outputs/reward_based_selector.json'


def load_flat_expert(cfg, expert_name):
    m = Diffusion_Planner(cfg).to(DEVICE).eval()
    base = torch.load(BASE_PTH, map_location=DEVICE, weights_only=False)
    bsd = base.get('ema_state_dict') or base['model']
    bsd = {k.replace('module.', ''): v for k, v in bsd.items()}
    m.load_state_dict(bsd, strict=False)
    path = f'{SELORA_DIR}/merged_{expert_name}.pth'
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    sd = ck.get('model', ck)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    return m


def reward_for(traj_xy_hsin, route_xy, neighbor_future):
    """Compute the 10-term reward on a sampled trajectory using
    ODD-default weights (Aichi scenes have no matching ODD weight entry)."""
    return float(odd_conditioned_reward(traj_xy_hsin, route_xy, neighbor_future,
                                          deltas=None, map_name='unknown').item())


def extract_route_xy(inputs):
    """Extract route centreline xy points (in base_link frame, flattened) for reward."""
    route = inputs.get('route_lanes')  # (B=1, P=25, V=20, D=33)
    if route is None: return None
    r = route[0].detach().cpu().numpy()  # (25, 20, 33)
    mask = (np.abs(r[:, :, :2]).sum(axis=2) > 0.1)
    pts = r[:, :, :2][mask]
    if pts.shape[0] < 2: return None
    return torch.tensor(pts, dtype=torch.float32, device=DEVICE)


def extract_neighbor_future(inputs):
    """Use neighbor_agents_past (last frame x,y) extrapolated as a coarse 'future' proxy.
    For Aichi scenes neighbor data is essentially absent (zero-filled), so the
    reward's collision/RSS terms will mostly be zero. This is fine for routing:
    progress, route_dev, comfort, and rss_route still discriminate."""
    nb = inputs.get('neighbor_agents_past')
    if nb is None: return None
    n = nb[0].detach().cpu().numpy()  # (32, 31, 11)
    # Use last frame's xy as a constant-extrapolation 80-frame future
    last_xy = n[:, -1, :2]
    mask = (np.abs(last_xy).sum(axis=1) > 0.1)
    if not mask.any(): return None
    out = np.zeros((32, 80, 2), dtype=np.float32)
    out[:, :, :] = last_xy[:, None, :]   # broadcast last position 80 times
    return torch.tensor(out, dtype=torch.float32, device=DEVICE)


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
            heading = float(np.arctan2(cl[1, 1] - cl[0, 1], cl[1, 0] - cl[0, 0]))
            try:
                inputs = build_inputs(llmap, (cx, cy), heading)
                scenes.append(dict(map=tag, ego_idx=ego_idx, inputs=inputs))
            except Exception as e:
                print(f"  skip {tag}/{ego_idx}: {e}")
    return scenes


def rollout_xy_hsin(model, cfg, inputs, ns):
    """Run inference, return trajectory tensor (T, 4) where columns are
    (x, y, cos_h, sin_h) — the format reward function expects."""
    inputs_norm = cfg.observation_normalizer({k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()})
    with torch.no_grad():
        tj = differentiable_dpm_solver_sample(model, cfg, inputs_norm, ns, device=DEVICE)
    return tj  # already (T, 4) including (x, y, cos_h, sin_h)


def main():
    cfg = Config(ARGS); ns = NoiseScheduleVP()
    scenes = build_aichi_scenes(cfg)
    print(f"[G1] {len(scenes)} Aichi scenes")

    # Load 4 expert models
    print("Loading 4 expert models (flat merged checkpoints)...")
    models = {e: load_flat_expert(cfg, e) for e in EXPERTS}
    print("  done")

    # Stage 1 cached max_lat for ground truth comparison
    s1 = json.load(open(S1))
    lats = {}
    for r in s1['rows']:
        if r.get('finite') and r['expert'] in EXPERTS:
            lats.setdefault((r['map'], r['ego_idx']), {})[r['expert']] = float(r['max_lat'])

    rows = []
    for i, s in enumerate(scenes):
        # Skip scenes missing stage1 entries
        key = (s['map'], s['ego_idx'])
        if key not in lats or len(lats[key]) < 4: continue
        oracle = min(lats[key], key=lats[key].get)

        route_xy = extract_route_xy(s['inputs'])
        nb_fut = extract_neighbor_future(s['inputs'])

        rewards = {}
        max_lat_under = {}
        for e in EXPERTS:
            try:
                traj = rollout_xy_hsin(models[e], cfg, s['inputs'], ns)
                r_val = reward_for(traj, route_xy, nb_fut)
                rewards[e] = r_val
                # also compute this run's max_lat to verify against stage1
                xy = traj[:, :2].detach().cpu().numpy()
                max_lat_under[e] = float(np.max(np.abs(xy[:, 1])))
            except Exception as ex:
                rewards[e] = -1e9
                max_lat_under[e] = float('nan')
        picked = max(rewards, key=rewards.get)
        rows.append(dict(
            map=s['map'], ego_idx=s['ego_idx'],
            rewards=rewards, oracle=oracle, picked_reward=picked,
            max_lat_under_picked=lats[key].get(picked),  # use stage1 ref number for consistency
            max_lat_oracle=lats[key].get(oracle),
            max_lat_static_unagoya=lats[key].get('unagoya'),
        ))
        if (i+1) % 6 == 0:
            print(f"  scene {i+1}/{len(scenes)}: rewards = {[f'{e}:{rewards[e]:.1f}' for e in EXPERTS]}, "
                  f"picked={picked}, oracle={oracle}")

    # Aggregate
    correct = sum(1 for r in rows if r['picked_reward'] == r['oracle'])
    static_correct = sum(1 for r in rows if r['oracle'] == 'unagoya')
    induced_offset = float(np.mean([r['max_lat_under_picked'] for r in rows if r['max_lat_under_picked'] is not None]))
    oracle_offset = float(np.mean([r['max_lat_oracle'] for r in rows if r['max_lat_oracle'] is not None]))
    static_offset = float(np.mean([r['max_lat_static_unagoya'] for r in rows if r['max_lat_static_unagoya'] is not None]))

    # Per-map breakdown
    by_map = defaultdict(list)
    for r in rows:
        by_map[r['map']].append(r)
    print("\n=== Per-map routing decisions ===")
    for m, rs in by_map.items():
        from collections import Counter
        picks = Counter(r['picked_reward'] for r in rs)
        oracles = Counter(r['oracle'] for r in rs)
        print(f"  {m}: reward-picker={dict(picks)} | oracle={dict(oracles)}")

    out = dict(
        description='Reward-based Selector: argmax_e R(traj_e) over 4 experts × 24 Aichi scenes',
        n_scenes=len(rows), n_experts=len(EXPERTS),
        routing_acc_vs_oracle=float(correct / max(1, len(rows))),
        static_rule_acc_vs_oracle=float(static_correct / max(1, len(rows))),
        induced_offset_m=induced_offset,
        oracle_offset_m=oracle_offset,
        static_offset_m=static_offset,
        rows=rows,
    )
    json.dump(out, open(OUT, 'w'), indent=2, default=float)
    print(f"\n=== SUMMARY ===")
    print(f"  reward-picker routing acc vs oracle: {out['routing_acc_vs_oracle']:.3f}")
    print(f"  static rule routing acc vs oracle:   {out['static_rule_acc_vs_oracle']:.3f}")
    print(f"  oracle lower-bound mean max_lat:     {oracle_offset:.2f} m")
    print(f"  static rule (always unagoya) mean:   {static_offset:.2f} m")
    print(f"  reward-picker induced mean max_lat:  {induced_offset:.2f} m")
    print(f"\nSaved {OUT}")


if __name__ == '__main__':
    main()
