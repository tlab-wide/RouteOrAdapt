"""
DLR: Denoising-Loss Router.

Per scene + per expert, compute the diffusion training loss against an "anchor"
trajectory (the route-lane centerline path, treated as a pseudo-ground-truth).
The expert whose model best explains this anchor (lowest denoising loss)
is selected. This is Idea 1 from brainstorm_v2.md — inspired by Diff-DAgger
which uses diffusion loss for OOD uncertainty estimation.

Distinction from prior SCR / SCES / LFS:
  - SCR: feature MLP, no expert-specific signal — fails per Finding 2.
  - SCES: pairwise traj distance, biased toward low-magnitude shared_only.
  - LFS: distance of expert's free generation to lane — measured: also fails.
  - DLR: scores experts by how well they model an in-distribution-like
        target, using each expert's own training objective.

Pick the anchor = route-lane interpolated polyline (the planner's input is
already in ego frame, so we sample 80 points along the visible route).
"""
import os, sys, json, time
sys.path.insert(0, '/opt/autoware/local/lib/python3.10/dist-packages')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner_ros')
sys.path.insert(0, 'third_party/Diffusion-Planner/preference_optimization')
sys.path.insert(0, '.')

import numpy as np
import torch
from diffusion_planner.utils.config import Config
from diffusion_planner.dimensions import OUTPUT_T, POSE_DIM, MAX_NUM_AGENTS
from diffusion_planner_ros.lanelet2_utils.lanelet_converter import convert_lanelet
from diffusion_planner.model.diffusion_utils.dpm_solver_pytorch import NoiseScheduleVP
from overnight_stage1_multiego import load_model, build_inputs, pick_n_egos, UNSEEN, N_EGO, SEED

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ARGS = 'third_party/Diffusion-Planner/args.json'
EXPERTS = ['shared_only','hongo','kashiwa','shalun','unagoya']
NOISE_LEVELS = [0.3, 0.5, 0.7]  # multiple sigma to average over

def build_anchor_from_route(route_lanes_np):
    """Pick the most-occupied route lane and interpolate to 80 points."""
    lanes = []
    for i in range(route_lanes_np.shape[0]):
        pts = route_lanes_np[i, :, :2]
        if np.abs(pts).sum() < 0.1: continue
        # prefer lanes that go forward (positive x at end)
        if pts[-1, 0] < pts[0, 0]: continue
        lanes.append(pts)
    if not lanes: return None
    # Pick the longest forward lane
    best = max(lanes, key=lambda p: np.linalg.norm(p[-1] - p[0]))
    # Interpolate to 80 points
    cum = np.cumsum(np.linalg.norm(np.diff(best, axis=0), axis=1))
    total = float(cum[-1])
    target_dists = np.linspace(0, total, OUTPUT_T)
    cum = np.concatenate([[0], cum])
    new_pts = np.zeros((OUTPUT_T, 2))
    for i, td in enumerate(target_dists):
        idx = np.searchsorted(cum, td)
        if idx == 0: new_pts[i] = best[0]
        elif idx >= len(best): new_pts[i] = best[-1]
        else:
            t = (td - cum[idx-1]) / max(cum[idx] - cum[idx-1], 1e-6)
            new_pts[i] = best[idx-1] + t * (best[idx] - best[idx-1])
    return new_pts


def denoising_loss(model, cfg, inputs_norm, anchor_xy, noise_levels):
    """Compute mean denoising MSE over `noise_levels` for the model's DiT
    given the anchor trajectory as pseudo-ground-truth."""
    # Build target trajectory tensor matching the model's expected output format.
    # model.decoder.dit expects x of shape (B, P, (1+T)*POSE_DIM) where the first
    # block is ego current state followed by T future steps. We populate ego
    # row 0 with anchor and zero out neighbors.
    P = 1 + cfg.predicted_neighbor_num
    T = OUTPUT_T
    B = 1
    x_dim = (1 + T) * 4   # 4 = ego state dim per timestep (x, y, cos_h, sin_h)
    x_clean = torch.zeros(B, P, x_dim, device=DEVICE)
    # heading from finite difference
    h = np.zeros(T+1)
    for i in range(1, T+1):
        h[i] = np.arctan2(anchor_xy[i-1, 1] - (anchor_xy[i-2,1] if i>=2 else 0),
                          anchor_xy[i-1, 0] - (anchor_xy[i-2,0] if i>=2 else 0))
    cos_h = np.cos(h); sin_h = np.sin(h)
    # current state row 0 → zero (ego at origin, heading 0)
    # future T steps for ego
    for t in range(T):
        base = (t+1) * 4
        x_clean[0, 0, base:base+4] = torch.tensor([anchor_xy[t, 0], anchor_xy[t, 1],
                                                     cos_h[t+1], sin_h[t+1]], device=DEVICE)

    # Encode
    with torch.no_grad():
        ctx = model.encoder(inputs_norm)
    # neighbor_current_mask: all-False = all neighbors active (so model attends)
    neighbor_mask = torch.zeros(B, P - 1, dtype=torch.bool, device=DEVICE)

    losses = []
    for sigma in noise_levels:
        with torch.no_grad():
            noise = torch.randn_like(x_clean) * sigma
            x_noisy = x_clean + noise
            t_tensor = torch.tensor([sigma], device=DEVICE)  # use sigma as proxy for t
            x_pred = model.decoder.dit(x_noisy, t_tensor, ctx, neighbor_mask)
        # Only score the ego (row 0) — others are noise
        mse = float(((x_pred[0, 0] - x_clean[0, 0]) ** 2).mean().item())
        losses.append(mse)
    return float(np.mean(losses))


def main():
    cfg = Config(ARGS)
    print("Loading 5 experts...")
    models = {e: load_model(e, cfg) for e in EXPERTS}
    models = {e: m for e, m in models.items() if m is not None}

    s1 = json.load(open('checkpoints/overnight_20260510/stage1_multiego.json'))
    s1_idx = {(r['map'], r['ego_idx'], r['expert']): r for r in s1['rows']}
    out = {'description': 'DLR — Denoising-Loss Router (Idea 1)', 'rows': []}

    for tag, osm in UNSEEN:
        try: llmap = convert_lanelet(osm)
        except: continue
        if not llmap.lanelets: continue
        egos = pick_n_egos(llmap, N_EGO, seed=SEED)
        print(f"\n=== {tag} ===")
        for ego_idx, ego_ll in enumerate(egos):
            cl = ego_ll.centerline
            if cl.shape[0] < 2: continue
            cx, cy = float(cl[0,0]), float(cl[0,1])
            heading = float(np.arctan2(cl[1,1]-cl[0,1], cl[1,0]-cl[0,0]))
            inputs = build_inputs(llmap, (cx, cy), heading)
            inputs_norm = cfg.observation_normalizer({k: (v.clone() if isinstance(v, torch.Tensor) else v) for k,v in inputs.items()})

            anchor = build_anchor_from_route(inputs['route_lanes'].cpu().numpy()[0])
            if anchor is None: continue

            scene = {'map': tag, 'ego_idx': ego_idx, 'losses': {}, 'pick': None, 'max_lat': {}}
            for expert, model in models.items():
                try:
                    loss = denoising_loss(model, cfg, inputs_norm, anchor, NOISE_LEVELS)
                    scene['losses'][expert] = loss
                    r1 = s1_idx.get((tag, ego_idx, expert))
                    if r1 and r1.get('max_lat') is not None:
                        scene['max_lat'][expert] = r1['max_lat']
                except Exception as e:
                    if ego_idx == 0: print(f"    {expert} failed: {e}")
            if scene['losses']:
                scene['pick'] = min(scene['losses'].items(), key=lambda kv: kv[1])[0]
            out['rows'].append(scene)

        from collections import Counter
        picks = Counter(r['pick'] for r in out['rows'] if r['map']==tag and r.get('pick'))
        print(f"  DLR pick distribution: {dict(picks)}")

    out_path = 'checkpoints/overnight_20260511/stage9_dlr.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\n=== Per-map summary ===")
    for tag in ['aichi_hub_v2','ishio_dai','morikoro']:
        rows = [r for r in out['rows'] if r['map']==tag]
        dlr_lats = [r['max_lat'].get(r['pick']) for r in rows if r.get('pick') and r['pick'] in r.get('max_lat', {})]
        unagoya_lats = [r['max_lat'].get('unagoya') for r in rows if 'unagoya' in r.get('max_lat', {})]
        dlr_lats = [x for x in dlr_lats if x is not None]
        unagoya_lats = [x for x in unagoya_lats if x is not None]
        print(f"\n  {tag}:  n={len(rows)}")
        if dlr_lats: print(f"    DLR lat:           {np.mean(dlr_lats):.1f} ± {np.std(dlr_lats,ddof=1) if len(dlr_lats)>1 else 0:.1f}")
        if unagoya_lats: print(f"    best-fixed unagoya: {np.mean(unagoya_lats):.1f} ± {np.std(unagoya_lats,ddof=1) if len(unagoya_lats)>1 else 0:.1f}")

main()
