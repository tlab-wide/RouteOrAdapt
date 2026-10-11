#!/usr/bin/env python3
"""
Stage 10 (Idea 3): GPS-Prior Adapter Mixing.

For each Aichi map, compute Haversine distance to each of the 4 training-ODD
centers; build a softmax with temperature tau over -distance/km; produce
alpha_i weights summing to 1; linearly average the 4 merged-expert state dicts
with these weights to form a mixed model; run inference and record max_lat.

Sweep tau in {1, 10, 50, 200} km. Also test pessimistic mixing (top-2 only).

Outputs checkpoints/overnight_20260513/stage10_gps_mix.json.
"""
import os, sys, json, random, time
os.environ.setdefault('LD_LIBRARY_PATH', '/opt/autoware/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
sys.path.insert(0, '/opt/autoware/local/lib/python3.10/dist-packages')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner_ros')
sys.path.insert(0, 'third_party/Diffusion-Planner/preference_optimization')
sys.path.insert(0, '.')

import numpy as np
import torch
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from train_reward_backprop import differentiable_dpm_solver_sample, NoiseScheduleVP
from diffusion_planner_ros.lanelet2_utils.lanelet_converter import convert_lanelet
from overnight_stage1_multiego import (
    build_inputs, pick_n_egos, metrics_for_traj, UNSEEN, N_EGO, SEED,
    SELORA_DIR, ARGS,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = 'checkpoints/overnight_20260513'
os.makedirs(OUT_DIR, exist_ok=True)

EXPERTS = ['hongo', 'kashiwa', 'shalun', 'unagoya']
# Geographic centers of the 4 training ODDs (Wikipedia / OSM coordinates)
ODD_CENTERS = {
    'hongo':   (35.7128, 139.7626),   # UTokyo Hongo Campus
    'kashiwa': (35.8949, 139.9521),   # UTokyo Kashiwanoha Campus
    'shalun':  (24.8300, 120.9400),   # ARTC Shalun test track
    'unagoya': (35.1551, 136.9690),   # Nagoya University
}

TAUS = [1.0, 10.0, 50.0, 200.0]  # km

# Aichi map centers (verified from OSM node medians; same values as prev. stage7)
AICHI_MAP_CENTERS = {
    'aichi_hub_v2': (35.1828, 137.1131),
    'ishio_dai':    (35.2310, 137.0240),
    'morikoro':     (35.1820, 137.0780),
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam/2)**2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    ex = np.exp(x)
    return ex / ex.sum()


def compute_alphas(map_lat, map_lon, tau_km, top2_only=False):
    """alpha_i = softmax(-d_i / tau) over the 4 training ODDs."""
    dists = np.array([haversine_km(map_lat, map_lon, *ODD_CENTERS[e]) for e in EXPERTS])
    if top2_only:
        # Mask to top-2 nearest only
        order = np.argsort(dists)
        mask = np.zeros_like(dists, dtype=bool)
        mask[order[:2]] = True
        scores = -dists / tau_km
        scores[~mask] = -1e9
        return softmax(scores), dists
    scores = -dists / tau_km
    return softmax(scores), dists


def load_state_dict_only(expert):
    """Load a merged_<expert>.pth checkpoint to a state_dict (no Diffusion_Planner instance)."""
    path = f'{SELORA_DIR}/merged_{expert}.pth'
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sd = ckpt.get('model', ckpt)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    return sd


def mix_state_dicts(state_dicts, alphas):
    """Linear blend of state dicts by alphas (sum=1, same keys assumed)."""
    keys = state_dicts[0].keys()
    out = {}
    for k in keys:
        ref = state_dicts[0][k]
        if not torch.is_tensor(ref) or not torch.is_floating_point(ref):
            out[k] = ref  # buffer / non-float: take from first
            continue
        acc = torch.zeros_like(ref, dtype=torch.float32)
        for i, sd in enumerate(state_dicts):
            v = sd[k]
            if v.shape != ref.shape:
                acc = ref.clone().float(); break
            acc = acc + float(alphas[i]) * v.float()
        out[k] = acc.to(ref.dtype)
    return out


def main():
    cfg = Config(ARGS)
    ns = NoiseScheduleVP()

    print("[stage10] Loading 4 expert state_dicts...")
    expert_sds = {e: load_state_dict_only(e) for e in EXPERTS}
    print("  loaded.")

    # Build one base model instance for re-loading mixed weights
    model = Diffusion_Planner(cfg).to(DEVICE).eval()
    # Sanity load (just to populate keys); we re-load mixed sd per (map, tau)
    model.load_state_dict(expert_sds[EXPERTS[0]], strict=False)

    out = {
        'description': 'Stage 10 (Idea 3): GPS soft-mix LoRA averaging',
        'odd_centers': ODD_CENTERS,
        'aichi_centers': AICHI_MAP_CENTERS,
        'taus_km': TAUS,
        'experts': EXPERTS,
        'rows': [],   # one row per (map, tau, mode, ego_idx)
        'alphas': {}, # per (map, tau, mode)
    }

    for tag, osm in UNSEEN:
        m_lat, m_lon = AICHI_MAP_CENTERS[tag]
        print(f"\n=== {tag} ({m_lat:.3f},{m_lon:.3f}) ===")
        try:
            llmap = convert_lanelet(osm)
        except Exception as e:
            print(f"  convert failed: {e}"); continue
        egos = pick_n_egos(llmap, N_EGO, seed=SEED)
        print(f"  {len(egos)} ego starts")

        for tau in TAUS:
            for mode in ['soft', 'top2']:
                alphas, dists = compute_alphas(m_lat, m_lon, tau, top2_only=(mode == 'top2'))
                alpha_dict = {EXPERTS[i]: float(alphas[i]) for i in range(len(EXPERTS))}
                key = f"{tag}|tau={tau}|{mode}"
                out['alphas'][key] = {'alpha': alpha_dict,
                                      'haversine_km': {EXPERTS[i]: float(dists[i]) for i in range(len(EXPERTS))}}
                # Mix state dicts and load
                mixed = mix_state_dicts([expert_sds[e] for e in EXPERTS], alphas)
                model.load_state_dict(mixed, strict=False)
                model.eval()

                lat_list = []
                for ego_idx, ego_ll in enumerate(egos):
                    cl = ego_ll.centerline
                    if cl.shape[0] < 2: continue
                    cx, cy = float(cl[0, 0]), float(cl[0, 1])
                    heading = float(np.arctan2(cl[1, 1] - cl[0, 1], cl[1, 0] - cl[0, 0]))
                    inputs = build_inputs(llmap, (cx, cy), heading)
                    inputs_norm = cfg.observation_normalizer({
                        k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()})
                    try:
                        with torch.no_grad():
                            ego_traj = differentiable_dpm_solver_sample(model, cfg, inputs_norm, ns, device=DEVICE)
                        xy = ego_traj[:, :2].cpu().numpy()
                        m = metrics_for_traj(xy)
                    except Exception as e:
                        m = {'error': str(e)[:120]}
                    row = {'map': tag, 'tau': tau, 'mode': mode, 'ego_idx': ego_idx,
                           'alpha': alpha_dict, **m}
                    out['rows'].append(row)
                    if m.get('finite'):
                        lat_list.append(m['max_lat'])
                if lat_list:
                    print(f"  tau={tau} {mode}  n={len(lat_list)}  "
                          f"alpha={ {k: round(v,2) for k,v in alpha_dict.items()} }  "
                          f"lat={np.mean(lat_list):.2f}±"
                          f"{np.std(lat_list, ddof=1) if len(lat_list)>1 else 0:.2f}")

    out_path = f'{OUT_DIR}/stage10_gps_mix.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\n[stage10] saved {out_path}  ({len(out['rows'])} rows)")

    # Summary table
    print("\n=== Summary (mean max_lat per map per tau, mode=soft) ===")
    print(f"{'map':<16}", end='')
    for tau in TAUS: print(f"tau={tau:<8.0f}", end='')
    print()
    for tag in [m for m,_ in UNSEEN]:
        print(f"{tag:<16}", end='')
        for tau in TAUS:
            lats = [r['max_lat'] for r in out['rows']
                    if r['map'] == tag and r['tau'] == tau and r['mode'] == 'soft' and r.get('finite')]
            if lats:
                print(f"{np.mean(lats):>5.1f}±{np.std(lats,ddof=1) if len(lats)>1 else 0:.1f} ", end='')
            else:
                print('       ? ', end='')
        print()


if __name__ == '__main__':
    main()
