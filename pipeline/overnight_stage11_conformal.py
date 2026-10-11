#!/usr/bin/env python3
"""
Stage 11 (Idea 2): Per-Expert Split-Conformal Corridor Predictor.

For each expert e in {hongo, kashiwa, shalun, unagoya}:
  - Run e on a calibration set of N npz scenes from each of 4 prod ODDs
  - Extract 8 SCR features per scene
  - Fit a Ridge regressor: features -> max_lat
  - Compute 90% split-conformal residual quantile q_e on own-ODD residuals
  - Define u_e(x) = max(predicted_lat_e(x), 0) + q_e

Then on the 3 Aichi unseen maps (already in stage1_multiego.json), build u_e(x)
for each scene and produce the abstention decision (Idea 5 triage stage 1).

Outputs:
  checkpoints/overnight_20260513/stage11_conformal.json
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
from diffusion_planner_ros.lanelet2_utils.lanelet_converter import convert_lanelet
from utils import load_npz_data, generate_deterministic_trajectory
from overnight_stage1_multiego import (
    load_model, build_inputs, pick_n_egos, UNSEEN, N_EGO, SEED,
    SELORA_DIR, ARGS,
)
from overnight_stage3_scr import features_from_npz, NUM_FEATS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = 'checkpoints/overnight_20260513'
os.makedirs(OUT_DIR, exist_ok=True)

EXPERTS = ['hongo', 'kashiwa', 'shalun', 'unagoya']
N_CALIB_PER_ODD = 30   # calibration scenes per (expert × ODD)
SEED_NPZ = 0


def features_from_inputs(inputs):
    """Extract the same 8 SCR features but from the inputs-dict format used
    for the Aichi unseen-map scenes (lanes is a (1, 140, 20, 33) tensor)."""
    lanes = inputs['lanes'][0].cpu().numpy()  # (140, 20, 33)
    nb_pasts = inputs['neighbor_agents_past'][0].cpu().numpy()  # (32, 31, 11)

    lane_mask = (np.abs(lanes[:, :, :2]).sum(axis=(1, 2)) > 0.1)
    n_lanes = float(lane_mask.sum())
    if n_lanes == 0:
        return np.zeros(NUM_FEATS, dtype=np.float32)

    mids = lanes[lane_mask, :, :2].mean(axis=1)
    lane_y_spread = float(mids[:, 1].std()) if mids.shape[0] > 1 else 0.0
    lane_x_spread = float(mids[:, 0].std()) if mids.shape[0] > 1 else 0.0
    lane_lengths = np.linalg.norm(
        lanes[lane_mask, -1, :2] - lanes[lane_mask, 0, :2], axis=1)
    mean_len = float(lane_lengths.mean())

    curvs = []
    for i in range(lanes.shape[0]):
        if not lane_mask[i]: continue
        pts = lanes[i, :, :2]
        diffs = np.diff(pts, axis=0)
        headings = np.arctan2(diffs[:, 1], diffs[:, 0])
        dh = np.abs(np.diff(headings))
        if dh.size: curvs.append(float(np.mean(dh)))
    mean_curv = float(np.mean(curvs)) if curvs else 0.0

    nb_mask = (np.abs(nb_pasts[:, -1, :2]).sum(axis=1) > 0.1)
    nb_speed = 0.0
    if nb_mask.any() and nb_pasts.shape[1] >= 2:
        last2 = nb_pasts[nb_mask, -2:, :2]
        speeds = np.linalg.norm(last2[:, 1] - last2[:, 0], axis=1) / 0.1
        nb_speed = float(speeds.mean())

    route = inputs['route_lanes'][0].cpu().numpy()
    n_route = float((np.abs(route[:, :, :2]).sum(axis=(1, 2)) > 0.1).sum())
    ego0 = float(inputs['ego_current_state'][0, 0].cpu().numpy())
    return np.array([n_lanes, mean_curv, mean_len, lane_y_spread, lane_x_spread,
                     nb_speed, n_route, ego0], dtype=np.float32)


def sample_calib_paths(odd_name, n, seed):
    """Pick n npz paths from the prod-ODD training list for `odd_name`."""
    paths = json.load(open('checkpoints/npz_multimap/path_list_train_v3.json'))
    sel = [p for p in paths if odd_name in p.lower()]
    rng = random.Random(seed + hash(odd_name) % 10**6)
    rng.shuffle(sel)
    return sel[:n]


def rollout_max_lat(model, cfg, data):
    """Run the model deterministically on a data dict; return max_lat of ego."""
    ego = generate_deterministic_trajectory(model, cfg, data, device=DEVICE)
    xy = ego[:, :2]
    if not np.all(np.isfinite(xy)):
        return None
    return float(np.max(np.abs(xy[:, 1])))


class RidgeRegressor:
    """Closed-form ridge: minimizes ||Xw - y||^2 + lam ||w||^2."""
    def __init__(self, lam=1.0):
        self.lam = lam
        self.w = None
        self.mu = None
        self.sd = None
    def fit(self, X, y):
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6
        Xn = (X - self.mu) / self.sd
        Xb = np.concatenate([Xn, np.ones((Xn.shape[0], 1))], axis=1)
        n, d = Xb.shape
        A = Xb.T @ Xb + self.lam * np.eye(d)
        A[-1, -1] -= self.lam  # don't regularize bias
        b = Xb.T @ y
        self.w = np.linalg.solve(A, b)
        return self
    def predict(self, X):
        Xn = (X - self.mu) / self.sd
        Xb = np.concatenate([Xn, np.ones((Xn.shape[0], 1))], axis=1)
        return Xb @ self.w


def main():
    cfg = Config(ARGS)

    out = {
        'description': 'Stage 11 (Idea 2): per-expert split-conformal corridor predictor + Idea 5 triage',
        'n_calib_per_odd': N_CALIB_PER_ODD,
        'experts': EXPERTS,
        'calibration': {},   # per expert: own-ODD residual quantile + per-ODD lat distribution
        'aichi_scenes': [],  # per (map, ego): features, u_e, abstain, idea5 pick
    }

    # ── 1. Calibration: per expert, run on calibration npz from 4 ODDs ──────
    print(f"[stage11] Building calibration set: {N_CALIB_PER_ODD} npz/ODD × 4 ODDs × {len(EXPERTS)} experts")
    per_expert_data = {e: {'feat': [], 'lat': [], 'odd': []} for e in EXPERTS}

    for expert in EXPERTS:
        print(f"\n[stage11] Expert {expert}")
        model = load_model(expert, cfg)
        if model is None:
            print(f"  load_model({expert}) failed; skip")
            continue
        for odd in EXPERTS:  # 4 ODDs == 4 experts (one ODD per expert)
            paths = sample_calib_paths(odd, N_CALIB_PER_ODD, SEED_NPZ)
            print(f"  on ODD={odd}: {len(paths)} npz")
            for p in paths:
                try:
                    data = load_npz_data(p, DEVICE)
                    lat = rollout_max_lat(model, cfg, data)
                    if lat is None: continue
                    feats = features_from_npz(p)
                    per_expert_data[expert]['feat'].append(feats)
                    per_expert_data[expert]['lat'].append(lat)
                    per_expert_data[expert]['odd'].append(odd)
                except Exception as e:
                    pass
        n_total = len(per_expert_data[expert]['lat'])
        print(f"  total {n_total} pairs")
        # Free GPU
        del model
        torch.cuda.empty_cache()

    # ── 2. Per expert ridge + split-conformal q_e ───────────────────────────
    print(f"\n[stage11] Fitting per-expert Ridge regressors + split-conformal q_e^0.9")
    regressors = {}
    q_e = {}
    for e in EXPERTS:
        d = per_expert_data[e]
        if not d['lat']:
            print(f"  {e}: no data — skip")
            continue
        X = np.array(d['feat'], dtype=np.float32)
        y = np.array(d['lat'], dtype=np.float32)
        odd_arr = np.array(d['odd'])
        own_mask = (odd_arr == e)
        n_own, n_total = int(own_mask.sum()), len(y)

        # Train on all calibration data (cross-ODD informs feature -> lat regressor)
        reg = RidgeRegressor(lam=1.0).fit(X, y)
        regressors[e] = reg

        # Split-conformal residuals on own-ODD only
        own_X = X[own_mask]
        own_y = y[own_mask]
        own_pred = reg.predict(own_X)
        residuals = np.abs(own_y - own_pred)
        q90 = float(np.quantile(residuals, 0.9)) if len(residuals) >= 5 else float('nan')
        q_e[e] = q90

        out['calibration'][e] = {
            'n_calib': n_total, 'n_own_odd': n_own,
            'mean_own_lat': float(own_y.mean()) if n_own else None,
            'std_own_lat': float(own_y.std(ddof=1)) if n_own > 1 else None,
            'mean_lat_per_odd': {o: float(y[odd_arr == o].mean())
                                  for o in EXPERTS if (odd_arr == o).any()},
            'q_e_0.9': q90,
            'mean_residual_own': float(residuals.mean()) if len(residuals) else None,
        }
        print(f"  {e}: n_own={n_own}/{n_total}  mean_own_lat={own_y.mean():.2f}  q90={q90:.2f}")

    # ── 3. Compute u_e(x) on each Aichi scene using stage1 trajectories ─────
    print(f"\n[stage11] Computing u_e(x) on 24 Aichi scenes from stage1_multiego.json")
    s1 = json.load(open('checkpoints/overnight_20260510/stage1_multiego.json'))
    s1_by_scene = {}
    for r in s1['rows']:
        key = (r['map'], r['ego_idx'])
        s1_by_scene.setdefault(key, {})[r['expert']] = r

    # Need feature vectors for each Aichi scene -- rebuild inputs from osm
    aichi_features = {}
    for tag, osm in UNSEEN:
        try:
            llmap = convert_lanelet(osm)
        except Exception as e:
            print(f"  convert {tag} failed: {e}"); continue
        egos = pick_n_egos(llmap, N_EGO, seed=SEED)
        for ego_idx, ego_ll in enumerate(egos):
            cl = ego_ll.centerline
            if cl.shape[0] < 2: continue
            cx, cy = float(cl[0, 0]), float(cl[0, 1])
            heading = float(np.arctan2(cl[1, 1] - cl[0, 1], cl[1, 0] - cl[0, 0]))
            inputs = build_inputs(llmap, (cx, cy), heading)
            feats = features_from_inputs(inputs)
            aichi_features[(tag, ego_idx)] = feats

    print(f"  extracted features for {len(aichi_features)} Aichi scenes")

    # ── 4. Idea 5 triage rule per scene ─────────────────────────────────────
    ABSTAIN_THRESH = 5.0  # m
    for (tag, ego_idx), feats in aichi_features.items():
        u_e = {}
        for e, reg in regressors.items():
            pred = float(reg.predict(feats[None])[0])
            u_e[e] = max(pred, 0.0) + q_e[e]
        # Surviving experts: u_e <= threshold
        surv = [e for e, u in u_e.items() if u <= ABSTAIN_THRESH]
        abstain = (len(surv) == 0)
        # Best non-abstained pick: argmin_e u_e
        if not abstain:
            best_pick = min(surv, key=lambda e: u_e[e])
        else:
            best_pick = None
        # Realized lat per expert from stage1
        realized = {e: s1_by_scene.get((tag, ego_idx), {}).get(e, {}).get('max_lat') for e in EXPERTS}
        out['aichi_scenes'].append({
            'map': tag, 'ego_idx': ego_idx,
            'features': feats.tolist(),
            'u_e': u_e,
            'survivors': surv,
            'abstain': abstain,
            'idea5_pick': best_pick,
            'realized_max_lat_per_expert': realized,
        })

    # ── 5. Stats: abstention rate per map, deployed lat per map ─────────────
    print("\n=== Idea 5 Triage Stats (abstain threshold = 5m) ===")
    for tag in [m for m, _ in UNSEEN]:
        scenes = [s for s in out['aichi_scenes'] if s['map'] == tag]
        n_total = len(scenes)
        n_abstain = sum(1 for s in scenes if s['abstain'])
        deployed_lats = []
        for s in scenes:
            if s['abstain']: continue
            pick = s['idea5_pick']
            if pick and s['realized_max_lat_per_expert'][pick] is not None:
                deployed_lats.append(s['realized_max_lat_per_expert'][pick])
        ar = n_abstain / max(n_total, 1)
        print(f"  {tag}: n={n_total}, abstain={n_abstain} ({ar:.0%})")
        if deployed_lats:
            print(f"    deployed lat: {np.mean(deployed_lats):.2f} ± "
                  f"{np.std(deployed_lats, ddof=1) if len(deployed_lats)>1 else 0:.2f}  "
                  f"(n_accepted={len(deployed_lats)})")
        # Also report what unagoya would have given (baseline)
        un_lats = [s['realized_max_lat_per_expert']['unagoya'] for s in scenes
                   if s['realized_max_lat_per_expert']['unagoya'] is not None]
        print(f"    unagoya baseline: {np.mean(un_lats):.2f} ± "
              f"{np.std(un_lats, ddof=1) if len(un_lats)>1 else 0:.2f}")

    out_path = f'{OUT_DIR}/stage11_conformal.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\n[stage11] saved {out_path}")


if __name__ == '__main__':
    main()
