#!/usr/bin/env python3
"""
Stage 11 v3: same as v2 but with lanes-tensor fallback when route_lanes is empty.

In v2, shalun calibration yielded 0/75 ok because its npz files have empty
route_lanes (test-track scenes; the planner uses lanes-only). The fallback
uses the full lanes tensor when route_lanes contains no nonzero rows, giving
a complete 4-expert calibration set.
"""
import os, sys, json, random
os.environ.setdefault('LD_LIBRARY_PATH', '/opt/autoware/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
sys.path.insert(0, '/opt/autoware/local/lib/python3.10/dist-packages')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner_ros')
sys.path.insert(0, 'third_party/Diffusion-Planner/preference_optimization')
sys.path.insert(0, '.')
import numpy as np
import torch
from diffusion_planner.utils.config import Config
from diffusion_planner_ros.lanelet2_utils.lanelet_converter import convert_lanelet
from utils import load_npz_data, generate_deterministic_trajectory
from overnight_stage1_multiego import (
    load_model, build_inputs, pick_n_egos, UNSEEN, N_EGO, SEED, ARGS,
)
from overnight_stage3_scr import features_from_npz, NUM_FEATS
from overnight_stage11_conformal import features_from_inputs, sample_calib_paths, RidgeRegressor, EXPERTS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = 'checkpoints/overnight_20260513'
N_CALIB_PER_ODD = 25
ABSTAIN_THRESH = 15.0  # m of route_dev (raised from 5 to match calibration scale)


def route_dev_with_fallback(traj_xy, data_or_inputs):
    """Mean perp dist from waypoint to nearest reference lane.
    Use route_lanes if nonzero, else fall back to full lanes tensor."""
    rl = data_or_inputs.get('route_lanes')
    if rl is not None:
        rl_np = rl[0].detach().cpu().numpy() if torch.is_tensor(rl) else (rl[0] if rl.ndim == 4 else rl)
        if np.abs(rl_np[:, :, :2]).max() > 0.1:
            pts = rl_np[:, :, :2].reshape(-1, 2)
            pts = pts[np.any(np.abs(pts) > 0.1, axis=1)]
            if len(pts) >= 2:
                return _devmean(traj_xy, pts), 'route'
    ln = data_or_inputs.get('lanes')
    if ln is not None:
        ln_np = ln[0].detach().cpu().numpy() if torch.is_tensor(ln) else (ln[0] if ln.ndim == 4 else ln)
        pts = ln_np[:, :, :2].reshape(-1, 2)
        pts = pts[np.any(np.abs(pts) > 0.1, axis=1)]
        if len(pts) >= 2:
            return _devmean(traj_xy, pts), 'lanes'
    return None, None


def _devmean(traj_xy, pts):
    if not np.all(np.isfinite(traj_xy)): return None
    idx = np.linspace(0, traj_xy.shape[0]-1, min(20, traj_xy.shape[0]), dtype=int)
    sampled = traj_xy[idx]
    return float(np.mean([np.linalg.norm(pts - s, axis=1).min() for s in sampled]))


def rollout_npz(model, cfg, data):
    ego = generate_deterministic_trajectory(model, cfg, data, device=DEVICE)
    return route_dev_with_fallback(ego[:, :2], data)


def rollout_inputs(model, cfg, inputs):
    from train_reward_backprop import differentiable_dpm_solver_sample, NoiseScheduleVP
    ns = NoiseScheduleVP()
    inputs_norm = cfg.observation_normalizer({
        k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()})
    with torch.no_grad():
        ego_traj = differentiable_dpm_solver_sample(model, cfg, inputs_norm, ns, device=DEVICE)
    return route_dev_with_fallback(ego_traj[:, :2].cpu().numpy(), inputs)


def main():
    cfg = Config(ARGS)
    out = {
        'description': 'Stage 11 v3: per-expert split-conformal with lanes fallback',
        'n_calib_per_odd': N_CALIB_PER_ODD,
        'abstain_thresh_m': ABSTAIN_THRESH,
        'experts': EXPERTS, 'calibration': {}, 'aichi_scenes': [],
    }

    print(f"[stage11_v3] Calibration set: {N_CALIB_PER_ODD} npz/ODD × 4 ODDs × {len(EXPERTS)} experts")
    per_expert = {e: {'feat': [], 'rd': [], 'odd': [], 'ref': []} for e in EXPERTS}
    for expert in EXPERTS:
        print(f"\n[stage11_v3] Expert {expert}")
        model = load_model(expert, cfg)
        if model is None: continue
        for odd in EXPERTS:
            paths = sample_calib_paths(odd, N_CALIB_PER_ODD, 0)
            n_ok = 0
            for p in paths:
                try:
                    data = load_npz_data(p, DEVICE)
                    rd_pair = rollout_npz(model, cfg, data)
                    if rd_pair is None or rd_pair[0] is None: continue
                    rd, ref = rd_pair
                    feats = features_from_npz(p)
                    per_expert[expert]['feat'].append(feats)
                    per_expert[expert]['rd'].append(rd)
                    per_expert[expert]['odd'].append(odd)
                    per_expert[expert]['ref'].append(ref)
                    n_ok += 1
                except Exception:
                    pass
            print(f"  on ODD={odd}: {n_ok}/{len(paths)} ok")
        n_total = len(per_expert[expert]['rd'])
        ref_count = {'route': sum(1 for r in per_expert[expert]['ref'] if r == 'route'),
                     'lanes': sum(1 for r in per_expert[expert]['ref'] if r == 'lanes')}
        print(f"  total {n_total} pairs  ref={ref_count}")
        del model; torch.cuda.empty_cache()

    regressors, q_e = {}, {}
    print(f"\n[stage11_v3] Per-expert Ridge + split-conformal q_e^0.9")
    for e in EXPERTS:
        d = per_expert[e]
        if not d['rd']: continue
        X = np.array(d['feat'], dtype=np.float32)
        y = np.array(d['rd'], dtype=np.float32)
        odd_arr = np.array(d['odd'])
        own = odd_arr == e
        reg = RidgeRegressor(lam=1.0).fit(X, y)
        regressors[e] = reg
        residuals = np.abs(y[own] - reg.predict(X[own])) if own.sum() >= 5 else np.abs(y - reg.predict(X))
        q90 = float(np.quantile(residuals, 0.9)) if len(residuals) >= 5 else float('nan')
        q_e[e] = q90
        out['calibration'][e] = {
            'n_calib': len(y), 'n_own_odd': int(own.sum()),
            'mean_own_rd': float(y[own].mean()) if own.any() else float(y.mean()),
            'std_own_rd':  float(y[own].std(ddof=1)) if own.sum() > 1 else 0.0,
            'mean_rd_per_odd': {o: float(y[odd_arr == o].mean())
                                 for o in EXPERTS if (odd_arr == o).any()},
            'q_e_0.9': q90,
        }
        print(f"  {e}: n_own={int(own.sum())}/{len(y)}  mean_own_rd="
              f"{y[own].mean() if own.any() else y.mean():.2f}  q90={q90:.2f}")

    print("\n[stage11_v3] Aichi: per-scene route_dev for each expert")
    s1 = json.load(open('checkpoints/overnight_20260510/stage1_multiego.json'))
    s1_by_scene = {}
    for r in s1['rows']: s1_by_scene.setdefault((r['map'], r['ego_idx']), {})[r['expert']] = r

    aichi_models = {e: load_model(e, cfg) for e in EXPERTS}
    for tag, osm in UNSEEN:
        try: llmap = convert_lanelet(osm)
        except Exception as e: print(f"  convert {tag} fail: {e}"); continue
        egos = pick_n_egos(llmap, N_EGO, seed=SEED)
        for ego_idx, ego_ll in enumerate(egos):
            cl = ego_ll.centerline
            if cl.shape[0] < 2: continue
            cx, cy = float(cl[0, 0]), float(cl[0, 1])
            heading = float(np.arctan2(cl[1, 1] - cl[0, 1], cl[1, 0] - cl[0, 0]))
            inputs = build_inputs(llmap, (cx, cy), heading)
            feats = features_from_inputs(inputs)
            per_expert_rd = {}
            for e, m in aichi_models.items():
                if m is None: continue
                try:
                    pair = rollout_inputs(m, cfg, inputs)
                    if pair and pair[0] is not None: per_expert_rd[e] = pair[0]
                except Exception: pass
            u_e = {e: max(float(reg.predict(feats[None])[0]), 0.0) + q_e[e]
                   for e, reg in regressors.items()}
            surv = [e for e, u in u_e.items() if not np.isnan(u) and u <= ABSTAIN_THRESH]
            best = min(surv, key=lambda e: u_e[e]) if surv else None
            out['aichi_scenes'].append({
                'map': tag, 'ego_idx': ego_idx, 'features': feats.tolist(),
                'per_expert_rd': per_expert_rd, 'u_e': u_e,
                'survivors': surv, 'abstain': not surv, 'idea5_pick': best,
                'realized_max_lat_per_expert': {
                    e: s1_by_scene.get((tag, ego_idx), {}).get(e, {}).get('max_lat')
                    for e in EXPERTS},
            })

    print("\n=== Idea 5 Triage Stats (threshold = 15m route_dev) ===")
    for tag in [m for m, _ in UNSEEN]:
        scenes = [s for s in out['aichi_scenes'] if s['map'] == tag]
        n = len(scenes); n_abst = sum(1 for s in scenes if s['abstain'])
        dep = [s['realized_max_lat_per_expert'].get(s['idea5_pick'])
               for s in scenes if not s['abstain'] and s.get('idea5_pick')]
        dep = [x for x in dep if x is not None]
        un = [s['realized_max_lat_per_expert']['unagoya'] for s in scenes
              if s['realized_max_lat_per_expert']['unagoya'] is not None]
        print(f"  {tag}: n={n}  abstain={n_abst} ({n_abst/max(n,1):.0%})")
        if dep:
            print(f"    deployed max_lat: {np.mean(dep):.2f} ± "
                  f"{np.std(dep, ddof=1) if len(dep)>1 else 0:.2f}")
        if un:
            print(f"    unagoya baseline: {np.mean(un):.2f} ± "
                  f"{np.std(un, ddof=1) if len(un)>1 else 0:.2f}")

    with open(f'{OUT_DIR}/stage11_conformal_v3.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[stage11_v3] saved {OUT_DIR}/stage11_conformal_v3.json")


if __name__ == '__main__':
    main()
