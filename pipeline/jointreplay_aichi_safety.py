"""Closed-loop safety metrics for joint-replay vs best-fixed on Aichi held-out.

Answers reviewer Q3: "Beyond max lateral offset, what are the closed-loop
safety/comfort/compliance outcomes of joint replay on held-out regions?"

For each (model, Aichi scene), compute trajectory-level metrics:
  - max_lat:        canonical lateral offset
  - jerk_p50, p95:  comfort
  - lat_acc_max:    comfort/safety
  - rss_corridor_exit:  RSS-style corridor exit binary
  - corridor_exit_m:    max distance outside route corridor
  - fwd_progress, mean_speed:  progress
  - finite (NaN-free):  validity

Models compared:
  - shared_only        (frozen MoLoRA, no per-expert)
  - best_fixed         (always-activate unagoya = ODD-D)
  - joint_replay_v2    (stage24_v2_joint_replay.pth, MoLoRA-format)
  - joint_replay_scr   (stage32_joint_replay_from_scratch_r4.pth)

Output: outputs/jointreplay_aichi_safety.json
"""
import os, sys, json, math, time
os.environ.setdefault('LD_LIBRARY_PATH', '/opt/autoware/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
sys.path.insert(0, '/opt/autoware/local/lib/python3.10/dist-packages')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner_ros')
sys.path.insert(0, 'third_party/Diffusion-Planner/preference_optimization')
sys.path.insert(0, '.')
sys.path.insert(0, '.')

import numpy as np
import torch
from collections import defaultdict

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from diffusion_planner.model.diffusion_utils.dpm_solver_pytorch import NoiseScheduleVP
from diffusion_planner_ros.lanelet2_utils.lanelet_converter import convert_lanelet
from train_molora import apply_molora, set_active_expert, N_EXPERTS
from train_reward_backprop import differentiable_dpm_solver_sample
from overnight_stage1_multiego import (build_inputs, pick_n_egos, UNSEEN, N_EGO, SEED,
                                         ARGS, BASE_PTH, metrics_for_traj)

DEVICE = torch.device('cuda')
OUT = 'outputs/jointreplay_aichi_safety.json'


def load_molora_model(cfg, lora_path=None):
    model = Diffusion_Planner(cfg).to(DEVICE).eval()
    ckpt = torch.load(BASE_PTH, map_location=DEVICE, weights_only=False)
    sd = ckpt.get('ema_state_dict') or ckpt['model']
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    apply_molora(model.decoder.dit, n_experts=N_EXPERTS, shared_rank=4, expert_rank=8)
    if lora_path:
        ckpt = torch.load(lora_path, map_location=DEVICE, weights_only=False)
        msd = ckpt.get('model', ckpt)
        msd = {k.replace('module.', ''): v for k, v in msd.items()}
        model.load_state_dict(msd, strict=False)
    return model


def trajectory_metrics(xy, dt=0.1, corridor_half_width=3.0):
    """Extended metrics beyond max_lat: jerk, lat_acc, RSS corridor exit."""
    out = {'finite': bool(np.all(np.isfinite(xy)))}
    if not out['finite']:
        return out

    out['path_len'] = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    out['fwd_progress'] = float(xy[-1, 0] - xy[0, 0])
    out['max_lat'] = float(np.max(np.abs(xy[:, 1])))

    # mean speed
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    out['mean_speed'] = float(step.mean() / dt)

    # acceleration: 2nd difference of xy
    accel = (xy[2:] - 2 * xy[1:-1] + xy[:-2]) / (dt * dt)
    lat_acc = np.abs(accel[:, 1])
    out['lat_acc_max'] = float(lat_acc.max())
    out['lat_acc_p95'] = float(np.percentile(lat_acc, 95))

    # jerk: 3rd difference
    jerk = (accel[1:] - accel[:-1]) / dt
    jerk_mag = np.linalg.norm(jerk, axis=1)
    out['jerk_p50'] = float(np.median(jerk_mag))
    out['jerk_p95'] = float(np.percentile(jerk_mag, 95))
    out['jerk_max'] = float(jerk_mag.max())

    # RSS-style corridor exit: any point with |y| > corridor_half_width
    lat_excess = np.maximum(0, np.abs(xy[:, 1]) - corridor_half_width)
    out['corridor_exit_m'] = float(lat_excess.max())  # max excess beyond corridor
    out['rss_corridor_exit'] = int(out['corridor_exit_m'] > 0.0)
    return out


def rollout_metrics(model, cfg, inputs, ns):
    """Run inference, return extended trajectory metrics."""
    try:
        inputs_norm = cfg.observation_normalizer({k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()})
        with torch.no_grad():
            tj = differentiable_dpm_solver_sample(model, cfg, inputs_norm, ns, device=DEVICE)
        xy = tj[:, :2].cpu().numpy()
        return trajectory_metrics(xy)
    except Exception as e:
        return {'finite': False, 'error': str(e)}


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
            except Exception:
                pass
    return scenes


def main():
    cfg = Config(ARGS)
    ns = NoiseScheduleVP()
    aichi_scenes = build_aichi_scenes(cfg)
    print(f"[Q3] {len(aichi_scenes)} Aichi scenes")

    # 'merged' configs load flat state-dicts (no apply_molora needed); 'molora' configs
    # apply_molora + load MoLoRA-format checkpoint + activate specific expert.
    SELORA_DIR = 'checkpoints/molora_v2_output'
    configs = {
        'shared_only':         dict(type='flat',   path=f'{SELORA_DIR}/merged_shared_only.pth',                                   expert=None),
        'best_fixed_unagoya':  dict(type='flat',   path=f'{SELORA_DIR}/merged_unagoya.pth',                                       expert=None),
        'joint_replay_v2':     dict(type='molora', path='checkpoints/overnight_20260513/stage24_v2_joint_replay.pth',         expert=3),
        'joint_replay_scr':    dict(type='molora', path='checkpoints/overnight_20260513/stage32_joint_replay_from_scratch_r4.pth', expert=3),
    }

    def make_model(c):
        m = Diffusion_Planner(cfg).to(DEVICE).eval()
        ckpt = torch.load(BASE_PTH, map_location=DEVICE, weights_only=False)
        sd = ckpt.get('ema_state_dict') or ckpt['model']
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        m.load_state_dict(sd, strict=False)
        if c['type'] == 'flat':
            # Load merged flat state-dict directly (no MoLoRA layer wrapping)
            ck = torch.load(c['path'], map_location=DEVICE, weights_only=False)
            msd = ck.get('model', ck)
            msd = {k.replace('module.', ''): v for k, v in msd.items()}
            m.load_state_dict(msd, strict=False)
        elif c['type'] == 'molora':
            apply_molora(m.decoder.dit, n_experts=N_EXPERTS, shared_rank=4, expert_rank=8)
            ck = torch.load(c['path'], map_location=DEVICE, weights_only=False)
            msd = ck.get('model', ck)
            msd = {k.replace('module.', ''): v for k, v in msd.items()}
            m.load_state_dict(msd, strict=False)
            set_active_expert(m, c['expert'])
        return m

    results = {}
    for cfg_name, c in configs.items():
        print(f"\n=== {cfg_name} (type={c['type']}, path={os.path.basename(c['path'])}, expert={c['expert']}) ===")
        base_model = make_model(c)
        base_model.eval()

        per_scene = []
        for i, s in enumerate(aichi_scenes):
            m = rollout_metrics(base_model, cfg, s['inputs'], ns)
            m['map'] = s['map']; m['ego_idx'] = s['ego_idx']
            per_scene.append(m)
            if (i+1) % 6 == 0:
                print(f"  scene {i+1}/{len(aichi_scenes)} done")

        # aggregate by map + fleet
        by_map = defaultdict(list)
        for r in per_scene:
            if r.get('finite'):
                by_map[r['map']].append(r)
        agg_map = {}
        for m, rs in by_map.items():
            agg_map[m] = {
                'n': len(rs),
                'max_lat_mean':       float(np.mean([r['max_lat'] for r in rs])),
                'jerk_p50_mean':      float(np.mean([r['jerk_p50'] for r in rs])),
                'jerk_p95_mean':      float(np.mean([r['jerk_p95'] for r in rs])),
                'lat_acc_max_mean':   float(np.mean([r['lat_acc_max'] for r in rs])),
                'lat_acc_p95_mean':   float(np.mean([r['lat_acc_p95'] for r in rs])),
                'corridor_exit_pct':  100.0 * float(np.mean([r['rss_corridor_exit'] for r in rs])),
                'corridor_excess_m':  float(np.mean([r['corridor_exit_m'] for r in rs])),
                'fwd_progress_mean':  float(np.mean([r['fwd_progress'] for r in rs])),
                'mean_speed_mean':    float(np.mean([r['mean_speed'] for r in rs])),
            }
        finite_all = [r for r in per_scene if r.get('finite')]
        fleet = {}
        for k in ['max_lat', 'jerk_p50', 'jerk_p95', 'lat_acc_max', 'lat_acc_p95',
                  'rss_corridor_exit', 'corridor_exit_m', 'fwd_progress', 'mean_speed']:
            fleet[f'{k}_mean'] = float(np.mean([r[k] for r in finite_all])) if finite_all else None
        fleet['n_finite'] = len(finite_all)
        results[cfg_name] = dict(per_map=agg_map, fleet=fleet, per_scene=per_scene)
        print(f"  fleet: max_lat={fleet['max_lat_mean']:.2f}  jerk_p50={fleet['jerk_p50_mean']:.2f}  "
              f"lat_acc_p95={fleet['lat_acc_p95_mean']:.2f}  corridor_exit={fleet['rss_corridor_exit_mean']*100:.1f}%  "
              f"progress={fleet['fwd_progress_mean']:.1f}m  speed={fleet['mean_speed_mean']:.2f}m/s")

    # save
    out = dict(description='Closed-loop safety/comfort metrics on Aichi held-out (reviewer Q3)',
               n_scenes=len(aichi_scenes), configs=list(configs.keys()), results=results)
    json.dump(out, open(OUT, 'w'), indent=2, default=float)
    print(f"\nSaved {OUT}")

    # print final compact comparison
    print("\n========================================================================")
    print(f"{'config':<22s} {'max_lat':>9s} {'jerk_p95':>9s} {'lat_ac_p95':>11s} {'corr_exit%':>11s} {'fwd_prog':>9s} {'speed':>7s}")
    print('-' * 92)
    for cfg_name in configs:
        f = results[cfg_name]['fleet']
        print(f"{cfg_name:<22s} {f['max_lat_mean']:>9.2f} {f['jerk_p95_mean']:>9.2f} "
              f"{f['lat_acc_p95_mean']:>11.2f} {f['rss_corridor_exit_mean']*100:>10.1f}% "
              f"{f['fwd_progress_mean']:>9.1f} {f['mean_speed_mean']:>7.2f}")


if __name__ == '__main__':
    main()
