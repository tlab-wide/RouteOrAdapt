#!/usr/bin/env python3
"""
Stage 5: Apply SCR to unseen-map scenes; build a comparison table:
  random expert | best-fixed | SCES | SCR-routed | shared_only | mean-of-experts

For each (map, ego) scene:
  1. Compute scene features (reuse extractor) from synthetic scene tensor
  2. SCR → predicted ODD label → use that expert's metrics from stage 1
  3. Compare to alternatives

Saves checkpoints/overnight_20260510/stage5_scr_eval.json
"""
import os, sys, json, random, time
os.environ.setdefault('LD_LIBRARY_PATH', '/opt/autoware/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
sys.path.insert(0, '/opt/autoware/local/lib/python3.10/dist-packages')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner_ros')
sys.path.insert(0, '.')

import numpy as np
import torch
import lanelet2
from autoware_lanelet2_extension_python.projection import MGRSProjector

from diffusion_planner.utils.config import Config
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.dimensions import (
    INPUT_T, MAX_NUM_AGENTS, MAX_NUM_NEIGHBORS, NUM_LINE_STRINGS,
    NUM_POLYGONS, NUM_SEGMENTS_IN_LANE, NUM_SEGMENTS_IN_ROUTE,
    POSE_DIM, POINTS_PER_LANELET, POINTS_PER_LINE_STRING, POINTS_PER_POLYGON,
)
from diffusion_planner_ros.lanelet2_utils.lanelet_converter import (
    convert_lanelet, create_lane_tensor)
from overnight_stage1_multiego import build_inputs, pick_n_egos, UNSEEN, N_EGO, SEED
from overnight_stage3_scr import SCR, ODD_LABELS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ARGS = 'third_party/Diffusion-Planner/args.json'
OUT_DIR = 'checkpoints/overnight_20260510'

# Load SCR
scr_pkg = torch.load(f'{OUT_DIR}/stage3_scr.pth', map_location='cpu', weights_only=False)
scr = SCR().to(DEVICE)
scr.load_state_dict(scr_pkg['model']); scr.eval()
mu, sd = scr_pkg['mu'].to(DEVICE), scr_pkg['sd'].to(DEVICE)


def features_from_lanes_tensor(lanes_t, route_t):
    """Compute the same 8-feature vector used to train SCR, from the lane tensors
    that were fed to the planner (no npz needed)."""
    lanes = lanes_t[0].cpu().numpy()  # (140, 20, 33)
    route = route_t[0].cpu().numpy() if route_t is not None else None
    lane_mask = (np.abs(lanes[:, :, :2]).sum(axis=(1,2)) > 0.1)
    n_lanes = float(lane_mask.sum())
    if n_lanes == 0:
        return np.zeros(8, dtype=np.float32)
    mids = lanes[lane_mask, :, :2].mean(axis=1)
    y_sp = float(mids[:, 1].std()) if mids.shape[0] > 1 else 0.0
    x_sp = float(mids[:, 0].std()) if mids.shape[0] > 1 else 0.0
    lengths = np.linalg.norm(lanes[lane_mask, -1, :2] - lanes[lane_mask, 0, :2], axis=1)
    mean_len = float(lengths.mean())
    curvs = []
    for i in range(lanes.shape[0]):
        if not lane_mask[i]: continue
        pts = lanes[i, :, :2]
        if pts.shape[0] < 3: continue
        diffs = np.diff(pts, axis=0)
        h = np.arctan2(diffs[:,1], diffs[:,0])
        curvs.append(float(np.mean(np.abs(np.diff(h)))))
    mean_curv = float(np.mean(curvs)) if curvs else 0.0
    n_route = float((np.abs(route[:, :, :2]).sum(axis=(1,2)) > 0.1).sum()) if route is not None else 0.0
    return np.array([n_lanes, mean_curv, mean_len, y_sp, x_sp, 0.0, n_route, 0.0],
                    dtype=np.float32)


def main():
    cfg = Config(ARGS)
    s1 = json.load(open(f'{OUT_DIR}/stage1_multiego.json'))
    s2 = json.load(open(f'{OUT_DIR}/stage2_sces.json'))

    # Index stage1 metrics by (map, ego_idx, expert)
    s1_idx = {}
    for r in s1['rows']:
        s1_idx[(r['map'], r['ego_idx'], r['expert'])] = r
    sces_idx = {(r['map'], r['ego_idx']): r.get('sces_pick') for r in s2['rows']}

    # Compute SCR prediction per scene
    print("[stage5] computing SCR predictions on unseen-map scenes...")
    scr_predictions = {}  # (map, ego_idx) -> {odd: prob_dist, pick: argmax_label}
    for tag, osm in UNSEEN:
        try: llmap = convert_lanelet(osm)
        except Exception as e: print(f"  failed: {e}"); continue
        if not llmap.lanelets: continue
        egos = pick_n_egos(llmap, N_EGO, seed=SEED)
        for ego_idx, ego_ll in enumerate(egos):
            cl = ego_ll.centerline
            if cl.shape[0] < 2: continue
            cx, cy = float(cl[0,0]), float(cl[0,1])
            heading = float(np.arctan2(cl[1,1]-cl[0,1], cl[1,0]-cl[0,0]))
            inputs = build_inputs(llmap, (cx, cy), heading)
            feats = features_from_lanes_tensor(inputs['lanes'], inputs['route_lanes'])
            x = torch.from_numpy(feats).float().to(DEVICE).unsqueeze(0)
            x_n = (x - mu) / sd
            with torch.no_grad():
                logits = scr(x_n).squeeze(0)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                pred_idx = int(np.argmax(probs))
            scr_predictions[(tag, ego_idx)] = {
                'features': feats.tolist(),
                'probs': {ODD_LABELS[i]: float(probs[i]) for i in range(len(ODD_LABELS))},
                'pick': ODD_LABELS[pred_idx],
            }

    # Build comparison table per scene
    rows = []
    for (m, e_idx), pred in scr_predictions.items():
        scr_expert = pred['pick']
        sces_expert = sces_idx.get((m, e_idx))

        # Look up per-method metrics
        def get(expert):
            r = s1_idx.get((m, e_idx, expert))
            if r is None: return None
            return {'expert': expert,
                    'path_len': r.get('path_len'),
                    'fwd_progress': r.get('fwd_progress'),
                    'max_lat': r.get('max_lat')}

        # Best fixed = unagoya (universal winner per stage1 analysis)
        # mean-of-experts: average max_lat over the 4 SE-LoRA experts
        expert_metrics = {e: get(e) for e in ['hongo','kashiwa','shalun','unagoya','shared_only','baseline']}
        moe_lats = [m_['max_lat'] for m_ in expert_metrics.values() if m_ and m_['max_lat'] is not None and expert_metrics[m_['expert']]['expert'] != 'baseline' and m_['expert'] != 'shared_only']
        moe_fwds = [m_['fwd_progress'] for m_ in expert_metrics.values() if m_ and m_['fwd_progress'] is not None and m_['expert'] not in ('baseline','shared_only')]

        rows.append({
            'map': m, 'ego_idx': e_idx,
            'scr_pick': scr_expert,
            'scr_pick_metrics': get(scr_expert),
            'sces_pick': sces_expert,
            'sces_pick_metrics': get(sces_expert) if sces_expert else None,
            'best_fixed_unagoya': get('unagoya'),
            'shared_only': get('shared_only'),
            'baseline_BC': get('baseline'),
            'mean_of_experts_lat': float(np.mean(moe_lats)) if moe_lats else None,
            'mean_of_experts_fwd': float(np.mean(moe_fwds)) if moe_fwds else None,
            'all_experts': expert_metrics,
            'scr_probs': pred['probs'],
        })

    # Aggregate per map
    summary = {}
    for m in set(r['map'] for r in rows):
        scn = [r for r in rows if r['map'] == m]
        def avg(field, key='max_lat'):
            vals = [r[field][key] for r in scn if r.get(field) and r[field].get(key) is not None]
            return (float(np.mean(vals)), float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0, len(vals)) if vals else (None,None,0)
        summary[m] = {
            'scr_pick_lat':         avg('scr_pick_metrics'),
            'sces_pick_lat':        avg('sces_pick_metrics'),
            'best_fixed_unagoya':   avg('best_fixed_unagoya'),
            'shared_only':          avg('shared_only'),
            'baseline_BC':          avg('baseline_BC'),
            'mean_of_experts_lat':  (float(np.mean([r['mean_of_experts_lat'] for r in scn if r['mean_of_experts_lat'] is not None])),0.0,len(scn)),
            'scr_pick_distribution': {ODD_LABELS[i]: sum(1 for r in scn if r['scr_pick']==ODD_LABELS[i]) for i in range(len(ODD_LABELS))},
        }

    out = {'description': 'Stage 5: SCR applied to unseen-map scenes + comparison table',
           'rows': rows, 'summary': summary}
    json.dump(out, open(f'{OUT_DIR}/stage5_scr_eval.json', 'w'), indent=2)

    # Print readable summary
    print("\n" + "="*78)
    print("STAGE 5 — SCR ON UNSEEN MAPS")
    print("="*78)
    for m, s in summary.items():
        print(f"\n  {m}:")
        print(f"    SCR pick distribution: {s['scr_pick_distribution']}")
        print(f"    Method                              n   mean lat (m) ± std")
        for k in ['baseline_BC','shared_only','best_fixed_unagoya','mean_of_experts_lat','sces_pick_lat','scr_pick_lat']:
            v = s[k]
            if v[0] is None: continue
            print(f"    {k:<35s}   {v[2]:>2}    {v[0]:>6.1f} ± {v[1]:.1f}")

    print(f"\nSaved: {OUT_DIR}/stage5_scr_eval.json")


if __name__ == '__main__':
    main()
