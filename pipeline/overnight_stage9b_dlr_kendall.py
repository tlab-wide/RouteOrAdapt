#!/usr/bin/env python3
"""
Stage 9b: post-hoc Kendall-tau analysis on the DLR (denoising-loss router) data.

The brainstorm v2 (Idea 1) specified Kendall-tau between the per-expert
denoising loss L_e and the realized lateral offset lat_e as the primary
falsifier (threshold 0.2). The Stage 9 implementation reported pick-distribution
and mean lat only. This script computes Kendall-tau on the saved stage9_dlr.json
and reports per-map / aggregate values.
"""
import os, json
import numpy as np
from itertools import combinations

S9 = 'checkpoints/overnight_20260511/stage9_dlr.json'
OUT = 'checkpoints/overnight_20260513/stage9b_kendall_tau.json'
MAPS = ['aichi_hub_v2', 'ishio_dai', 'morikoro']


def kendall_tau(x, y):
    """Concordant - discordant / (n choose 2). x, y are arrays of equal length."""
    n = len(x)
    if n < 2: return float('nan')
    c = d = 0
    for i, j in combinations(range(n), 2):
        dx, dy = x[i] - x[j], y[i] - y[j]
        s = dx * dy
        if s > 0: c += 1
        elif s < 0: d += 1
    total = n * (n - 1) // 2
    return (c - d) / total if total else float('nan')


def main():
    s9 = json.load(open(S9))
    per_map_taus = {}
    all_pairs = []
    for tag in MAPS:
        rows = [r for r in s9['rows'] if r['map'] == tag and r.get('losses') and r.get('max_lat')]
        scene_taus = []
        for r in rows:
            shared = set(r['losses']) & set(r['max_lat'])
            if len(shared) < 3: continue
            experts = sorted(shared)
            losses = [r['losses'][e] for e in experts]
            lats   = [r['max_lat'][e] for e in experts]
            tau = kendall_tau(losses, lats)
            if not np.isnan(tau):
                scene_taus.append(tau)
                # Also build cross-map pairs for aggregate
                for e in experts:
                    all_pairs.append((r['losses'][e], r['max_lat'][e]))
        per_map_taus[tag] = {
            'n_scenes': len(scene_taus),
            'mean_tau': float(np.mean(scene_taus)) if scene_taus else None,
            'std_tau': float(np.std(scene_taus, ddof=1)) if len(scene_taus) > 1 else 0.0,
        }
        print(f"{tag}: n_scenes={len(scene_taus)}  mean Kendall-tau={per_map_taus[tag]['mean_tau']:.3f}")

    # Aggregate Kendall-tau across all (loss, lat) pairs (pooled)
    if all_pairs:
        L = np.array([p[0] for p in all_pairs])
        Y = np.array([p[1] for p in all_pairs])
        pooled = kendall_tau(L, Y)
    else:
        pooled = float('nan')
    print(f"\nPooled Kendall-tau over all (loss, lat) pairs: {pooled:.3f}")
    print(f"Brainstorm falsification threshold: |tau| >= 0.2 to support DLR hypothesis")
    print(f"Verdict: DLR Kendall-tau is {'<' if abs(pooled) < 0.2 else '>='} 0.2 -> "
          f"{'DLR claim REFUTED' if abs(pooled) < 0.2 else 'DLR claim survives this falsifier'}")

    out = {'per_map_taus': per_map_taus, 'pooled_kendall_tau': pooled,
           'n_total_pairs': len(all_pairs),
           'falsification_threshold': 0.2,
           'verdict': 'refuted' if abs(pooled) < 0.2 else 'survives'}
    with open(OUT, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nsaved {OUT}")


if __name__ == '__main__':
    main()
