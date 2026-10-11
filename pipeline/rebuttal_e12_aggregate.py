#!/usr/bin/env python3
"""Aggregate E1/E2/JR matched-baseline runs (3 seeds each) into the rebuttal table.
Also runs Pool-B paired Wilcoxon: each method's post per-scene vs its PRE (static/shared)
and cross-method where per-scene rows align on (map, lanelet_idx)."""
import json, glob
import numpy as np
from scipy import stats

OUT = 'outputs/rebuttal'
MODES = ['target_only', 'target_only_scratch', 'ft_best_expert',
         'joint_replay', 'joint_replay_scratch',
         'target_only_l2sp0.001', 'target_only_l2sp0.1']

runs = {}
for p in sorted(glob.glob(f'{OUT}/e12_*.json')):
    if p.endswith('_lora.json'):
        continue
    d = json.load(open(p))
    if 'mode' in d:
        runs.setdefault(d['mode'], []).append(d)

print(f"{'mode':16s} {'seeds':>5s} {'target fleet (m)':>18s} {'forget Δroute_dev (m)':>22s} {'forget Δprogress (m)':>21s}")
summary = {}
for mode in MODES:
    rs = runs.get(mode, [])
    if not rs:
        print(f'{mode:16s}  MISSING'); continue
    fleet = np.array([r['post_fleet'] for r in rs])
    forg = np.array([r['forgetting_sum_route_dev_m'] for r in rs])
    fpr = np.array([r['forgetting_sum_progress_m'] for r in rs])
    summary[mode] = dict(n_seeds=len(rs),
                         fleet_mean=float(fleet.mean()), fleet_std=float(fleet.std(ddof=1)) if len(rs) > 1 else 0.0,
                         forget_rd_mean=float(forg.mean()), forget_rd_std=float(forg.std(ddof=1)) if len(rs) > 1 else 0.0,
                         forget_pr_mean=float(fpr.mean()), forget_pr_std=float(fpr.std(ddof=1)) if len(rs) > 1 else 0.0,
                         pre_fleet=rs[0]['pre_fleet'])
    s = summary[mode]
    print(f"{mode:16s} {len(rs):>5d} {s['fleet_mean']:>9.2f}±{s['fleet_std']:<7.2f} "
          f"{s['forget_rd_mean']:>13.2f}±{s['forget_rd_std']:<7.2f} {s['forget_pr_mean']:>13.1f}±{s['forget_pr_std']:<6.1f}")

# Pool-B paired test: joint_replay vs target_only per scene (seed-matched, then pooled)
def rows_of(d):
    return {(r['map'], r['lanelet_idx']): r['max_lat'] for r in d['post_rows'] if r.get('finite')}

pairs = []
for mode_a, mode_b in [('joint_replay', 'target_only'), ('joint_replay', 'ft_best_expert'),
                       ('joint_replay_scratch', 'target_only'),
                       ('joint_replay_scratch', 'target_only_scratch'),
                       ('joint_replay_scratch', 'ft_best_expert')]:
    da = {r['seed']: r for r in runs.get(mode_a, [])}
    db = {r['seed']: r for r in runs.get(mode_b, [])}
    diffs = []
    for seed in sorted(set(da) & set(db)):
        ra, rb = rows_of(da[seed]), rows_of(db[seed])
        ks = sorted(set(ra) & set(rb))
        diffs.extend(ra[k] - rb[k] for k in ks)
    if diffs:
        diffs = np.array(diffs)
        w = stats.wilcoxon(diffs)
        print(f'\n{mode_a} vs {mode_b}: n={len(diffs)} paired (scene x seed), '
              f'mean diff {diffs.mean():+.2f} m, Wilcoxon p={w.pvalue:.4f}')
        pairs.append(dict(a=mode_a, b=mode_b, n=len(diffs),
                          mean_diff=float(diffs.mean()), wilcoxon_p=float(w.pvalue)))

# PRE references
if 'target_only' in runs:
    print(f"\nPRE shared-only fleet: {runs['target_only'][0]['pre_fleet']:.2f} m")
if 'ft_best_expert' in runs:
    print(f"PRE static-rule (unagoya expert) fleet: {runs['ft_best_expert'][0]['pre_fleet']:.2f} m")

with open(f'{OUT}/e12_summary.json', 'w') as f:
    json.dump(dict(summary=summary, paired=pairs), f, indent=1)
print(f'\nsaved {OUT}/e12_summary.json')
