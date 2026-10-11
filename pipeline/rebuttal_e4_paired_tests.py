#!/usr/bin/env python3
"""E4 (rebuttal): paired per-scene Wilcoxon signed-rank tests, each routing/selection
method vs the static geographic rule (unagoya / ODD-D), on the Pool-A seed-42
24-scene held-out set. Answers Nkac Q2.

All per-scene vectors are keyed by (map, ego_idx) and drawn ONLY from Pool-A files
(stage1/5/8/9/10/11, forensics_full, reward_based_selector, jointreplay_aichi_safety)
so pairing is valid. Pool-B files (learned_coef etc.) are per-map-mean only and are
NOT mixed in here.
"""
import json, math
import numpy as np
from scipy import stats

GD = 'checkpoints'
CW = 'outputs'
OUT = f'{CW}/rebuttal/e4_paired_tests'

def load(p):
    with open(p) as f:
        return json.load(f)

def key(r):
    return (r['map'], r['ego_idx'])

# ---------- static rule + per-expert reference (stage1) ----------
s1 = load(f'{GD}/overnight_20260510/stage1_multiego.json')['rows']
per_expert = {}          # (map,ego) -> {expert: max_lat}
for r in s1:
    per_expert.setdefault(key(r), {})[r['expert']] = r['max_lat']
scenes = sorted(per_expert.keys())
assert len(scenes) == 24, len(scenes)

static = {k: per_expert[k]['unagoya'] for k in scenes}
oracle = {k: min(per_expert[k][e] for e in ('hongo', 'kashiwa', 'shalun', 'unagoya'))
          for k in scenes}

methods = {}             # name -> {(map,ego): max_lat}
methods['shared_only'] = {k: per_expert[k]['shared_only'] for k in scenes}
methods['oracle_per_scene'] = oracle

# ---------- SCR / SCES (stage5) ----------
s5 = load(f'{GD}/overnight_20260510/stage5_scr_eval.json')['rows']
methods['scr_router'] = {key(r): r['scr_pick_metrics']['max_lat'] for r in s5}
methods['sces_self_consistency'] = {key(r): r['sces_pick_metrics']['max_lat'] for r in s5}

# ---------- LFS router (stage8): pick expert with min lane-follow score ----------
s8 = load(f'{GD}/overnight_20260511/stage8_lfs.json')['rows']
for r in s8:
    pick = min(r['lfs_score'], key=r['lfs_score'].get)
    methods.setdefault('lfs_router', {})[key(r)] = r['lfs_max_lat'][pick]

# ---------- DLR router (stage9) ----------
s9 = load(f'{GD}/overnight_20260511/stage9_dlr.json')['rows']
methods['dlr_router'] = {key(r): r['max_lat'][r['pick']] for r in s9}

# ---------- vMF / GPS mixes (stage10): 4 taus x 2 modes ----------
s10 = load(f'{GD}/overnight_20260513/stage10_gps_mix.json')['rows']
for r in s10:
    name = f"vmf_{r['mode']}_tau{int(r['tau'])}km"
    methods.setdefault(name, {})[key(r)] = r['max_lat']

# ---------- conformal idea5 (stage11 v3) ----------
s11 = load(f'{GD}/overnight_20260513/stage11_conformal_v3.json')['aichi_scenes']
n_abstain = 0
for r in s11:
    pick = r.get('idea5_pick')
    if pick is None or pick not in r['realized_max_lat_per_expert']:
        # abstention -> falls back to static rule per paper protocol
        pick = 'unagoya'
        n_abstain += 1
    methods.setdefault('conformal_triage', {})[key(r)] = r['realized_max_lat_per_expert'][pick]

# ---------- reward-based selector ----------
rb = load(f'{CW}/reward_based_selector.json')['rows']
methods['reward_selector'] = {key(r): r['max_lat_under_picked'] for r in rb}

# ---------- forensics: hypernet / vmf softmix / tent ----------
fo = {e['label']: e for e in load(f'{CW}/forensics_full.json')['results']}
fmap = {'3_vmf_softmix_tau200': 'vmf_softmix_tau200_forensics',
        '4_hypernet_ln': 'hypernet_ln',
        '5_hypernet_plus_tent_polish': 'hypernet_plus_tent',
        '6_best_fixed_plus_tent_polish': 'static_plus_tent'}
for src, name in fmap.items():
    methods[name] = {(r['map'], r['ego_idx']): r['max_lat'] for r in fo[src]['rows']}

# ---------- joint replay (Pool-A safety eval) ----------
jr = load(f'{CW}/jointreplay_aichi_safety.json')['results']
jr_rows = jr['joint_replay_v2']['per_scene']
if 'map' in jr_rows[0]:
    methods['joint_replay_v2'] = {(r['map'], r['ego_idx']): r['max_lat'] for r in jr_rows}
    # sanity: its best_fixed_unagoya must match stage1 static
    chk = {(r['map'], r['ego_idx']): r['max_lat'] for r in jr['best_fixed_unagoya']['per_scene']}
    mism = max(abs(chk[k] - static[k]) for k in scenes)
    print(f'[sanity] jointreplay best_fixed vs stage1 static max |diff| = {mism:.4g} m')
else:
    raise RuntimeError('jointreplay per_scene rows lack map/ego_idx keys')

# ---------- paired tests ----------
res = []
static_v = np.array([static[k] for k in scenes])
for name, mv in sorted(methods.items()):
    missing = [k for k in scenes if k not in mv]
    if missing:
        print(f'[warn] {name}: missing {len(missing)} scenes, skipped'); continue
    v = np.array([mv[k] for k in scenes])
    d = v - static_v
    nz = d[np.abs(d) > 1e-9]
    if len(nz) == 0:
        w_p = 1.0; w_stat = float('nan')
    else:
        w_stat, w_p = stats.wilcoxon(v, static_v, zero_method='wilcox', mode='exact'
                                     if len(nz) <= 25 else 'approx')
    # sign test on non-zero diffs
    n_worse = int((nz > 0).sum()); n_better = int((nz < 0).sum())
    sign_p = stats.binomtest(min(n_better, n_worse), n_better + n_worse).pvalue if len(nz) else 1.0
    res.append(dict(method=name, n=24, mean=float(v.mean()), std=float(v.std(ddof=1)),
                    mean_diff_vs_static=float(d.mean()), median_diff=float(np.median(d)),
                    n_better=n_better, n_worse=n_worse, n_tied=int(24 - len(nz)),
                    wilcoxon_p=float(w_p), sign_test_p=float(sign_p)))

res.sort(key=lambda r: r['mean'])
static_row = dict(method='static_rule_unagoya (reference)', n=24,
                  mean=float(static_v.mean()), std=float(static_v.std(ddof=1)))

with open(f'{OUT}.json', 'w') as f:
    json.dump(dict(description=__doc__, pool='Pool-A seed-42, 24 scenes',
                   static=static_row, conformal_abstain_fallbacks=n_abstain,
                   results=res), f, indent=1)

hdr = f"{'method':34s} {'mean':>7s} {'Δvs-static':>10s} {'med Δ':>7s} {'B/W/T':>8s} {'Wilcoxon p':>10s} {'sign p':>7s}"
lines = [hdr, '-' * len(hdr)]
lines.append(f"{static_row['method']:34s} {static_row['mean']:7.2f} {'—':>10s} {'—':>7s} {'—':>8s} {'—':>10s} {'—':>7s}")
for r in res:
    lines.append(f"{r['method']:34s} {r['mean']:7.2f} {r['mean_diff_vs_static']:+10.2f} "
                 f"{r['median_diff']:+7.2f} {r['n_better']}/{r['n_worse']}/{r['n_tied']:>2d} "
                 f"{r['wilcoxon_p']:10.4f} {r['sign_test_p']:7.4f}")
table = '\n'.join(lines)
with open(f'{OUT}.txt', 'w') as f:
    f.write(table + '\n')
print(table)
