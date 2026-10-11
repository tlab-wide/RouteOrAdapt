"""Recompute the numbers reported in the paper from the released per-scene results.

    python analysis/reproduce_paper_numbers.py

Every line prints the value quoted in the paper next to the value recomputed here.
Map keys in the result files are the internal names:
    held-out maps   aichi_hub_v2 = Knowledge Hub, ishio_dai = Ishiodai, morikoro = Morikoro
    training ODDs   hongo = ODD-Hongo, kashiwa = ODD-Kashiwa, shalun = ODD-Shalun, unagoya = ODD-Nagoya
The static single-expert default is the unagoya (ODD-Nagoya) expert.
"""
import csv
import glob
import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
R = os.path.join(ROOT, 'results')
MAPS = ['aichi_hub_v2', 'ishio_dai', 'morikoro']


def load(path):
    with open(os.path.join(R, path)) as f:
        return json.load(f)


def show(label, paper, value):
    print(f'  {label:58s} paper: {paper:>18s}   recomputed: {value}')


def key(r):
    return (r['map'], r['ego_idx'])


def per_map(d):
    """mean +- sample std per held-out map, and the mean over the three maps."""
    out = []
    for m in MAPS:
        v = np.array([x for k, x in d.items() if k[0] == m])
        out.append((v.mean(), v.std(ddof=1)))
    return out, np.mean([o[0] for o in out])


# ---------------------------------------------------------------- routing pool (Sec. 4, Table 2)
s1 = load('routing_pool/per_expert_rollouts.json')['rows']
expert = {}
for r in s1:
    expert.setdefault(key(r), {})[r['expert']] = r['max_lat']
scenes = sorted(expert)
static = {k: expert[k]['unagoya'] for k in scenes}
oracle = {k: min(expert[k][e] for e in ('hongo', 'kashiwa', 'shalun', 'unagoya')) for k in scenes}
s5 = load('routing_pool/scene_mlp_and_self_consistency.json')['rows']
s8 = load('routing_pool/lane_following.json')['rows']
s9 = load('routing_pool/denoising_loss.json')['rows']
s10 = load('routing_pool/vmf_mixture.json')['rows']
rb = load('routing_pool/reward_as_scorer.json')['rows']
schemes = {
    'Scene-conditioned MLP': {key(r): r['scr_pick_metrics']['max_lat'] for r in s5},
    'Self-consistency': {key(r): r['sces_pick_metrics']['max_lat'] for r in s5},
    'Lane-following': {key(r): r['lfs_max_lat'][min(r['lfs_score'], key=r['lfs_score'].get)] for r in s8},
    'Denoising-loss': {key(r): r['max_lat'][r['pick']] for r in s9},
    'vMF mixture, tau = 200 km': {key(r): r['max_lat'] for r in s10 if r['mode'] == 'soft' and int(r['tau']) == 200},
    'Reward-as-scorer': {key(r): r['max_lat_under_picked'] for r in rb},
    'Static single-expert default': static,
}
PAPER_T2 = {'Scene-conditioned MLP': '25.61', 'Self-consistency': '22.76', 'Lane-following': '18.29',
            'Denoising-loss': '21.67', 'vMF mixture, tau = 200 km': '15.86', 'Reward-as-scorer': '26.57',
            'Static single-expert default': '13.74'}
print('Table 2: per-map mean +- std over scenes, and the mean over the three maps (m)')
for name, d in schemes.items():
    pm, fleet = per_map(d)
    cells = '  '.join(f'{m:5.2f}+-{s:5.2f}' for m, s in pm)
    show(name, PAPER_T2[name], f'{cells}   fleet {fleet:.2f}')

print('\nSec. 4: paired Wilcoxon tests against the static default (24 scenes)')
sv = np.array([static[k] for k in scenes])
for name, d in schemes.items():
    if name.startswith('Static'):
        continue
    v = np.array([d[k] for k in scenes])
    p = stats.wilcoxon(v, sv, zero_method='wilcox').pvalue if np.any(v != sv) else 1.0
    show(f'{name}: mean diff vs default', 'six worse, p<=0.012', f'{(v - sv).mean():+6.2f} m, p = {p:.4f}')
# Holm correction over the eight schemes of Table 2. The reward-supervised gate has no
# 24-scene vector and enters as p = 1; the normalised self-consistency variant equals the
# vanilla one and enters with the same p. Both choices make the correction stricter.
pt = {r['method']: r for r in load('routing_pool/paired_tests.json')['results']}
family = {'Scene-conditioned MLP': 'scr_router', 'Self-consistency': 'sces_self_consistency',
          'Self-consistency (normalised)': 'sces_self_consistency', 'Lane-following': 'lfs_router',
          'Denoising-loss': 'dlr_router', 'Conformal corridor': 'conformal_triage',
          'Reward-as-scorer': 'reward_selector'}
fam = {k: (pt[v]['wilcoxon_p'], pt[v]['mean_diff_vs_static']) for k, v in family.items()}
fam['Reward-supervised gate'] = (1.0, 0.0)
order = sorted(fam, key=lambda k: fam[k][0])
holm, running = {}, 0.0
for i, k in enumerate(order):
    running = max(running, min(1.0, (len(order) - i) * fam[k][0]))
    holm[k] = running
worse = [k for k in order if fam[k][1] > 0 and holm[k] < 0.05]
show('worse after Holm correction over 8 schemes', 'six, also after Holm', f'{len(worse)}: ' + ', '.join(worse))

ov = np.array([oracle[k] for k in scenes])
print('\nSec. 4: oracle headroom')
show('scene mean, static default -> oracle (24 scenes)', '16.04 -> 15.15', f'{sv.mean():.2f} -> {ov.mean():.2f}')
show('Wilcoxon p', '0.012', f'{stats.wilcoxon(ov, sv).pvalue:.4f}')
e18 = load('expanded_pool_104/expanded_pool.json')
show('104-scene headroom (bootstrap CI)', '1.34 [0.7, 2.2]',
     f"{e18['headroom']:.2f} [{e18['headroom_ci'][0]:.2f}, {e18['headroom_ci'][1]:.2f}]")
e6 = load('matched_baselines/oracle_per_seed_slice.json')['summary']['headroom']
show('per-seed slices of Sec. 5.3', '1.1 +- 0.5', f"{e6['mean']:.2f} +- {e6['std']:.2f}")

jr_rows = load('routing_pool/joint_replay_and_safety_proxies.json')['results']['joint_replay_v2']['per_scene']
jr = {key(r): r['max_lat'] for r in jr_rows}
jv = np.array([jr[k] for k in scenes])
show('joint replay on the 24 scenes (scene mean)', '16.04 -> 6.95', f'{sv.mean():.2f} -> {jv.mean():.2f}')
show('Wilcoxon p, scenes improved', '0.0005, 19/24', f'{stats.wilcoxon(jv, sv).pvalue:.4f}, {(jv < sv).sum()}/24')
pool = list(csv.DictReader(open(os.path.join(ROOT, 'splits/routing_pool_24.csv'))))
train = {(r['map_key'], r['lanelet_index'])
         for r in csv.DictReader(open(os.path.join(ROOT, 'splits/joint_replay_pool_a_checkpoints_train20.csv')))}
clean = [(r['map_key'], int(r['ego_idx'])) for r in pool if (r['map_key'], r['lanelet_index']) not in train]
cs, cj = np.array([static[k] for k in clean]), np.array([jr[k] for k in clean])
show('excluding the 7 training scenes: gain, p, improved', '12.7, 0.0008, 14/17',
     f'{cs.mean() - cj.mean():.2f}, {stats.wilcoxon(cj, cs).pvalue:.4f}, {(cj < cs).sum()}/{len(clean)}')

print('\nSec. 4: can scene features predict the oracle expert? (leave-one-out accuracy, permutation p)')
for path, lab in [('routing_pool/loocv_handcrafted_features.json', '8 hand-crafted features, 24 scenes'),
                  ('routing_pool/loocv_encoder_embedding.json', '256-d encoder embedding, 24 scenes')]:
    d = load(path)
    accs = {k: v['loocv_acc'] for k, v in d['results'].items()}
    ps = [v['perm_p'] for v in d['results'].values()]
    show(lab, 'max acc <= 0.667', f"majority {d['majority_baseline']:.3f}, best {max(accs.values()):.3f}, min perm p {min(ps):.3f}")
show('104 scenes', 'max acc <= 0.625', f"majority {e18['majority']:.3f}, best {max(v['acc'] for v in e18['loocv'].values()):.3f}, "
     f"min perm p {min(v['perm_p'] for v in e18['loocv'].values()):.3f}")

# ---------------------------------------------------------------- shared-adapter repair (Sec. 5.2, appendix table)
print('\nSec. 5.2: shared-adapter repair, per-map means (m)')
rows = load('shared_adapter_repair/per_expert_compositions.json')['rows']
PAPER_REP = {'shared_only_park': '6.0 / 6.1 / 21.4', 'hongo_park': '24.5 / 17.6 / 50.3', 'kashiwa_park': '7.9 / 7.8 / 11.1',
             'shalun_park': '6.0 / 5.8 / 18.3', 'unagoya_park': '11.0 / 9.5 / 12.2'}
for cfg, paper in PAPER_REP.items():
    pm = [np.mean([r['max_lat'] for r in rows if r['config'] == cfg and r['map'] == m and r['finite']]) for m in MAPS]
    show(f'{cfg} + repaired shared module', paper, ' / '.join(f'{v:.1f}' for v in pm) + f'   fleet {np.mean(pm):.2f}')

# ---------------------------------------------------------------- Table 3 (80/10 split, eight seeds)
print('\nTable 3: target fleet max-lat (mean over the three maps) and summed source forgetting (m)')
def runs(mode):
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(R, f'matched_baselines/runs/{mode}_s*.json')))]
def mapmean(d):
    return np.mean([d[m]['mean'] for m in MAPS])
def rd(prod):
    return sum(v['route_dev_mean'] for v in prod.values())
prod_pre = {r['seed']: rd(r['pre_prod']) for r in runs('target_only_sp80')}
TABLE3 = [('Fine-tune the best fixed expert', 'ft_best_expert_sp80', '4.69+-1.90 / +22.6+-2.9'),
          ('Target-only LoRA', 'target_only_sp80', '5.34+-4.46 / +37.4+-11.9'),
          ('Target-only + L2-SP', 'target_only_l2sp0.001_sp80', '4.04+-0.77 / +37.8+-13.2'),
          ('Adapter trained from scratch', 'target_only_scratch_sp80', '6.03+-2.64 / +17.3+-23.4'),
          ('EWC, lambda = 1e2 (3 seeds)', 'target_only_ewc100_sp80', '8.20+-0.83 / +4.7+-1.3'),
          ('Prioritised replay', 'joint_replay_per0.6_sp80_frng', '6.78+-1.33 / +1.7+-0.8'),
          ('Joint replay', 'joint_replay_sp80_frng', '6.91+-2.16 / +0.9+-1.4')]
base = runs('target_only_sp80')
pre = np.array([mapmean(r['pre_aichi']) for r in base])
show('No adaptation (frozen substrate)', '20.0+-2.4', f'{pre.mean():.2f}+-{pre.std(ddof=1):.2f}')
stat = np.array([mapmean(r['pre_aichi']) for r in runs('ft_best_expert_sp80')])
show('Static single-expert default', '12.4+-3.0', f'{stat.mean():.2f}+-{stat.std(ddof=1):.2f}')
for name, mode, paper in TABLE3:
    rs = runs(mode)
    t = np.array([mapmean(r['post_aichi']) for r in rs])
    if 'scratch' in mode:
        # The from-scratch runs store forgetting relative to their own zero-initialised shared module;
        # the paper measures every method against the production adapter stack instead.
        f = np.array([rd(r['post_prod']) - prod_pre[r['seed']] for r in rs])
    else:
        f = np.array([r['forgetting_sum_route_dev_m'] for r in rs])
    show(name, paper, f'{t.mean():.2f}+-{t.std(ddof=1):.2f} / {f.mean():+.1f}+-{f.std(ddof=1):.1f}  (n={len(rs)})')
print('  EWC over lambda (3 seeds each):')
for lam in (100, 300, 1000, 3000, 10000):
    rs = runs(f'target_only_ewc{lam}_sp80')
    t = np.array([mapmean(r['post_aichi']) for r in rs]); f = np.array([r['forgetting_sum_route_dev_m'] for r in rs])
    print(f'    lambda = {lam:>5d}: target {t.mean():5.2f}+-{t.std(ddof=1):.2f}, forgetting {f.mean():+5.1f}+-{f.std(ddof=1):.1f}')
e10 = load('matched_baselines/source_forgetting_per_scene.json')
seed_means = np.array([v['mean'] for v in e10['per_seed'].values()])
show('joint replay source regression per scene, tested across seeds', '+0.22, 5/8 seeds positive, p=0.15',
     f"{seed_means.mean():+.2f}, {(seed_means > 0).sum()}/{len(seed_means)} seeds positive, "
     f"p = {stats.wilcoxon(seed_means).pvalue:.2f}")
e13 = load('matched_baselines/safety_proxies.json')['results']
show('jerk p95, L2-SP relative to joint replay', '+27%',
     f"{(e13['l2sp']['jerk_p95']['mean'] / e13['joint_replay']['jerk_p95']['mean'] - 1) * 100:+.0f}%")

# ---------------------------------------------------------------- safety proxies on the routing pool
print('\nSec. 5.3: trajectory-level proxies on the 24 scenes (static default -> joint replay)')
sp = load('routing_pool/joint_replay_and_safety_proxies.json')['results']
def mean(c, k):
    return np.mean([r[k] for r in sp[c]['per_scene']])
# All four are computed on the raw 10 Hz plan in the ego frame; jerk and lateral acceleration are
# unsmoothed finite differences, so only the relative change between models is meaningful.
show('lateral offset in excess of the straight 3 m band (m)', '13.0 -> 3.9',
     f"{mean('best_fixed_unagoya', 'corridor_exit_m'):.1f} -> {mean('joint_replay_v2', 'corridor_exit_m'):.1f}")
for k, lab, paper in [('jerk_p95', 'jerk p95, relative change', '-26%'),
                      ('lat_acc_p95', 'lateral acceleration p95, relative change', '-4%')]:
    a0, a1 = mean('best_fixed_unagoya', k), mean('joint_replay_v2', k)
    show(lab, paper, f'{(a1 / a0 - 1) * 100:+.0f}%  ({a0:.1f} -> {a1:.1f})')
show('forward progress (m)', '164 -> 187',
     f"{mean('best_fixed_unagoya', 'fwd_progress'):.1f} -> {mean('joint_replay_v2', 'fwd_progress'):.1f}")
show('rollouts exceeding the band, from-scratch variant', '100% -> 50%',
     f"{mean('best_fixed_unagoya', 'rss_corridor_exit') * 100:.0f}% -> {mean('joint_replay_scr', 'rss_corridor_exit') * 100:.0f}%")
a0, a1 = mean('best_fixed_unagoya', 'lat_acc_p95'), mean('joint_replay_scr', 'lat_acc_p95')
show('lateral acceleration p95, from-scratch variant', '-57%', f'{(a1 / a0 - 1) * 100:+.0f}%')

# ---------------------------------------------------------------- switching regions (Sec. 7, appendix)
print('\nSec. 7: regions where the nearest expert is not ODD-Nagoya')
for path, near, best, paper in [('switching_regions/suburb_adjacent_to_kashiwa.json', 'kashiwa', 'unagoya', '+1.37, p=0.001, n=30'),
                                ('switching_regions/dense_urban_nearest_hongo.json', 'hongo', 'kashiwa', '+30.0, p=0.084, n=10')]:
    rows = load(path)['rows']
    a, b = np.array([r[near] for r in rows]), np.array([r[best] for r in rows])
    show(f'{near} (nearest) minus {best} (best fixed)', paper,
         f'{a.mean() - b.mean():+.2f}, p = {stats.wilcoxon(a, b).pvalue:.3f}, n = {len(rows)}')

# ---------------------------------------------------------------- sample complexity (Fig. 5, appendix)
print('\nFig. 5: joint-replay sample complexity, mean over training seeds 42-44')
vals = {}
for entry in load('sample_complexity/three_seed.json')['seeds']:
    for r in entry['rows']:
        vals.setdefault(r['n_aichi'], []).append(r['fleet_mean'])
for n, paper in zip(sorted(vals), ['9.79+-5.14', '10.14+-6.62', '6.17+-1.23', '5.67+-1.06']):
    v = np.array(vals[n]); show(f'n = {n}', paper, f'{v.mean():.2f}+-{v.std(ddof=1):.2f}')

# ---------------------------------------------------------------- nuPlan (Sec. 6, Limitations)
print('\nnuPlan')
cl = load('nuplan/closed_loop_reactive_cls_r.json')['north_s42']
show('closed loop, Singapore held out: base -> adapted', '0.858 -> 0.598', f"{cl['base']:.3f} -> {cl['adapted']:.3f}")
cld = load('nuplan/closed_loop_reactive_cls_r.json')
cities = [v for v in cld.values() if isinstance(v, dict) and 'adapted' in v]
show('closed loop: cities where the adapted model scores lower (significant, p < 0.05)', 'all four (three)',
     f"{sum(c['adapted'] < c['base'] for c in cities)} of {len(cities)} "
     f"({sum(c['adapted'] < c['base'] and c['p'] < 0.05 for c in cities)})")
show('closed loop, Singapore drivable-area compliance', '0.97 -> 0.76',
     f"{cld['north_s42']['base_drivable']:.2f} -> {cld['north_s42']['adapted_drivable']:.2f}")
ol = load('nuplan/open_loop_leave_one_city_out_3seed.json')
print('  open loop: training cities of each held-out run (no held-out-city data):')
for city, runs_ in ol['per_run'].items():
    print(f"    {city:28s} held out -> trained on {sorted(set(c for r in runs_ for c in r['train_cities']))}")
