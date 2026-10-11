"""Appendix Pareto figure: target-region fleet mean vs source-domain regression.

The data-gathering block is kept verbatim from the original renderer; only the drawing is new.
Encoding: methods that use no target data are gray (shape = method); joint replay is orange
as everywhere in the paper (filled = inherit-shared, hollow = sample-complexity sweep on its
own slice); from-scratch joint replay is aqua; the distilled adapter is violet.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')
OUT = os.path.join(HERE, 'out')
os.makedirs(OUT, exist_ok=True)
import paperstyle as ps

ps.setup()

# Data gathering (verbatim from the original renderer; colour names are only category tags here).
C_EXPERT = 'C_EXPERT'
C_SOUP_UNIFORM = 'C_SOUP_UNIFORM'
C_SOUP_GREEDY = 'C_SOUP_GREEDY'
C_TIES = 'C_TIES'
C_GPS = 'C_GPS'
C_DISTILL = 'C_DISTILL'
C_INHERIT = 'C_INHERIT'
C_FROMSCR = 'C_FROMSCR'
C_DEPLOY = 'C_DEPLOY'
C_SAMPLE = 'C_SAMPLE'
OD = os.path.join(RESULTS, 'weight_space')

# Reference values
PRE_FLEET   = 24.80  # frozen MoLoRA on canonical CRC32 pool
PRE_ROUTEDV = 40.62  # sum_route_dev across 4 prod ODDs

# ----- Anonymization: map data-source ODD codes to neutral labels -----
ODD_ANON = {
    'aichi_hub_v2': 'ODD-A',
    'aichi':        'ODD-A',
    'ishio_dai':    'ODD-B',
    'morikoro':     'ODD-C',
    'kashiwa':      'ODD-D',
}
def anon_odd(name):
    key = str(name).lower()
    for src, dst in ODD_ANON.items():
        if src in key:
            return dst
    return name

# ----- Gather points (label, target_fleet, prod_sum_route_dev_delta, marker, color, cat) -----
P = []

# Real measured prod-domain Δsum_route_dev for soup configs
d_soup = json.load(open(f'{OD}/soups_prod_eval.json'))
PRE_ROUTE = d_soup['pre']['sum_route_dev']
def soup_delta(key):
    return d_soup['soups'][key]['sum_route_dev'] - PRE_ROUTE

# Per-ODD experts
d30 = json.load(open(f'{OD}/stage30_greedy_soup.json'))
for odd, info in d30['per_expert_alone'].items():
    P.append((f'expert: {anon_odd(odd)}', info['fleet_mean'], 0.0, 'd', C_EXPERT, 'expert'))

# Uniform soup
P.append(('uniform soup', d30['uniform_soup_all4']['fleet_mean'],
          soup_delta('uniform_soup_all4'), 'o', C_SOUP_UNIFORM, 'soup'))
# Greedy soup
P.append(('greedy soup', d30['final_soup']['fleet_mean'],
          soup_delta('greedy_soup'), '*', C_SOUP_GREEDY, 'soup'))

# TIES merging
d31 = json.load(open(f'{OD}/stage31_ties_merging.json'))
for key, info in d31['results'].items():
    if not key.startswith('TIES_'): continue
    dens = info.get('keep_density', 'unk')
    P.append((f'TIES d={dens}', info['fleet_mean'], soup_delta(key), 'D', C_TIES, 'soup'))

# GPS soft mix (stage10)
try:
    d10 = json.load(open(f'{OD}/stage10_gps_mix.json'))
    for entry in d10.get('rows', d10.get('sweep', [])):
        tau = entry.get('tau_km', entry.get('tau'))
        if tau in (50, 100, 200, 500, 1000):
            fm = entry.get('fleet_mean') or entry.get('post_fleet')
            if fm is not None and tau == 200:
                P.append((f'GPS mix tau=200km', fm, 0.0, 's', C_GPS, 'gps'))
                break
except Exception as e:
    print(f'stage10 gps_mix: {e}')

# GeoFlex distillation (stage14_v2)
d14 = json.load(open(f'{OD}/stage14_v2_eval.json'))
ev = d14['eval']
hub = ev.get('aichi_hub_v2', {}).get('mean')
ish = ev.get('ishio_dai',    {}).get('mean')
mor = ev.get('morikoro',     {}).get('mean')
if hub and ish and mor:
    fm14 = (hub + ish + mor) / 3
    P.append(('GeoFlex distill', fm14, 0.0, 'P', C_DISTILL, 'distill'))

# Inherit-shared joint-replay (Stage 24 v2)
d24 = json.load(open(f'{OD}/stage24_v2_joint_replay.json'))
hub = d24['post_aichi'].get('aichi_hub_v2',{}).get('mean')
ish = d24['post_aichi'].get('ishio_dai',{}).get('mean')
mor = d24['post_aichi'].get('morikoro', {}).get('mean')
fm24 = (hub + ish + mor) / 3
delta24 = d24['forgetting_sum_route_dev_m']
P.append(('joint-replay inherit', fm24, delta24, '^', C_INHERIT, 'joint_inherit'))

# From-scratch joint-replay rank sweep (3 runs)
RUN_VALS = {
    'r=2':  [8.95, 6.06, 6.06],
    'r=4':  [4.92, 7.35, 9.91],
    'r=8':  [5.83, 7.49, 4.43],
    'r=16': [6.75, 5.21, 6.75],
}
d32 = json.load(open(f'{OD}/stage32_rank_sweep_run3.json'))
prod_delta_run3 = {f"r={r['rank']}": r['delta_sum_route_dev'] for r in d32['rows']}
prod_delta_log = {
    'r=2':  [11.26, 14.11, 14.52],
    'r=4':  [13.87, 13.30, 13.64],
    'r=8':  [14.82, 15.64, 16.18],
    'r=16': [16.09, 15.95, 15.56],
}
for rank in ['r=2','r=4','r=8','r=16']:
    for i, (fm, pd) in enumerate(zip(RUN_VALS[rank], prod_delta_log[rank])):
        is_deploy = (rank == 'r=4' and i == 1)
        marker = '*' if is_deploy else 'v'
        size   = 360 if is_deploy else 70
        color  = C_DEPLOY if is_deploy else C_FROMSCR
        label  = f'joint-replay from-scratch ({rank})' if (rank == 'r=8' and i == 0) else None
        P.append((label or '', fm, pd, marker, color, 'joint_fs' if not is_deploy else 'deploy'))

# Sample-complexity sweep (Stage 26)
d26 = json.load(open(f'{OD}/stage26_sample_complexity.json'))
for s in d26['sweep']:
    fm = s['post_aichi'].get('aichi_hub_v2',{}).get('mean'), \
         s['post_aichi'].get('ishio_dai',   {}).get('mean'), \
         s['post_aichi'].get('morikoro',   {}).get('mean')
    fm = sum(fm)/len(fm)
    delta = s['delta_sum_route_dev']
    P.append((f"sample n={s['n_aichi']}", fm, delta, 'X', C_SAMPLE, 'samplecomp'))



WS = '#a3a199'   # weight-space methods (no target data)
STYLE = {        # category -> (marker, face, edge, size)
    'expert':        ('D', WS, 'white', 20),
    'gps':           ('^', WS, 'white', 26),
    'distill':       ('P', ps.VIOLET, 'white', 40),
    'joint_inherit': ('o', ps.ORANGE, 'white', 34),
    'samplecomp':    ('o', 'white', ps.ORANGE, 24),
    'joint_fs':      ('o', ps.AQUA, 'white', 22),
    'deploy':        ('*', ps.AQUA, 'white', 90),
}
SOUP = {'uniform soup': ('o', WS, 'white', 24), 'greedy soup': ('s', WS, 'white', 22)}

def style_of(label, cat):
    if cat == 'soup':
        return SOUP.get(label, ('X', WS, 'white', 26))   # TIES variants
    return STYLE[cat]

fig, ax = plt.subplots(figsize=(5.4, 2.75))
fig.subplots_adjust(left=0.1, right=0.66, bottom=0.16, top=0.97)

ax.axhline(PRE_FLEET, color=ps.MUTED, linewidth=0.7, zorder=1)
ax.text(16.9, PRE_FLEET + 0.5, 'frozen adapter stack', ha='right', va='bottom', fontsize=6, color=ps.MUTED)
ax.axvline(0, color=ps.MUTED, linewidth=0.7, zorder=1)

for label, fm, pd, _, _, cat in P:
    m, face, edge, s = style_of(label, cat)
    lw = 1.0 if face == 'white' else 0.5
    ax.scatter(pd, fm, s=s, marker=m, c=face, edgecolors=edge, linewidths=lw, zorder=4 if cat == 'deploy' else 3)

def pareto_front(points):
    nd = [p for i, p in enumerate(points)
          if not any((q[0] <= p[0] and q[1] <= p[1] and q != p) for j, q in enumerate(points) if j != i)]
    return sorted(nd)
same = {'expert', 'soup', 'gps', 'distill', 'joint_inherit', 'joint_fs', 'deploy'}
front = pareto_front([(p[2], p[1]) for p in P if p[5] in same])
xs, ys = [front[0][0]], [front[0][1]]
for x, y in front[1:]:
    xs += [x, x]; ys += [ys[-1], y]
ax.plot(xs, ys, color=ps.INK2, linewidth=0.8, zorder=2, solid_joinstyle='miter')

ax.set_xlim(-1.2, 17.2); ax.set_ylim(0, 36)
ax.set_xticks([0, 5, 10, 15]); ax.set_yticks([0, 10, 20, 30])
ax.set_xlabel(r'Source-domain regression, $\Sigma\Delta$ route deviation (m)')
ax.set_ylabel('Target-region fleet mean (m)')
ps.hairline_grid(ax, 'y')
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)

def mk(m, face, edge, s, label):
    return Line2D([0], [0], marker=m, linestyle='', markerfacecolor=face, markeredgecolor=edge,
                  markeredgewidth=1.0 if face == 'white' else 0.5, markersize=np.sqrt(s) * 1.05, label=label)
handles = [
    mk('D', WS, 'white', 20, 'per-ODD expert alone'),
    mk('o', WS, 'white', 24, 'uniform soup'),
    mk('s', WS, 'white', 22, 'greedy soup'),
    mk('X', WS, 'white', 26, 'TIES merging'),
    mk('o', ps.ORANGE, 'white', 34, 'joint replay (inherit shared)'),
    mk('o', 'white', ps.ORANGE, 24, 'joint replay, sample sweep\n($n = 5$ to $80$, own slice)'),
    mk('o', ps.AQUA, 'white', 22, 'joint replay from scratch\n(rank sweep, 3 runs)'),
    mk('*', ps.AQUA, 'white', 90, r'from scratch, $r = 4$ (Run 2)'),
    Line2D([0], [0], color=ps.INK2, linewidth=0.8, label='Pareto frontier'),
]
fig.legend(handles=handles, loc='center left', bbox_to_anchor=(0.675, 0.55), labelspacing=0.55,
           handletextpad=0.5, borderaxespad=0)

if __name__ == '__main__':
    import sys
    ps.save(fig, os.path.join(OUT, 'soup_pareto.pdf'), sys.argv[1] if len(sys.argv) > 1 else None)
    print('saved soup_pareto.pdf; frontier:', [(round(x, 2), round(y, 2)) for x, y in front])
