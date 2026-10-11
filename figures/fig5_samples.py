"""Fig. 4: joint-replay sample complexity over three training seeds.

Data: stage26_multiseed.json (seeds 42-44, n = 5..40, each seed scored on its own held-out
slice) and the single-seed n = 80 reference (4.44 m). The static single-expert default on
the same kind of slice is 12.4 m (Table 3, eight seeds). Drawn at the width it is printed
(0.415 of the text width, next to Table 3).
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import os
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')
OUT = os.path.join(HERE, 'out')
os.makedirs(OUT, exist_ok=True)
import paperstyle as ps

ps.setup()

d = json.load(open(os.path.join(RESULTS, 'sample_complexity', 'three_seed.json')))
seed_vals = {}
for entry in d['seeds']:
    for row in entry['rows']:
        if 'fleet_mean' in row:
            seed_vals.setdefault(row['n_aichi'], []).append(row['fleet_mean'])
ns = np.array(sorted(seed_vals))
mean = np.array([np.mean(seed_vals[n]) for n in ns])
std = np.array([np.std(seed_vals[n], ddof=1) for n in ns])
N80, STATIC = 4.44, 12.4

fig, ax = plt.subplots(figsize=(2.28, 1.72))
ax.set_xscale('log', base=2)
ax.axhline(STATIC, color=ps.BLUE, linewidth=1.0, zorder=2)
ax.text(95, STATIC + 0.6, 'static default', color=ps.BLUE, fontsize=6.3, ha='right', va='bottom')
for i, jit in enumerate((0.93, 1.0, 1.07)):
    ax.scatter(ns * jit, [seed_vals[n][i] for n in ns], s=8, color=ps.GRAY, linewidth=0, zorder=3)
ax.errorbar(ns, mean, yerr=std, color=ps.ORANGE, linewidth=1.4, elinewidth=1.0, capsize=2.2,
            capthick=1.0, marker='o', markersize=3.8, markerfacecolor=ps.ORANGE,
            markeredgecolor='white', markeredgewidth=0.6, zorder=4)
ax.plot([80], [N80], linestyle='none', marker='o', markersize=3.8, markerfacecolor='white',
        markeredgecolor=ps.ORANGE, markeredgewidth=1.0, zorder=4)
ax.text(5.2, 1.4, 'joint replay', color=ps.ORANGE, fontsize=6.3, ha='left', va='bottom',
        fontweight='bold')

ax.set_xticks([5, 10, 20, 40, 80]); ax.set_xticklabels(['5', '10', '20', '40', '80'])
ax.minorticks_off()
ax.set_xlim(4.2, 98)
ax.set_ylim(0, 20); ax.set_yticks([0, 5, 10, 15, 20])
ax.set_xlabel('Target-region training scenes $n$')
ax.set_ylabel('Fleet-mean offset (m)')
ps.hairline_grid(ax, 'y')
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)

if __name__ == '__main__':
    import sys
    ps.save(fig, os.path.join(OUT, 'fig5_sample_complexity.pdf'), sys.argv[1] if len(sys.argv) > 1 else None)
    print('saved fig5_sample_complexity.pdf')
