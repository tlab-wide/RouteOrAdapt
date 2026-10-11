"""Fig. 4: (a) routing schemes per held-out map, (b) fleet mean of the constructive mechanisms.

All numbers are on the 24-scene routing pool (Pool A). Per-map values are the means of the
per-scene result files used for the paired tests (stage1/5/8/9/10 + reward_based_selector);
they match Table 2. Panel (b) averages over the three maps; for joint replay the Ishiodai mean
excludes the 7 scenes that are in its training set (see Sec. 4).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')
OUT = os.path.join(HERE, 'out')
os.makedirs(OUT, exist_ok=True)
import paperstyle as ps

ps.setup()

MAPS = ['Knowledge Hub', 'Ishiodai', 'Morikoro']
SCHEMES = [  # name, per-map max lateral offset (m)
    ('Scene-MLP router',     [4.51, 8.99, 63.34]),
    ('Self-consistency',     [14.26, 7.72, 46.30]),
    ('Lane-following',       [5.99, 7.76, 41.12]),
    ('Denoising-loss',       [13.96, 8.11, 42.93]),
    ('Reward-as-scorer',     [13.60, 9.20, 56.91]),
    ('Conformal corridor',   [4.02, 4.63, 38.92]),
]
STATIC = ('Static default', [4.51, 4.82, 31.88])
XMAX = [16, 10.5, 70]
XTICKS = [[0, 5, 10, 15], [0, 5, 10], [0, 20, 40, 60]]

FLEET = [  # name, fleet mean over the three maps (m), colour
    ('Shared module only',    22.76, ps.GRAY_LIGHT),
    ('vMF mixture',           15.86, ps.GRAY),
    ('Static default',        13.74, ps.BLUE),
    ('Shared-adapter repair',  8.91, ps.GRAY),
    ('Joint replay',           6.51, ps.ORANGE),
]

fig = plt.figure(figsize=(5.45, 1.9))
gs = GridSpec(1, 5, figure=fig, width_ratios=[0.82, 0.82, 0.82, 1.02, 1.12],
              left=0.155, right=0.985, bottom=0.2, top=0.8, wspace=0.12)
axes = [fig.add_subplot(gs[0, 0])]
axes += [fig.add_subplot(gs[0, i], sharey=axes[0]) for i in (1, 2)]
axb = fig.add_subplot(gs[0, 4])

ypos = list(range(len(SCHEMES))) + [len(SCHEMES) + 0.6]
labels = [s[0] for s in SCHEMES] + [STATIC[0]]
for j, ax in enumerate(axes):
    vals = [s[1][j] for s in SCHEMES] + [STATIC[1][j]]
    colors = [ps.GRAY] * len(SCHEMES) + [ps.BLUE]
    ax.barh(ypos, vals, height=0.62, color=colors, linewidth=0, zorder=2)
    ax.axvline(STATIC[1][j], color=ps.BLUE, linewidth=0.7, alpha=0.75, zorder=3)
    ax.text(STATIC[1][j] + XMAX[j] * 0.03, ypos[-1], f'{STATIC[1][j]:.1f}',
            va='center', ha='left', fontsize=6.3, color=ps.INK2)
    ax.set_xlim(0, XMAX[j]); ax.set_xticks(XTICKS[j])
    ax.set_title(MAPS[j], fontsize=7, fontweight='normal', color=ps.INK, pad=3)
    ps.hairline_grid(ax, 'x')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if j:
        ax.tick_params(axis='y', length=0, labelleft=False)
axes[0].set_yticks(ypos); axes[0].set_yticklabels(labels)
axes[0].tick_params(axis='y', length=0)
axes[0].invert_yaxis()
axes[1].set_xlabel('Max. lateral offset (m)')
fig.text(np.mean([axes[0].get_position().x0, axes[2].get_position().x1]), 0.955,
         '(a) Routing schemes, per held-out map', ha='center', va='top',
         fontsize=7.5, fontweight='bold', color=ps.INK)

yb = np.arange(len(FLEET))
axb.barh(yb, [f[1] for f in FLEET], height=0.62, color=[f[2] for f in FLEET], linewidth=0, zorder=2)
base = FLEET[0][1]
for y, (name, v, _) in zip(yb, FLEET):
    pct = '' if name == 'Shared module only' else f'  ({(v - base) / base * 100:+.0f}%)'.replace('-', '\u2212')
    axb.text(v + 0.7, y, f'{v:.2f}{pct}', va='center', ha='left', fontsize=6.3, color=ps.INK2)
axb.set_yticks(yb); axb.set_yticklabels([f[0] for f in FLEET])
axb.tick_params(axis='y', length=0)
axb.invert_yaxis()
axb.set_xlim(0, 36); axb.set_xticks([0, 10, 20, 30])
axb.set_xlabel('Fleet-mean offset (m)')
ps.hairline_grid(axb, 'x')
for side in ('top', 'right'):
    axb.spines[side].set_visible(False)
axb.set_title('(b) Mean over the three maps', fontsize=7.5, fontweight='bold', color=ps.INK, pad=11.5,
              loc='left', x=-0.62)

if __name__ == '__main__':
    import sys
    ps.save(fig, os.path.join(OUT, 'fig4_results.pdf'), sys.argv[1] if len(sys.argv) > 1 else None)
    print('saved fig4_results.pdf')
