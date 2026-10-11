"""Appendix soup-landscape figure, redrawn in the paper style.

The surface is a thin-plate-spline interpolation through four evaluated weight vectors
(anchor values copied from autoware_ws/scripts/plot_landscape.py); only those four points
are measured. Sequential encoding: one hue (blue), darker = larger value.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.interpolate import RBFInterpolator
import os
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')
OUT = os.path.join(HERE, 'out')
os.makedirs(OUT, exist_ok=True)
import paperstyle as ps

ps.setup()

ANCHORS = {
    'theta_0':    {'xy': (0, 0),     'train_loss': 1.10e-2, 'test_err': 20.11, 'm3_err': 39.67},
    'theta_1':    {'xy': (120, 0),   'train_loss': 8.70e-4, 'test_err': 19.41, 'm3_err': 34.94},
    'theta_2':    {'xy': (0, 120),   'train_loss': 8.70e-4, 'test_err': 22.96, 'm3_err': 41.16},
    'theta_soup': {'xy': (60, 60),   'train_loss': 1.60e-3, 'test_err': 20.70, 'm3_err': 37.64},
}
G = 200
gx = np.linspace(-15, 135, G)
XX, YY = np.meshgrid(gx, gx)

def surface(field):
    pts = np.array([a['xy'] for a in ANCHORS.values()])
    vals = np.array([a[field] for a in ANCHORS.values()])
    z = RBFInterpolator(pts, vals, kernel='thin_plate_spline', smoothing=0.0)(np.column_stack([XX.ravel(), YY.ravel()]))
    return z.reshape(G, G)

CMAP = LinearSegmentedColormap.from_list('blue_seq', ps.BLUE_RAMP[:11])
PANELS = [
    ('train_loss', 'Training loss', lambda v: f'{v * 1e3:.1f}', r'loss ($\times 10^{-3}$)', True),
    ('test_err', 'Target-region error', lambda v: f'{v:.0f}', 'fleet mean (m)', False),
    ('m3_err', 'Error on Morikoro', lambda v: f'{v:.0f}', 'Morikoro mean (m)', False),
]
MARK = {'theta_0': ('*', 70), 'theta_1': ('s', 26), 'theta_2': ('D', 24), 'theta_soup': ('o', 28)}
TEXT = {'theta_0': r'$\theta_0$', 'theta_1': r'$\theta_1$', 'theta_2': r'$\theta_2$', 'theta_soup': r'$\theta_{\rm soup}$'}

fig, axes = plt.subplots(1, 3, figsize=(5.45, 2.05))
fig.subplots_adjust(left=0.07, right=0.93, bottom=0.25, top=0.9, wspace=0.55)
for ax, (field, title, fmt, clabel, clip) in zip(axes, PANELS):
    Z = surface(field)
    if clip:
        Z = np.clip(Z, 0, None)
    levels = np.linspace(Z.min(), Z.max(), 11)
    cf = ax.contourf(XX, YY, Z, levels=levels, cmap=CMAP)
    ax.contour(XX, YY, Z, levels=levels, colors='white', linewidths=0.35, alpha=0.6)
    p0, p1, p2, pso = (ANCHORS[k]['xy'] for k in ('theta_0', 'theta_1', 'theta_2', 'theta_soup'))
    for end in (p1, p2):
        ax.annotate('', xy=end, xytext=p0, arrowprops=dict(arrowstyle='-|>', color=ps.INK2, lw=0.7,
                                                         shrinkA=3, shrinkB=4, mutation_scale=6))
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=ps.INK2, linewidth=0.6, alpha=0.7)
    for k, (m, s) in MARK.items():
        x, y = ANCHORS[k]['xy']
        ax.scatter(x, y, marker=m, s=s, c=ps.INK, edgecolors='white', linewidths=0.8, zorder=5)
        off, ha = ((-4, 4), 'right') if k == 'theta_1' else ((4, 3), 'left')
        ax.annotate(TEXT[k], (x, y), xytext=off, textcoords='offset points', fontsize=6.3, color=ps.INK, ha=ha)
    ax.set_aspect('equal'); ax.set_xlim(-15, 135); ax.set_ylim(-15, 135)
    ax.set_xticks([0, 60, 120]); ax.set_yticks([0, 60, 120])
    ax.set_title(title, fontsize=7.2, fontweight='bold', pad=4)
    ax.set_xlabel(r'$\theta_1$ direction', labelpad=1.5)
    for side in ax.spines.values():
        side.set_visible(False)
    cb = fig.colorbar(cf, ax=ax, fraction=0.05, pad=0.04)
    ticks = np.linspace(Z.min(), Z.max(), 4)
    cb.set_ticks(ticks); cb.set_ticklabels([fmt(t) for t in ticks])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, length=1.5, width=0.4, color=ps.MUTED)
    cb.set_label(clabel, fontsize=6.3, color=ps.INK2, labelpad=2)
axes[0].set_ylabel(r'$\theta_2$ direction', labelpad=1.5)

handles = [Line2D([0], [0], marker=MARK[k][0], linestyle='', markerfacecolor=ps.INK, markeredgecolor='white',
                  markersize=np.sqrt(MARK[k][1]) * 1.05, label=lab)
           for k, lab in [('theta_0', r'$\theta_0$ shared only'), ('theta_1', r'$\theta_1$ ODD-Nagoya expert (best)'),
                          ('theta_2', r'$\theta_2$ ODD-Kashiwa expert (second)'), ('theta_soup', r'$\theta_{\rm soup}$ uniform soup')]]
fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.0), columnspacing=1.2,
           handletextpad=0.3)

if __name__ == '__main__':
    import sys
    ps.save(fig, os.path.join(OUT, 'model_soup_landscape.pdf'), sys.argv[1] if len(sys.argv) > 1 else None)
    print('saved model_soup_landscape.pdf')
