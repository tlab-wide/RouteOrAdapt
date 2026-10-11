"""Appendix figures of the real-vehicle / DPO study, redrawn in the paper style.

Same data as the original renderers in corl_work/scripts (phase4_paper_fig_v2_render.py,
final_results_figures.py, awsim_n200_paper_fig.py, awsim_n200_heatmap_fig.py,
awsim_heatmap_fig.py, awsim_adv_train_fig_n200.py); only the drawing changes.
Categorical slots are taken in the documented order (1 blue, 2 orange, 3 aqua, 4 yellow);
magnitudes use the single-hue blue ramp.
"""
import json
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')
OUT = os.path.join(HERE, 'out')
os.makedirs(OUT, exist_ok=True)
import paperstyle as ps

ps.setup()
PREVIEW = sys.argv[1] if len(sys.argv) > 1 else None
DATA = os.path.join(RESULTS, 'dpo_study')
SEQ = LinearSegmentedColormap.from_list('blue_seq', ['#f3f7fd'] + ps.BLUE_RAMP)


def finish(ax, grid='y'):
    ps.hairline_grid(ax, grid)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def save(fig, stem):
    ps.save(fig, f'{OUT}/{stem}.pdf', f'{PREVIEW}/{stem}.png' if PREVIEW else None)
    plt.close(fig)
    print('saved', stem)


def ink_for(rgba):
    r, g, b = rgba[:3]
    return 'white' if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.55 else ps.INK


# ---------------------------------------------------------------- Phase 4 (n = 500)
P4 = [('base_nolora', 'Base (no LoRA)'), ('v3_dpo_march_production', 'Production DPO LoRA'),
      ('v2_jr_paper', 'Joint replay (reward-trained)'), ('v5_pure', 'Reward-back-prop LoRA'),
      ('v5_surgical_zero_preproj', 'Reward-back-prop, preproj.fc1 zeroed'),
      ('phase2_sam', 'DPO+L2+SAM (headline)'), ('phase2_w0.9_mostly_imit', 'DPO+L2, w = 0.9'),
      ('phase2_beta_001', 'DPO+L2, beta = 0.01'), ('phase2_tgt_blocks', 'DPO+L2, block LoRA only'),
      ('phase2_alpha_64', 'DPO+L2, alpha = 64'), ('phase3_sam_seed42_best', 'DPO+L2+SAM, seed 42, best'),
      ('phase3_sam_seed42_final', 'DPO+L2+SAM, seed 42, final'), ('v8_deep_soup', 'Deep soup'),
      ('soup_v3_comfort', 'Comfort soup'), ('phase2_w0.0_pure_dpo', 'Pure DPO, no L2 anchor')]
d = json.load(open(f'{DATA}/phase4_n500.json'))
rows = [(lab, d[k]['_overall']['comfort_violation_rate'] * 100, d[k]['_overall']['strict_safety_violation_rate'] * 100)
        for k, lab in P4 if k in d and '_error' not in d[k]]
fig, ax = plt.subplots(figsize=(5.2, 3.0))
y = np.arange(len(rows))
h = 0.36
ax.barh(y - h / 2, [r[1] for r in rows], height=h, color=ps.BLUE, linewidth=0, label='comfort violation', zorder=2)
ax.barh(y + h / 2, [r[2] for r in rows], height=h, color=ps.ORANGE, linewidth=0, label='strict-safety violation', zorder=2)
for yi, (_, c, s) in zip(y, rows):
    if s > 0 and abs(c - s) < 5:
        ax.text(max(c, s) + 1.2, yi, f'{c:.0f} (both)', va='center', fontsize=5.8, color=ps.INK2)
        continue
    ax.text(c + 1.2, yi - h / 2, f'{c:.1f}', va='center', fontsize=5.8, color=ps.INK2)
    if s > 0:
        ax.text(s + 1.2, yi + h / 2, f'{s:.0f}', va='center', fontsize=5.8, color=ps.INK2)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
for t in ax.get_yticklabels():
    if 'headline' in t.get_text():
        t.set_fontweight('bold'); t.set_color(ps.INK)
ax.tick_params(axis='y', length=0)
ax.set_xlim(0, 118); ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel('Violation rate on $n = 500$ held-out ODD-Kashiwa scenes (%)')
finish(ax, 'x')
ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.55), handlelength=1.0)
save(fig, 'phase4_paper_fig_v2')

# ---------------------------------------------------------------- per-layer ablation
pl = json.load(open(f'{DATA}/per_layer_lora_ablation.json'))
layers = list(pl['per_layer'].items())
names = [k.replace('decoder.dit.', '') for k, _ in layers]
delta = [v['delta'] * 100 for _, v in layers]
fig, ax = plt.subplots(figsize=(4.6, 2.45))
y = np.arange(len(names))
cols = [ps.BLUE if abs(v) == max(abs(x) for x in delta) else ps.GRAY for v in delta]
ax.barh(y, delta, height=0.6, color=cols, linewidth=0, zorder=2)
ax.axvline(0, color=ps.AXIS, linewidth=0.6, zorder=1)
for yi, v in zip(y, delta):
    if abs(v) > 0.05:
        ax.text(v - 0.8, yi, f'{v:.1f}', va='center', ha='right', fontsize=5.8, color=ps.INK2)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=5.8); ax.invert_yaxis()
ax.tick_params(axis='y', length=0)
ax.set_xlim(-50, 5); ax.set_xticks([-40, -30, -20, -10, 0])
ax.set_xlabel('Change in comfort-violation rate when the layer is zeroed (pp)')
ax.text(0.02, 0.03, f"full LoRA {pl['_full_lora'] * 100:.1f}%, all layers zeroed {pl['_all_zero'] * 100:.1f}%",
        transform=ax.transAxes, fontsize=5.8, color=ps.MUTED)
finish(ax, 'x')
save(fig, 'surgical_lora_paper_fig')

# ---------------------------------------------------------------- AWSIM-Script n = 200
MODELS = [('base_nolora', 'Base (no LoRA)', ps.BLUE), ('v5_pure', 'Reward-back-prop LoRA', ps.ORANGE),
          ('v5_surgical', 'Reward-back-prop, preproj.fc1 zeroed', ps.AQUA), ('phase2_sam', 'DPO+L2+SAM', ps.YELLOW)]
aw = json.load(open(f'{DATA}/awsim_script_n200.json'))

def family(s):
    return 'cut-in' if s.startswith('cutin') else 'cut-out' if s.startswith('cutout') else 'deceleration'

FAMS = ['cut-in', 'cut-out', 'deceleration']
nfam = {f: len({r['scenario'] for r in aw if family(r['scenario']) == f}) for f in FAMS}
fig, ax = plt.subplots(figsize=(4.8, 2.05))
x = np.arange(len(FAMS)); w = 0.19
for i, (m, lab, col) in enumerate(MODELS):
    vals = [np.mean([r['collision_pct'] for r in aw if r['model'] == m and family(r['scenario']) == f]) for f in FAMS]
    pos = x + (i - 1.5) * (w + 0.012)
    ax.bar(pos, vals, width=w, color=col, linewidth=0, label=lab, zorder=2)
    for p, v in zip(pos, vals):
        ax.text(p, v + 1.2, f'{v:.1f}', ha='center', va='bottom', fontsize=5.4, color=ps.INK2)
ax.set_xticks(x); ax.set_xticklabels([f'{f} ({nfam[f]} scripts)' for f in FAMS])
ax.tick_params(axis='x', length=0)
ax.set_ylim(0, 84); ax.set_yticks([0, 20, 40, 60, 80])
ax.set_ylabel('Collision rate (%)')
finish(ax, 'y')
ax.legend(loc='upper left', ncol=1, handlelength=1.0, borderaxespad=0.1)
save(fig, 'awsim_n200_family_fig')

def group(s):
    m = re.match(r'^cutin(\d+)-(\d+)-\d+$', s)
    if m:
        return (0, int(m.group(1)), int(m.group(2))), f'cut-in {m.group(1)}/{m.group(2)}'
    m = re.match(r'^cutout(\d+)-\d+$', s)
    if m:
        return (1, int(m.group(1)), 0), f'cut-out {m.group(1)}'
    m = re.match(r'^deceleration(\d+)', s)
    return (2, int(m.group(1)), 0), f'decel {m.group(1)}'

keys = sorted({group(r['scenario']) for r in aw})
M = np.array([[np.mean([r['collision_pct'] for r in aw if r['model'] == m and group(r['scenario']) == k] or [np.nan])
               for k in keys] for m, _, _ in MODELS])
cnt = [sum(1 for r in aw if r['model'] == 'base_nolora' and group(r['scenario']) == k) for k in keys]
fig, ax = plt.subplots(figsize=(4.75, 1.75))
im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=100, aspect='auto')
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f'{M[i, j]:.0f}', ha='center', va='center', fontsize=5.4, color=ink_for(SEQ(M[i, j] / 100)))
ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([m[1].replace(', preproj', ',\npreproj') for m in MODELS], fontsize=6.2)
ax.set_xticks(range(len(keys))); ax.set_xticklabels([f'{k[1]} ({c})' for k, c in zip(keys, cnt)], rotation=40, ha='right',
                                                    fontsize=5.6)
ax.tick_params(length=0)
for side in ax.spines.values():
    side.set_visible(False)
for b in (6.5, 10.5):
    ax.axvline(b, color='white', linewidth=1.5)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5.6, length=1.5, width=0.4)
cb.set_label('collision rate (%)', fontsize=6, color=ps.INK2)
save(fig, 'awsim_n200_heatmap_fig')

# ---------------------------------------------------------------- TTC p25 grid
tt = json.load(open(f'{DATA}/awsim_cutin_ttc_grid.json'))
dx0 = tt['dx0']; vo = [round(v * 3.6) for v in tt['Vo']]
TTC_MAP = LinearSegmentedColormap.from_list('ttc', ps.BLUE_RAMP[::-1][:-1] + ['#f3f7fd'])
fig, axes = plt.subplots(2, 2, figsize=(5.4, 3.55), sharex=True, sharey=True)
fig.subplots_adjust(left=0.1, right=0.9, bottom=0.12, top=0.94, wspace=0.08, hspace=0.25)
for ax, (m, lab, _) in zip(axes.ravel(), MODELS):
    T = np.array(tt['models'][m]['ttc_p25'])
    im = ax.imshow(np.clip(T, 0, 5), cmap=TTC_MAP, vmin=0, vmax=5, origin='lower', aspect='auto')
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            txt = '5+' if T[i, j] >= 5 else f'{T[i, j]:.1f}'
            ax.text(j, i, txt, ha='center', va='center', fontsize=5.2, color=ink_for(TTC_MAP(min(T[i, j], 5) / 5)))
    ax.set_title(lab, fontsize=6.6, fontweight='bold', pad=3)
    ax.set_xticks(range(len(dx0))); ax.set_xticklabels([f'{v:g}' for v in dx0])
    ax.set_yticks(range(len(vo))); ax.set_yticklabels(vo)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
for ax in axes[1]:
    ax.set_xlabel('initial gap $dx_0$ (m)')
for ax in axes[:, 0]:
    ax.set_ylabel('NPC speed $V_o$ (km/h)')
cax = fig.add_axes([0.915, 0.25, 0.012, 0.55])
cb = fig.colorbar(im, cax=cax)
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5.6, length=1.5, width=0.4)
cb.set_ticks([0, 1, 2, 3, 4, 5]); cb.set_ticklabels(['0', '1', '2', '3', '4', '5+'])
cb.set_label('TTC p25 (s); darker is more dangerous', fontsize=6, color=ps.INK2)
save(fig, 'awsim_ttc_heatmap_fig')

# ---------------------------------------------------------------- adversarial-aware training (n = 200)
base = {r['scenario']: r for r in aw if r['model'] == 'phase2_sam'}
adv = {r['scenario']: r for r in json.load(open(f'{DATA}/awsim_script_n200_adv162.json'))}
scl = {r['scenario']: r for r in json.load(open(f'{DATA}/awsim_script_n200_adv500.json'))}
TRIO = [(base, 'DPO+L2+SAM', ps.BLUE), (adv, '+ 162 adversarial pairs', ps.ORANGE), (scl, '+ 500 adversarial pairs', ps.AQUA)]
fig = plt.figure(figsize=(5.4, 2.05))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.55], wspace=0.28, left=0.08, right=0.99, bottom=0.2, top=0.86)
axl, axr = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
x = np.arange(len(FAMS)); w = 0.25
for i, (src, lab, col) in enumerate(TRIO):
    vals = [np.mean([v['collision_pct'] for s, v in src.items() if family(s) == f]) for f in FAMS]
    pos = x + (i - 1) * (w + 0.015)
    axl.bar(pos, vals, width=w, color=col, linewidth=0, label=lab, zorder=2)
    for p, v in zip(pos, vals):
        axl.text(p, v + 1.2, f'{v:.1f}', ha='center', va='bottom', fontsize=5.2, color=ps.INK2)
axl.set_xticks(x); axl.set_xticklabels(FAMS); axl.tick_params(axis='x', length=0)
axl.set_ylim(0, 70); axl.set_yticks([0, 20, 40, 60]); axl.set_ylabel('Collision rate (%)')
finish(axl, 'y')
fig.legend(*axl.get_legend_handles_labels(), loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.02),
           handlelength=1.0, columnspacing=1.2)
order = sorted(base, key=lambda s: (FAMS.index(family(s)), scl[s]['collision_pct'] - base[s]['collision_pct']))
dv = [scl[s]['collision_pct'] - base[s]['collision_pct'] for s in order]
axr.bar(np.arange(len(dv)), dv, width=0.72, color=ps.AQUA, linewidth=0, zorder=2)
axr.axhline(0, color=ps.AXIS, linewidth=0.6, zorder=1)
start = 0
for f in FAMS:
    n = sum(1 for s in order if family(s) == f)
    axr.text(start + n / 2 - 0.5, 0.985, {'deceleration': 'decel.'}.get(f, f), transform=axr.get_xaxis_transform(),
             ha='center', va='top', fontsize=6.0, color=ps.INK2)
    if start:
        axr.axvline(start - 0.5, color=ps.GRID, linewidth=0.8, zorder=0)
    start += n
improved = sum(v < 0 for v in dv)
axr.text(0.02, 0.04, f'{improved} of {len(dv)} scripts improve', transform=axr.transAxes, ha='left', va='bottom',
         fontsize=6, color=ps.MUTED)
axr.set_xticks([]); axr.set_xlim(-0.8, len(dv) - 0.2)
axr.set_ylabel('Change vs DPO+L2+SAM (pp)')
finish(axr, 'y')
save(fig, 'awsim_adv_train_fig')
