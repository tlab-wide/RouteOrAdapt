"""Shared style for the camera-ready data figures.

Palette slots and chart ink follow the validated reference palette of the dataviz
method (slot 1 blue, slot 2 orange; adjacent and all-pairs checks pass for the
slots used). Colour follows the entity across every figure of the paper:
  BLUE   = static single-expert default
  ORANGE = joint replay (the proposed recipe)
  GRAY   = every other method (emphasis form: highlight one, gray the rest)
Figures are drawn at their final printed size, so font sizes are real points.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm

import glob as _glob
import os as _os
# Liberation Sans (metric-compatible with Arial). Set LIBERATION_SANS_DIR if it is not found.
_cands = [_os.environ.get('LIBERATION_SANS_DIR', ''), '/usr/share/fonts/truetype/liberation',
          '/opt/ros/humble/share/rviz_rendering/ogre_media/fonts/liberation-sans']
FONT_FAMILY = 'DejaVu Sans'
for _d in _cands:
    _files = _glob.glob(_os.path.join(_d, 'LiberationSans-*.ttf')) if _d else []
    if _files:
        for _f in _files:
            fm.fontManager.addfont(_f)
        FONT_FAMILY = 'Liberation Sans'
        break

# Validated categorical slots (light mode) and chart chrome.
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
MAGENTA, GREEN, VIOLET, RED = '#e87ba4', '#008300', '#4a3aa7', '#e34948'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS = '#e1e0d9', '#c3c2b7'
GRAY, GRAY_LIGHT = '#c5c3bb', '#dddbd4'          # de-emphasis marks
BLUE_RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
             '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']

BASE_PT = 7.0


def setup():
    sns.set_theme(context='paper', style='white')
    mpl.rcParams.update({
        'font.family': FONT_FAMILY,
        'mathtext.fontset': 'custom',
        'mathtext.rm': FONT_FAMILY,
        'mathtext.it': FONT_FAMILY + ':italic',
        'mathtext.bf': FONT_FAMILY + ':bold',
        'font.size': BASE_PT,
        'axes.titlesize': BASE_PT + 0.5,
        'axes.titleweight': 'bold',
        'axes.titlecolor': INK,
        'axes.labelsize': BASE_PT,
        'axes.labelcolor': INK2,
        'axes.edgecolor': AXIS,
        'axes.linewidth': 0.6,
        'axes.grid': False,
        'grid.color': GRID,
        'grid.linewidth': 0.5,
        'grid.linestyle': '-',
        'xtick.labelsize': BASE_PT - 0.5,
        'ytick.labelsize': BASE_PT - 0.5,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'xtick.labelcolor': INK2,
        'ytick.labelcolor': INK2,
        'xtick.major.size': 2.0,
        'ytick.major.size': 2.0,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'legend.fontsize': BASE_PT - 0.5,
        'legend.frameon': False,
        'text.color': INK,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def hairline_grid(ax, axis='x'):
    """Solid, recessive gridlines behind the data on one axis."""
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5, linestyle='-', zorder=0)
    ax.set_axisbelow(True)


def save(fig, path, preview=None):
    fig.savefig(path, bbox_inches='tight', pad_inches=0.02)
    if preview:
        fig.savefig(preview, bbox_inches='tight', pad_inches=0.02, dpi=200)
