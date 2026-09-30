"""Figure style of the paper (formerly general_settings.py).

apply_style() switches matplotlib to LaTeX text rendering with Arial/Helvetica,
as used for the published figures. Set the environment variable MCT_USETEX=0
(or call apply_style(usetex=False)) to render without a LaTeX installation;
labels then fall back to matplotlib mathtext.
"""
import os

import matplotlib
import numpy as np
from matplotlib.ticker import AutoMinorLocator

from ..units import eV_to_nm

FIGSIZE = (297/25.4, 210/25.4)   # A4 landscape, in inches
DPI = 300
FONTSIZE = 16

SHADE_COLORS = ['#F1B4B4', '#7FA0CE', '#EBEBEB']   # backward / beyond k_c / within k_c
LABEL_COLORS = ['#A41717', 'black', '#3C3319']
MODEAX_COLOR = '#669999'

_LATEX_PREAMBLE = r"""
    \usepackage{amsmath}
    \usepackage{sansmath}
    \sansmath
    \renewcommand{\familydefault}{\sfdefault}
    \usepackage{helvet}
    \renewcommand{\rmdefault}{phv}
"""

USETEX = os.environ.get('MCT_USETEX', '1') != '0'


def apply_style(usetex=None, fontsize=FONTSIZE):
    global USETEX
    if usetex is not None:
        USETEX = usetex
    matplotlib.rcParams.update({
        "text.usetex": USETEX,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": fontsize,
        "mathtext.fontset": "custom",
        "mathtext.it": "Arial:italic",
        "mathtext.rm": "Arial",
        "text.latex.preamble": _LATEX_PREAMBLE,
    })


def tex(latex, plain):
    """Pick the LaTeX or the mathtext version of a label."""
    return latex if USETEX else plain


def kpar_label():
    return tex(r'$k_\parallel$ ($\text{\textmu}$m$^{-1}$)', r'$k_\parallel$ ($\mu$m$^{-1}$)')


def percent():
    return tex(r'\%', '%')


def custom_formatter(x, pos):
    """Tick labels without trailing zeros for integers, one decimal otherwise."""
    if int(x) == x:
        return '{}'.format(int(x))
    return f'{x:.1f}'


def ticks_all_around(ax, formatter=True):
    """Ticks on all four sides (major in-out, minor in) with automatic minor locators."""
    ax.tick_params(axis='x', direction='inout')
    ax.tick_params(axis='x', which='minor', direction='inout')
    ax.tick_params(axis='y', direction='inout')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(top=True, right=True)
    ax.tick_params(which='minor', top=True, right=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if formatter:
        ax.yaxis.set_major_formatter(matplotlib.pyplot.FuncFormatter(custom_formatter))
    return ax


def wavelength_axis(ax, labelpad=8):
    """Secondary x axis on top of an energy axis, labelled in wavelength (nm)."""
    top = ax.twiny()
    top.set_xlabel('Wavelength (nm)', labelpad=labelpad)
    top.set_xlim(ax.get_xlim())
    top.set_xticks(ax.get_xticks())
    top.set_xticklabels([f'${eV_to_nm(t):.0f}$' for t in ax.get_xticks()], fontsize=FONTSIZE)
    top.tick_params(axis='x', direction='inout')
    top.tick_params(axis='x', which='minor', direction='in')
    return top


def colorbar(fig, ax, im, box, vmax=None, label=None, labelpad=-14, rotate=True):
    """Horizontal colorbar inside ax; box = (x0, y0, width, height) as fractions of ax.

    With vmax, the ticks read "0" and ">vmax".
    """
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0 + box[0]*pos.width, pos.y0 + box[1]*pos.height,
                        box[2]*pos.width, box[3]*pos.height])
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    if vmax is not None:
        cbar.set_ticks([0, vmax])
        cbar.set_ticklabels(['0', tex('$>${}', '>{}').format(vmax)])
    cbar.set_label(label if label is not None else f'Power ({percent()})', labelpad=labelpad)
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')
    if rotate:
        cbar.ax.yaxis.label.set_rotation(0)
    return cbar


def distinct_colors(num):
    """num distinct RGB colors via a golden-ratio hue spread (mode curves)."""
    import colorsys
    import math
    out = []
    for i in range(num):
        hue = (i*0.618033988749895) % 1.0
        sat = 0.7 + 0.3*(i % 2)
        light = 0.5 + 0.2*abs(math.sin(i*2))
        out.append(colorsys.hls_to_rgb(hue, light, sat))
    return out


def normalize(a):
    a = np.asarray(a, dtype=float)
    return (a - a.min())/(a.max() - a.min())
