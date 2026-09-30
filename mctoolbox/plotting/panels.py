"""Panels shared by several figures."""
import numpy as np
from scipy.interpolate import griddata

from ..units import k0, light_line
from .style import LABEL_COLORS, SHADE_COLORS


def heatmap_image(ax, ff, n, vmax, grid=2000, interpolation='bilinear', vmin=None, zorder=1, air_line=True):
    """P(E, k) in percent, interpolated onto a regular grid; the region beyond the
    material light line n(E)*k0 is black.

    grid: number of grid points per axis, or 'x4' for four times the number of energies.
    Returns the AxesImage (for a colorbar).
    """
    x, y, z = ff.rows[:, 0], ff.rows[:, 1], ff.rows[:, 2]*100
    energies = np.arange(min(x), max(x) + .01, 0.01)
    ax.fill_between(energies, 0, light_line(energies, n), edgecolor=None, facecolor='#000000',
                    interpolate=True, color='#000000')

    pts = 4*len(np.unique(x)) if grid == 'x4' else grid
    X, Y = np.meshgrid(np.linspace(min(x), max(x), pts), np.linspace(min(y), max(y), pts))
    Z = griddata((x, y), z, (X, Y), method='linear')
    Z[Y > light_line(X, n)] = np.nan
    im = ax.imshow(Z, extent=(min(x), max(x), min(y), max(y)), origin='lower', aspect='auto',
                   interpolation=interpolation, cmap='viridis', zorder=zorder,
                   vmin=np.nanmin(Z) if vmin == 'data' else vmin, vmax=vmax)
    if air_line:
        ax.plot(energies, k0(energies), linewidth=0.5, color='w')
    return im


def mode_lines(ax, curves, colors):
    """Dashed dispersion curves of the unpatterned-slab modes (from heatmap.mode_curves)."""
    for m, (xm, ym) in curves.items():
        ax.plot(xm, ym, '--', color=colors[m], linewidth=0.5)


def power_fraction_panel(ax, energies, within, beyond):
    """Stacked shading: beyond k_c (bottom), within k_c, backward (1 - within - beyond, top)."""
    ax.fill_between(energies, 1, within + beyond, edgecolor=None, facecolor=SHADE_COLORS[0],
                    interpolate=True, color=SHADE_COLORS[0])
    ax.fill_between(energies, beyond + within, beyond, edgecolor=None, facecolor=SHADE_COLORS[2],
                    interpolate=True, color=SHADE_COLORS[2])
    ax.fill_between(energies, beyond, 0, edgecolor=None, facecolor=SHADE_COLORS[1],
                    interpolate=True, color=SHADE_COLORS[1])


def power_fraction_labels(ax, positions, texts=None):
    """Colored text labels for the three shaded regions."""
    texts = texts or ['backward', r'beyond $k_\mathrm{c}$', r'within $k_\mathrm{c}$']
    for text, pos, col in zip(texts, positions, LABEL_COLORS):
        ax.annotate(text, xy=(1.5, .8), xytext=pos, color=col,
                    horizontalalignment='left', verticalalignment='center')


def mode_absorptance_panel(ax, modes, colors, linewidth=1.3/.8*.5, **kw):
    """Absorptance per mode vs photon energy (from heatmap.attribute_modes)."""
    for m, xz in modes.items():
        ax.plot(xz[:, 0], xz[:, 1], '-', color=colors[m], linewidth=linewidth, **kw)

