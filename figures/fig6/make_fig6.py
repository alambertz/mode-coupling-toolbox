#!/usr/bin/env python3
"""Fig. 6: heatmaps, pattern PSDs and power fractions of the disordered patterns.

    python make_fig6.py            -> fig_6.pdf, fig_6.png in this folder
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, FixedLocator

from mctoolbox.config import load_config, path
from mctoolbox.heatmap import beer_lambert_lambertian, power_fractions
from mctoolbox.io import load_nk, load_nk_from_permittivity, read_farfield
from mctoolbox.plotting import panels
from mctoolbox.plotting.style import (DPI, FIGSIZE, apply_style, colorbar, custom_formatter, kpar_label,
                                      normalize, wavelength_axis)
from mctoolbox.units import k0, light_line

HERE = Path(__file__).resolve().parent


def compute(cfg):
    cols = []
    for col in cfg['column']:
        ff = read_farfield(path(cfg, col['farfield']), norm=cfg['norm'])
        cols.append(dict(col, ff=ff, fractions=power_fractions(ff)))
    return dict(columns=cols,
                si=load_nk(path(cfg, cfg['silicon_light_line'])),
                si_eps=load_nk_from_permittivity(path(cfg, cfg['silicon_lambertian'])))


def sparse_heatmap(ax, ff, n, vmax, spacing):
    """Scatter of P(E, k) at photon energies at least `spacing` eV apart."""
    x, y, z = ff.rows[:, 0], ff.rows[:, 1], ff.rows[:, 2]*100
    energies = np.arange(min(x), max(x) + .01, 0.01)
    ax.fill_between(energies, 0, light_line(energies, n), edgecolor=None, facecolor='#000000',
                    interpolate=True, color='#000000')
    shown, last = [], min(x)
    for e in np.unique(x):
        if abs(e - last) >= spacing:
            last = e
            shown.append(e)
    sel = np.isin(x, shown)
    im = ax.scatter(x[sel], y[sel], c=z[sel], cmap='viridis', s=0.2, zorder=1, edgecolors=None, vmax=vmax)
    ax.plot(energies, k0(energies), linewidth=0.5, color='w')
    ax.set_ylabel(kpar_label(), labelpad=10)
    ax.xaxis.set_ticks_position('both')
    ax.set_yticks(range(0, 80, 10))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))
    return im


def psd_panel(ax, cfg):
    c = cfg['psd']
    rand = np.loadtxt(path(cfg, c['random']), skiprows=1, delimiter=',')
    hud = np.loadtxt(path(cfg, c['hyperuniform']), skiprows=1, delimiter=',')
    for d, sign, color in ((rand, -1, c['random_color']), (hud, 1, c['hyperuniform_color'])):
        ax.plot(sign*normalize(d[:, 1]), d[:, 0], color=color)
        ax.fill_between(sign*normalize(d[:, 1]), d[:, 0], edgecolor=None, facecolor=color, interpolate=True, color=color)
    ax.axvline(0, color='black', linewidth=.2)
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels([1, 0, 1])
    ax.tick_params(axis='y', which='major', direction='inout', color='black', length=4)
    ax.spines['left'].set_position(('data', 0))
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_yticks([10, 20, 30, 40, 50])
    ax.tick_params(axis='y', which='minor', direction='inout', size=0)
    ax.set_xlabel('PSD (a.u.)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def energy_ticks(ax, xlim):
    ax.set_xticks(np.arange(1, 3.2, .2))
    ax.set_xlim(xlim)
    ax.tick_params(axis='x', direction='inout')
    ax.tick_params(axis='x', which='minor', direction='inout')
    ax.tick_params(axis='y', direction='inout')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(top=True, right=True)
    ax.tick_params(which='minor', top=True, right=True)
    ax.xaxis.set_minor_locator(FixedLocator(np.arange(1., 3.05, .05)))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))


def plot(cfg, r, out=HERE/'fig_6'):
    p = cfg['plot']
    apply_style()
    cols = r['columns']
    figx, figy = 1.2*FIGSIZE[1]*.8, 1.2*FIGSIZE[0]*.7
    fig = plt.figure(figsize=(len(cols)*figx, figy*.85))
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 0.2, 1], height_ratios=[1, 0.05, 1])
    paxes = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 2])]

    for pax, c in zip(paxes, cols):
        panels.power_fraction_panel(pax, *c['fractions'])
        pax.xaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))
        pax.yaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))
        pax.tick_params(axis='x', direction='in')
        pax.tick_params(axis='x', which='minor', direction='in')
        pax.tick_params(axis='y', direction='inout')
        pax.tick_params(axis='y', which='minor', direction='in')
        pax.tick_params(top=True, right=True)
        pax.tick_params(which='minor', top=True, right=True)
        pax.yaxis.set_minor_locator(AutoMinorLocator())
        pax.xaxis.set_minor_locator(AutoMinorLocator())
        pax.yaxis.set_tick_params(labelleft=True, pad=2)
        pax.set_ylim([0, 1])
        panels.power_fraction_labels(pax, [(2.55, .92), (2.55, .1), (2.55, .78)])
    paxes[0].set_ylabel('Power fraction', labelpad=10)

    haxes = [fig.add_subplot(gs[0, 0])]
    haxes.append(fig.add_subplot(gs[0, 2], sharex=paxes[0], sharey=haxes[0]))
    plt.subplots_adjust(wspace=0.085, hspace=0.03)
    psd_ax = fig.add_subplot(gs[0, 1], sharey=haxes[0])
    psd_panel(psd_ax, cfg)

    for hax, c in zip(haxes, cols):
        im = sparse_heatmap(hax, c['ff'], r['si'].n_um, p['vmax'], p['min_energy_spacing'])
        hax.set_ylim([0, p['kmax']])
        colorbar(fig, hax, im, (.05, .95, .3, .05), vmax=p['vmax'])

    for ct, (pax, hax, c) in enumerate(zip(paxes, haxes, cols)):
        energy_ticks(hax, p['xlim'])
        hax.set_xlim(p['xlim'])
        hax.xaxis.set_tick_params(labelbottom=False)
        hax.set_title(c['label'], pad=15, color=c['title_color'])
        energy_ticks(pax, p['xlim'])
        pax.set_xlabel('Photon energy (eV)')
        if ct > 0:
            hax.yaxis.set_tick_params(labelleft=False)
            pax.yaxis.set_tick_params(labelleft=False)
            hax.yaxis.set_tick_params(pad=10)
            hax.set_ylabel('')
        pax.set_aspect('auto')
        hax.set_aspect('auto')

    for hax in haxes:
        wavelength_axis(hax, labelpad=7)

    xx = np.arange(1, 4, 0.01)
    for pax in paxes:
        pax.plot(xx, beer_lambert_lambertian(xx, cfg['thickness_nm'], r['si_eps']), '--', color='green')

    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}.{ext}', dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def main():
    cfg = load_config(HERE)
    plot(cfg, compute(cfg))


if __name__ == '__main__':
    main()
