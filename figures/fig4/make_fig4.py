#!/usr/bin/env python3
"""Fig. 4: heatmap, power fractions and mode-resolved absorptance of the periodic grating.

    python make_fig4.py            -> fig_4.pdf, fig_4.png in this folder
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

from mctoolbox.config import load_config, path
from mctoolbox.heatmap import attribute_modes, beer_lambert_lambertian, mode_curves, power_fractions
from mctoolbox.io import load_nk, read_farfield
from mctoolbox.modes import ModeLibrary
from mctoolbox.plotting import panels
from mctoolbox.plotting.style import (DPI, FIGSIZE, MODEAX_COLOR, apply_style, colorbar, custom_formatter,
                                      distinct_colors, kpar_label, ticks_all_around, wavelength_axis)

HERE = Path(__file__).resolve().parent


def compute(cfg):
    ff = read_farfield(path(cfg, cfg['farfield']), norm=cfg['norm'], lambda_min=cfg['lambda_min'])
    si = load_nk(*[path(cfg, f) for f in cfg['silicon']])
    lib = ModeLibrary(path(cfg, cfg['mode_library']))
    return dict(ff=ff, si=si, lib=lib,
                fractions=power_fractions(ff),
                modes=attribute_modes(ff, lib, lib.n_modes, cfg['plot']['krad']))


def plot(cfg, r, out=HERE/'fig_4'):
    p = cfg['plot']
    apply_style()
    colors = distinct_colors(r['lib'].n_modes)
    figx, figy = .95*FIGSIZE[1]*.9, .95*FIGSIZE[0]
    fig = plt.figure(figsize=(figx, figy))
    gs = fig.add_gridspec(4, 1)
    pax = fig.add_subplot(gs[1, 0])
    hax = fig.add_subplot(gs[0, 0])
    max_ = fig.add_subplot(gs[2, 0])
    plt.subplots_adjust(wspace=0.07, hspace=.11)

    # power fractions
    e, within, beyond = r['fractions']
    panels.power_fraction_panel(pax, e, within, beyond)
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
    panels.power_fraction_labels(pax, [(2.55, .92), (2.55, .1), (2.55, .74)])
    pax.set_ylabel('Power fraction')
    xx = np.arange(1, 4, 0.01)
    pax.plot(xx, beer_lambert_lambertian(xx, cfg['thickness_nm'], r['si']), '--', color='green')

    # heatmap + mode curves
    im = panels.heatmap_image(hax, r['ff'], r['si'].n_um, p['vmax'])
    panels.mode_lines(hax, mode_curves(r['lib'], r['lib'].n_modes), colors)
    colorbar(fig, hax, im, (.015, .95, .28, .05), vmax=p['vmax'])

    # mode absorptance
    panels.mode_absorptance_panel(max_, r['modes'], colors)
    max_.fill_between([0, 5], 0, 1, edgecolor=None, facecolor=MODEAX_COLOR, color=MODEAX_COLOR)

    max_.set_xlabel('Photon energy (eV)', labelpad=2)
    max_.set_xlim(p['xlim'])
    max_.xaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))
    max_.set_ylim([0, 1])
    max_.set_ylabel('Mode absorptance', labelpad=2)
    hax.set_xlim(p['xlim'])
    hax.set_ylim([0, p['kmax']])
    hax.xaxis.set_tick_params(labelbottom=False)
    hax.set_ylabel(kpar_label(), labelpad=7)
    hax.xaxis.set_ticks_position('both')
    for ax in (hax, max_):
        ticks_all_around(ax)
    pax.set_xlim(p['xlim'])
    pax.xaxis.set_tick_params(labelbottom=False)
    pax.set_ylim([0, 1])
    for ax in (pax, hax, max_):
        ax.set_aspect('auto')
    wavelength_axis(hax, labelpad=8)

    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}.{ext}', dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def main():
    cfg = load_config(HERE)
    plot(cfg, compute(cfg))


if __name__ == '__main__':
    main()
