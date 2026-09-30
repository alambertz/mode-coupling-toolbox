#!/usr/bin/env python3
"""Fig. 5: influence of the pillar height (localized resonances) on scattering and mode absorptance.

    python make_fig5.py            -> fig_5.pdf, fig_5.png in this folder
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as scinter
from matplotlib.ticker import AutoMinorLocator

from mctoolbox.config import load_config, path
from mctoolbox.heatmap import attribute_modes, beer_lambert_lambertian, power_fractions
from mctoolbox.io import load_nk, read_cone_file, read_farfield, read_scattering_cross_section
from mctoolbox.modes import ModeLibrary
from mctoolbox.plotting import panels
from mctoolbox.plotting.style import (DPI, FIGSIZE, LABEL_COLORS, MODEAX_COLOR, SHADE_COLORS, apply_style,
                                      custom_formatter, distinct_colors, tex, ticks_all_around, wavelength_axis)
from mctoolbox.units import nm_to_eV

HERE = Path(__file__).resolve().parent


def substrate_index(cfg):
    """Index of the single-pillar substrate: a number, or a refractive-index file (wl in um, n)."""
    n = cfg['substrate_index']
    return n if isinstance(n, (int, float)) else load_nk(path(cfg, n)).n_um


def scattering_split(cfg, col):
    """Scattering cross-section (sigma_scat/sigma_geo) and its split into backward scattering and forward
    scattering within / beyond the escape cone (critical angle evaluated at every photon energy)."""
    x_s, sigma = read_scattering_cross_section(path(cfg, col['scattering_cross_section']), cfg['pillar_radius_nm'])
    sigma = scinter.interp1d(x_s, sigma)
    n = substrate_index(cfg)
    x_f, forward, inside, theta_c = read_cone_file(path(cfg, col['cone_transmission']), n)
    _, backward, _, _ = read_cone_file(path(cfg, col['cone_reflection']), n)
    total = forward + backward
    return dict(x_scat=x_s, sigma=sigma, x=x_f, theta_c=theta_c, backward_frac=backward/total,
                within_frac=forward*inside/total, beyond_frac=forward*(1 - inside)/total)


def compute(cfg):
    si = load_nk(*[path(cfg, f) for f in cfg['silicon']])
    lib = ModeLibrary(path(cfg, cfg['mode_library']))
    cols = []
    for col in cfg['column']:
        ff = read_farfield(path(cfg, col['farfield']), norm=cfg['norm'])
        cols.append(dict(height=col['height_nm'], ff=ff, fractions=power_fractions(ff),
                         modes=attribute_modes(ff, lib),
                         scattering=scattering_split(cfg, col)))
    return dict(si=si, lib=lib, columns=cols)


def scattering_panel(ax, s, labels):
    x, sig = s['x'], s['sigma']
    ax.plot(s['x_scat'], sig(s['x_scat']), color='black')
    ax.set_ylabel(tex(r'$\mathrm{\sigma}_\mathrm{scat} / \mathrm{\sigma}_\mathrm{geo}$', r'$\sigma_\mathrm{scat}/\sigma_\mathrm{geo}$'),
                  fontsize=18, labelpad=5)
    back, beyond = s['backward_frac'], s['beyond_frac']
    # top: backward; middle: forward within the escape cone; bottom: forward beyond k_c
    ax.fill_between(x, sig(x), sig(x)*(1 - back), facecolor=SHADE_COLORS[0], color=SHADE_COLORS[0],
                    edgecolor=None, interpolate=True)
    ax.fill_between(x, sig(x)*(1 - back), sig(x)*beyond, facecolor=SHADE_COLORS[2],
                    color=SHADE_COLORS[2], edgecolor=None, interpolate=True)
    ax.fill_between(x, sig(x)*beyond, 0*x, facecolor=SHADE_COLORS[1], color=SHADE_COLORS[1],
                    edgecolor=None, interpolate=True)
    if labels:
        xp = nm_to_eV(700)
        b, w, bb = (np.interp(xp, x[::-1], f[::-1]) for f in (back, s['within_frac'], beyond))
        pos = [(xp, sig(xp)*(1 - b/2)), (xp, sig(xp)*(bb + w/2)), (xp, sig(xp)*bb/2)]
        for text, p, c in zip(['backward', r'within $k_\mathrm{c}$', r'beyond $k_\mathrm{c}$'], pos, LABEL_COLORS):
            ax.annotate(text, xy=(nm_to_eV(800), 4.8), xytext=p, color=c,
                        horizontalalignment='left', verticalalignment='center')


def plot(cfg, r, out=HERE/'fig_5'):
    p = cfg['plot']
    apply_style()
    colors = distinct_colors(r['lib'].n_modes + 3)
    cols = r['columns']
    number = len(cols)
    figx, figy = .8*FIGSIZE[1], .8*FIGSIZE[0]*0.5
    fig = plt.figure(figsize=(number*figx, figy*2))
    gs = fig.add_gridspec(3, number, height_ratios=[0.67, 1., 1])
    aaxes, paxes, maxes = [], [], []
    for i in range(number):
        paxes.append(fig.add_subplot(gs[1, i]))
        if i == 0:
            aaxes.append(fig.add_subplot(gs[0, i]))
            maxes.append(fig.add_subplot(gs[2, i]))
        else:
            aaxes.append(fig.add_subplot(gs[0, i], sharey=aaxes[0]))
            maxes.append(fig.add_subplot(gs[2, i], sharex=aaxes[0]))
    plt.subplots_adjust(wspace=0.1, hspace=.08)

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
        pax.yaxis.set_tick_params(labelleft=True, pad=2)
    paxes[0].set_ylabel('Power fraction')

    for aax, max_, c in zip(aaxes, maxes, cols):
        aax.set_xlim(p['xlim'])
        scattering_panel(aax, c['scattering'], labels=c['height'] == p['labels_on_height'])
        panels.mode_absorptance_panel(max_, c['modes'], colors, linewidth=None)
        max_.fill_between([0, 5], 0, 1, edgecolor=None, facecolor=MODEAX_COLOR, color=MODEAX_COLOR)

    for aax in aaxes:
        wavelength_axis(aax, labelpad=4)

    for ct, (pax, hax, max_) in enumerate(zip(paxes, aaxes, maxes)):
        max_.set_xlabel('Photon energy (eV)', labelpad=2)
        max_.set_xlim(hax.get_xlim())
        max_.set_xticks(hax.get_xticks())
        max_.xaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))
        max_.set_ylim([0, 1])
        max_.set_ylabel('Mode absorptance', labelpad=2)
        hax.set_xticklabels([])
        hax.xaxis.set_tick_params(labelbottom=False)
        if ct == 0:
            hax.set_ylabel(tex(r'$\mathrm{\sigma}_\mathrm{scat} / \mathrm{\sigma}_\mathrm{geo}$',
                               r'$\sigma_\mathrm{scat}/\sigma_\mathrm{geo}$'), labelpad=2, fontsize=20)
        hax.xaxis.set_ticks_position('both')
        for ax in (hax, max_):
            ticks_all_around(ax)
        pax.set_xticklabels([])
        pax.set_xlim(hax.get_xlim())
        pax.set_xticks(hax.get_xticks())
        pax.xaxis.set_minor_locator(AutoMinorLocator())
        pax.set_ylim([0, 1])
        if ct > 0:
            for ax in (hax, pax, max_):
                ax.yaxis.set_tick_params(labelleft=False)
            hax.yaxis.set_tick_params(pad=10)
            hax.set_ylabel('')
            max_.set_ylabel('')
        for ax in (pax, hax, max_):
            ax.set_aspect('auto')

    xx = np.arange(1, 4, 0.01)
    for pax in paxes:
        pax.plot(xx, beer_lambert_lambertian(xx, cfg['thickness_nm'], r['si']), '--', color='green')

    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}.{ext}', dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def main():
    cfg = load_config(HERE)
    plot(cfg, compute(cfg))


if __name__ == '__main__':
    main()
