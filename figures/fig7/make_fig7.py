#!/usr/bin/env python3
"""Fig. 7: GMR fits in the momentum and energy domain and the coupling condition gamma_e/gamma_i.

    python make_fig7.py            -> fig_7.pdf, fig_7.png, fig7_kspace_fits.csv, fig7_espace_fits.csv
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.colors import BoundaryNorm

from mctoolbox.config import load_config, path
from mctoolbox.fit import EFitSettings, KFitSettings, LINESHAPES, analyse_coupling, lorentzian
from mctoolbox.io import load_nk, read_farfield
from mctoolbox.plotting.style import (DPI, FIGSIZE, apply_style, colorbar, custom_formatter, kpar_label, percent,
                                      tex, ticks_all_around)
from mctoolbox.units import eV_to_nm, light_line

HERE = Path(__file__).resolve().parent
PALETTE = ['#8750CC', '#FF7F00', '#FFFF00', '#FF3D33', '#FF3D33']   # energy slices use [0:], momenta [4:]


def settings(cfg):
    es = dict(cfg['efit'])
    es['k_grid'] = tuple(es['k_grid'])
    return KFitSettings(**cfg['kfit']), EFitSettings(**es)


def compute(cfg, split='auto'):
    ff = read_farfield(path(cfg, cfg['farfield']), norm=cfg['norm'], max_energy=cfg['max_energy'],
                       energy_decimals=cfg['energy_decimals'], split=split)
    si = load_nk(*[path(cfg, f) for f in cfg['silicon']])
    ks, es = settings(cfg)
    return dict(ff=ff, si=si, es=es, fits=analyse_coupling(ff, si, ks, es))


def write_tables(r, out_dir=HERE):
    f = r['fits']
    np.savetxt(out_dir/'fig7_kspace_fits.csv', f.kspace_table(), delimiter=',', comments='',
               header='energy_eV,center_k_um-1,amplitude,hwhm_k_um-1')
    np.savetxt(out_dir/'fig7_espace_fits.csv', np.column_stack([f.espace_table(), f.group_velocity()]),
               delimiter=',', comments='',
               header='k_um-1,E0_eV,peak_absorptance,gamma_i_eV,gamma_e_eV,N,branch_unique,gamma_e_over_gamma_i,'
                      'v_group_m_per_s,n_group')


def heatmap_panel(ax, r, cfg):
    p = cfg['plot']
    ff, n = r['ff'], r['si'].n_um
    x, y, z = ff.rows[:, 0], ff.rows[:, 1], ff.rows[:, 2]*100
    energies = np.arange(min(x), max(x) + .01, 0.01)
    ax.fill_between(energies, 0, light_line(energies, n) + 0.1, edgecolor=None, facecolor='#000000',
                    interpolate=True, color='#000000')
    im = ax.scatter(x, y, c=z, cmap='viridis', s=50, zorder=1, marker='.', edgecolors=None, vmax=p['vmax'])
    slice_guides(ax, cfg, linewidth=1)
    ax.set_xlabel('Photon energy (eV)')
    ax.set_ylabel(kpar_label(), labelpad=0)
    ax.xaxis.set_ticks_position('both')
    ax.set_yticks(range(0, 80, 4))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))
    ax.set_ylim(p['klim'])
    ax.set_xlim(p['xlim'])
    return im


def slice_guides(ax, cfg, linewidth):
    """Vertical lines at the energy slices (up to the Si light line), horizontal lines at the momentum slices."""
    k0_, k1_ = cfg['plot']['klim']
    si_n = load_nk(path(cfg, cfg['silicon'][0])).n_um
    for e, col in zip(cfg['slice_energies'], PALETTE):
        ax.axvline(x=e, ymax=(light_line(e, si_n) - k0_)/(k1_ - k0_), color=col, linewidth=linewidth)
    for k, col in zip(cfg['slice_momenta'], PALETTE[4:]):
        ax.axhline(y=k, color=col, linewidth=linewidth)


def momentum_slice_panel(ax, r, cfg):
    fits = r['fits'].kfits
    for e, col in zip(cfg['slice_energies'], PALETTE):
        sub = r['ff'].at_energy(e)
        x = np.arange(sub[:, 1].min(), sub[:, 1].max(), .05)
        ax.plot(100*sub[:, 2], sub[:, 1], color=col, linewidth=4,
                label=r'E$_\mathrm{ph}$: ' + str(round(e, 3)) + tex(r'\,eV', ' eV'))
        for fit in fits[min(fits, key=lambda k: abs(k - e))]:
            ax.plot(100*lorentzian(x, *fit), x, '--', color='grey')
    ax.set_xlabel(tex(r'P$_\mathrm{abs}$ (\%)', r'P$_\mathrm{abs}$ (%)'))
    ax.set_ylabel(kpar_label())
    legend = ax.legend(frameon=False, bbox_to_anchor=(0, 0, 0.28, 1))
    for text in legend.get_texts():
        text.set_color(PALETTE[0])
    for handle in legend.legend_handles:
        handle.set_visible(False)
    ticks_all_around(ax)


def energy_slice_panel(ax, r, cfg):
    fits, model = r['fits'].efits, LINESHAPES[r['es'].lineshape]
    for k, col in zip(cfg['slice_momenta'], PALETTE[4:]):
        sub = r['ff'].at_momentum(k)
        x = np.arange(sub[:, 0].min(), sub[:, 0].max(), .001)
        ax.plot(sub[:, 0], 100*sub[:, 2], color=col, linewidth=4,
                label=r'$k_\parallel$: ' + str(round(k, 2)) + tex(r'\,\textmu m$^{-1}$', r' $\mu$m$^{-1}$'))
        for e0, _, gi, ge, N, *_ in fits[min(fits, key=lambda q: abs(q - k))]:
            ax.plot(x, 100*model(x, e0, gi, ge, N), '--', color='grey')
    ax.set_xlabel('Photon energy (eV)')
    ax.set_ylabel(tex(r'P$_\mathrm{abs}$ (\%)', r'P$_\mathrm{abs}$ (%)'), labelpad=10)
    legend = ax.legend(frameon=False, bbox_to_anchor=(0, 0, 0.28, 1.02))
    for text in legend.get_texts():
        text.set_color(PALETTE[4])
    for handle in legend.legend_handles:
        handle.set_visible(False)
    ticks_all_around(ax)


def coupling_panel(fig, ax, r, cfg):
    p = cfg['plot']
    t = r['fits'].espace_table()
    cmap = plt.get_cmap('coolwarm')
    sc = ax.scatter(t[:, 1], t[:, 0], c=t[:, 7], cmap=cmap, s=150, marker='.', edgecolor=None,
                    norm=BoundaryNorm(p['ratio_boundaries'], cmap.N))
    colorbar(fig, ax, sc, (.025, .93, .35, .05), label=r'$\gamma_e/\gamma_i$', labelpad=1, rotate=False)
    slice_guides(ax, cfg, linewidth=0.25)
    xx = np.arange(1, 4, .01)
    ax.plot(xx, light_line(xx, r['si'].n_um), color='black')
    ax.set_xlabel('Photon Energy (eV)')
    ax.set_ylabel(tex(r'k$_\parallel$ (\textmu m$^{-1}$)', r'k$_\parallel$ ($\mu$m$^{-1}$)'))
    ax.set_ylim(p['klim'])
    ax.set_yticks([12, 16, 20, 24])
    ax.set_xlim(p['xlim'])
    ticks_all_around(ax)


def plot(cfg, r, out=HERE/'fig_7'):
    apply_style()
    figsize = (0.7*FIGSIZE[0], 0.7*FIGSIZE[1])
    fig = plt.figure(figsize=(figsize[0]*2, figsize[1]*2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, .1])
    heat_ax = fig.add_subplot(gs[0, 0])
    avk_ax = fig.add_subplot(gs[0, 1], sharey=heat_ax)
    avp_ax = fig.add_subplot(gs[1, 0], sharex=heat_ax)
    res_ax = fig.add_subplot(gs[1, 1])
    avk_ax.set_zorder(10)
    plt.subplots_adjust(wspace=0.15, hspace=.25)

    im = heatmap_panel(heat_ax, r, cfg)
    colorbar(fig, heat_ax, im, (.025, .93, .4, .05), vmax=cfg['plot']['vmax'], label=f'Power ({percent()})')
    momentum_slice_panel(avk_ax, r, cfg)
    energy_slice_panel(avp_ax, r, cfg)
    coupling_panel(fig, res_ax, r, cfg)
    avk_ax.set_xlim([-0.01, 5])
    avp_ax.set_ylim([-0.01, 5])

    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}.{ext}', dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def main():
    matplotlib.use('Agg')
    cfg = load_config(HERE)
    r = compute(cfg)
    write_tables(r)
    plot(cfg, r)
    t = r['fits'].espace_table()
    ratio = t[:, 7]
    print(f'{len(ratio)} GMR fits ({int(t[:, 6].sum())} with a unique coupling branch); gamma_e/gamma_i: '
          f'median {np.median(ratio):.0f}, 10-90 % range {np.percentile(ratio, 10):.0f}-{np.percentile(ratio, 90):.0f}')


if __name__ == '__main__':
    main()
