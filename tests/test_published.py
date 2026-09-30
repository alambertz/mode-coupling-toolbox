"""The package reproduces the numbers behind the figures of the first submission.

Reference tables in reference/ were exported from the original figure scripts
(plot_fig4_clean_V161, plot_fig5_weeded_V156, plot_fig6_N_v129_V213,
plot_fig7_clean_V188), each of which reproduces its archived PNG pixel for pixel.
Differences to the current defaults are corrections, and each is pinned here:
  - power fractions: published values are shifted by one energy step;
  - Figs. 5 and 7: published reading pairs k and power 3 bins apart (split='half');
  - Fig. 7: published energy-domain line shape ('legacy').
"""
import dataclasses

import numpy as np
import pytest
from conftest import read_reference, shifted_to_published

from mctoolbox.config import load_config, path
from mctoolbox.fit import analyse_coupling
from mctoolbox.heatmap import attribute_modes, power_fractions
from mctoolbox.io import load_nk, read_farfield
from mctoolbox.modes import ModeLibrary
from conftest import FIGURES


def assert_powerfrac(fractions, ref):
    e, within, beyond = fractions
    i = shifted_to_published(e, ref['energy_eV'])
    np.testing.assert_allclose(within[i], ref['within_kc'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(beyond[i], ref['beyond_kc'], rtol=0, atol=1e-12)


def assert_modes(modes, ref):
    ours = np.array([[m, *row] for m in sorted(modes) for row in modes[m]])
    np.testing.assert_array_equal(ours, np.c_[ref['mode'], ref['energy_eV'], ref['absorptance']])


def test_fig4():
    cfg = load_config(FIGURES/'fig4')
    ff = read_farfield(path(cfg, cfg['farfield']), norm=cfg['norm'], lambda_min=cfg['lambda_min'])
    lib = ModeLibrary(path(cfg, cfg['mode_library']))
    assert_powerfrac(power_fractions(ff), read_reference('fig4_powerfrac_H75.csv'))
    assert_modes(attribute_modes(ff, lib, lib.n_modes, cfg['plot']['krad']), read_reference('fig4_modeabs_H75.csv'))


@pytest.mark.parametrize('height', [75, 175])
def test_fig5_published_reading(height):
    cfg = load_config(FIGURES/'fig5')
    col = next(c for c in cfg['column'] if c['height_nm'] == height)
    ff = read_farfield(path(cfg, col['farfield']), norm=cfg['norm'], split='half')
    lib = ModeLibrary(path(cfg, cfg['mode_library']))
    assert_powerfrac(power_fractions(ff), read_reference(f'fig5_powerfrac_H{height}.csv'))
    assert_modes(attribute_modes(ff, lib, lib.n_modes, cfg['plot']['krad']),
                 read_reference(f'fig5_modeabs_H{height}.csv'))


@pytest.mark.parametrize('label,tag', [('Poisson random', 'poisson'), ('Hyperuniform disordered', 'hyperuniform')])
def test_fig6(label, tag):
    cfg = load_config(FIGURES/'fig6')
    col = next(c for c in cfg['column'] if c['label'] == label)
    ff = read_farfield(path(cfg, col['farfield']), norm=cfg['norm'])
    assert_powerfrac(power_fractions(ff), read_reference(f'fig6_powerfrac_{tag}.csv'))


def test_fig7_published_pipeline(fig_module):
    f7 = fig_module(7)
    cfg = load_config(FIGURES/'fig7')
    ff = read_farfield(path(cfg, cfg['farfield']), norm=cfg['norm'], max_energy=cfg['max_energy'],
                       energy_decimals=cfg['energy_decimals'], split='half')
    si = load_nk(*[path(cfg, f) for f in cfg['silicon']])
    ks, es = f7.settings(cfg)
    res = analyse_coupling(ff, si, ks, dataclasses.replace(es, lineshape='legacy', guided_only=False))
    kref = np.genfromtxt(FIGURES.parent/'reference'/'fig7_kfits.csv', delimiter=',', skip_header=1)
    eref = np.genfromtxt(FIGURES.parent/'reference'/'fig7_efits_legacy_lineshape.csv', delimiter=',', skip_header=1)
    np.testing.assert_array_equal(res.kspace_table(), kref)
    np.testing.assert_array_equal(res.espace_table()[:, :6], eref)


def test_fig7_current_defaults_are_over_coupled(fig_module):
    """Regression guard for the corrected analysis (reader, Eq. HH, bulk-bound branch)."""
    f7 = fig_module(7)
    cfg = load_config(FIGURES/'fig7')
    t = f7.compute(cfg)['fits'].espace_table()
    ratio = t[:, 7]
    assert len(t) == 70
    assert np.all(ratio >= 1)
    assert 150 < np.median(ratio) < 250
