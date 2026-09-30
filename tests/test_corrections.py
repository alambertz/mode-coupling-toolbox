"""Behaviour of the corrected analysis (see README, "Changes since the submitted version")."""
import numpy as np
import pytest
from conftest import FIGURES, load_figure_module

from mctoolbox.config import load_config, path
from mctoolbox.fit import CouplingResult
from mctoolbox.heatmap import attribute_modes, power_fractions
from mctoolbox.io import load_nk, read_cone_file, read_farfield
from mctoolbox.modes import ModeLibrary
from mctoolbox.units import E0, HBAR, C0


@pytest.mark.parametrize('height,expected', [(75, (8.3, 21.7, 70.0)), (175, (22.8, 27.3, 49.9))])
def test_fig5_scattering_split(height, expected):
    """Share of the total scattered power (backward, escape cone, beyond k_c), mean over 1.2-2.0 eV."""
    f5 = load_figure_module(5)
    cfg = load_config(FIGURES/'fig5')
    col = next(c for c in cfg['column'] if c['height_nm'] == height)
    s = f5.scattering_split(cfg, col)
    sel = (s['x'] >= 1.2) & (s['x'] <= 2.0)
    got = [100*np.mean(s[k][sel]) for k in ('backward_frac', 'within_frac', 'beyond_frac')]
    np.testing.assert_allclose(got, expected, atol=1.0)
    np.testing.assert_allclose(s['backward_frac'] + s['within_frac'] + s['beyond_frac'], 1)


def test_critical_angle_follows_substrate_index():
    f = FIGURES/'fig5'/'Pillar_on_Si_forFarField-r75h75-p4000nm-5M_V06.fsp_monTransmission3_full_out.dat'
    _, _, _, theta_const = read_cone_file(f, 3.55)
    assert np.allclose(theta_const, np.degrees(np.arcsin(1/3.55)))
    si = load_nk(FIGURES/'fig5'/'siliconR.txt')
    e, _, inside, theta = read_cone_file(f, si.n_um)
    assert theta.max() - theta.min() > 3            # dispersive Si: theta_c changes with photon energy
    assert np.all((inside >= 0) & (inside <= 1))


@pytest.mark.parametrize('fig,files', [(4, None), (5, 'column')])
def test_all_guided_power_is_attributed(fig, files):
    cfg = load_config(FIGURES/f'fig{fig}')
    lib = ModeLibrary(path(cfg, cfg['mode_library']))
    names = [cfg['farfield']] if files is None else [c['farfield'] for c in cfg[files]]
    for name in names:
        ff = read_farfield(path(cfg, name), norm=cfg['norm'], lambda_min=cfg.get('lambda_min'))
        e, _, beyond = power_fractions(ff)
        modes = attribute_modes(ff, lib)
        total = np.zeros(len(e))
        for rows in modes.values():
            total[np.searchsorted(e, rows[:, 0])] += rows[:, 1]
        np.testing.assert_allclose(total, beyond, rtol=0, atol=1e-12)


def test_group_velocity_from_widths():
    gi, ge, hwhm_k = 2e-4, 3e-3, 0.1
    res = CouplingResult(kfits={1.25: [[20.0, 0.01, hwhm_k]], 1.26: [[25.0, 0.01, 0.5]]},
                         efits={20.0: np.array([[1.2502, 0.02, gi, ge, 1.0, 1]])}, k_step=0.25)
    v, ng = res.group_velocity()[0]
    expected = ((gi + ge)*E0/HBAR)/(2*hwhm_k*1e6)
    assert np.isclose(v, expected) and np.isclose(ng, C0/expected)


def test_group_velocity_nan_without_match():
    res = CouplingResult(kfits={1.25: [[10.0, 0.01, 0.1]]}, efits={20.0: np.array([[1.25, 0.02, 1e-4, 1e-3, 1.0, 1]])},
                         k_step=0.25)
    assert np.all(np.isnan(res.group_velocity()))


@pytest.mark.parametrize('n', [4, 5, 7])
def test_finite_slab_panels_use_dispersive_silicon(n):
    """Light line, mode windows and Lambertian curve of the slab figures use n_Si(lambda), not 3.55."""
    mod = load_figure_module(n)
    cfg = load_config(FIGURES/f'fig{n}')
    si = load_nk(*[path(cfg, f) for f in cfg['silicon']])
    assert si.n_um(0.5) - si.n_um(1.0) > 0.5         # clearly dispersive
    assert abs(si.n_um(1.0) - 3.55) > 0.01
    r = mod.compute(cfg)
    assert np.isclose(r['si'].n_um(1.0), si.n_um(1.0))
    if n == 7:                                        # guided-only filter against the Si light line
        from mctoolbox.units import light_line
        t = r['fits'].espace_table()
        assert np.all(t[:, 0] < light_line(t[:, 1], si.n_um))
