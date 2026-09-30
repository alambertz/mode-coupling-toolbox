import numpy as np
import pytest
from conftest import FIGURES

from mctoolbox.fit import (EFitSettings, KFitSettings, cmt_absorptance, fit_espace, fit_kspace,
                           gamma_i_bulk, legacy_pi_lineshape, lorentzian)
from mctoolbox.io import load_nk

SI = load_nk(FIGURES/'fig7'/'siliconR.txt', FIGURES/'fig7'/'siliconI.txt')
E = np.linspace(1.20, 1.30, 400)


def test_cmt_peak_is_one_at_critical_coupling():
    assert np.isclose(cmt_absorptance(1.25, 1.25, 1e-3, 1e-3), 1.0)
    assert np.isclose(cmt_absorptance(1.25, 1.25, 1e-3, 2e-3, N=2), 4*1e-3*2e-3/(1e-3 + 2*2e-3)**2)


def test_cmt_fwhm_is_sum_of_rates():
    gi, ge = 1e-3, 3e-3
    x = np.linspace(1.24, 1.26, 200001)
    y = cmt_absorptance(x, 1.25, gi, ge)
    above = x[y >= y.max()/2]
    assert np.isclose(above[-1] - above[0], gi + ge, rtol=1e-3)


def test_legacy_lineshape_differs_from_eq_hh():
    assert not np.isclose(legacy_pi_lineshape(1.25, 1.25, 1e-3, 1e-3), cmt_absorptance(1.25, 1.25, 1e-3, 1e-3))


def test_lorentzian_area():
    x = np.linspace(-200, 200, 400001)
    assert np.isclose(np.trapezoid(lorentzian(x, 0, 2.5, 0.3), x), 2.5, rtol=1e-3)


@pytest.mark.parametrize('ratio', [0.2, 5.0, 150.0])
@pytest.mark.parametrize('branch', ['over', 'under'])
def test_espace_fit_recovers_rates(ratio, branch):
    """Synthetic Eq. HH peaks are recovered; the branch setting picks r or 1/r."""
    gi = 2e-4
    ge = ratio*gi
    y = cmt_absorptance(E, 1.25, gi, ge)
    s = EFitSettings(prominence=1e-4, branch=branch, half_window=150)
    fits = fit_espace(E, y, SI, s)
    assert len(fits) == 1
    e0, _, gi_fit, ge_fit, n, _ = fits[0]
    assert np.isclose(e0, 1.25, atol=1e-5)
    expected = max(ratio, 1/ratio) if branch == 'over' else min(ratio, 1/ratio)
    assert np.isclose(ge_fit/gi_fit, expected, rtol=0.01)
    assert np.isclose(gi_fit + ge_fit, gi + ge, rtol=0.01)


def test_branch_bulk_bound():
    """bulk-bound keeps the branch whose gamma_i does not exceed the bulk rate."""
    g = gamma_i_bulk(1.25, SI)
    gi, ge = 0.05*g, 10*g                       # over-coupled; under branch would need gamma_i = 10 g
    fits = fit_espace(E, cmt_absorptance(E, 1.25, gi, ge), SI,
                      EFitSettings(prominence=1e-4, half_window=150, max_gamma_e=1))
    assert np.isclose(fits[0][3]/fits[0][2], ge/gi, rtol=0.01)
    assert fits[0][5] == 1


def test_kspace_fit_recovers_lorentzian():
    k = np.arange(0.25, 30, 0.25)
    p = lorentzian(k, 20.1, 0.02, 0.3) + 1e-5
    fits, peaks = fit_kspace(1.25, k, p, KFitSettings())
    assert len(fits) == 1
    assert np.isclose(fits[0][0], 20.1, atol=0.02)
    assert np.isclose(fits[0][2], 0.3, rtol=0.05)
