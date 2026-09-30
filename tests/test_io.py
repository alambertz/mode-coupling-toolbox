import numpy as np
import pytest
from conftest import FIGURES

from mctoolbox.io import load_nk, power_column_offset, read_farfield
from mctoolbox.modes import ModeLibrary

F4 = FIGURES/'fig4'/'500nmSi-wAbs-withBumpH75R75nm-BS350nm-Acc4-4M-Conf0-lambda300-1200_1nmSTEP_V15.fsp_15UMlateral_full_out_v127.dat'
F7_025 = FIGURES/'fig7'/'500nmSi-wAbs-HUDPillar-H175R150nm-BS5.0um-Acc4-8M4400nmfit-lambda413-1127_V23.fsp_0.25um1_full_out_v128.dat'
F7_050 = FIGURES/'fig7'/'500nmSi-wAbs-HUDPillar-H175R150nm-BS5.0um-Acc4-8M4400nmfit-lambda413-1127_V23.fsp_0.5um1_full_out_v128.dat'


def test_read_farfield_shape_and_grid():
    ff = read_farfield(F4, norm='R', lambda_min=400)
    assert ff.rows.shape[1] == 3
    assert np.isclose(ff.k_step, 0.25)
    assert len(ff.energies) == 600          # wavelengths >= 400 nm
    assert np.all(ff.rows[:, 1] > 0)


def test_unknown_normalisation_rejected():
    with pytest.raises(ValueError):
        read_farfield(F4, norm='X')


def test_power_offset_regular_file_is_half():
    raw = np.loadtxt(F7_050, delimiter='\t', skiprows=1)[:, 5:]
    assert power_column_offset(raw) == raw.shape[1]//2


def test_power_offset_detected_for_irregular_file():
    raw = np.loadtxt(F7_025, delimiter='\t', skiprows=1)[:, 5:]
    assert raw.shape[1] == 647
    assert power_column_offset(raw) == 326       # half would be 323


def test_bins_agree_between_bin_widths():
    """The 0.5 um^-1 file of the same simulation must equal pairs of 0.25 um^-1 bins
    (to within the far-field discretisation) only with the detected offset."""
    def mismatch(split):
        q = read_farfield(F7_025, norm=None, split=split)
        h = read_farfield(F7_050, norm=None)
        errs = []
        for e in h.energies[::10]:
            pq, ph = q.at_energy(e)[:, 2], h.at_energy(e)[:, 2]
            m = min(len(ph), len(pq)//2)
            pairs = pq[0:2*m:2] + pq[1:2*m:2]
            errs.append(np.abs(pairs - ph[:m]).sum()/np.abs(ph[:m]).sum())
        return np.median(errs)
    assert mismatch('auto') < 0.2
    assert mismatch('half') > 0.4


def test_mode_library_segments():
    lib = ModeLibrary(FIGURES/'fig4'/'500nmSi-highP.dat')
    assert lib.thickness_um == 0.5
    m0 = lib.mode(0)
    assert m0.shape[1] == 3 and len(m0) > 100
    assert np.all(np.diff(m0[:, 0]) > 0)          # wavelength increases within a mode


def test_material_interpolation():
    si = load_nk(FIGURES/'fig4'/'siliconR.txt', FIGURES/'fig4'/'siliconI.txt')
    assert 3.5 < si.n_nm(1000.) < 3.7
    assert 0.005 < si.k_nm(700.) < 0.02
