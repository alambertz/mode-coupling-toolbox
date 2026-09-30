"""Every figure script and the GUI's headless mode run end to end (mathtext, no LaTeX)."""
import filecmp
import subprocess
import sys

import pytest
from conftest import FIGURES, load_figure_module

from mctoolbox.config import load_config


@pytest.mark.parametrize('n', [4, 5, 6, 7])
def test_figure_renders(n, tmp_path):
    mod = load_figure_module(n)
    cfg = load_config(FIGURES/f'fig{n}')
    mod.plot(cfg, mod.compute(cfg), out=tmp_path/f'fig_{n}')
    assert (tmp_path/f'fig_{n}.png').stat().st_size > 10_000
    assert (tmp_path/f'fig_{n}.pdf').exists()


def test_gui_headless(tmp_path):
    env = {'TOOLBOX_TEST_OUT': str(tmp_path), 'MCT_USETEX': '0', 'MPLBACKEND': 'Agg'}
    import os
    r = subprocess.run([sys.executable, str(FIGURES/'fig7'/'gui.py'), '--test'], capture_output=True, text=True,
                       env={**os.environ, **env}, timeout=600)
    assert r.returncode == 0, r.stderr
    assert len(list(tmp_path.glob('*.png'))) == 4


def test_shared_material_files_identical():
    """Figure folders carry their own copies of shared inputs; they must not drift apart."""
    for name in ['siliconR.txt', 'siliconI.txt', '500nmSi-highP.dat']:
        copies = sorted(FIGURES.glob(f'*/{name}'))
        assert len(copies) >= 2
        assert all(filecmp.cmp(copies[0], c, shallow=False) for c in copies[1:]), name
