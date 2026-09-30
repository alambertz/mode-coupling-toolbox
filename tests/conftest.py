import importlib.util
import os
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
os.environ.setdefault('MCT_USETEX', '0')

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT/'figures'
REFERENCE = ROOT/'reference'


def load_figure_module(n):
    """Import figures/figN/make_figN.py as a module."""
    p = FIGURES/f'fig{n}'/f'make_fig{n}.py'
    spec = importlib.util.spec_from_file_location(f'make_fig{n}', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_reference(name):
    return np.genfromtxt(REFERENCE/name, delimiter=',', names=True)


def shifted_to_published(energies, ref_energies):
    """Index into `energies` of the values the published power fractions show at `ref_energies`.

    The published extract_k_array labelled the sum of energy i with energy i+1
    (and skipped zero sums), so published(E_{i+1}) = ours(E_i).
    """
    idx = {e: i for i, e in enumerate(energies)}
    return np.array([idx[e] - 1 for e in ref_energies])


@pytest.fixture(scope='session')
def fig_module():
    return load_figure_module
