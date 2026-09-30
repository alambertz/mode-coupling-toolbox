"""The code cells of examples/minimal_example.ipynb run without errors."""
import json
import os

import matplotlib.pyplot as plt
from conftest import ROOT


def test_minimal_example_notebook(monkeypatch):
    nb = json.loads((ROOT/'examples'/'minimal_example.ipynb').read_text())
    monkeypatch.chdir(ROOT/'examples')
    ns = {}
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            exec(compile(''.join(cell['source']), 'minimal_example.ipynb', 'exec'), ns)
    plt.close('all')
    assert len(ns['t']) > 10
