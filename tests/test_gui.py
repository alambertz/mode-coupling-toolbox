"""GUI logic without a display: settings round trip, click-to-slice and exports."""
import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import FIGURES

from mctoolbox.config import dump_config, load_config


@pytest.fixture(scope='module')
def app():
    sys.path.insert(0, str(FIGURES/'fig7'))
    spec = importlib.util.spec_from_file_location('fig7_gui', FIGURES/'fig7'/'gui.py')
    gui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gui)
    a = gui.ToolboxGUI()
    a.cfg, a.result = a.analyse()
    a.gui = gui
    return a


def test_config_round_trip(tmp_path):
    cfg = load_config(FIGURES/'fig7')
    dump_config(cfg, tmp_path/'config.toml')
    again = load_config(tmp_path)
    cfg.pop('dir'), again.pop('dir')
    assert cfg == again


def test_settings_save_and_load(app, tmp_path):
    label = next(lbl for lbl, keys, _ in app.gui.FIELDS if keys == ('efit', 'max_gamma_e'))
    app.entries[label].delete(0, 'end')
    app.entries[label].insert(0, '0.02')
    app.save_settings(tmp_path/'mine.toml')
    app.entries[label].delete(0, 'end')
    app.entries[label].insert(0, '0.05')
    app.load_settings(tmp_path/'mine.toml')
    assert app.settings()['efit']['max_gamma_e'] == 0.02
    assert (FIGURES/'fig7'/app.base_cfg['farfield']).exists()


def test_click_sets_both_slices(app):
    app.panel = 'heatmap'
    app.draw()
    event = SimpleNamespace(inaxes=app.heat_ax, button=1, xdata=1.2612, ydata=18.63)
    app.on_click(event)
    assert app.cfg['slice_energies'] == [1.2612]
    assert app.cfg['slice_momenta'] == [18.75]          # snapped to the 0.25 um^-1 fit grid
    s = app.settings()
    assert s['slice_momenta'] == [18.75]


def test_click_ignored_outside_heatmap(app):
    app.panel = 'coupling'
    app.draw()
    before = list(app.cfg['slice_momenta'])
    app.on_click(SimpleNamespace(inaxes=None, button=1, xdata=1.3, ydata=12.0))
    assert app.cfg['slice_momenta'] == before


def test_exports(app, tmp_path):
    app.export_tables(tmp_path)
    t = np.genfromtxt(tmp_path/'fig7_espace_fits.csv', delimiter=',', names=True)
    assert 'v_group_m_per_s' in t.dtype.names and len(t) > 10
    app.panel = 'energy slice'
    app.export_panel(tmp_path/'panel.pdf')
    app.export_figure(tmp_path/'fig_7.pdf')
    assert (tmp_path/'panel.pdf').stat().st_size > 5000
    assert (tmp_path/'fig_7.png').exists()
