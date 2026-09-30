#!/usr/bin/env python3
"""Interactive explorer for the Fig. 7 analysis (tkinter).

    python gui.py                    # window with heatmap / momentum slice / energy slice / coupling view
    python gui.py --test [file]      # headless: runs the analysis, saves the four panels as PNG

Settings start from config.toml in this folder; any far-field .dat file can be analysed.
Panels are rendered with matplotlib mathtext (no LaTeX needed).
"""
import copy
import os
import sys
import threading
import traceback
from pathlib import Path

import matplotlib

HERE = Path(__file__).resolve().parent
HEADLESS = '--test' in sys.argv
if not HEADLESS:
    matplotlib.use('TkAgg')
from matplotlib.figure import Figure

sys.path.insert(0, str(HERE))
import make_fig7 as f7                            # noqa: E402
from mctoolbox.config import load_config         # noqa: E402
from mctoolbox.plotting.style import apply_style, colorbar  # noqa: E402

PANELS = ('heatmap', 'momentum slice', 'energy slice', 'coupling view')

# (label, config key path, type)
FIELDS = [
    ('Data file', ('farfield',), str),
    ('Normalisation (R or T)', ('norm',), str),
    ('Max energy (eV)', ('max_energy',), float),
    ('Energy slices (eV)', ('slice_energies',), list),
    ('Momentum slices (um^-1)', ('slice_momenta',), list),
    ('k-fit prominence', ('kfit', 'prominence'), float),
    ('k-fit max FWHM (um^-1)', ('kfit', 'max_fwhm'), float),
    ('k-fit max center variance', ('kfit', 'max_center_var'), float),
    ('E-fit prominence', ('efit', 'prominence'), float),
    ('E-fit max gamma_e (eV)', ('efit', 'max_gamma_e'), float),
    ('E-fit max center variance', ('efit', 'max_center_var'), float),
    ('E-fit N', ('efit', 'N'), float),
    ('E-fit branch (bulk-bound/over/under)', ('efit', 'branch'), str),
    ('Colorbar max (%)', ('plot', 'vmax'), float),
]


def _get(cfg, keys):
    for k in keys:
        cfg = cfg[k]
    return cfg


def _set(cfg, keys, value):
    for k in keys[:-1]:
        cfg = cfg[k]
    cfg[keys[-1]] = value


def _fmt(v):
    return ', '.join(f'{x:g}' for x in v) if isinstance(v, list) else str(v)


def _parse(text, typ):
    text = text.strip()
    if typ is list:
        return [float(x) for x in text.replace(',', ' ').split()]
    return typ(text)


class _Var:
    """Stand-in for a tk entry in headless mode."""

    def __init__(self, v):
        self.v = v

    def get(self):
        return self.v


class ToolboxGUI:
    def __init__(self, root=None, data_file=None):
        self.root = root
        self.base_cfg = load_config(HERE)
        if data_file:
            self.base_cfg['farfield'] = data_file
        self.result = None
        self.cfg = None
        self.panel = 'heatmap'
        apply_style(usetex=False, fontsize=11)
        if root is None:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            self.fig = Figure(figsize=(8.5, 6.5), dpi=100)
            self.canvas = FigureCanvasAgg(self.fig)
            self.entries = {label: _Var(_fmt(_get(self.base_cfg, keys))) for label, keys, _ in FIELDS}
        else:
            self._build()

    def _build(self):
        import tkinter as tk
        from tkinter import filedialog, ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        root = self.root
        root.title('Mode-coupling toolbox: GMR analysis')
        root.geometry('1320x820')
        self.status_var = tk.StringVar(value='Press "Run analysis".')
        ttk.Label(root, textvariable=self.status_var, anchor='w', relief='sunken').pack(side='bottom', fill='x')
        top = ttk.Frame(root)
        top.pack(fill='both', expand=True)
        left = ttk.Frame(top)
        left.pack(side='left', fill='both', expand=True)
        bar = ttk.Frame(left)
        bar.pack(fill='x')
        for name in PANELS:
            ttk.Button(bar, text=name.title(), command=lambda n=name: self.select(n)).pack(side='left', padx=3, pady=2)
        self.fig = Figure(figsize=(8.5, 6.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        NavigationToolbar2Tk(self.canvas, left).update()

        right = ttk.Frame(top)
        right.pack(side='right', fill='y', padx=8)
        self.entries = {}
        dats = sorted(p.name for p in HERE.glob('*.dat'))
        for label, keys, _ in FIELDS:
            row = ttk.Frame(right)
            row.pack(fill='x', pady=3)
            ttk.Label(row, text=label, width=32, anchor='w').pack(side='left')
            e = ttk.Combobox(row, values=dats, width=40) if keys == ('farfield',) else ttk.Entry(row, width=22)
            e.insert(0, _fmt(_get(self.base_cfg, keys)))
            e.pack(side='left', fill='x', expand=True)
            self.entries[label] = e

        def browse():
            p = filedialog.askopenfilename(initialdir=HERE, filetypes=[('far-field output', '*.dat')])
            if p:
                self.entries['Data file'].delete(0, 'end')
                self.entries['Data file'].insert(0, p)
        ttk.Button(right, text='Browse data file...', command=browse).pack(fill='x', pady=4)
        self.run_btn = ttk.Button(right, text='Run analysis', command=self.on_run)
        self.run_btn.pack(side='bottom', fill='x', pady=10)

    def settings(self):
        cfg = copy.deepcopy(self.base_cfg)
        for label, keys, typ in FIELDS:
            _set(cfg, keys, _parse(self.entries[label].get(), typ))
        cfg['norm'] = cfg['norm'].upper()
        return cfg

    def status(self, text):
        if self.root is None:
            print('STATUS:', text)
        else:
            self.status_var.set(text)

    def analyse(self):
        cfg = self.settings()
        return cfg, f7.compute(cfg)

    def on_run(self):
        try:
            self.settings()
        except (ValueError, KeyError) as e:
            self.status(f'Invalid setting: {e}')
            return
        self.run_btn.configure(state='disabled')
        self.status('Fitting...')

        def work():
            try:
                cfg, r = self.analyse()
                self.root.after(0, lambda: self._done(cfg, r))
            except Exception:
                msg = traceback.format_exc().strip().splitlines()[-1]
                self.root.after(0, lambda: self._failed(msg))
        threading.Thread(target=work, daemon=True).start()

    def _done(self, cfg, r):
        self.cfg, self.result = cfg, r
        self.run_btn.configure(state='normal')
        t = r['fits'].espace_table()
        self.status(f"{len(r['fits'].kspace_table())} momentum-domain fits, {len(t)} energy-domain fits.")
        self.draw()

    def _failed(self, msg):
        self.run_btn.configure(state='normal')
        self.status(f'Run failed: {msg}')

    def select(self, name):
        self.panel = name
        self.draw()

    def draw(self):
        if self.result is None:
            self.status('No results yet: press "Run analysis".')
            return
        self.fig.clear()
        ax = self.fig.add_axes([0.10, 0.12, 0.84, 0.76])
        cfg, r = self.cfg, self.result
        if self.panel == 'heatmap':
            im = f7.heatmap_panel(ax, r, cfg)
            colorbar(self.fig, ax, im, (.025, .93, .4, .05), vmax=cfg['plot']['vmax'])
        elif self.panel == 'momentum slice':
            f7.momentum_slice_panel(ax, r, cfg)
        elif self.panel == 'energy slice':
            f7.energy_slice_panel(ax, r, cfg)
        else:
            f7.coupling_panel(self.fig, ax, r, cfg)
        self.canvas.draw_idle()


def main():
    argv = sys.argv[1:]
    if HEADLESS:
        i = argv.index('--test')
        data = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith('-') else None
        app = ToolboxGUI(data_file=data)
        app.cfg, app.result = app.analyse()
        out = Path(os.environ.get('TOOLBOX_TEST_OUT', HERE/'gui_test'))
        out.mkdir(exist_ok=True)
        for name in PANELS:
            app.panel = name
            app.draw()
            app.fig.savefig(out/(name.replace(' ', '_') + '.png'), dpi=100)
        print('TEST OK:', out)
        return
    import tkinter as tk
    root = tk.Tk()
    ToolboxGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
