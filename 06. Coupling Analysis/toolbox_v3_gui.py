#!/usr/bin/env python3
"""
toolbox_v3_gui.py -- tkinter + matplotlib GUI for the MC7 class (Fig 7 toolbox), v3.

v3 changes vs toolbox_v2_gui.py (so far; all in plot_fig7_class.py, no GUI-side changes):
- legend formatting in both slice panels: each entry's line sample and label takes
  the color of its ax.plot curve, auto-placed with loc='best'.
- energy slice (plot_AbsVpe): one curve per requested momentum — first keeps its color,
  extras cycle the existing PALETTE; fit overlay snaps to the nearest k-grid key.

v2 changes vs toolbox_v1_gui.py:
- new "max_energy (eV)" setting field (default 1.5): filters data lines during the fit
  and sets the upper bound of every photon-energy axis; the lower bound is the lowest
  energy present in the data file (E_min, computed in import_full_data).
- energy axes of heatmap / energy slice / coupling view now span [E_min, max_energy]
  (momentum slice has no energy axis — unchanged).

Layout
------
left : one subpanel at a time in a live matplotlib canvas, with four buttons above
       to choose the panel type: heatmap / momentum slice / energy slice / coupling view.
right: labels + text input fields for every MC7 setting, and a "Run analysis" button
       that re-imports the data file and refits everything with the current settings.

Panel switching re-renders from the cached fit results (no refit). "Run analysis"
recomputes all fits in a background thread; the canvas and matplotlib navigation
toolbar stay live.

Run (from the mc-7 folder):
    ../mc-4/venv/bin/python toolbox_v3_gui.py                 # GUI (needs tkinter + display)
    ../mc-4/venv/bin/python toolbox_v3_gui.py --test [file]   # headless smoke test; saves
                                                              # all four panels to /tmp/toolbox_test/

Notes
-----
* Uses text.usetex (Arial) via general_settings.py -> needs a LaTeX install.
* The class reads siliconR.txt / siliconI.txt and *.dat relative to the cwd, so this
  script chdir()s into its own folder at startup.
"""

import os
import sys
import threading
import traceback
from glob import glob as gl

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HEADLESS = '--test' in sys.argv

import matplotlib
if not HEADLESS:
    matplotlib.use('TkAgg')          # before pyplot is imported anywhere
from matplotlib.figure import Figure

sys.path.insert(0, HERE)
from plot_fig7_class import MC7      # noqa: E402  (also loads general_settings rcParams)

if HEADLESS:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
else:
    import tkinter as tk
    from tkinter import ttk, filedialog
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

PANELS = ('heatmap', 'momentum slice', 'energy slice', 'coupling view')

# settings fields (label, default string) -- defaults match MC7.initialize()
FIELDS = [
    ('Data file', ''),
    ('slice_energies (eV)', '1.247434'),
    ('momenta (um^-1)', '20.5'),
    ('max_energy (eV)', '1.5'),
    ('N (auto or value)', 'auto'),
    ('promincence', '3e-3'),
    ('fhm_thres', '2.'),
    ('cov_thres_espace', '1e-4'),
    ('cov_thres_kspace', '1e-2'),
    ('vmax', '2.'),
    ('norm (R or T)', 'R'),
]


class _StrVal:
    """StringVar stand-in so headless mode shares the entry API."""

    def __init__(self, value=''):
        self._v = str(value)

    def get(self):
        return self._v

    def set(self, v):
        self._v = str(v)


class ToolboxGUI:
    """tkinter front-end for one MC7 instance (live single-panel display)."""

    def __init__(self, root=None, test_file=None):
        self.headless = root is None
        self.root = root
        self.mc7 = MC7()
        self.results = None          # (data, mdic, edic, peakd) after a successful run
        self.panel = 'heatmap'
        self.running = False

        if self.headless:
            self.fig = Figure(figsize=(8.5, 6.5), dpi=100)
            self.canvas = FigureCanvasAgg(self.fig)
            self.entries = {label: _StrVal(default) for label, default in FIELDS}
            dats = sorted(gl(os.path.join(HERE, '*.dat')))
            self.entries['Data file'].set(test_file or (dats[0] if dats else ''))
        else:
            self._build_tk()

    # ───────────────────────── UI construction ─────────────────────────
    def _dat_files(self):
        return sorted(gl(os.path.join(HERE, '*.dat')))

    def _browse_file(self):
        p = filedialog.askopenfilename(initialdir=HERE, title='Select data file',
                                       filetypes=[('FSP output', '*.dat'), ('All files', '*')])
        if p:
            self.entries['Data file'].delete(0, 'end')
            self.entries['Data file'].insert(0, os.path.basename(p))
    def _build_tk(self):
        root = self.root
        root.title('MC7 toolbox v3 — mode coupling Fig 7')
        root.geometry('1320x820')

        style = ttk.Style(root)
        for theme in ('clam', 'default'):
            try:
                style.theme_use(theme)
                break
            except tk.TclError:
                continue
        style.configure('Panel.TButton', padding=5)
        style.configure('PanelActive.TButton', padding=5, font=('TkDefaultFont', 9, 'bold'))

        # status bar (bottom of window)
        self.status_var = tk.StringVar(value='Ready — pick a data file, then press "Run analysis".')
        ttk.Label(root, textvariable=self.status_var, anchor='w', relief='sunken',
                  padding=(6, 3)).pack(side='bottom', fill='x')

        top = ttk.Frame(root)
        top.pack(side='top', fill='both', expand=True)

        # ── left: panel buttons + live canvas + navigation toolbar ──
        left = ttk.Frame(top)
        left.pack(side='left', fill='both', expand=True)

        btns = ttk.Frame(left)
        btns.pack(side='top', fill='x')
        self.panel_btns = {}
        for name in PANELS:
            b = ttk.Button(btns, text=name.title(), style='Panel.TButton',
                           command=lambda n=name: self.select_panel(n))
            b.pack(side='left', padx=3, pady=2)
            self.panel_btns[name] = b

        self.fig = Figure(figsize=(8.5, 6.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, left)
        self.toolbar.update()

        # ── right: settings fields (scrollable) + run button ──
        right = ttk.Frame(top)
        right.pack(side='right', fill='y')

        cont = ttk.Frame(right)
        cont.pack(side='top', fill='both', expand=True)
        cv = tk.Canvas(cont, highlightthickness=0, width=380)
        sb = ttk.Scrollbar(cont, orient='vertical', command=cv.yview)
        inner = ttk.Frame(cv)
        inner.bind('<Configure>', lambda e: cv.configure(scrollregion=cv.bbox('all')))
        win = cv.create_window((0, 0), window=inner, anchor='nw')
        cv.configure(yscrollcommand=sb.set)
        cv.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        cv.bind('<Configure>', lambda e: cv.itemconfigure(win, width=e.width))

        self.entries = {}
        dats = self._dat_files()
        for label, default in FIELDS:
            row = ttk.Frame(inner)
            row.pack(fill='x', padx=8, pady=4)
            ttk.Label(row, text=label, width=21, anchor='w').pack(side='left')
            if label == 'Data file':
                e = ttk.Combobox(row, values=[os.path.basename(d) for d in dats])
                e.pack(side='left', fill='x', expand=True)
                ttk.Button(row, text='…', width=3, command=self._browse_file).pack(
                    side='left', padx=(4, 0))
                if default == '' and dats:
                    default = os.path.basename(dats[0])
            elif label == 'norm (R or T)':
                e = ttk.Combobox(row, values=['R', 'T'])
                e.pack(side='left', fill='x', expand=True)
            else:
                e = ttk.Entry(row)
                e.pack(side='left', fill='x', expand=True)
            e.insert(0, default)
            self.entries[label] = e

        self.run_btn = ttk.Button(right, text='Run analysis', command=self.on_run)
        self.run_btn.pack(side='bottom', fill='x', padx=8, pady=10)
    # ───────────────────────── settings ─────────────────────────
    def _settings(self):
        """Parse the entry fields into a settings dict (raises ValueError on bad input)."""
        g = lambda k: self.entries[k].get().strip()
        n = g('N (auto or value)').lower()
        return {
            'filename': g('Data file'),
            'slice_energies': MC7.parse_float_list(g('slice_energies (eV)')),
            'momenta': MC7.parse_float_list(g('momenta (um^-1)')),
            'max_energy': float(g('max_energy (eV)')),
            'Nfix': None if n in ('', 'auto', 'fit') else float(n),
            'promincence': float(g('promincence')),
            'fhm_thres': float(g('fhm_thres')),
            'cov_thres_espace': float(g('cov_thres_espace')),
            'cov_thres_kspace': float(g('cov_thres_kspace')),
            'vmax': float(g('vmax')),
            't': g('norm (R or T)').upper(),
        }

    def _resolve_file(self, name):
        return name if os.path.isabs(name) else os.path.join(HERE, name)

    # ───────────────────────── run / draw ─────────────────────────
    def on_run(self):
        if self.running:
            self.status('Run already in progress…')
            return
        try:
            st = self._settings()
        except ValueError as e:
            self.status(f'Setting parse error: {e}')
            return
        if not st['slice_energies']:
            self.status('slice_energies (eV) needs at least one value.')
            return
        if st['t'] not in ('R', 'T'):
            self.status("norm must be 'R' or 'T'.")
            return
        if st['max_energy'] <= 0:
            self.status('max_energy must be > 0 eV.')
            return
        fname = self._resolve_file(st['filename'])
        if not os.path.exists(fname):
            self.status(f'Data file not found: {st["filename"]!r}')
            return

        self.running = True
        self.run_btn.configure(state='disabled')
        self.status(f'Importing + fitting {os.path.basename(fname)} … (can take a minute)')
        self.root.update_idletasks()

        def worker():
            try:
                self.results = self._do_import(st, fname)
                self.root.after(0, self._run_done)
            except Exception:
                tb = traceback.format_exc()
                print(tb)
                self.root.after(0, lambda: self._run_failed(tb))

        threading.Thread(target=worker, daemon=True).start()

    def _do_import(self, st, fname):
        """import_full_data with the GUI settings; stray rp.plot figures stay off-screen."""
        import matplotlib.pyplot as plt
        import ramanspy as rp
        mc7 = self.mc7
        if not hasattr(mc7, 'n1ri'):            # one-time static init (material data etc.)
            mc7.initialize(interactive=False)

        old_backend = matplotlib.get_backend()
        matplotlib.use('Agg', force=True)       # rp.plot.spectra figures -> off-screen
        orig_spectra = rp.plot.spectra

        def quiet_spectra(*a, **k):              # keep the figure pile at ~1 during the fit loop
            orig_spectra(*a, **k)
            plt.close('all')

        rp.plot.spectra = quiet_spectra
        mc7.max_energy = st['max_energy']    # line filter in import_full_data + panel axes
        lam_max = np.loadtxt(fname, delimiter='\t', skiprows=1, usecols=(0,), ndmin=2)[:, 0].max()
        if st['max_energy'] < mc7.nm_to_eV(lam_max):
            raise ValueError(f'no data at/below max_energy={st["max_energy"]} eV '
                             f'(file E_min={round(mc7.nm_to_eV(lam_max), 6)} eV)')
        try:
            results = mc7.import_full_data(
                fname, norm_power=st['t'], slice_energies=st['slice_energies'],
                momenta=st['momenta'], Nfix=st['Nfix'], promincence=st['promincence'],
                fhm_thres=st['fhm_thres'], cov_thres_espace=st['cov_thres_espace'],
                cov_thres_kspace=st['cov_thres_kspace'])
        finally:
            rp.plot.spectra = orig_spectra
            plt.close('all')
            matplotlib.use(old_backend, force=True)

        data = results[0]
        if len(data) < 1:
            raise ValueError(f'no data at/below max_energy={st["max_energy"]} eV '
                             f'(file E_min={mc7.E_min} eV)')
        return results

    def _run_done(self):
        self.running = False
        self.run_btn.configure(state='normal')
        data, mdic, edic, peakd = self.results
        n_espace = sum(1 for v in edic.values() if len(v) > 0)
        self.status(f'Done — {len(data):,} rows, kspace fits at {len(mdic)} energies, '
                    f'espace fits at {n_espace} momenta. Pick a panel above.')
        self.draw_current()

    def _run_failed(self, tb):
        self.running = False
        self.run_btn.configure(state='normal')
        last = tb.strip().splitlines()[-1]
        self.status(f'Run failed: {last}')
    # ───────────────────────── panels ─────────────────────────
    def select_panel(self, name):
        self.panel = name
        for n, b in self.panel_btns.items():
            b.configure(style='PanelActive.TButton' if n == name else 'Panel.TButton')
        self.draw_current()

    def draw_current(self):
        if self.results is None:
            self.status('No results yet — press "Run analysis" first.')
            return
        try:
            self.draw_panel(self.panel)
        except KeyError as e:
            self.status(f'Panel "{self.panel}" needs a value present in the fit grid (missing key {e}).')
        except Exception as e:
            self.status(f'Render error ({self.panel}): {e}')

    def draw_panel(self, name):
        """Render one subpanel on the shared figure (live canvas)."""
        data, mdic, edic, peakd = self.results
        st = self._settings()
        self.fig.clear()
        ax = self.fig.add_axes([0.10, 0.12, 0.84, 0.76])
        if name == 'heatmap':
            ax, im = self.mc7.heatmap(ax, data, slice_energies=st['slice_energies'],
                                      momenta=st['momenta'], vmax=st['vmax'])
            self.mc7.colorbar(self.fig, ax, im, st['vmax'])
        elif name == 'momentum slice':
            self.mc7.plot_AbsVk(ax, slice_energies=st['slice_energies'], data=data, d=mdic)
        elif name == 'energy slice':
            self.mc7.plot_AbsVpe(ax, momenta=st['momenta'], data=data, d=edic)
        elif name == 'coupling view':
            self.mc7.plot_Coupling(self.fig, ax, edic, slice_energies=st['slice_energies'],
                                   momenta=st['momenta'])
        else:
            raise ValueError(f'unknown panel {name!r}')
        self.canvas.draw_idle()

    # ───────────────────────── misc ─────────────────────────
    def status(self, text):
        if self.headless:
            print('STATUS:', text)
        else:
            self.status_var.set(text)


def main():
    os.chdir(HERE)          # MC7 reads siliconR/I.txt and *.dat relative to the cwd
    argv = sys.argv[1:]

    if HEADLESS:
        i = argv.index('--test')
        test_file = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith('-') else None
        app = ToolboxGUI(root=None, test_file=test_file)
        st = app._settings()
        fname = app._resolve_file(st['filename'])
        app.results = app._do_import(st, fname)
        outdir = os.environ.get('TOOLBOX_TEST_OUT', '/tmp/toolbox_test')
        os.makedirs(outdir, exist_ok=True)
        for name in PANELS:
            app.draw_panel(name)
            p = os.path.join(outdir, name.replace(' ', '_') + '.png')
            app.fig.savefig(p, dpi=100)
            print('saved', p)
        print('TEST OK')
        return

    root = tk.Tk()
    ToolboxGUI(root=root)
    root.mainloop()


if __name__ == '__main__':
    main()
