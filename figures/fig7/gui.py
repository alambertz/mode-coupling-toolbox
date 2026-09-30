#!/usr/bin/env python3
"""Interactive explorer for the GMR analysis of Fig. 7 (tkinter).

    python gui.py                    # window: heatmap / momentum slice / energy slice / coupling panels
    python gui.py --test [file]      # headless: runs the analysis and saves the four panels as PNG

Settings start from config.toml in this folder and can be saved to / loaded from other
config files. Any far-field .dat file written by farfield_power_analysis.lsf can be analysed.
Click in the heatmap to move the energy and momentum slices. Panels use matplotlib mathtext,
so no LaTeX installation is needed.
"""
import copy
import os
import sys
import threading
import traceback
from pathlib import Path

import matplotlib
import numpy as np

HERE = Path(__file__).resolve().parent
from matplotlib.figure import Figure

sys.path.insert(0, str(HERE))
import make_fig7 as f7                                        # noqa: E402
from mctoolbox.config import dump_config, load_config          # noqa: E402
from mctoolbox.plotting.style import apply_style, colorbar      # noqa: E402

PANELS = ('heatmap', 'momentum slice', 'energy slice', 'coupling')
PANEL_TITLES = {'heatmap': 'Heatmap P(E, k)', 'momentum slice': 'Slice P(k)',
                'energy slice': 'Slice P(E)', 'coupling': 'Coupling γe/γi'}

# (group title, hint, [(label, config keys, type), ...])
GROUPS = [
    ('Data', 'Far-field summary file of farfield_power_analysis.lsf.', [
        ('Far-field file', ('farfield',), str),
        ('Normalisation: R = (1−R) power, T = transmitted', ('norm',), str),
        ('Max photon energy (eV)', ('max_energy',), float),
    ]),
    ('Slices', 'Comma-separated values; click in the heatmap to set both.', [
        ('Photon energies (eV)', ('slice_energies',), list),
        ('In-plane momenta (µm⁻¹)', ('slice_momenta',), list),
    ]),
    ('Momentum-domain fit: Lorentzians in P(k)', 'Peaks beyond the air light line.', [
        ('Min. prominence (fraction of P)', ('kfit', 'prominence'), float),
        ('Max. FWHM (µm⁻¹)', ('kfit', 'max_fwhm'), float),
        ('Max. variance of center (µm⁻²)', ('kfit', 'max_center_var'), float),
    ]),
    ('Energy-domain fit: Eq. HH in P(E)', 'Branch: bulk-bound, over or under (N = 1 is symmetric).', [
        ('Min. prominence (fraction of P)', ('efit', 'prominence'), float),
        ('Max. γe (eV)', ('efit', 'max_gamma_e'), float),
        ('Max. variance of center (eV²)', ('efit', 'max_center_var'), float),
        ('Free-space channels N', ('efit', 'N'), float),
        ('Coupling branch', ('efit', 'branch'), str),
    ]),
    ('Display', '', [
        ('Colorbar maximum (%)', ('plot', 'vmax'), float),
    ]),
]
FIELDS = [f for _, _, fields in GROUPS for f in fields]
CHOICES = {('norm',): ['R', 'T'], ('efit', 'branch'): ['bulk-bound', 'over', 'under']}


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

    def delete(self, *_):
        self.v = ''

    def insert(self, _, v):
        self.v = v


class ToolboxGUI:
    def __init__(self, root=None, data_file=None):
        self.root = root
        self.base_cfg = load_config(HERE)
        if data_file:
            self.base_cfg['farfield'] = data_file
        self.result, self.cfg = None, None
        self.panel = 'heatmap'
        self.heat_ax = None
        apply_style(usetex=False, fontsize=11)
        if root is None:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            self.fig = Figure(figsize=(8.5, 6.5), dpi=100)
            self.canvas = FigureCanvasAgg(self.fig)
            self.toolbar = None
            self.entries = {label: _Var(_fmt(_get(self.base_cfg, keys))) for label, keys, _ in FIELDS}
        else:
            self._build()

    # ------------------------------------------------------------------ window
    def _build(self):
        import tkinter as tk
        from tkinter import font as tkfont
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        root = self.root
        root.title('Mode-coupling toolbox — guided-mode resonance analysis')
        # UI scale for high-resolution screens (many X setups report 96 dpi regardless);
        # override with the environment variable MCT_GUI_SCALE, e.g. 1.5
        self.scale = float(os.environ.get('MCT_GUI_SCALE', 0)) or max(1.0, round(root.winfo_screenwidth()/1920*4)/4)
        root.tk.call('tk', 'scaling', self.scale*96/72)
        for name in ('TkDefaultFont', 'TkTextFont', 'TkHeadingFont', 'TkMenuFont'):
            try:
                f = tkfont.nametofont(name)
                if f.cget('size') < 0:                        # pixel sizes do not follow tk scaling
                    f.configure(size=int(round(f.cget('size')*self.scale)))
            except tk.TclError:
                pass
        root.geometry(f'{int(1400*self.scale)}x{int(860*self.scale)}')
        root.minsize(int(1000*self.scale), int(600*self.scale))
        style = ttk.Style(root)
        if 'clam' in style.theme_names():
            style.theme_use('clam')
        style.configure('Panel.TButton', padding=(10, 4))
        style.configure('Active.TButton', padding=(10, 4), font=tkfont.Font(font='TkDefaultFont', weight='bold'))
        style.configure('Hint.TLabel', foreground='#555555')

        self.status_var = tk.StringVar(value='Press "Run analysis" to fit the data.')
        ttk.Label(root, textvariable=self.status_var, anchor='w', relief='sunken', padding=(8, 3)).pack(
            side='bottom', fill='x')
        pane = ttk.PanedWindow(root, orient='horizontal')
        pane.pack(fill='both', expand=True)

        left = ttk.Frame(pane)
        pane.add(left, weight=3)
        bar = ttk.Frame(left)
        bar.pack(fill='x', pady=(4, 0))
        self.panel_buttons = {}
        for name in PANELS:
            b = ttk.Button(bar, text=PANEL_TITLES[name], style='Panel.TButton', command=lambda n=name: self.select(n))
            b.pack(side='left', padx=3)
            self.panel_buttons[name] = b
        self.fig = Figure(figsize=(8.5, 6.5), dpi=96*self.scale)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, left)
        self.toolbar.update()
        self.canvas.mpl_connect('button_press_event', self.on_click)

        right = ttk.Frame(pane, padding=(8, 4))
        pane.add(right, weight=1)
        self.entries = {}
        dats = sorted(p.name for p in HERE.glob('*.dat'))
        for title, hint, fields in GROUPS:
            box = ttk.LabelFrame(right, text=title, padding=(8, 4))
            box.pack(fill='x', pady=4)
            if hint:
                ttk.Label(box, text=hint, style='Hint.TLabel', wraplength=int(380*self.scale)).pack(anchor='w', pady=(0, 3))
            for label, keys, _ in fields:
                row = ttk.Frame(box)
                row.pack(fill='x', pady=2)
                ttk.Label(row, text=label, anchor='w').pack(side='top', anchor='w')
                if keys == ('farfield',):
                    e = ttk.Combobox(row, values=dats)
                elif keys in CHOICES:
                    e = ttk.Combobox(row, values=CHOICES[keys], state='normal', width=14)
                else:
                    e = ttk.Entry(row, width=16)
                e.insert(0, _fmt(_get(self.base_cfg, keys)))
                e.pack(side='top', fill='x' if keys == ('farfield',) else None, anchor='w')
                self.entries[label] = e
        acts = ttk.LabelFrame(right, text='Actions', padding=(8, 4))
        acts.pack(fill='x', pady=4)
        self.run_btn = ttk.Button(acts, text='Run analysis', command=self.on_run)
        self.run_btn.pack(fill='x', pady=(0, 6))
        grid = ttk.Frame(acts)
        grid.pack(fill='x')
        for i, (text, cmd) in enumerate([('Browse data file…', self._ask_data_file),
                                         ('Load settings…', self._ask_load_settings),
                                         ('Save settings…', self._ask_save_settings),
                                         ('Save fit tables…', self._ask_export_tables),
                                         ('Save panel…', self._ask_export_panel),
                                         ('Save Fig. 7…', self._ask_export_figure)]):
            ttk.Button(grid, text=text, command=cmd).grid(row=i//2, column=i % 2, sticky='ew', padx=2, pady=2)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        self._highlight()

    def _highlight(self):
        for name, b in getattr(self, 'panel_buttons', {}).items():
            b.configure(style='Active.TButton' if name == self.panel else 'Panel.TButton')

    # ---------------------------------------------------------------- settings
    def settings(self):
        """Current field values merged into the base configuration."""
        cfg = copy.deepcopy(self.base_cfg)
        for label, keys, typ in FIELDS:
            _set(cfg, keys, _parse(self.entries[label].get(), typ))
        cfg['norm'] = cfg['norm'].upper()
        if cfg['efit']['branch'] not in CHOICES[('efit', 'branch')]:
            raise ValueError(f"unknown branch {cfg['efit']['branch']!r}")
        return cfg

    def _fill(self, cfg):
        for label, keys, _ in FIELDS:
            self.entries[label].delete(0, 'end')
            self.entries[label].insert(0, _fmt(_get(cfg, keys)))

    def save_settings(self, file):
        cfg = self.settings()
        far = Path(cfg['farfield'])
        far = far if far.is_absolute() else (Path(cfg['dir'])/far).resolve()
        target = Path(file).resolve().parent
        cfg['farfield'] = far.name if far.parent == target else str(far)   # valid wherever the file is loaded
        dump_config(cfg, file)
        self.status(f'Settings saved to {file}')

    def load_settings(self, file):
        cfg = load_config(Path(file).parent) if Path(file).name == 'config.toml' else None
        if cfg is None:
            import tomllib
            with open(file, 'rb') as fh:
                cfg = tomllib.load(fh)
            cfg['dir'] = Path(file).parent
        far = Path(cfg['farfield'])
        if not far.is_absolute():               # keep data paths valid relative to this folder
            cfg['farfield'] = str((Path(cfg['dir'])/far).resolve())
        merged = copy.deepcopy(self.base_cfg)
        for k, v in cfg.items():
            if k != 'dir':
                merged[k] = v
        self.base_cfg = merged
        self._fill(merged)
        self.status(f'Settings loaded from {file}')

    # --------------------------------------------------------------- analysis
    def status(self, text):
        if self.root is None:
            print('STATUS:', text)
        else:
            self.status_var.set(text)

    def analyse(self):
        cfg = self.settings()
        return cfg, f7.compute(cfg)

    def summary(self):
        t = self.result['fits'].espace_table()
        nk = len(self.result['fits'].kspace_table())
        if len(t) == 0:
            return f'{nk} momentum-domain fits, no energy-domain fits.'
        amb = int((t[:, 6] == 0).sum())
        return (f'{nk} momentum-domain fits, {len(t)} energy-domain fits; γe/γi median {np.median(t[:, 7]):.0f} '
                f'(10–90 %: {np.percentile(t[:, 7], 10):.0f}–{np.percentile(t[:, 7], 90):.0f}); '
                f'{amb} with ambiguous branch.')

    def on_run(self):
        try:
            self.settings()
        except (ValueError, KeyError) as e:
            self.status(f'Invalid setting: {e}')
            return
        self.run_btn.configure(state='disabled')
        self.status('Fitting…')

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
        self.status(self.summary())
        self.draw()

    def _failed(self, msg):
        self.run_btn.configure(state='normal')
        self.status(f'Run failed: {msg}')

    # ----------------------------------------------------------------- slices
    def set_slice(self, energy, k):
        """Move the energy and momentum slices (k snapped to the fit grid); no refit needed."""
        grid = np.arange(*self.cfg['efit']['k_grid'])
        k = float(grid[np.argmin(np.abs(grid - k))])
        energy = round(float(energy), 4)
        for key, value in (('slice_energies', [energy]), ('slice_momenta', [k])):
            self.cfg[key] = value
            label = next(lbl for lbl, keys, _ in FIELDS if keys == (key,))
            self.entries[label].delete(0, 'end')
            self.entries[label].insert(0, _fmt(value))
        self.status(f'Slices: E = {energy:.4f} eV, k = {k:g} µm⁻¹. {self.summary()}')
        self.draw()

    def on_click(self, event):
        if (self.result is None or self.panel != 'heatmap' or event.inaxes is not self.heat_ax
                or event.button != 1 or event.xdata is None):
            return
        if self.toolbar is not None and str(getattr(self.toolbar, 'mode', '')):
            return                              # zoom / pan active
        self.set_slice(event.xdata, event.ydata)

    # ---------------------------------------------------------------- drawing
    def select(self, name):
        self.panel = name
        self._highlight()
        self.draw()

    def draw_panel(self, fig, name):
        fig.clear()
        ax = fig.add_axes([0.11, 0.12, 0.84, 0.76])
        cfg, r = self.cfg, self.result
        if name == 'heatmap':
            im = f7.heatmap_panel(ax, r, cfg)
            colorbar(fig, ax, im, (.025, .93, .4, .05), vmax=cfg['plot']['vmax'])
            self.heat_ax = ax
        elif name == 'momentum slice':
            f7.momentum_slice_panel(ax, r, cfg)
            ax.set_xlim(left=-0.01)
        elif name == 'energy slice':
            f7.energy_slice_panel(ax, r, cfg)
            ax.set_ylim(bottom=-0.01)
        else:
            f7.coupling_panel(fig, ax, r, cfg)
        ax.set_title(PANEL_TITLES[name], loc='right', fontsize=11)
        return ax

    def draw(self):
        if self.result is None:
            self.status('No results yet: press "Run analysis".')
            return
        self.draw_panel(self.fig, self.panel)
        self.canvas.draw_idle()

    # ---------------------------------------------------------------- exports
    def export_tables(self, folder):
        folder = Path(folder)
        f7.write_tables(self.result, folder)
        self.status(f'Fit tables written to {folder}/fig7_kspace_fits.csv and fig7_espace_fits.csv')

    def export_panel(self, file):
        fig = Figure(figsize=(6.5, 5), dpi=300)
        self.draw_panel(fig, self.panel)
        fig.savefig(file, bbox_inches='tight')
        self.status(f'Panel saved to {file}')

    def export_figure(self, file):
        base = Path(file).with_suffix('')
        f7.plot(self.cfg, self.result, out=base)
        apply_style(usetex=False, fontsize=11)
        self.status(f'Fig. 7 saved to {base}.pdf / .png')

    # ---------------------------------------------------------------- dialogs
    def _need_result(self):
        if self.result is None:
            self.status('Run the analysis first.')
            return False
        return True

    def _ask_data_file(self):
        from tkinter import filedialog
        p = filedialog.askopenfilename(initialdir=HERE, filetypes=[('far-field output', '*.dat'), ('all', '*')])
        if p:
            label = FIELDS[0][0]
            self.entries[label].delete(0, 'end')
            self.entries[label].insert(0, p)

    def _ask_load_settings(self):
        from tkinter import filedialog
        p = filedialog.askopenfilename(initialdir=HERE, filetypes=[('settings', '*.toml')])
        if p:
            self._guard(self.load_settings, p)

    def _ask_save_settings(self):
        from tkinter import filedialog
        p = filedialog.asksaveasfilename(initialdir=HERE, defaultextension='.toml', filetypes=[('settings', '*.toml')])
        if p:
            self._guard(self.save_settings, p)

    def _ask_export_tables(self):
        from tkinter import filedialog
        if self._need_result():
            p = filedialog.askdirectory(initialdir=HERE)
            if p:
                self._guard(self.export_tables, p)

    def _ask_export_panel(self):
        from tkinter import filedialog
        if self._need_result():
            p = filedialog.asksaveasfilename(initialdir=HERE, defaultextension='.pdf',
                                             filetypes=[('PDF', '*.pdf'), ('PNG', '*.png')])
            if p:
                self._guard(self.export_panel, p)

    def _ask_export_figure(self):
        from tkinter import filedialog
        if self._need_result():
            p = filedialog.asksaveasfilename(initialdir=HERE, initialfile='fig_7', defaultextension='.pdf',
                                             filetypes=[('PDF and PNG', '*.pdf')])
            if p:
                self._guard(self.export_figure, p)

    def _guard(self, fn, *args):
        try:
            fn(*args)
        except Exception as e:
            self.status(f'{fn.__name__} failed: {e}')


def main():
    argv = sys.argv[1:]
    if '--test' in argv:
        i = argv.index('--test')
        data = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith('-') else None
        app = ToolboxGUI(data_file=data)
        app.cfg, app.result = app.analyse()
        print(app.summary())
        out = Path(os.environ.get('TOOLBOX_TEST_OUT', HERE/'gui_test'))
        out.mkdir(exist_ok=True)
        for name in PANELS:
            app.panel = name
            app.draw()
            app.fig.savefig(out/(name.replace(' ', '_') + '.png'), dpi=100)
        print('TEST OK:', out)
        return
    matplotlib.use('TkAgg')
    import tkinter as tk
    root = tk.Tk()
    ToolboxGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
