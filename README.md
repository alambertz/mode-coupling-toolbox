# mode-coupling-toolbox

Post-processing pipeline for full-wave (FDTD) simulations that quantifies how light couples to the guided
modes of a semiconductor slab with an arbitrary, periodic or disordered, surface pattern. From a single
simulation it produces the energy- and momentum-resolved power distribution, the power trapped beyond the
escape cone, the absorptance per guided mode, and the internal and external loss rates of individual
guided-mode resonances (GMRs).

It accompanies the article *Quantifying light coupling to guided modes in semiconductor slabs with
arbitrary scattering patterns* by A. Lambertz, E. Alarcon-Llado and J. van de Groep (Optics Express,
in review, 2026). The state of the repository at submission is tagged `v1.0-submission`.

## Modules

| # | Module | Where |
|---|---|---|
| 1 | FDTD simulation of the 3D structure (ANSYS Lumerical) | `01. FDTD simulation/` |
| 2 | Far-field projection of the monitor fields and integration into momentum bins | `02. Far field Transform/farfield_power_analysis.lsf` |
| 3 | Energy-momentum heatmaps | `mctoolbox.io`, `mctoolbox.heatmap` |
| 4 | Power within / beyond the escape cone; absorptance per guided mode | `mctoolbox.heatmap` |
| 5 | Peak finding in momentum and energy slices | `mctoolbox.peaks` |
| 6 | Peak fitting: mode momentum, FWHM, loss rates gamma_i and gamma_e | `mctoolbox.fit` |

Modules 1-2 need a Lumerical licence. Modules 3-6 are a small Python package (numpy, scipy, matplotlib) and
run on the far-field files included here.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"
```

The figure scripts render text with LaTeX, as in the paper (TeX Live with `helvet`, `sansmath`, `cm-super`).
Without LaTeX, set `MCT_USETEX=0`.

## Quick start

`examples/minimal_example.ipynb` walks through modules 3-6 on one dataset (hyperuniform pattern, 1.4 MB):
reading, heatmap, escape-cone fractions, momentum-domain fits and coupling rates.

```python
from mctoolbox.io import read_farfield, load_nk
from mctoolbox.fit import analyse_coupling

ff = read_farfield('figures/fig7/<file>_0.5um1_full_out_v128.dat', norm='R', max_energy=1.35)
si = load_nk('figures/fig7/siliconR.txt', 'figures/fig7/siliconI.txt')
table = analyse_coupling(ff, si).espace_table()   # k, E0, amplitude, gamma_i, gamma_e, N, unique, gamma_e/gamma_i
```

## Reproducing the figures

```bash
python -m mctoolbox.figures          # Figs. 4-7, or e.g.: python -m mctoolbox.figures 7
```

Each `figures/figN/` folder is self-contained: the data it uses, a `config.toml` with every parameter, and
`make_figN.py`. Outputs (`fig_N.pdf/png`, and for Fig. 7 the fit tables `fig7_*_fits.csv`) are written into
the same folder.

`figures/fig7/gui.py` opens an interactive explorer for the GMR analysis (needs tkinter): change the fit
settings, click in the heatmap to move the slices, save the fit tables, single panels or the full figure,
and save or load the settings as a `config.toml`. `gui.py --test` runs it without a display;
`MCT_GUI_SCALE=1.5` enlarges the interface on high-resolution screens.

## Running your own simulation

1. Build the simulation with one of the scripts in `01. FDTD simulation/` (they use `lumapi`; set
   `LUMAPI_PATH` if your installation is not found automatically) or open one of the `.fsp` projects.
   `material-setup.fsp` holds the silicon material fit.
2. After the run, execute `farfield_power_analysis.lsf` in the same project. Set the monitor names, the
   momentum bin width `step_size` and the substrate material at the top of the script.
3. Read the resulting `*_full_out_v130.dat` with `mctoolbox.io.read_farfield` and continue as in the example.

## Tests

```bash
pytest
```

The tests check the file readers, recover known loss rates from synthetic resonances, run all figure
scripts, the GUI and the example notebook, and verify against tables exported from the original scripts
(`reference/`) that the package reproduces the numbers of the submitted figures exactly when run with the
submitted settings.

## Changes since the submitted version (v1.0-submission)

* The analysis code is one package instead of per-figure scripts; parameters live in `config.toml` files.
  Peak finding uses `scipy.signal.find_peaks` directly (ramanspy is no longer needed).
* **Far-field reader:** in the files with 0.25 um^-1 bins (script version v128) the power columns do not
  start at half of the row. The previous scripts paired every k value with the power of the bin three
  steps lower, shifting features by +0.75 um^-1 (Figs. 5 and 7). The reader now locates the power columns
  from the data (`mctoolbox.io.power_column_offset`); `split='half'` reproduces the old behaviour.
* **Energy-domain fit:** uses Eq. (HH) of the paper. The previous line shape had an extra factor 1/pi and
  a width twice as large. For N = 1 the line shape cannot tell over- from under-coupling; the branch is now
  chosen explicitly (`EFitSettings.branch`, default: gamma_i may not exceed the bulk absorption rate).
  Momentum- and energy-domain thresholds are separate settings with their own units.
* **Power fractions:** each energy used to be labelled with the next energy (one-step offset).
* **Fig. 5, top row:** the bands were computed swapped (the blue "beyond k_c" band showed the power inside
  the escape cone). The critical angle is now evaluated at every photon energy from the substrate index and
  the cone power is interpolated at it (previously a fixed 14-degree column).
* **Mode absorptance (Figs. 4, 5):** all power in the guided-mode regime is attributed to a mode, with the
  windows described in the Methods (previously about 1 % was left unassigned).
* **Group velocity:** v_G = FWHM_e/FWHM_m and n_g are computed from matched energy- and momentum-domain fits
  and written to `fig7_espace_fits.csv`.
* File-to-panel assignments in Figs. 5 and 6 are explicit instead of depending on file order.
* **FDTD scripts:** `lumapi` is found via `LUMAPI_PATH` or the standard install folders; the material
  project `material-setup.fsp` is loaded from the script folder.
* **`farfield_power_analysis.lsf` (v130):** stores the source power at each frequency (previously
  evaluated at the loop index; this did not affect the included, CW-normalised plane-wave results) and
  writes the summary file once.

## Licence

GPL-3.0-or-later, see `LICENSE`. Please cite the article when you use the toolbox (`CITATION.cff`).
