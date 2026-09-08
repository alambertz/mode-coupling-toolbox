# mode-coupling-toolbox
ANSYS-Lumerical FDTD simulation setup and post-processing pipeline for quantifying light coupling to guided modes in semiconductor slabs with arbitrary scattering patterns

This repository accompanies the publication "Quantifying light coupling to guided modes in semiconductor slabs with arbitrary scattering patterns" by A. Lambertz, E. Alarcon-Llado, and Jorik van de Groep, [in review at Optics Express], 2026.

"Toolbox" here refers to the complete post-processing pipeline — not a single script or program. It comprises six modules:

1. FDTD simulation of arbitrary 3D structures with frequency-domain power-dissipation monitors.
2. Far-field projection to obtain momentum-resolved spectral decomposition of absorptance.
3. Construction of k∥–E_ph heatmaps from the spatially resolved dissipation data.
4. Internal loss decomposition into material absorption (γ_i) and radiative channels (γ_r).
5. Peak-finding algorithm to match guided-mode dispersion curves obtained from mode solvers.
6. GMR fitting to obtain physical quantities including the coupling rates.

# Description of the content

The folders are numbered in pipeline order and map onto the six modules: `01` → 1, `02` → 2, `03-05` → 3–4, `06` → 5–6.

| Folder | Content |
|---|---|
| `01. FDTD simulation/` | Lumerical FDTD projects (`.fsp`) for 500 nm Si slabs with absorption and periodic, HUD-pillar, or random surface patterns, plus an infinite-slab reference. Python scripts rebuild the same setups via lumapi; GDS files define the surface patterns; `material-setup.fsp` holds the material data. |
| `02. Far field Transform/` | Lumerical script `farfield_power_analysis.lsf`: after the FDTD run it extracts the far field for every frequency and integrates the radiated power into momentum bins (um⁻¹) around the normal direction. Writes one per-frequency `.dat` and one full summary `.dat`. Example outputs included. |
| `03-05. Plotting Heatmaps and quantifying absorptance/` | Python scripts for Figs. 4–6: `Periodic/plot_fig4.py` (far-field heatmaps, power integrals, and per-mode absorptance for the periodic structures) and `Disordered/plot_fig6.py` (random / HUD-pillar patterns). Shared `general_settings.py` (figure style, A6 size, 300 dpi), material data files (Si, Ag), mode libraries, AM15G solar spectrum, and the far-field `.dat` inputs. |
| `06. Coupling Analysis/` | Python toolbox for Fig. 7: `plot_fig7_class.py` (MC7 analysis class: imports far-field `.dat`, fits mode dispersions, computes coupling), `plot_fig7_interactive_v2.py` (command-line script) and `toolbox_v3_gui.py` (tkinter GUI with heatmap / momentum-slice / energy-slice / coupling-view panels). Includes material data and example far-field outputs. |

# Requirements

- **Step 01:** ANSYS Lumerical FDTD v22.2 (the Python setup scripts assume the lumapi path `/opt/lumerical/v222/api/python/lumapi.py` on Linux — adjust for other installations).
- **Step 02:** run inside Lumerical with the corresponding FDTD project open.
- **Steps 03–06:** Python 3 with `numpy`, `matplotlib`, `scipy`, `Pillow`; a LaTeX installation (TeX Live + cm-super — figures use `text.usetex` with Arial); `ramanspy` for the Fig. 7 fits; `tkinter` + a display for the GUI (step 06 only).
- All input data files (material data, mode libraries, AM15G spectrum, far-field outputs) are included in the repository.

# Usage

The folders are numbered in execution order:

1. **FDTD simulation** — open one of the `.fsp` projects in Lumerical and run it (or rebuild a setup with the corresponding Python script via lumapi), e.g. `500nmSi-wAbs-HUDPillar-...-V23.fsp`.
2. **Far field transform** — in the same project, run `farfield_power_analysis.lsf` (set `step_size`, monitor names, and refractive index at the top of the script). This produces `<project>_<step>um1_..._full_out_v1xx.dat` plus one file per frequency. Copy the full-output `.dat` into the folders of steps 3–4 that need it.
3. **Figs. 4–6** — from the respective folder: `python plot_fig4.py` (Periodic) or `python plot_fig6.py` (Disordered). The scripts pick up the far-field `.dat` files in the same folder via glob, plus the mode libraries, material data, and AM15G.dat; figures are saved as PDF/PNG with a version suffix.
4. **Fig. 7** — from `06. Coupling Analysis/`:
   - GUI: `python toolbox_v3_gui.py` (needs tkinter + display), or `python toolbox_v3_gui.py --test [file]` for a headless smoke test that saves all four panels to `/tmp/toolbox_test/`;
   - command line: `python plot_fig7_interactive_v2.py`.
   Both read the far-field `.dat` file(s) plus Si/Ag material data from the folder.
