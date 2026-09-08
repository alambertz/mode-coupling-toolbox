#!/usr/bin/env python3
"""
plot_fig7_class.py -- Interactive Fig 7 (mode-coupling rebuttal) as a class.

Based on plot_fig7_interactive_v2_for_class.py. All module-level functions are
now methods of MC7; all settings are set in MC7.initialize() and flow through
import_full_data() into the fit subroutines (fit_GMR_kspace / fit_GMR_espace).

Usage:
    python plot_fig7_class.py            # fully interactive
"""

import re
import sys
from glob import glob as gl

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as cnt
import scipy.interpolate as scinter
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import AutoMinorLocator

# ramanspy 0.2.10 calls plt.cm.get_cmap(), removed in matplotlib >= 3.9 --
# restore it via the modern ColormapRegistry API before any rp.plot call.
import ramanspy as rp
if not hasattr(matplotlib.cm, 'get_cmap'):
    matplotlib.cm.get_cmap = lambda name=None: matplotlib.colormaps.get_cmap(name or 'viridis')

# Shared settings: figsize, dpi, custom_formatter, C0/e0/Pi/h0, TeX/Arial rcParams.
from general_settings import *


class MC7:
    """Fig 7 analysis + plotting for the mode-coupling rebuttal.

    Workflow:
        mc7 = MC7()
        mc7.initialize()   # sets all settings (interactive where marked)
        mc7()              # or mc7.run(): import + fit data, plot all panels
    """

    def __init__(self):
        # No state yet -- everything is set in initialize().
        pass

    # ────────────────────── SETTINGS ──────────────────────
    def initialize(self, interactive=True):
        """Set all settings: material data, run parameters, fit parameters.

        interactive=False skips the input() prompts and uses defaults (used by the
        GUI, which then overrides the run parameters from its entry fields).
        """
        # --- 1. material data (Si refractive index) ---
        self.n1r = np.loadtxt("siliconR.txt", skiprows=1)
        self.n1i = np.loadtxt("siliconI.txt", skiprows=1)
        self.n1ri = scinter.interp1d(self.n1r[:, 0], self.n1r[:, 1], fill_value='extrapolate')
        self.n1ii = scinter.interp1d(self.n1i[:, 0], self.n1i[:, 1], fill_value='extrapolate')
        self.siN = scinter.interp1d(self.n1r[:, 0] * 1000, self.n1r[:, 1])
        self.siK = scinter.interp1d(self.n1i[:, 0] * 1000, self.n1i[:, 1])

        # --- 2. static settings ---
        _m = re.search(r'_v(\d+)\.py$', sys.argv[0])
        self.VER = sys.argv[0][-5:-3] if '_V' in sys.argv[0] else (_m.group(1) if _m else 'int')
        print('Version', self.VER)

        self.PALETTE = np.array(['#8750CC', '#FF7F00', '#FFFF00', '#FF3D33', '#FF3D33'])

        # Reference k_parallel grid; the actual espace fit grid is detected
        # per file in import_full_data (k step may differ between data files).
        self.MOMENTUM_GRID = np.arange(10, 25.5, 0.5)

        # N (line-shape parameter): None -> fit determines N within this range;
        # float -> N pinned to that value.
        self.N_AUTO_RANGE = (1., 1000.)
        self.t = 'R'  # normalize power to transmission 'T' or (1-R) 'R'

        # Scale the A4 figure down to 70% (same as plot_fig7_clean_V188.py).
        self.figsize = (0.7 * figsize[0], 0.7 * figsize[1])

        # Material: real silicon refractive index (interp vs um) used for the light line.
        self.n = self.n1ri

        # --- 3. run parameters (interactive unless initialize(interactive=False)) ---
        candidates = sorted(gl('*.dat'))
        if interactive:
            print('\nAvailable data files:')
            for i, f in enumerate(candidates, 1):
                print(f'  [{i}] {f}')
            choice = input('Select data file (number or path) [1]: ').strip() or '1'
            if choice.isdigit() and 1 <= int(choice) <= len(candidates):
                self.filename = candidates[int(choice) - 1]
            else:
                matches = gl(choice)
                self.filename = sorted(matches)[0] if matches else choice

            e_input = input('Photon-energy slices (eV), comma-separated [1.247434]: ').strip() or '1.247434'
            self.slice_energies = self.parse_float_list(e_input)

            m_input = input("k_parallel slices (um^-1), comma-separated [20.5]: ").strip() or '20.5'
            self.momenta = self.parse_float_list(m_input)

            n_input = input("N (line-shape parameter): value to pin, or 'auto' for the fit to determine [auto]: ").strip().lower()
        else:
            # non-interactive defaults (the GUI overrides these from its entry fields)
            self.filename = candidates[0] if candidates else None
            self.slice_energies = [1.247434]
            self.momenta = [20.5]
            n_input = '1'

        if n_input in ('', 'auto', 'fit'):
            self.Nfix = None
            n_desc = f"auto (fit within [{self.N_AUTO_RANGE[0]:g}, {self.N_AUTO_RANGE[1]:g}])"
        else:
            self.Nfix = float(n_input)
            n_desc = f"pinned to {self.Nfix:g}"

        # --- 4. fit settings (passed to import_full_data -> both fits; edit here) ---
        # promincence / fwhm_thres are shared by BOTH fit_GMR_kspace and
        # fit_GMR_espace. Previous per-fit defaults: kspace prominence=3e-3,
        # fwhm=2. (um^-1); espace prominence=1e-2, fwhm=.05 (eV).
        self.promincence = 3e-3
        self.fwhm_thres = 2.
        self.cov_thres_espace = 5e-6
        self.cov_thres_kspace = 1e-2
        self.vmax = 2.

        # max fit energy (eV): filters data lines in import_full_data and sets the
        # upper bound of every photon-energy axis (lower bound = file E_min).
        if interactive:
            me_input = input("Max fit energy (eV) [1.5]: ").strip() or '1.5'
            self.max_energy = float(me_input)
        else:
            self.max_energy = 1.35

        print(f'\nfile={self.filename}')
        print(f'slice_energies={self.slice_energies} eV')
        print(f'momenta={self.momenta} um^-1')
        print(f'N={n_desc}, norm={self.t}, max_energy={self.max_energy} eV\n')

    # ────────────────────── HELPERS ──────────────────────
    @staticmethod
    def parse_float_list(s):
        """'1.247434, 20.5' style input -> list of floats."""
        return [float(v) for v in re.split(r'[,\s]+', s.strip()) if v]

    def nm_to_eV(self, lam):
        """Photon energy [eV] from wavelength [nm]."""
        return C0/lam*1e9/e0*h0

    def eV_to_nm(self, e):
        """Wavelength [nm] from photon energy [eV]."""
        return C0/e*1e9/e0*h0

    def find_and_extract_lines(self, arr, x, col=1):
        """Row block of `arr` whose column `col` is closest to `x`, sorted by column 1."""
        differences = np.abs(arr[:, col] - x)
        closest_line_index = np.argmin(differences)
        closest_val = arr[closest_line_index, 0]
        matching_lines_mask = arr[:, col] == closest_val
        matching_lines = arr[matching_lines_mask]
        sorted_indices = matching_lines[:, 1].argsort()
        return matching_lines[sorted_indices]

    def find_and_extract_klines(self, arr, x, col=1):
        """For each unique energy in `arr`, the row whose column `col` is closest to `x`.
        Returns one [energy, k, power] row per energy, sorted by energy."""
        out_array = []
        energies = np.unique(arr[:, 0])
        for energy in energies:
            subarray_mask = arr[:, 0] == energy
            subarray = arr[subarray_mask]
            differences = np.abs(subarray[:, col] - x)
            closest_line_index = np.argmin(differences)
            out_array.append(subarray[closest_line_index])
        out_array = np.asarray(out_array)
        sorted_indices = out_array[:, 0].argsort()
        return out_array[sorted_indices]

    # ────────────────────── PHYSICS & FITTING ──────────────────────
    def lorentzian(self, x, center, amplitude, width):
        """Normalized Lorentzian (area = amplitude)."""
        return amplitude * (width) / ((x - center)**2 + width**2) / Pi

    def do_gmr_fit(self, x, center, gamma_i, gamma_e, N):
        """Absorption line shape with internal (gamma_i) and external (gamma_e) damping."""
        return 4 * gamma_i*gamma_e / ((x - center)**2 + (gamma_i + N*gamma_e)**2) / Pi

    def gammai_bulk(self, lamda):
        """Bulk internal damping [eV] at wavelength lamda [nm], from the Si imaginary index."""
        alpha = 4*Pi/lamda*1e9*self.siK(lamda)
        return alpha * C0 / self.siN(lamda)

    def calculate_fwhm(self, x_axis, y_axis, peak_index, left_base, right_base):
        """Full width at half maximum of the peak at `peak_index`, scanned within [left_base, right_base]."""
        peak_height = y_axis[peak_index]
        half_height = peak_height / 2.0
        left_half_index = left_base
        right_half_index = right_base
        for i in range(peak_index, left_base, -1):
            if y_axis[i] <= half_height:
                left_half_index = i
                break
        for i in range(peak_index, right_base):
            if y_axis[i] <= half_height:
                right_half_index = i
                break
        return x_axis[right_half_index] - x_axis[left_half_index]

    def fit_GMR_kspace(self, energy, momentum, power, fwhm_thres=2., prominence=3e-3, cov_thres=1e-2):
        """Lorentzian fits of the guided-mode resonances in P_abs vs k at one photon energy."""
        x_axis = momentum
        y_axis = power
        data = rp.Spectrum(y_axis, x_axis)

        rp.plot.spectra(data)
        peaks = rp.plot.peaks(data, prominence=prominence, return_peaks=True)

        prominence = prominence*np.max(power)
        peak_positions = peaks[1]
        prominences = peaks[2]['prominences']
        left_bases = peaks[2]['left_bases']
        right_bases = peaks[2]['right_bases']

        lightline = 2*Pi/self.eV_to_nm(energy)*1000
        lightline_mat = 2*Pi/self.eV_to_nm(energy)*1000*self.n(self.eV_to_nm(energy)/1000)

        filtered_peaks = []
        for peak, prom, left_base, right_base in zip(peak_positions, prominences, left_bases, right_bases):
            closest_index = (np.abs(x_axis - peak)).argmin()
            peak_height = y_axis[closest_index]
            fwhm = self.calculate_fwhm(x_axis, y_axis, closest_index, left_base, right_base)
            if prom > prominence and fwhm < fwhm_thres and peak > lightline:
                filtered_peaks.append([peak, peak_height, fwhm])

        lorentzian_fits = []
        for peak, _, _ in filtered_peaks:
            closest_index = (np.abs(x_axis - peak)).argmin()
            peak_range = (max(0, closest_index - 5), min(len(x_axis) - 1, closest_index + 5))
            x_fit = x_axis[peak_range[0]:peak_range[1]]
            y_fit = y_axis[peak_range[0]:peak_range[1]]
            initial_guess = [peak, y_fit.max(), 1.0]
            try:
                popt, cov = curve_fit(self.lorentzian, x_fit, y_fit, p0=initial_guess)
                center, amplitude, width = popt
                if width > 0 and width < fwhm_thres and cov[0,0] < cov_thres and center < lightline_mat:
                    lorentzian_fits.append([center, amplitude, width])
            except RuntimeError:
                pass

        return np.asarray(lorentzian_fits), np.asarray(filtered_peaks)

    def fit_GMR_espace(self, energies, power, momentum=None, fwhm_thres=.05, prominence=1e-2, cov_thres=1e-4, Nfix=None):
        """Line-shape fits of P_abs vs photon energy at one momentum (coupling panel).

        Nfix: None -> N determined by the fit within N_AUTO_RANGE; float -> N pinned to that value.
        """
        x_axis = energies
        y_axis = power
        data = rp.Spectrum(y_axis, x_axis)

        rp.plot.spectra(data)
        peaks = rp.plot.peaks(data, prominence=prominence, return_peaks=True)
        peak_positions = peaks[1]
        prominences = peaks[2]['prominences']

        filtered_peaks = []
        for peak, prom in zip(peak_positions, prominences):
            if prom > prominence:
                filtered_peaks.append(peak)

        lorentzian_fits = []
        for peak in filtered_peaks:
            closest_index = (np.abs(x_axis - peak)).argmin()
            peak_range = (max(0, closest_index - 10), min(len(x_axis) - 1, closest_index + 10))
            x_fit = x_axis[peak_range[0]:peak_range[1]]
            y_fit = y_axis[peak_range[0]:peak_range[1]]

            gammai_bulk_bulk = self.gammai_bulk(self.eV_to_nm(peak))*h0/e0
            if Nfix is None:
                n_lo, n_hi, n0 = self.N_AUTO_RANGE[0], self.N_AUTO_RANGE[1], 1.0
            else:
                n_lo = n_hi = n0 = float(Nfix)
                n_hi += .001
            initial_guess = [peak, gammai_bulk_bulk, gammai_bulk_bulk, n0]
            bounds = ([peak/2, 0, 0, n_lo], [peak*2, 1e16, 1e16, n_hi])

            try:
                popt, cov = curve_fit(self.do_gmr_fit, x_fit, y_fit, p0=initial_guess, bounds=bounds)
                if popt[2] > 0 and popt[2] < fwhm_thres and cov[0,0] < cov_thres:
                    center, gamma_i, gamma_e, N = popt
                    amplitude = np.max(self.do_gmr_fit(x_fit, *popt))
                    lightline_mat = 2*Pi/self.eV_to_nm(center)*1000*self.n(self.eV_to_nm(center)/1000)
                    if momentum < lightline_mat:
                        print(f"GMR at {center:.3e} eV, K|| {momentum} $\mu$m^(-1), Amplitude {amplitude:.2e}, gamma_i: {gamma_i*e0/h0:.3e}, gamma_e: {gamma_e*e0/h0:.3e}, N: {N:.1e}, Regime: {gamma_e/gamma_i:.2f}")
                        lorentzian_fits.append([round(center,6), amplitude, gamma_i, gamma_e, N])
            except RuntimeError:
                pass
        return np.asarray(lorentzian_fits)

    def import_full_data(self, filename, k_min=0, norm_power='R', slice_energies=None, momenta=None, Nfix=None,
                         promincence=3e-3, fwhm_thres=2., cov_thres_espace=1e-4, cov_thres_kspace=1e-2):
        """Load one FSP output file and fit all GMR lines.

        Fit settings (promincence, fwhm_thres, cov_thres_*) are passed on to the
        relevant subfunction calls: fit_GMR_kspace and fit_GMR_espace.
        """
        if slice_energies is None:
            slice_energies = [1.247434]
        if momenta is None:
            momenta = [20.5]

        data_raw = np.loadtxt(filename, delimiter='\t', skiprows=1)
        self.E_min = round(self.nm_to_eV(data_raw[:, 0].max()), 6)   # lowest photon energy in file

        data = []
        mdic = {}
        peakd = {}
        max_k, step_k = 0, 100

        for line in data_raw:
            lamda = line[0]
            energy = round(self.nm_to_eV(lamda), 6)
            if energy > self.max_energy:
                continue
            ff_tot = line[4]
            k_ffpower = line[5:]
            k_values = k_ffpower[:int(len(k_ffpower)/2)]
            ff_power = k_ffpower[int(len(k_ffpower)/2):]/ff_tot
            max_k, step_k = max(*k_values, max_k), min(step_k, k_values[1]-k_values[0])
            if norm_power == 'T':
                power_norm = float(line[2]/line[1])   # transmission
            elif norm_power == 'R':
                power_norm = float(line[3]/line[1])   # 1-R
            else:
                power_norm = 1

            lam_data = []
            for k, ff in zip(k_values, ff_power*power_norm):
                if k == 0:
                    continue
                lam_data.append([energy, k, ff])
            lam_data = np.asarray(lam_data)

            data = lam_data if len(data) < 1 else np.vstack([data, lam_data])

            momentum_fits, momentum_peaks = self.fit_GMR_kspace(energy, lam_data[:,1], lam_data[:,2],
                                                                fwhm_thres=fwhm_thres, prominence=promincence,
                                                                cov_thres=cov_thres_kspace)
            mdic[energy] = momentum_fits
            peakd[energy] = momentum_peaks

            #print(f'Energy: {energy} eV, Abs: {np.sum(lam_data[:,2])*100:.1f} %, num fits: {len(momentum_fits)}')

        edic = {}
        # k grid detected from the data file (may differ per file)
        momenta_grid = np.arange(10, max_k+step_k, step_k)
        print('Detected k-step', step_k)
        for mom in momenta_grid:
            subarray = self.find_and_extract_klines(data, mom, 1)
            edic[mom] = self.fit_GMR_espace(subarray[:,0], subarray[:,2], momentum=mom, Nfix=Nfix,
                                            fwhm_thres=fwhm_thres, prominence=promincence,
                                            cov_thres=cov_thres_espace)

        plt.close()
        return data, mdic, edic, peakd

    # ────────────────────── PLOTTING ──────────────────────
    def style_axes(self, ax):
        """Shared axis styling: ticks on all four sides, minor ticks, custom formatter."""
        ax.tick_params(axis='x', direction='inout')
        ax.tick_params(axis='x', which='minor', direction='inout')
        ax.tick_params(axis='y', direction='inout')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(top=True, right=True)
        ax.tick_params(which='minor', top=True, right=True)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))

    def colorbar(self, fig, ax, im, vmax=-1, label='Power (\\%)'):
        """Horizontal colorbar in the upper-left corner of `ax`."""
        pos = ax.get_position()
        cbar_pos = [pos.x0+.025*pos.width, pos.y0 + 0.93*pos.height, 0.4*pos.width, 0.05*pos.height]
        cbar = fig.colorbar(im, cax=fig.add_axes(cbar_pos), orientation='horizontal')
        if vmax != -1:
            cbar.set_ticks([0, vmax])
            cbar.set_ticklabels(['0', '$>${}'.format(vmax)])
        cbar.set_label(label, labelpad=-14)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.yaxis.label.set_rotation(0)
        return fig, ax, cbar

    def colorbar_C(self, fig, ax, im, vmin=None, vmax=None, label='Power (\\%)'):
        """Horizontal colorbar for the coupling-regime scatter."""
        pos = ax.get_position()
        cbar_pos = [pos.x0+.025*pos.width, pos.y0 + 0.93*pos.height, 0.35*pos.width, 0.05*pos.height]
        cbar = fig.colorbar(im, cax=fig.add_axes(cbar_pos), orientation='horizontal')
        cbar.set_label(label, labelpad=1)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
        return fig, ax, cbar

    def heatmap(self, ax, data, slice_energies=None, momenta=None, vmax=0.5, lightline=True, log=False, axv=False, scatter=True):
        """Panel [0,0]: P_abs [%] vs (E_ph, k_parallel), Si light line + slice guides."""
        if slice_energies is None:
            slice_energies = [1.247434]
        if momenta is None:
            momenta = [20.5]

        xlim0, xlim1 = self.E_min, self.max_energy
        ylim0, ylim1 = 10, 25

        data = np.asarray(data)
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]*100

        energies_axis = np.arange(min(x), max(x)+.01, 0.01)
        kn = 2*Pi*energies_axis*e0/cnt.h/C0/1e6*self.n(self.eV_to_nm(energies_axis)/1000)+0.1
        ax.fill_between(energies_axis, 0, kn, edgecolor=None, facecolor='#000000', interpolate=True, color='#000000')

        if scatter:
            im = ax.scatter(x, y, c=z, cmap='viridis', s=50, zorder=1, marker='.', edgecolors=None, vmax=vmax)

        clrs = self.PALETTE
        ecol = [clrs[i % len(clrs)] for i in range(len(slice_energies))]
        mcol = [clrs[(4 + i) % len(clrs)] for i in range(len(momenta))]
        for e, col in zip(slice_energies, ecol):
            l = self.eV_to_nm(e)
            k_light = 2000*Pi/l*self.siN(l)
            ymax = (k_light-ylim0) / (ylim1-ylim0)
            ax.axvline(x=e, ymax=ymax, color=col, linewidth=1)
            ax.axhline(y=k_light, color=col, linewidth=1)

        for mom, col in zip(momenta, mcol):
            ax.axhline(y=mom, color=col, linewidth=1)

        ax.set_xlabel(r'Photon energy (eV)')
        ax.set_ylabel(r'$k_\parallel$ ($\text{\textmu}$m$^{-1}$)', labelpad=0)
        ax.xaxis.set_ticks_position('both')
        ax.set_yticks(range(0, 80, 4))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))
        ax.set_ylim([ylim0, ylim1])
        ax.set_xlim([xlim0, xlim1])

        return ax, im

    def plot_AbsVk(self, ax, slice_energies=None, data=None, d=None):
        """Panel [0,1]: absorption vs k_parallel at the sliced photon energies."""
        if slice_energies is None:
            slice_energies = [1.247434]

        clrs = self.PALETTE
        for e, col in zip(slice_energies, clrs[:]):
            subarray = self.find_and_extract_lines(data, e, 0)
            x = np.arange(min(subarray[:,1]), max(subarray[:,1]), .05)
            ax.plot(100*subarray[:,2], subarray[:,1], color=col, linewidth=4,
                    label=r'E$_\mathrm{ph}$: ' + str(round(e, 3)) + r'\,eV')
            closest_key = min(d.keys(), key=lambda k: abs(k - e))
            fit_array = np.asarray(d[closest_key])
            for fit in fit_array:
                ax.plot(100*self.lorentzian(x, *fit), x, '--', color='grey')

        ax.set_xlabel(r'P$_\mathrm{abs}$ (\%)')
        ax.set_ylabel(r'$k_\parallel$ ($\text{\textmu}$m$^{-1}$)')

        legend = ax.legend(frameon=False, loc='best')
        for text, handle, col in zip(legend.get_texts(), legend.legend_handles, clrs[:]):
            text.set_color(col)
            handle.set_color(col)

        self.style_axes(ax)
        return ax

    def plot_AbsVpe(self, ax, momenta=None, data=None, d=None):
        """Panel [1,0]: absorption vs photon energy at the sliced momenta."""
        if momenta is None:
            momenta = [20.5]

        clrs = self.PALETTE
        cols = [clrs[(4 + i) % len(clrs)] for i in range(len(momenta))]
        for mom, col in zip(momenta, cols):
            subarray = self.find_and_extract_klines(data, mom, 1)
            x = np.arange(min(subarray[:,0]), max(subarray[:,0]), .001)
            ax.plot(subarray[:,0], 100*subarray[:,2], color=col, linewidth=4,
                    label=r'$k_\parallel$: ' + str(round(mom, 2)) + r'\,\textmu m$^{-1}$')
            closest_key = min(d.keys(), key=lambda k: abs(k - mom))
            for k in d[closest_key]:
                center, amplitude, gamma_i, gamma_e, N = k
                ax.plot(x, 100*self.do_gmr_fit(x, center, gamma_i, gamma_e, N), '--', color='grey')

        ax.set_xlabel(r'Photon energy (eV)')
        ax.set_ylabel(r'P$_\mathrm{abs}$ (\%)', labelpad=10)
        ax.set_xlim([self.E_min, self.max_energy])

        legend = ax.legend(frameon=False, loc='best')
        for text, handle, col in zip(legend.get_texts(), legend.legend_handles, cols):
            text.set_color(col)
            handle.set_color(col)

        self.style_axes(ax)
        return ax

    def plot_Coupling(self, fig, ax, dic, slice_energies=None, momenta=None, vlines=True):
        """Panel [1,1]: gamma_e/gamma_i scatter."""
        if slice_energies is None:
            slice_energies = [1.247434]
        if momenta is None:
            momenta = [20.5]

        ylim = [10, 25]
        xlim = [self.E_min, self.max_energy]
        
        plot_data = []
        for mom in dic.keys():
            for gmr_fits in dic[mom]:
                if len(gmr_fits) < 1:
                    continue
                energy, A, gamma_i, gamma_e, N = gmr_fits
                ratio = gamma_e / gamma_i
                plot_data.append([energy, mom, ratio])
        plot_data = np.asarray(plot_data)

        boundaries = [0, 1, 10, 100, 1000]
        cmap = plt.get_cmap('coolwarm')
        norm = BoundaryNorm(boundaries, cmap.N)
        scatter = ax.scatter(plot_data[:, 0], plot_data[:, 1], c=plot_data[:, 2],
                             cmap=cmap, s=150, marker='.', edgecolor=None, norm=norm)

        fig, res_ax, cbar2 = self.colorbar_C(fig, ax, scatter, label=r'$\gamma_e/\gamma_i$')

        if vlines:
            ylim0, ylim1 = ylim
            clrs = self.PALETTE
            ecol = [clrs[i % len(clrs)] for i in range(len(slice_energies))]
            mcol = [clrs[(4 + i) % len(clrs)] for i in range(len(momenta))]
            for e, col in zip(slice_energies, ecol):
                l = self.eV_to_nm(e)
                ymax = (2000*Pi/l*self.siN(l) - ylim0) / (ylim1 - ylim0)
                ax.axvline(x=e, ymax=ymax, color=col, linewidth=0.25)
            for mom, col in zip(momenta, mcol):
                ax.axhline(y=mom, color=col, linewidth=0.25)
            xx = np.arange(self.E_min, self.max_energy, .01)
            ax.plot(xx, 2000*Pi*self.siN(l)/self.eV_to_nm(xx), color='black')

        ax.set_xlabel('Photon Energy (eV)')
        ax.set_xlim(xlim)
        ax.set_ylabel(r'k$_\parallel$ (\textmu m$^{-1}$)')
        ax.set_ylim(ylim)
        ax.set_yticks([12, 16, 20, 24])

        ax.tick_params(axis='x', direction='inout')
        ax.tick_params(axis='x', which='minor', direction='inout')
        ax.tick_params(axis='y', direction='inout')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(top=True, right=True)
        ax.tick_params(which='minor', top=True, right=True)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))
        return fig, ax

    def plot_figure(self, data, dict_momentum, dict_energy, slice_energies=None, momenta=None, name='test', vmaxs=None):
        """Assemble the 4-panel Fig 7 and save it as pdf (x2) + png."""
        if slice_energies is None:
            slice_energies = [1.247434]
        if momenta is None:
            momenta = [20.5]
        if vmaxs is None:
            vmaxs = [self.vmax]

        fig = plt.figure(figsize=(self.figsize[0]*2, self.figsize[1]*2), constrained_layout=False)
        gs = fig.add_gridspec(3, 2, width_ratios=[1, 1], height_ratios=[1, 1, .1])

        heat_ax = fig.add_subplot(gs[0, 0])
        avk_ax = fig.add_subplot(gs[0, 1], sharey=heat_ax)
        avp_ax = fig.add_subplot(gs[1, 0], sharex=heat_ax)
        res_ax = fig.add_subplot(gs[1, 1])
        avk_ax.set_zorder(10)

        plt.subplots_adjust(wspace=0.15)
        plt.subplots_adjust(hspace=.25)

        heat_ax, heat_im = self.heatmap(heat_ax, data, slice_energies=slice_energies, momenta=momenta, vmax=vmaxs[0])
        fig, heat_ax, cbar = self.colorbar(fig, heat_ax, heat_im, vmaxs[0])

        avk_ax = self.plot_AbsVk(avk_ax, slice_energies=slice_energies, data=data, d=dict_momentum)

        avp_ax = self.plot_AbsVpe(avp_ax, momenta=momenta, data=data, d=dict_energy)

        fig, res_ax = self.plot_Coupling(fig, res_ax, dict_energy, slice_energies=slice_energies, momenta=momenta)
        res_ax.set_xlim([self.E_min, self.max_energy])

        avk_ax.set_xlim([-0.01, 5])
        avp_ax.set_ylim([-0.01, 5])

        ### SAVING
        plt.savefig(name + '.pdf', dpi=dpi, bbox_inches='tight')
        plt.savefig(name + '_V' + self.VER + '.pdf', dpi=dpi, bbox_inches='tight')
        plt.savefig(name + '_V' + self.VER + '.png', dpi=dpi, bbox_inches='tight')
        plt.close()

    # ────────────────────── PIPELINE ──────────────────────
    def run(self):
        """Run the full pipeline: import + fit all data, then plot Fig 7."""
        data, dict_momentum, dict_energy, peaks = self.import_full_data(
            self.filename, norm_power=self.t, slice_energies=self.slice_energies, momenta=self.momenta,
            Nfix=self.Nfix, promincence=self.promincence, fwhm_thres=self.fwhm_thres,
            cov_thres_espace=self.cov_thres_espace, cov_thres_kspace=self.cov_thres_kspace)

        self.plot_figure(data, dict_momentum, dict_energy, slice_energies=self.slice_energies,
                         momenta=self.momenta, name='fig_7', vmaxs=[self.vmax])
        print('Done')

    def __call__(self):
        """Allow calling the instance directly: mc7() == mc7.run()."""
        self.run()


def main():
    mc7 = MC7()
    mc7.initialize()
    mc7()


if __name__ == '__main__':
    main()
