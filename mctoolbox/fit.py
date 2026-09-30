"""Peak fitting of guided-mode resonances (GMRs) (module 6).

Momentum domain: Lorentzians in P(k) at fixed photon energy give the mode's
in-plane momentum and FWHM_m.

Energy domain: temporal coupled-mode theory (Eq. HH of the paper) in P(E) at
fixed k_parallel gives the internal (gamma_i) and external (gamma_e) loss rates:

    A(E) = gamma_i*gamma_e / ((E - E0)^2 + (gamma_i + N*gamma_e)^2 / 4)

with all rates expressed as energies (hbar*gamma, eV). The peak value is
4*gamma_i*gamma_e/(gamma_i + N*gamma_e)^2, i.e. 1 at critical coupling.

For N = 1 the line shape depends on gamma_i and gamma_e only through their sum
and product: an over-coupled solution (gamma_e/gamma_i = r) and the under-coupled
one (1/r) fit a single peak equally well. EFitSettings.branch chooses between
them; the default uses that the internal loss rate of a guided mode cannot
exceed the bulk absorption rate of the material (gamma_i_bulk).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit

from .peaks import find_peaks, fwhm
from .units import C0, E0, H, HBAR, eV_to_nm, k0, light_line


def lorentzian(x, center, amplitude, width):
    """Area-normalised Lorentzian: area = amplitude, half width at half maximum = width."""
    return amplitude*width/((x - center)**2 + width**2)/np.pi


def cmt_absorptance(E, E0_, gamma_i, gamma_e, N=1.0):
    """Eq. HH: absorptance of a single GMR coupled to N free-space channels."""
    return gamma_i*gamma_e/((E - E0_)**2 + (gamma_i + N*gamma_e)**2/4)


def legacy_pi_lineshape(E, E0_, gamma_i, gamma_e, N=1.0):
    """Line shape used for the Fig. 7d values of the first submission (for comparison only).

    Its width parameter is 2x that of Eq. HH and its amplitude carries an
    extra 1/pi, so it yields gamma_e/gamma_i about 3x smaller than Eq. HH.
    """
    return 4*gamma_i*gamma_e/((E - E0_)**2 + (gamma_i + N*gamma_e)**2)/np.pi


LINESHAPES = {'cmt': cmt_absorptance, 'legacy': legacy_pi_lineshape}


def gamma_i_bulk(energy, material, planck=HBAR):
    """Bulk absorption rate alpha*c/n of the material, as an energy (eV).

    Used only as the starting value of the energy-domain fit. The first
    submission multiplied by h instead of hbar (planck=H reproduces it).
    """
    lam = eV_to_nm(energy)
    alpha = 4*np.pi/lam*1e9*material.k_nm(lam)
    return alpha*C0/material.n_nm(lam)*planck/E0


@dataclass
class KFitSettings:
    """Lorentzian fits of P(k) at fixed energy. Momenta in um^-1."""
    prominence: float = 3e-3      # peak finding; peaks must also exceed prominence*max(P)
    max_fwhm: float = 2.0         # um^-1, applied to the peak and to the fitted width
    max_center_var: float = 1e-2  # max variance of the fitted center
    half_window: int = 5          # fit window: +-half_window points around the peak


@dataclass
class EFitSettings:
    """Eq. HH fits of P(E) at fixed k. Energies in eV."""
    prominence: float = 1e-2      # absolute prominence for peak finding
    max_gamma_e: float = 0.05     # eV; fits with larger gamma_e are rejected
    max_center_var: float = 1e-4  # max variance of the fitted center
    half_window: int = 10
    N: float = 1.0                # number of free-space channels (fixed)
    lineshape: str = 'cmt'        # 'cmt' (Eq. HH) or 'legacy'
    branch: str = 'bulk-bound'    # 'bulk-bound', 'over' or 'under' (see module docstring); 'cmt' only
    guided_only: bool = True      # drop fits with k at or beyond the material light line (edge artefacts)
    k_grid: tuple = (10.0, 25.5, 0.25)   # np.arange(*k_grid): momenta at which P(E) is fitted


def fit_kspace(energy, k, power, s=KFitSettings()):
    """Lorentzian fits of the GMR peaks in P(k) beyond the air light line.

    Returns (fits [center, amplitude, width], peaks [position, height, fwhm]).
    """
    positions, props = find_peaks(k, power, s.prominence)
    min_prom = s.prominence*np.max(power)
    peaks = []
    for pos, prom, lb, rb in zip(positions, props['prominences'], props['left_bases'], props['right_bases']):
        i = np.argmin(np.abs(k - pos))
        w = fwhm(k, power, i, lb, rb)
        if prom > min_prom and w < s.max_fwhm and pos > k0(energy):
            peaks.append([pos, power[i], w])

    fits = []
    for pos, _, _ in peaks:
        i = np.argmin(np.abs(k - pos))
        lo, hi = max(0, i - s.half_window), min(len(k) - 1, i + s.half_window)
        x, y = k[lo:hi], power[lo:hi]
        try:
            popt, cov = curve_fit(lorentzian, x, y, p0=[pos, y.max(), 1.0])
        except RuntimeError:
            continue
        if 0 < popt[2] < s.max_fwhm and cov[0, 0] < s.max_center_var:
            fits.append(list(popt))
    return np.asarray(fits), np.asarray(peaks)


def fit_espace(energies, power, material, s=EFitSettings()):
    """Fit the GMR peaks in P(E) at one k_parallel.

    Returns an array of [E0, peak amplitude, gamma_i, gamma_e, N, unique] per accepted fit.
    unique is 1 when only one of the two coupling branches is compatible with
    s.branch (always 1 for the legacy line shape, which does not resolve branches).
    """
    model = LINESHAPES[s.lineshape]
    positions, props = find_peaks(energies, power, s.prominence)
    fits = []
    for pos, prom in zip(positions, props['prominences']):
        if prom <= s.prominence:
            continue
        i = np.argmin(np.abs(energies - pos))
        lo, hi = max(0, i - s.half_window), min(len(energies) - 1, i + s.half_window)
        x, y = energies[lo:hi], power[lo:hi]
        unique = 1
        try:
            if s.lineshape == 'legacy':
                # as published: N as a (nearly pinned) fit parameter, seed with h instead of hbar
                g = gamma_i_bulk(pos, material, planck=H)
                popt, cov = curve_fit(model, x, y, p0=[pos, g, g, 1.0],
                                      bounds=([pos/2, 0, 0, 1], [pos*2, 1e16, 1e16, s.N + .001]))
            else:
                g = gamma_i_bulk(pos, material)
                # asymmetric seed: an equal seed sits on the symmetry line of the two branches
                popt, cov = curve_fit(lambda E, e0, gi, ge: model(E, e0, gi, ge, s.N), x, y,
                                      p0=[pos, g, 10*g], bounds=([pos/2, 0, 0], [pos*2, np.inf, np.inf]))
                popt, unique = _choose_branch(popt, g, s)
                popt = np.append(popt, s.N)
        except RuntimeError:
            continue
        if 0 < popt[2] < s.max_gamma_e and cov[0, 0] < s.max_center_var:
            amplitude = np.max(model(x, *popt))
            fits.append([round(popt[0], 6), amplitude, popt[1], popt[2], popt[3], unique])
    return np.asarray(fits)


def _choose_branch(popt, gamma_bulk, s):
    """Order (gamma_i, gamma_e) of an N=1 fit according to s.branch. Returns (popt, unique)."""
    e0, a, b = popt
    small, large = sorted((a, b))
    over, under = (e0, small, large), (e0, large, small)
    if s.branch == 'over':
        return np.array(over), 1
    if s.branch == 'under':
        return np.array(under), 1
    # bulk-bound: gamma_i <= bulk absorption rate. Under-coupling needs gamma_i = large.
    under_allowed = large <= gamma_bulk
    over_allowed = small <= gamma_bulk
    if over_allowed and not under_allowed:
        return np.array(over), 1
    if under_allowed and not over_allowed:
        return np.array(under), 1
    return np.array(over), 0


@dataclass
class CouplingResult:
    kfits: dict = field(default_factory=dict)   # {energy: [[center, amplitude, width], ...]}
    kpeaks: dict = field(default_factory=dict)  # {energy: [[position, height, fwhm], ...]}
    efits: dict = field(default_factory=dict)   # {k: [[E0, amplitude, gamma_i, gamma_e, N, unique], ...]}
    k_step: float = float('nan')                # momentum bin width of the data (um^-1)

    def espace_table(self):
        """All energy-domain fits as rows [k, E0, amplitude, gamma_i, gamma_e, N, unique, gamma_e/gamma_i]."""
        rows = [[k, *f, f[3]/f[2]] for k in sorted(self.efits) for f in self.efits[k]]
        return np.asarray(rows).reshape(-1, 8)

    def group_velocity(self):
        """Group velocity of each energy-domain GMR from the widths in both domains.

        Each row of espace_table() is matched with the momentum-domain Lorentzian at the
        energy closest to E0 whose center lies within one bin of k. Then
        v_G = domega/dk = (FWHM_e/hbar)/FWHM_m with FWHM_e = gamma_i + N*gamma_e (Eq. HH)
        and FWHM_m = 2*HWHM of the Lorentzian.
        Returns rows [v_G (m/s), n_g = c/v_G]; NaN where no momentum-domain fit matches.
        """
        energies = np.array(sorted(self.kfits))
        out = []
        for k, e0, _, gi, ge, n, *_ in self.espace_table():
            e = energies[np.argmin(np.abs(energies - e0))] if len(energies) else None
            fits = np.asarray(self.kfits.get(e, [])).reshape(-1, 3)
            near = fits[np.abs(fits[:, 0] - k) <= self.k_step] if len(fits) else fits
            if len(near) == 0:
                out.append([np.nan, np.nan])
                continue
            hwhm_k = near[np.argmin(np.abs(near[:, 0] - k)), 2]
            v = ((gi + n*ge)*E0/HBAR)/(2*hwhm_k*1e6)
            out.append([v, C0/v])
        return np.asarray(out).reshape(-1, 2)

    def kspace_table(self):
        """All momentum-domain fits as rows [energy, center, amplitude, width]."""
        rows = [[e, *f] for e in sorted(self.kfits) for f in self.kfits[e]]
        return np.asarray(rows).reshape(-1, 4)


def analyse_coupling(ff, material, ks=KFitSettings(), es=EFitSettings()):
    """Fit all GMRs of a far-field dataset in both domains."""
    res = CouplingResult(k_step=ff.k_step)
    for e in ff.energies:
        sub = ff.at_energy(e)
        res.kfits[e], res.kpeaks[e] = fit_kspace(e, sub[:, 1], sub[:, 2], ks)
    for k in np.arange(*es.k_grid):
        sub = ff.at_momentum(k)
        fits = fit_espace(sub[:, 0], sub[:, 2], material, es)
        if es.guided_only and len(fits):
            fits = fits[k < light_line(fits[:, 0], material.n_um)]
        res.efits[k] = fits
    return res
