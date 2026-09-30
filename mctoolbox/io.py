"""Readers for the far-field output of farfield_power_analysis.lsf and for material data.

Far-field summary file (one row per frequency, tab-separated, one header line):
    lambda (nm) | source power (W) | transmitted power (W) | (1-R) power (W) |
    total far-field power (W) | k values (um^-1) ... | power per k-bin (W) ...
The k values and per-bin powers each take half of the remaining columns; rows of
lower frequencies are zero-padded at the end.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.interpolate as scinter

from .units import nm_to_eV

NORMALISATIONS = {
    'T': 2,     # transmitted power / source power
    'R': 3,     # (1-R) power / source power, i.e. power entering the structure
    None: None,  # raw fraction of the total far-field power
}


@dataclass
class FarField:
    """Momentum-resolved power of one simulation.

    rows: (N, 3) array of [photon energy (eV), k_parallel (um^-1), power fraction],
          in file order (one block of increasing k per wavelength).
    """
    rows: np.ndarray
    path: Path
    norm: str | None

    @property
    def energies(self):
        return np.unique(self.rows[:, 0])

    @property
    def k_step(self):
        """Momentum bin width (um^-1), from the highest-energy row."""
        return float(np.median(np.diff(self.at_energy(self.energies.max())[:, 1])))

    def at_energy(self, energy):
        """Rows [E, k, P] of the energy closest to `energy`, sorted by k."""
        e = self.energies[np.argmin(abs(self.energies - energy))]
        sub = self.rows[self.rows[:, 0] == e]
        return sub[sub[:, 1].argsort()]

    def at_momentum(self, k):
        """For every energy, the row whose k is closest to `k`; sorted by energy."""
        out = []
        for e in self.energies:
            sub = self.rows[self.rows[:, 0] == e]
            out.append(sub[np.argmin(np.abs(sub[:, 1] - k))])
        return np.asarray(out)


def read_farfield(path, norm='R', lambda_min=None, max_energy=None, energy_decimals=None, split='auto'):
    """Read a far-field summary file (script versions v125 and later).

    norm:            'R' (power entering the structure, 1-R), 'T' (transmitted) or None.
    lambda_min:      drop wavelengths below this value [nm].
    max_energy:      drop photon energies above this value [eV].
    energy_decimals: round photon energies (Fig. 7 used 6 decimals).
    split:           'auto' locates the power columns from the data (see power_column_offset);
                     'half' assumes they start at half of the row, as the scripts of the first
                     submission did. 'half' misreads the v128 files with 0.25 um^-1 bins
                     (powers shifted by 3 bins); it is kept only to reproduce published numbers.
    """
    path = Path(path)
    if norm not in NORMALISATIONS:
        raise ValueError(f"norm must be one of {list(NORMALISATIONS)}, got {norm!r}")
    with open(path) as fh:
        header = fh.readline()
    if 'k (um-1)' not in header:
        raise ValueError(f"{path.name}: not a momentum-binned far-field file (header: {header.strip()[:80]!r}). "
                         "Files from script versions before v125 store cone angles, not k bins.")

    col = NORMALISATIONS[norm]
    raw = np.loadtxt(path, delimiter='\t', skiprows=1)
    split = power_column_offset(raw[:, 5:]) if split == 'auto' else (raw.shape[1] - 5)//2
    blocks = []
    for line in raw:
        lam = line[0]
        if lambda_min is not None and lam < lambda_min:
            continue
        energy = nm_to_eV(lam)
        if energy_decimals is not None:
            energy = round(energy, energy_decimals)
        if max_energy is not None and energy > max_energy:
            continue
        kp = line[5:]
        nk = n_k_values(kp)
        k = kp[:nk]
        power = kp[split:split + nk]/line[4]
        if col is not None:
            power = power*float(line[col]/line[1])
        blocks.append(np.column_stack([np.full(nk, energy), k, power]))
    return FarField(np.vstack(blocks), path, norm)


def n_k_values(kp):
    """Number of k values at the start of a row (they are positive; zero padding follows)."""
    zeros = np.flatnonzero(kp <= 0)
    return int(zeros[0]) if len(zeros) else len(kp)


def power_column_offset(kp_rows):
    """Column (within the k/power part of a row) where the per-bin powers start.

    The .lsf script writes array_size k columns followed by array_size power
    columns, so the powers usually start at half of the row. Some files
    (v128 with 0.25 um^-1 bins) have a different number of columns; there the
    offset is found from the data: every row must have zeros between its last k
    value and the first power, and zeros after its last power.
    """
    width = kp_rows.shape[1]
    n = [n_k_values(r) for r in kp_rows]

    def valid(a):
        return all(a + nr <= width and not np.any(r[nr:a]) and not np.any(r[a + nr:])
                   for r, nr in zip(kp_rows, n))

    half = width//2
    if valid(half):
        return half
    top = kp_rows[int(np.argmax(n))]
    candidates = [a for a in range(max(n), width - max(n) + 1) if top[a] != 0 and valid(a)]
    if len(candidates) != 1:
        raise ValueError(f"cannot locate the power columns (candidates: {candidates})")
    return candidates[0]


@dataclass
class Material:
    """Complex refractive index n + ik with interpolants over wavelength."""
    wl_um: np.ndarray
    n_re: np.ndarray
    n_im: np.ndarray

    def n_um(self, wl_um):
        """Real index vs wavelength in um (extrapolating)."""
        return scinter.interp1d(self.wl_um, self.n_re, fill_value='extrapolate')(wl_um)

    def n_nm(self, wl_nm):
        """Real index vs wavelength in nm (no extrapolation)."""
        return scinter.interp1d(self.wl_um*1000, self.n_re)(wl_nm)

    def k_nm(self, wl_nm):
        """Imaginary index vs wavelength in nm (no extrapolation)."""
        return scinter.interp1d(self.wl_um*1000, self.n_im)(wl_nm)


def load_nk(real_file, imag_file=None):
    """Material from refractiveindex.info-style files (wl in um, n) and (wl in um, k)."""
    r = np.loadtxt(real_file, skiprows=1)
    if imag_file is None:
        return Material(r[:, 0], r[:, 1], np.zeros_like(r[:, 1]))
    i = np.loadtxt(imag_file, skiprows=1)
    if not np.array_equal(r[:, 0], i[:, 0]):
        i_on_r = np.interp(r[:, 0], i[:, 0], i[:, 1])
        return Material(r[:, 0], r[:, 1], i_on_r)
    return Material(r[:, 0], r[:, 1], i[:, 1])


def load_nk_from_permittivity(path):
    """Material from a tab-separated (wl in nm, eps', eps'') file such as Si.dat."""
    d = np.loadtxt(path, skiprows=1, delimiter='\t')
    nk = np.sqrt(d[:, 1] + 1j*d[:, 2])
    return Material(d[:, 0]/1000, np.real(nk), np.imag(nk))


def read_scattering_cross_section(path, radius_nm):
    """(photon energy eV, sigma_scat/sigma_geo) from a *_scatCS.dat file (lambda nm, sigma m^2, ...)."""
    d = np.loadtxt(path, skiprows=1, delimiter='\t')
    return nm_to_eV(d[:, 0]), d[:, 1]/(np.pi*(radius_nm*1e-9)**2)


def read_cone_file(path, n_substrate=3.55):
    """Single-particle far-field file with power in cones of 1..90 degrees.

    Columns: lambda (nm), source power, monitor power, total far-field power,
    then cumulative power in cones of half-angle 1, 2, ..., 90 deg.

    Returns (photon energy eV, monitor power / source power, fraction of the
    monitor power inside the escape cone of a substrate with index n_substrate).
    The escape-cone column is the cone that contains the critical angle,
    rounded up to the next full degree.
    """
    d = np.loadtxt(path, skiprows=1, delimiter='\t')
    lam = d[:, 0]
    frac = np.abs(np.minimum(1, d[:, 2]/d[:, 1]))
    theta_c = np.degrees(np.arcsin(1/n_substrate))
    col = 3 + int(theta_c + 1)           # cone of 1 deg is column 4
    within = d[:, 2]/d[:, 1]*d[:, col]/d[:, 3]
    return nm_to_eV(lam), frac, within


def load_am15(path):
    """AM1.5G spectrum: (wavelength nm, irradiance W/m^2/nm)."""
    d = np.loadtxt(path, skiprows=1)
    return d[:, 0], d[:, 1]
