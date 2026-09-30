"""Quantities derived from the energy-momentum resolved power (modules 3-4):
power fractions within/beyond the escape cone and mode-resolved absorptance."""
import numpy as np
import scipy.interpolate as scinter

from .units import eV_to_nm, k0, nm_to_eV


def power_fractions(ff):
    """Per photon energy: power inside and outside the escape cone of air.

    Returns (energies, within, beyond), sorted by energy, where
        within = sum of P(k) for k <  k0 (couples back out to free space)
        beyond = sum of P(k) for k >= k0 (trapped: guided-mode regime)
    """
    rows = ff.rows
    energies = ff.energies
    within, beyond = np.zeros(len(energies)), np.zeros(len(energies))
    for i, e in enumerate(energies):
        sub = rows[rows[:, 0] == e]
        cone = sub[:, 1] < k0(e)
        within[i] = sub[cone, 2].sum()
        beyond[i] = sub[~cone, 2].sum()
    return energies, within, beyond


def mode_curves(lib, n_modes):
    """Dispersion k(E) of the unpatterned-slab modes: {m: (energies eV, beta um^-1)}."""
    curves = {}
    for m in range(n_modes):
        seg = lib.mode(m)
        if len(seg) > 1:
            curves[m] = (nm_to_eV(seg[:, 0]*1000), seg[:, 1])
    return curves


def attribute_modes(ff, lib, dk=None):
    """Assign all power in the guided-mode regime (k >= k0) to the guided modes of the slab.

    At each photon energy the guided modes (beta > k0) are ordered by decreasing beta.
    Mode i collects the bins with beta_i - 2*dk <= k < beta_{i-1} - 2*dk; the mode with the
    largest beta has no upper limit and the one with the smallest beta extends down to k0.
    The shift of 2*dk towards smaller k accommodates the red shift of the modes caused by the
    pattern (see Methods, GMR modal attribution). Every bin with k >= k0 is counted exactly once.

    dk: momentum bin width (default: the bin width of the data).
    Returns {mode: array of [energy (eV), power fraction]} for modes with at least two points.
    """
    dk = ff.k_step if dk is None else dk
    betas = {m: (xm.min(), xm.max(), scinter.interp1d(xm, ym)) for m, (xm, ym) in mode_curves(lib, lib.n_modes).items()}
    out = {m: [] for m in betas}
    for e in ff.energies:
        rows = ff.rows[ff.rows[:, 0] == e]
        k, p = rows[:, 1], rows[:, 2]
        kc = k0(e)
        guided = sorted(((float(f(e)), m) for m, (lo, hi, f) in betas.items() if lo <= e <= hi), reverse=True)
        guided = [(b, m) for b, m in guided if b > kc]
        upper = np.inf
        for i, (beta, m) in enumerate(guided):
            lower = beta - 2*dk if i < len(guided) - 1 else kc
            take = (k >= max(lower, kc)) & (k < upper)
            out[m].append([e, p[take].sum()])
            upper = lower
    return {m: np.asarray(v) for m, v in out.items() if len(v) > 1}


def attribute_modes_legacy(ff, lib, n_modes, krad):
    """Mode attribution of the first submission (kept to reproduce its numbers; see attribute_modes).

    Leaves power between beta_0 + 3*krad/2 and the material light line, and below the highest
    mode, unassigned.

    Modes are processed from the lowest order (largest beta) upwards. Mode 0
    collects the bins with |k - (beta_0 + krad/2)| <= krad; mode m collects
    beta_m <= k <= beta_{m-1}. Assigned bins are removed, so no power is
    counted twice.

    Returns {mode: array of [energy (eV), absorptance]} for modes with at
    least two points.
    """
    data = ff.rows.copy()
    energies = np.unique(data[:, 0])
    out, previous = {}, None
    for m, (xm, ym) in mode_curves(lib, n_modes).items():
        beta = scinter.interp1d(xm, ym)
        result = []
        for e in energies:
            if e < xm.min() or e > xm.max():
                continue
            at_e = data[:, 0] == e
            if not np.any(at_e):
                continue
            k = data[at_e, 1]
            if previous is None:
                close = np.isclose(k, beta(e) + krad/2, atol=krad)
            else:
                close = (k >= beta(e)) & (k <= previous(e))
            take = at_e.copy()
            take[at_e] = close
            if np.any(take):
                result.append([e, data[take, 2].sum()])
                data = data[~take]
        previous = beta
        if len(result) > 1:
            out[m] = np.asarray(result)
    return out


def beer_lambert_lambertian(energy, thickness_nm, material):
    """Absorptance 1 - exp(-4 n^2 alpha d) of a slab with Lambertian path enhancement."""
    lam = eV_to_nm(energy)
    alpha = 4*np.pi/lam*material.k_nm(lam)
    return 1 - np.exp(-alpha*thickness_nm*4*material.n_nm(lam)**2)
