"""Unit conversions and light lines.

Conventions used throughout the toolbox:
    wavelength      nm
    photon energy   eV
    in-plane momentum k_parallel  um^-1
"""
import numpy as np
import scipy.constants as cnt

C0 = cnt.c
E0 = cnt.e
H = cnt.h
HBAR = cnt.hbar


def nm_to_eV(lam):
    """Photon energy [eV] from wavelength [nm]."""
    return C0/lam*1e9/E0*H


def eV_to_nm(e):
    """Wavelength [nm] from photon energy [eV]."""
    return C0/e*1e9/E0*H


def k0(energy):
    """Vacuum wavevector = air light line [um^-1] at photon energy [eV]."""
    return 2000*np.pi/eV_to_nm(energy)


def light_line(energy, n):
    """Light line k = n*k0 [um^-1] in a medium with index n(wavelength in um)."""
    return k0(energy)*n(eV_to_nm(energy)/1000)
