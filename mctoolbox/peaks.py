"""Peak detection in 1D slices of the heatmap (module 5)."""
import numpy as np
from scipy.signal import find_peaks as _find_peaks


def find_peaks(x, y, prominence):
    """Peaks of y(x) with at least the given (absolute) prominence.

    Returns (peak positions in x units, scipy properties dict with
    'prominences', 'left_bases', 'right_bases'). Equivalent to
    ramanspy.plot.peaks(..., return_peaks=True)[1:], without plotting.
    """
    idx, props = _find_peaks(y, prominence=prominence)
    return np.asarray(x)[idx], props


def fwhm(x, y, peak_index, left_base, right_base):
    """Full width at half maximum of the peak at peak_index, searched within [left_base, right_base]."""
    half = y[peak_index]/2.0
    left, right = left_base, right_base
    for i in range(peak_index, left_base, -1):
        if y[i] <= half:
            left = i
            break
    for i in range(peak_index, right_base):
        if y[i] <= half:
            right = i
            break
    return x[right] - x[left]
