"""Guided-mode library of the unpatterned slab (output of the simple mode solver,
https://doi.org/10.5281/zenodo.4589006 / MSclass19.Modedata.Export).

File format: header line ending in "Thickness=<um>; um", then comma- or
space-separated rows of
    mode number, lambda (um), beta (um^-1), alpha/2 (um^-1), k0 (um^-1)
Modes follow each other; a new mode starts where lambda stops increasing
(TE and TM of the same order share a mode number).
"""
from pathlib import Path

import numpy as np


class ModeLibrary:
    def __init__(self, path):
        path = Path(path)
        with open(path) as fh:
            self.thickness_um = float(fh.readline().split(';')[1].split('=')[1])
            delimiter = ',' if ',' in fh.readline() else None
        self.rows = np.loadtxt(path, delimiter=delimiter, skiprows=1)
        self.starts = self._segment_starts(self.rows[:, 1])

    @staticmethod
    def _segment_starts(wl_um):
        """Row indices where a new mode starts, plus the index of the last row.

        Same bookkeeping as MSclass19.Modedata.Import: each segment ends one row
        before the next start, so the final row of the file belongs to no mode.
        """
        starts, last = [0], round(wl_um[0], 5)
        for j in range(1, len(wl_um)):
            lam = round(wl_um[j], 5)
            if last >= lam:
                starts.append(j)
            last = lam
        starts.append(len(wl_um)-1)
        return starts

    @property
    def n_modes(self):
        """Number of mode segments that can be addressed with mode()."""
        return len(self.starts)-1

    def mode(self, m, kmin=0, kmax=1e3):
        """Rows [lambda (um), beta (um^-1), alpha/2 (um^-1)] of mode m with kmin <= beta <= kmax.

        Returns an empty array for segments with fewer than two points.
        """
        seg = self.rows[self.starts[m]:self.starts[m+1], 1:4]
        if len(seg) < 2:
            return np.empty((0, 3))
        return seg[(seg[:, 1] >= kmin) & (seg[:, 1] <= kmax)]
