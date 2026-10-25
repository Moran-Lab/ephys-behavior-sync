"""Peri-event time histograms and raster plots.

DRAFT: signatures only. Defaults follow the associated study: 50 ms bins, Gaussian smoothing
with sigma = 2 bins (100 ms), window from -1 s to +2 s, baseline = mean rate before onset.
"""

import numpy as np


def compute_psth(
    spike_times_s: np.ndarray,
    onsets_s: np.ndarray,
    pre_s: float = 1.0,
    post_s: float = 2.0,
    bin_s: float = 0.05,
    smooth_sigma_bins: float = 2.0,
    subtract_baseline: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin_centers_s, rate_hz) averaged over events."""
    raise NotImplementedError


def plot_raster_psth(spike_times_s, onsets_by_behavior, highlight=None, ax_raster=None, ax_psth=None):
    """Raster (top) and PSTH (bottom) for one unit, one color per behavior."""
    raise NotImplementedError
