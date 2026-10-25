"""Cleaning of raw multichannel recordings before spike sorting.

DRAFT: signatures only. Implemented in the next step, with tests on synthetic data.
"""

import numpy as np


def bandpass_filter(
    data: np.ndarray,
    fs: float,
    low_hz: float = 300.0,
    high_hz: float = 6000.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter, applied per channel.

    data : (n_channels, n_samples) float array
    Returns an array of the same shape (float, not cast back to integers).
    """
    raise NotImplementedError


def median_reference(data: np.ndarray) -> np.ndarray:
    """Subtract, sample by sample, the median of the *other* channels from each channel.

    Removes artifacts shared across channels while keeping spikes that appear on one channel.
    data : (n_channels, n_samples)
    """
    raise NotImplementedError
