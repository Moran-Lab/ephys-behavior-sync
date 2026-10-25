"""Cleaning of raw multichannel recordings before spike sorting.

All functions take and return arrays shaped ``(n_channels, n_samples)`` and never modify
their input. Data stay in floating point until the single, explicit cast in :func:`to_int16`,
so rounding happens once.

Memory note: the functions work on whole arrays. For recordings that do not fit in memory,
call them on slices of channels (``bandpass_filter``) or time chunks (``median_reference``
already chunks internally).
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_filter(
    data: np.ndarray,
    fs: float,
    low_hz: float = 300.0,
    high_hz: float = 6000.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter, applied to each channel independently.

    Filtering forward and backward (``sosfiltfilt``) cancels the phase shift, so spike
    shapes and timing are preserved, and doubles the effective filter order.

    Parameters
    ----------
    data : (n_channels, n_samples) array
    fs : sampling rate in Hz
    low_hz, high_hz : pass band edges in Hz
    order : Butterworth order of each pass (4 gives an effective order of 8)

    Returns
    -------
    (n_channels, n_samples) float64 array
    """
    if not 0 < low_hz < high_hz < fs / 2:
        raise ValueError(f"need 0 < low_hz < high_hz < fs/2, got {low_hz}, {high_hz}, fs={fs}")
    x = np.asarray(data, dtype=np.float64)
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def median_reference(data: np.ndarray, chunk_samples: int = 1_000_000) -> np.ndarray:
    """Subtract, sample by sample, the median of the *other* channels from each channel.

    Noise that appears on all channels at once (movement, chewing, electrical interference)
    is removed, while a spike seen on a single channel is kept. The channel's own value is
    excluded from its reference so that its spikes cannot leak into the reference.

    Parameters
    ----------
    data : (n_channels, n_samples) array, at least 3 channels
    chunk_samples : number of samples processed at a time (limits memory use)
    """
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("data must be 2-D (n_channels, n_samples)")
    n_channels, n_samples = x.shape
    if n_channels < 3:
        raise ValueError("median referencing needs at least 3 channels")
    out = np.empty_like(x)
    for start in range(0, n_samples, chunk_samples):
        block = x[:, start:start + chunk_samples]
        for ch in range(n_channels):
            others = np.delete(block, ch, axis=0)
            out[ch, start:start + chunk_samples] = block[ch] - np.median(others, axis=0)
    return out


def preprocess(
    data: np.ndarray,
    fs: float,
    low_hz: float = 300.0,
    high_hz: float = 6000.0,
    order: int = 4,
) -> np.ndarray:
    """Band-pass filter, then median-reference. Returns float64."""
    return median_reference(bandpass_filter(data, fs, low_hz, high_hz, order))


def to_int16(data: np.ndarray) -> np.ndarray:
    """Round and cast to int16, clipping values outside the int16 range (the only cast in the pipeline)."""
    info = np.iinfo(np.int16)
    return np.clip(np.rint(data), info.min, info.max).astype(np.int16)
