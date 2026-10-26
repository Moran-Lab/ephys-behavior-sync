"""Alignment of events recorded on one clock to the neural recording clock.

Assumptions (also stated in the README):

* The two clocks do not drift measurably over a recording (about 40 minutes), so a single
  trigger pulse defines the offset. :func:`estimate_clock_offset_s` warns if extra pulses
  disagree with that.
* Video frames are acquired at a constant rate, so the time of the first frame is enough to
  map annotation time (seconds from video start) onto the neural clock.

All times are in seconds.
"""

import warnings

import numpy as np


def detect_edges(digital_words: np.ndarray, bit: int, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Rising and falling edge times (seconds) of one bit of a digital-input word stream.

    Parameters
    ----------
    digital_words : 1-D integer array, one word per sample (bit ``k`` has value ``2**k``)
    bit : which bit carries the signal (0 is the least significant)
    fs : sampling rate in Hz

    Returns
    -------
    rising_s, falling_s : arrays of times. An edge is timed at the first sample with the new
    state, so its time is ``sample_index / fs``. A bit that is already high at sample 0 produces
    no rising edge there.
    """
    words = np.asarray(digital_words)
    if words.ndim != 1 or not np.issubdtype(words.dtype, np.integer):
        raise ValueError("digital_words must be a 1-D integer array")
    if not 0 <= bit < 8 * words.dtype.itemsize:
        raise ValueError(f"bit {bit} does not exist in {words.dtype} words")
    if len(words) == 0:
        return np.array([]), np.array([])
    level = ((words >> bit) & 1).astype(np.int8)
    step = np.diff(level, prepend=level[0])
    return np.flatnonzero(step == 1) / fs, np.flatnonzero(step == -1) / fs


def estimate_clock_offset_s(
    trigger_times_a_s: np.ndarray,
    trigger_times_b_s: np.ndarray,
    tolerance_s: float = 1e-3,
) -> float:
    """Offset to add to clock-B times to express them on clock A, from the first trigger pulse.

    Both inputs are the times of the *same* physical pulses, seen by each clock. The offset is
    ``first pulse on A - first pulse on B``. If both clocks recorded more than one pulse, the
    later pulses are used only as a check: a warning is raised when the intervals since the
    first pulse differ by more than ``tolerance_s`` (a sign of clock drift or a missed pulse).
    """
    a = np.asarray(trigger_times_a_s, dtype=float)
    b = np.asarray(trigger_times_b_s, dtype=float)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("both clocks need at least one trigger pulse")
    if len(a) != len(b) and min(len(a), len(b)) > 1:
        warnings.warn(f"different number of pulses on the two clocks ({len(a)} vs {len(b)})")
    n = min(len(a), len(b))
    if n > 1:
        discrepancy = np.max(np.abs((a[:n] - a[0]) - (b[:n] - b[0])))
        if discrepancy > tolerance_s:
            warnings.warn(
                f"clocks disagree by up to {discrepancy * 1e3:.1f} ms across {n} pulses; "
                "a single-pulse offset assumes no drift"
            )
    return float(a[0] - b[0])


def annotation_to_neural_time(annotation_times_s: np.ndarray, first_frame_neural_s: float) -> np.ndarray:
    """Map annotation times (seconds from video start) onto the neural clock.

    Assumes annotation time zero is the first video frame and the frame rate is constant.
    """
    return np.asarray(annotation_times_s, dtype=float) + float(first_frame_neural_s)
