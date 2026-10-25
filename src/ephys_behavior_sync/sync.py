"""Alignment of events recorded on one clock to the neural recording clock.

DRAFT: signatures only. Implemented in the next step, with tests that recover a known offset.

Assumptions (also stated in the README):
* The two clocks do not drift measurably over a recording (about 40 minutes), so a single
  trigger pulse is enough to define the offset.
* Video frames are acquired at a constant rate, so the first frame time is enough to map
  annotation time (seconds from video start) onto the neural clock.
"""

import numpy as np


def detect_edges(digital_words: np.ndarray, bit: int, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Rising and falling edge times (seconds) of one bit of a digital-input word stream."""
    raise NotImplementedError


def estimate_clock_offset_s(trigger_times_a_s: np.ndarray, trigger_times_b_s: np.ndarray) -> float:
    """Offset to add to clock-B times to express them on clock A, from the first trigger pulse."""
    raise NotImplementedError


def annotation_to_neural_time(annotation_times_s: np.ndarray, first_frame_neural_s: float) -> np.ndarray:
    """Map annotation times (seconds from video start) onto the neural clock."""
    raise NotImplementedError
