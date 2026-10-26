import warnings

import numpy as np
import pytest

from ephys_behavior_sync.sync import annotation_to_neural_time, detect_edges, estimate_clock_offset_s
from ephys_behavior_sync.synthetic import make_digital_words

FS = 32000.0


def test_detect_edges_recovers_pulse_times():
    pulses = [1.0, 2.5, 4.25]
    words = make_digital_words(pulses, duration_s=6.0, fs=FS, bit=3, pulse_width_s=0.01)
    rising, falling = detect_edges(words, bit=3, fs=FS)
    assert np.allclose(rising, pulses, atol=1 / FS)
    assert np.allclose(falling, np.array(pulses) + 0.01, atol=1 / FS)


def test_detect_edges_reads_only_the_requested_bit():
    # bit 4 toggles at 5 Hz as a distractor; the pulses are on bit 3
    words = make_digital_words([1.0], duration_s=3.0, fs=FS, bit=3)
    rising, _ = detect_edges(words, bit=3, fs=FS)
    assert len(rising) == 1
    distractor_rising, _ = detect_edges(words, bit=4, fs=FS)
    assert len(distractor_rising) > 5


def test_bit_already_high_at_start_gives_no_rising_edge_at_zero():
    words = make_digital_words([], duration_s=1.0, fs=FS, bit=3, start_high=True)
    rising, falling = detect_edges(words, bit=3, fs=FS)
    assert len(rising) == 0
    assert np.allclose(falling, 0.01, atol=1 / FS)


def test_detect_edges_handles_no_pulses_and_bad_input():
    words = make_digital_words([], duration_s=1.0, fs=FS)
    rising, falling = detect_edges(words, bit=3, fs=FS)
    assert len(rising) == 0 and len(falling) == 0
    with pytest.raises(ValueError):
        detect_edges(words.astype(float), bit=3, fs=FS)
    with pytest.raises(ValueError):
        detect_edges(words, bit=16, fs=FS)      # uint16 words only have bits 0-15


def test_offset_recovers_known_value():
    true_offset = 7.25                          # clock A = clock B + 7.25 s
    pulse_b = np.array([3.25])
    assert estimate_clock_offset_s(pulse_b + true_offset, pulse_b) == pytest.approx(true_offset)


def test_offset_from_two_simulated_recordings():
    # neural clock sees the pulse at 10.5 s, the logger at 3.25 s -> offset 7.25 s
    neural = make_digital_words([10.5], duration_s=15.0, fs=FS)
    logger = make_digital_words([3.25], duration_s=15.0, fs=FS)
    a, _ = detect_edges(neural, 3, FS)
    b, _ = detect_edges(logger, 3, FS)
    assert estimate_clock_offset_s(a, b) == pytest.approx(7.25, abs=2 / FS)


def test_offset_needs_pulses_on_both_clocks():
    with pytest.raises(ValueError):
        estimate_clock_offset_s(np.array([]), np.array([1.0]))


def test_offset_warns_when_clocks_drift_but_not_when_they_agree():
    a = np.array([10.0, 70.0, 130.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimate_clock_offset_s(a, a - 7.0)                         # perfect agreement
    assert len(caught) == 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimate_clock_offset_s(a, a - 7.0 + np.array([0, 0.002, 0.005]))   # 5 ms drift
    assert len(caught) == 1 and "drift" in str(caught[0].message)


def test_annotation_to_neural_time():
    out = annotation_to_neural_time(np.array([0.0, 1.5, 10.0]), first_frame_neural_s=9.25)
    assert np.allclose(out, [9.25, 10.75, 19.25])
