import numpy as np
import pytest
from scipy.signal import find_peaks

from ephys_behavior_sync.preprocessing import (
    bandpass_filter, median_reference, preprocess, to_int16,
)
from ephys_behavior_sync.synthetic import make_raw_snippet

FS = 32000.0


def _amplitude(x):
    """Amplitude of a sine wave from its RMS, ignoring the edges where filters misbehave."""
    core = x[..., int(0.2 * FS):-int(0.2 * FS)]
    return np.sqrt(2) * np.sqrt(np.mean(core ** 2, axis=-1))


def test_bandpass_removes_low_frequency_and_keeps_spike_band():
    t = np.arange(int(2 * FS)) / FS
    data = np.vstack([np.sin(2 * np.pi * 50 * t), np.sin(2 * np.pi * 1000 * t)])
    out = bandpass_filter(data, FS)
    low, mid = _amplitude(out)
    assert low < 0.01      # 50 Hz: more than 40 dB down
    assert mid > 0.95      # 1 kHz: passed almost unchanged


def test_bandpass_is_zero_phase():
    # a pulse's peak must not move (a causal filter would delay it)
    n = int(FS)
    t = (np.arange(n) - n // 2) / FS
    pulse = np.exp(-0.5 * (t / 0.3e-3) ** 2)[None, :]
    out = bandpass_filter(pulse, FS)
    assert abs(int(np.argmax(np.abs(out[0]))) - n // 2) <= 1


def test_bandpass_does_not_modify_input_and_returns_float():
    data = np.random.default_rng(0).integers(-100, 100, size=(4, 5000)).astype(np.int16)
    copy = data.copy()
    out = bandpass_filter(data, FS)
    assert np.array_equal(data, copy)
    assert out.dtype == np.float64


def test_bandpass_rejects_bad_band():
    with pytest.raises(ValueError):
        bandpass_filter(np.zeros((3, 100)), FS, low_hz=6000, high_hz=300)
    with pytest.raises(ValueError):
        bandpass_filter(np.zeros((3, 100)), FS, high_hz=FS)  # above Nyquist


def test_median_reference_exact_small_example():
    data = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [10.0, 10.0]])
    out = median_reference(data)
    # channel 0: median of the others [2, 3, 10] is 3, so 1 - 3 = -2
    assert out[0, 0] == -2.0
    # channel 3: median of [1, 2, 3] is 2, so 10 - 2 = 8
    assert out[3, 0] == 8.0


def test_median_reference_chunking_gives_same_result():
    data = np.random.default_rng(1).normal(size=(6, 1000))
    assert np.allclose(median_reference(data, chunk_samples=1000), median_reference(data, chunk_samples=77))


def test_median_reference_needs_three_channels():
    with pytest.raises(ValueError):
        median_reference(np.zeros((2, 10)))


def test_median_reference_removes_shared_signal_but_keeps_single_channel_spike():
    rng = np.random.default_rng(2)
    shared = rng.normal(0, 20, size=2000)
    data = np.tile(shared, (8, 1))
    data[3, 1000] += 100.0                    # a "spike" on one channel only
    out = median_reference(data)
    assert np.allclose(np.delete(out, 3, axis=0)[:, :999], 0.0)   # shared part gone
    assert out[3, 1000] == pytest.approx(100.0)                     # spike intact


def test_preprocess_decorrelates_channels():
    snip = make_raw_snippet(seed=3, spike_rate_hz=0.0)             # noise only
    def mean_offdiag_corr(x):
        c = np.corrcoef(x)
        return c[~np.eye(len(c), dtype=bool)].mean()
    filtered = bandpass_filter(snip["data"], FS)
    assert mean_offdiag_corr(filtered) > 0.8                       # shared noise dominates
    assert abs(mean_offdiag_corr(preprocess(snip["data"], FS))) < 0.1


def test_spikes_are_recovered_after_preprocessing():
    snip = make_raw_snippet(seed=4)
    clean = preprocess(snip["data"], FS)
    refractory = int(0.001 * FS)
    detected = []
    for ch in range(clean.shape[0]):
        x = clean[ch]
        sigma = np.median(np.abs(x)) / 0.6745                       # robust noise estimate
        peaks, _ = find_peaks(-x, height=4.5 * sigma, distance=refractory)
        detected.append(peaks / FS)

    hits = 0
    for t, ch in zip(snip["spike_times_s"], snip["spike_channels"]):
        hits += np.any(np.abs(detected[ch] - t) < 0.0005)
    n_true = len(snip["spike_times_s"])
    n_detected = sum(len(d) for d in detected)
    assert hits / n_true > 0.9                                      # recall
    assert (n_detected - hits) / n_detected < 0.05                  # few false detections


def test_to_int16_rounds_and_clips():
    out = to_int16(np.array([0.4, 0.6, -1.5, 1e6, -1e6]))
    assert out.dtype == np.int16
    assert list(out) == [0, 1, -2, 32767, -32768]
