import matplotlib
matplotlib.use("Agg")          # draw without opening windows
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ephys_behavior_sync.events import onsets_by_subject_behavior  # noqa: E402
from ephys_behavior_sync.psth import align_spikes, compute_psth, plot_raster_psth  # noqa: E402
from ephys_behavior_sync.sync import annotation_to_neural_time  # noqa: E402
from ephys_behavior_sync.synthetic import IGNORED_BEHAVIORS, make_session  # noqa: E402

ONSETS = np.array([10.0, 20.0, 30.0, 40.0])


def test_align_spikes_gives_times_relative_to_onset_and_respects_the_window():
    spikes = np.array([8.9, 9.0, 10.5, 12.0, 12.1])        # onset 10, window [9, 12]
    (trial,) = align_spikes(spikes, [10.0], pre_s=1.0, post_s=2.0)
    assert np.allclose(trial, [-1.0, 0.5, 2.0])              # 8.9 and 12.1 are outside


def test_bin_centers_and_known_rate():
    # one spike 0.525 s after every onset -> the bin [0.5, 0.55) holds 4 spikes / (4 events * 0.05 s) = 20 Hz
    spikes = ONSETS + 0.525
    centers, rate = compute_psth(spikes, ONSETS, bin_s=0.05, smooth_sigma_bins=0, subtract_baseline=False)
    assert len(centers) == 60 and centers[0] == pytest.approx(-0.975) and centers[-1] == pytest.approx(1.975)
    peak = int(np.argmax(rate))
    assert centers[peak] == pytest.approx(0.525)
    assert rate[peak] == pytest.approx(20.0)
    assert np.count_nonzero(rate) == 1


def test_baseline_subtraction_makes_pre_onset_mean_zero():
    rng = np.random.default_rng(0)
    spikes = np.sort(rng.uniform(0, 60, 600))                # 10 Hz background
    centers, rate = compute_psth(spikes, ONSETS, smooth_sigma_bins=0, subtract_baseline=True)
    assert np.mean(rate[centers < 0]) == pytest.approx(0.0, abs=1e-9)


def test_smoothing_keeps_the_total_number_of_spikes():
    spikes = ONSETS + 0.525                                  # mass far from the edges
    _, raw = compute_psth(spikes, ONSETS, smooth_sigma_bins=0, subtract_baseline=False)
    _, smooth = compute_psth(spikes, ONSETS, smooth_sigma_bins=2, subtract_baseline=False)
    assert smooth.sum() == pytest.approx(raw.sum())
    assert smooth.max() < raw.max()                          # but the peak is spread out


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        compute_psth(np.array([1.0]), np.array([]))
    with pytest.raises(ValueError):
        compute_psth(np.array([1.0]), ONSETS, pre_s=1.0, post_s=2.0, bin_s=0.07)


def test_simulated_responses_show_in_the_psth():
    s = make_session(seed=0)
    video = onsets_by_subject_behavior(s["events"], exclude_behaviors=IGNORED_BEHAVIORS)
    units = s["truth"]["units"]
    for _, row in units[units.role.isin(["up", "down"])].iterrows():
        onsets = annotation_to_neural_time(video[(row.subject, row.behavior)], s["first_frame_neural_s"])
        centers, rate = compute_psth(s["spike_times_s"][int(row.unit)], onsets)
        after = rate[(centers > 0) & (centers < 2)].mean()
        assert (after > 0) if row.role == "up" else (after < 0)


def test_plot_returns_figure_with_one_psth_line_per_behavior():
    s = make_session(seed=0)
    video = onsets_by_subject_behavior(s["events"], exclude_behaviors=IGNORED_BEHAVIORS)
    by_behavior = {
        b: annotation_to_neural_time(video[("rat_A", b)], s["first_frame_neural_s"])
        for b in ("allo-groom", "nose-to-nose")
    }
    fig, (ax_raster, ax_psth) = plot_raster_psth(s["spike_times_s"][0], by_behavior, title="unit 0")
    try:
        assert len(ax_psth.lines) == 2 + 1                   # 2 PSTH curves + the onset line
        assert ax_psth.get_xlim() == (-1.0, 2.0)
        assert ax_raster.get_title() == "unit 0"
    finally:
        plt.close(fig)
