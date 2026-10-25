import numpy as np
import pytest

from ephys_behavior_sync.synthetic import IGNORED_BEHAVIORS, make_raw_snippet, make_session


def test_same_seed_same_data_different_seed_different_data():
    a, b, c = make_session(seed=1), make_session(seed=1), make_session(seed=2)
    assert all(np.array_equal(x, y) for x, y in zip(a["spike_times_s"], b["spike_times_s"]))
    assert a["events"].equals(b["events"])
    assert not np.array_equal(a["spike_times_s"][0], c["spike_times_s"][0])
    r1, r2 = make_raw_snippet(seed=1, duration_s=1), make_raw_snippet(seed=1, duration_s=1)
    assert np.array_equal(r1["data"], r2["data"])


def test_events_table_is_well_formed():
    s = make_session(seed=0, n_events_per_key=8)
    ev = s["events"]
    assert list(ev.columns) == ["time_s", "subject", "behavior", "status"]
    assert ev["time_s"].is_monotonic_increasing
    # every key has 8 STARTs and 8 STOPs, each STOP after its START
    for (subject, behavior), g in ev.groupby(["subject", "behavior"]):
        starts = g.loc[g.status == "START", "time_s"].to_numpy()
        stops = g.loc[g.status == "STOP", "time_s"].to_numpy()
        assert len(starts) == len(stops) == 8
        assert np.all(stops > starts)


def test_onsets_are_separated_and_inside_the_recording():
    s = make_session(seed=0)
    all_onsets = np.sort(np.concatenate(list(s["truth"]["onsets_neural_s"].values())))
    assert np.all(np.diff(all_onsets) > 6.0)             # responses never overlap
    assert all_onsets[0] > 3.0 and all_onsets[-1] + 3.0 < s["duration_s"]


def test_neural_onsets_equal_video_starts_plus_first_frame():
    s = make_session(seed=0)
    starts = s["events"].query("status == 'START'")["time_s"].to_numpy() + s["first_frame_neural_s"]
    all_onsets = np.sort(np.concatenate(list(s["truth"]["onsets_neural_s"].values())))
    assert np.allclose(np.sort(starts), all_onsets)


def test_baseline_rates_match_truth():
    s = make_session(seed=0)
    units = s["truth"]["units"]
    for u, spikes in enumerate(s["spike_times_s"]):
        row = units.iloc[u]
        if row.role in ("none", "slow"):
            expected = row.baseline_hz * s["duration_s"]
            assert abs(len(spikes) - expected) < 5 * np.sqrt(expected)   # Poisson, 5 SD


def test_injected_responses_have_the_right_direction():
    s = make_session(seed=0)
    units = s["truth"]["units"]
    for _, row in units[units.role.isin(["up", "down"])].iterrows():
        spikes = s["spike_times_s"][int(row.unit)]
        onsets = s["truth"]["onsets_neural_s"][(row.subject, row.behavior)]
        before = np.mean([np.sum((spikes > t - 1) & (spikes <= t)) / 1.0 for t in onsets])
        after = np.mean([np.sum((spikes > t) & (spikes <= t + 2)) / 2.0 for t in onsets])
        if row.role == "up":
            assert after > 2 * before
        else:
            assert after < 0.5 * before


def test_roles_and_pairs():
    s = make_session(seed=0, n_slow_units=2, n_up_units=3, n_down_units=3, n_units=20)
    units = s["truth"]["units"]
    assert units.role.value_counts().to_dict() == {"none": 12, "up": 3, "down": 3, "slow": 2}
    resp = units[units.role.isin(["up", "down"])]
    assert len(set(zip(resp.subject, resp.behavior))) == 6          # distinct pairs
    assert not resp.behavior.isin(IGNORED_BEHAVIORS).any()
    assert (units[units.role == "slow"].baseline_hz == 0.3).all()


def test_too_many_events_for_the_recording_raises():
    with pytest.raises(ValueError):
        make_session(duration_s=200.0, n_events_per_key=8)


def test_raw_snippet_truth_is_consistent():
    snip = make_raw_snippet(seed=0, duration_s=2.0, n_channels=8)
    assert snip["data"].shape == (8, 64000)
    n = len(snip["spike_times_s"])
    assert len(snip["spike_channels"]) == len(snip["spike_amplitudes"]) == n
    assert np.all(np.diff(snip["spike_times_s"]) >= 0)
    assert np.all((snip["spike_amplitudes"] >= 60) & (snip["spike_amplitudes"] <= 120))
