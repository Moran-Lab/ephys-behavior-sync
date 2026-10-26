import numpy as np
import pytest

from ephys_behavior_sync.events import onsets_by_subject_behavior
from ephys_behavior_sync.responsivity import all_units_responsivity, unit_responsivity
from ephys_behavior_sync.sync import annotation_to_neural_time
from ephys_behavior_sync.synthetic import IGNORED_BEHAVIORS, make_session

KEY = ("rat_A", "allo-groom")


def _spikes_around(onsets, n_before, n_after, pre=1.0, post=2.0):
    """Evenly spaced spikes: n_before in the second before each onset, n_after in the 2 s after."""
    out = []
    for t in onsets:
        out += list(t - pre + (np.arange(n_before) + 0.5) * pre / max(n_before, 1)) if n_before else []
        out += list(t + (np.arange(n_after) + 0.5) * post / max(n_after, 1)) if n_after else []
    return np.sort(np.array(out))


def test_rates_are_counts_divided_by_window_length():
    onsets = np.array([10.0, 30.0, 50.0])
    spikes = _spikes_around(onsets, n_before=2, n_after=6)       # 2 Hz before, 3 Hz after
    row = unit_responsivity(spikes, {KEY: onsets}).iloc[0]
    assert row.mean_before_hz == pytest.approx(2.0)
    assert row.mean_after_hz == pytest.approx(3.0)
    assert row.n_events == 3 and row.direction == "up"


def test_window_edges_belong_to_the_documented_side():
    onsets = np.array([10.0, 30.0, 50.0])
    # one spike exactly at onset -> counted as BEFORE; one exactly at onset - 1 s -> outside;
    # one exactly at onset + 2 s -> counted as AFTER
    spikes = np.sort(np.concatenate([onsets, onsets - 1.0, onsets + 2.0]))
    row = unit_responsivity(spikes, {KEY: onsets}).iloc[0]
    assert row.mean_before_hz == pytest.approx(1.0)     # 1 spike / 1 s
    assert row.mean_after_hz == pytest.approx(0.5)      # 1 spike / 2 s
    assert row.direction == "down"


def test_identical_rates_give_p_equal_one_and_not_responsive():
    onsets = np.array([10.0, 30.0, 50.0])
    spikes = _spikes_around(onsets, n_before=2, n_after=4)       # 2 Hz both sides
    row = unit_responsivity(spikes, {KEY: onsets}).iloc[0]
    assert row.p == 1.0 and row.direction == "none" and not row.responsive


def test_slow_unit_is_flagged_and_never_responsive():
    onsets = np.arange(10.0, 90.0, 10.0)                        # 8 events
    spikes = _spikes_around(onsets, n_before=0, n_after=1)      # 0 Hz before, 0.5 Hz after
    row = unit_responsivity(spikes, {KEY: onsets}).iloc[0]
    assert row.slow and row.p < 0.05 and not row.responsive     # significant, but dropped as slow


def test_fast_unit_with_clear_increase_is_responsive():
    onsets = np.arange(10.0, 90.0, 10.0)
    rng = np.random.default_rng(0)
    spikes = np.sort(np.concatenate([_spikes_around(onsets, 5, 20), rng.uniform(0, 100, 20)]))
    row = unit_responsivity(spikes, {KEY: onsets}).iloc[0]
    assert row.responsive and row.direction == "up" and not row.slow


def test_pairs_with_too_few_events_are_skipped():
    onsets = {KEY: np.array([10.0, 20.0]), ("rat_B", "nose-to-nose"): np.array([30.0, 40.0, 50.0])}
    out = unit_responsivity(np.array([1.0, 2.0]), onsets, min_events=3)
    assert out[["subject", "behavior"]].values.tolist() == [["rat_B", "nose-to-nose"]]
    assert len(unit_responsivity(np.array([1.0]), {}).index) == 0


def test_correction_options():
    rng = np.random.default_rng(1)
    onsets = {("rat_A", "a"): np.arange(10.0, 90.0, 10.0), ("rat_A", "b"): np.arange(110.0, 190.0, 10.0)}
    spikes = np.sort(rng.uniform(0, 200, 800))
    spikes = np.sort(np.concatenate([spikes, _spikes_around(onsets[("rat_A", "a")], 0, 30)]))
    none = unit_responsivity(spikes, onsets)
    holm = unit_responsivity(spikes, onsets, correction="holm")
    fdr = unit_responsivity(spikes, onsets, correction="fdr_bh")
    assert (none.p_adj == none.p).all()
    assert holm.p.tolist() == none.p.tolist()                    # raw p unchanged
    assert holm.p_adj.min() == pytest.approx(min(1.0, 2 * none.p.min()))   # Holm, 2 tests
    assert (holm.p_adj >= none.p).all() and (fdr.p_adj >= none.p).all()
    with pytest.raises(ValueError):
        unit_responsivity(spikes, onsets, correction="bonferroni")


def test_recovers_injected_responses_from_a_synthetic_session():
    s = make_session(seed=0)
    video_onsets = onsets_by_subject_behavior(s["events"], exclude_behaviors=IGNORED_BEHAVIORS)
    onsets = {k: annotation_to_neural_time(v, s["first_frame_neural_s"]) for k, v in video_onsets.items()}
    res = all_units_responsivity(s["spike_times_s"], onsets)
    truth = s["truth"]["units"]

    res = res.merge(truth, on="unit", how="left", suffixes=("", "_true"))
    is_target = (res.subject == res.subject_true) & (res.behavior == res.behavior_true)
    targets = res[is_target]
    others = res[~is_target]

    # every injected response is found, with the right direction
    assert len(targets) == 6
    assert targets.responsive.all()
    assert (targets.direction == targets.role.map({"up": "up", "down": "down"})).all()
    # a 0.3 Hz unit is flagged "slow" in almost every pair (by chance a pair can exceed 1 Hz),
    # and rows flagged slow are never reported as responsive
    slow_unit_rows = res[res.role == "slow"]
    assert slow_unit_rows.slow.mean() > 0.9
    assert not res[res.slow].responsive.any()
    # elsewhere only chance-level false positives (5% expected without correction)
    assert others.responsive.mean() < 0.12
    # with Holm correction within each unit, false positives all but disappear
    res_holm = all_units_responsivity(s["spike_times_s"], onsets, correction="holm")
    res_holm = res_holm.merge(truth, on="unit", how="left", suffixes=("", "_true"))
    target_holm = (res_holm.subject == res_holm.subject_true) & (res_holm.behavior == res_holm.behavior_true)
    assert res_holm[target_holm].responsive.all()
    assert res_holm[~target_holm].responsive.sum() <= 2
