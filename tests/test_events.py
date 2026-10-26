import warnings

import numpy as np
import pandas as pd
import pytest

from ephys_behavior_sync.events import onsets_by_subject_behavior, validate_events
from ephys_behavior_sync.synthetic import IGNORED_BEHAVIORS, make_session
from ephys_behavior_sync.sync import annotation_to_neural_time


def _table(rows):
    return pd.DataFrame(rows, columns=["time_s", "subject", "behavior", "status"])


def _warnings_from(func, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = func(*args, **kwargs)
    return result, [str(w.message) for w in caught]


def test_labels_are_normalized_and_sorted():
    raw = _table([
        (5.0, " rat_A ", "Nose-To-Nose ", "stop"),
        (2.0, "rat_A", "nose-to-nose", "Start"),
    ])
    out, msgs = _warnings_from(validate_events, raw)
    assert msgs == []
    assert out["time_s"].tolist() == [2.0, 5.0]
    assert out["subject"].tolist() == ["rat_A", "rat_A"]
    assert out["behavior"].unique().tolist() == ["nose-to-nose"]
    assert out["status"].tolist() == ["START", "STOP"]


def test_input_table_is_not_modified():
    raw = _table([(2.0, "rat_A", "Nose-To-Nose", "start"), (5.0, "rat_A", "Nose-To-Nose", "stop")])
    copy = raw.copy()
    validate_events(raw)
    assert raw.equals(copy)


def test_aliases_merge_labels():
    raw = _table([
        (1.0, "rat_A", "Body-Sniff", "START"), (2.0, "rat_A", "Body-Sniff", "STOP"),
        (5.0, "rat_A", "gentle-sniff", "START"), (6.0, "rat_A", "gentle-sniff", "STOP"),
    ])
    out = onsets_by_subject_behavior(raw, behavior_aliases={"body-sniff": "gentle-sniff"})
    assert list(out) == [("rat_A", "gentle-sniff")]
    assert out[("rat_A", "gentle-sniff")].tolist() == [1.0, 5.0]


def test_missing_stop_gives_a_warning_naming_the_behavior():
    raw = _table([
        (1.0, "rat_A", "allo-groom", "START"), (2.0, "rat_A", "allo-groom", "STOP"),
        (5.0, "rat_A", "allo-groom", "START"),                      # STOP missing
        (7.0, "rat_B", "nose-to-nose", "START"), (8.0, "rat_B", "nose-to-nose", "STOP"),
    ])
    _, msgs = _warnings_from(validate_events, raw)
    assert len(msgs) == 1 and "rat_A / allo-groom" in msgs[0]


def test_two_starts_in_a_row_gives_a_warning():
    raw = _table([(1.0, "rat_A", "allo-groom", "START"), (2.0, "rat_A", "allo-groom", "START")])
    _, msgs = _warnings_from(validate_events, raw)
    assert len(msgs) == 1


def test_invalid_tables_raise():
    ok = [(1.0, "rat_A", "x", "START")]
    with pytest.raises(ValueError):
        validate_events(_table([(1.0, "rat_A", "x", "BEGIN")]))      # unknown status
    with pytest.raises(ValueError):
        validate_events(_table([(-1.0, "rat_A", "x", "START")]))     # negative time
    with pytest.raises(ValueError):
        validate_events(_table(ok).drop(columns="subject"))          # missing column


def test_onsets_use_start_rows_only_and_respect_exclusions():
    raw = _table([
        (1.0, "rat_A", "allo-groom", "START"), (2.0, "rat_A", "allo-groom", "STOP"),
        (9.0, "rat_A", "approaching", "START"), (9.5, "rat_A", "approaching", "STOP"),
        (4.0, "rat_B", "allo-groom", "START"), (5.0, "rat_B", "allo-groom", "STOP"),
    ])
    out = onsets_by_subject_behavior(raw, exclude_behaviors=("Approaching",))
    assert set(out) == {("rat_A", "allo-groom"), ("rat_B", "allo-groom")}
    assert out[("rat_A", "allo-groom")].tolist() == [1.0]


def test_synthetic_session_events_map_back_to_the_true_neural_onsets():
    # events table (video clock) -> onsets -> neural clock must equal the generator's ground truth
    s = make_session(seed=5)
    onsets_video = onsets_by_subject_behavior(s["events"], exclude_behaviors=IGNORED_BEHAVIORS)
    truth = {k: v for k, v in s["truth"]["onsets_neural_s"].items() if k[1] not in IGNORED_BEHAVIORS}
    assert set(onsets_video) == set(truth)
    for key, times in onsets_video.items():
        neural = annotation_to_neural_time(times, s["first_frame_neural_s"])
        assert np.allclose(neural, truth[key])
