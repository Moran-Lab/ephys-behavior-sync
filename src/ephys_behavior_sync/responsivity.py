"""Event-locked responsivity of single units.

Defaults reproduce the criteria used in the associated study:

* for each event, firing rate in the 1 s before onset vs. the 2 s after onset;
* paired t-test across events, for every (subject, behavior) with at least 3 events;
* significant when the raw p < 0.05 (no multiple-comparison correction by default;
  ``"holm"`` and ``"fdr_bh"`` are available, applied across the tests of one unit);
* a (unit, subject, behavior) row is flagged *slow* when the mean rate is below 1 Hz in both
  windows, and slow rows are never counted as responsive.

Window convention: the "before" window is (onset - pre, onset] and the "after" window is
(onset, onset + post], i.e. a spike exactly at onset counts as *before*.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

_CORRECTIONS = {None: None, "holm": "holm", "fdr_bh": "fdr_bh"}

COLUMNS = [
    "subject", "behavior", "n_events", "mean_before_hz", "mean_after_hz",
    "p", "p_adj", "direction", "slow", "responsive",
]


def _count_in(spikes: np.ndarray, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Number of spikes in each interval (start, stop]. ``spikes`` must be sorted."""
    return np.searchsorted(spikes, stops, side="right") - np.searchsorted(spikes, starts, side="right")


def unit_responsivity(
    spike_times_s: np.ndarray,
    onsets_by_key: dict[tuple[str, str], np.ndarray],
    pre_s: float = 1.0,
    post_s: float = 2.0,
    min_events: int = 3,
    correction: str | None = None,
    alpha: float = 0.05,
    slow_rate_hz: float = 1.0,
) -> pd.DataFrame:
    """Test one unit against every (subject, behavior) in ``onsets_by_key``.

    Parameters
    ----------
    spike_times_s : spike times of one unit, seconds, same clock as the onsets
    onsets_by_key : ``{(subject, behavior): onset times}``, e.g. from
        ``events.onsets_by_subject_behavior`` after ``sync.annotation_to_neural_time``
    pre_s, post_s : lengths of the windows before and after onset
    min_events : pairs with fewer events are skipped (no row)
    correction : None, "holm" or "fdr_bh". Applied across the rows (pairs) of this unit.
    alpha : significance threshold, applied to ``p_adj`` (equal to ``p`` without correction)
    slow_rate_hz : rows with mean rate below this in both windows are flagged ``slow``

    Returns
    -------
    DataFrame with one row per tested pair and columns ``COLUMNS``. ``direction`` is "up",
    "down" or "none" from the sign of (mean after - mean before), regardless of significance.
    If every event has identical before and after rates, p is set to 1.
    """
    if correction not in _CORRECTIONS:
        raise ValueError(f"correction must be None, 'holm' or 'fdr_bh', got {correction!r}")
    spikes = np.sort(np.asarray(spike_times_s, dtype=float))

    rows = []
    for (subject, behavior), onsets in onsets_by_key.items():
        onsets = np.asarray(onsets, dtype=float)
        if len(onsets) < min_events:
            continue
        before = _count_in(spikes, onsets - pre_s, onsets) / pre_s
        after = _count_in(spikes, onsets, onsets + post_s) / post_s
        if np.allclose(before, after):
            p = 1.0
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # near-identical differences
                p = float(ttest_rel(before, after).pvalue)
            if np.isnan(p):
                p = 1.0
        mean_before, mean_after = float(before.mean()), float(after.mean())
        direction = "up" if mean_after > mean_before else "down" if mean_after < mean_before else "none"
        rows.append((subject, behavior, len(onsets), mean_before, mean_after, p, p, direction,
                     bool(mean_before < slow_rate_hz and mean_after < slow_rate_hz)))

    df = pd.DataFrame(rows, columns=COLUMNS[:9])
    if correction is not None and len(df) > 0:
        df["p_adj"] = multipletests(df["p"].to_numpy(), method=correction)[1]
    df["responsive"] = (df["p_adj"] < alpha) & ~df["slow"]
    return df[COLUMNS]


def all_units_responsivity(
    spike_times_by_unit: list[np.ndarray],
    onsets_by_key: dict[tuple[str, str], np.ndarray],
    **kwargs,
) -> pd.DataFrame:
    """Run :func:`unit_responsivity` for each unit; adds a ``unit`` column (list position).

    Keyword arguments are passed on. Multiple-comparison correction, if requested,
    is applied within each unit.
    """
    tables = []
    for unit, spikes in enumerate(spike_times_by_unit):
        table = unit_responsivity(spikes, onsets_by_key, **kwargs)
        table.insert(0, "unit", unit)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)
