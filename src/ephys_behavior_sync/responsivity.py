"""Event-locked responsivity of single units.

DRAFT: signatures only.

Defaults reproduce the criteria used in the associated study:
* per event, firing rate in the 1 s before onset vs. the 2 s after onset;
* paired t-test across events, at least 3 events per (subject, behavior);
* significant when the raw p < 0.05 (no multiple-comparison correction by default;
  "holm" and "fdr_bh" are available as options);
* units with firing rate < 1 Hz in both windows are flagged as slow and dropped from the responsive set.
"""

import numpy as np
import pandas as pd


def test_unit_responsivity(
    spike_times_s: np.ndarray,
    onsets_by_key: dict[tuple[str, str], np.ndarray],
    pre_s: float = 1.0,
    post_s: float = 2.0,
    min_events: int = 3,
    correction: str | None = None,
    alpha: float = 0.05,
    slow_rate_hz: float = 1.0,
) -> pd.DataFrame:
    """One row per (subject, behavior): n_events, mean rates, p value, direction, slow flag, responsive flag."""
    raise NotImplementedError
