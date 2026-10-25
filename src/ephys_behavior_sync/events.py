"""Behavior annotations as a tidy table, and event onsets per (subject, behavior).

DRAFT: signatures only.

Expected columns of the events table:

    time_s   float  annotation time, seconds from video start
    subject  str    who performed the behavior
    behavior str    behavior label
    status   str    "START" or "STOP"
"""

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("time_s", "subject", "behavior", "status")


def validate_events(events: pd.DataFrame) -> pd.DataFrame:
    """Check columns, normalize labels (case, whitespace), and warn when STARTs and STOPs do not pair up."""
    raise NotImplementedError


def onsets_by_subject_behavior(
    events: pd.DataFrame,
    exclude_behaviors: tuple[str, ...] = (),
) -> dict[tuple[str, str], np.ndarray]:
    """Return {(subject, behavior): START times in seconds}, skipping excluded behaviors."""
    raise NotImplementedError
