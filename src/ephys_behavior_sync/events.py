"""Behavior annotations as a tidy table, and event onsets per (subject, behavior).

Expected columns of the events table:

    time_s   float  annotation time, seconds from video start
    subject  str    who performed the behavior
    behavior str    behavior label
    status   str    "START" or "STOP"

Getting an annotation export into this shape (renaming columns, skipping header lines) is
left to the user, since it depends on the annotation software.
"""

import warnings

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("time_s", "subject", "behavior", "status")
VALID_STATUS = ("START", "STOP")


def validate_events(events: pd.DataFrame, behavior_aliases: dict[str, str] | None = None) -> pd.DataFrame:
    """Check and normalize an events table. Returns a cleaned copy sorted by time.

    * columns must include ``REQUIRED_COLUMNS``; extra columns are dropped
    * ``time_s`` must be numeric and non-negative
    * subject and behavior lose surrounding spaces; behavior is lower-cased; status is upper-cased
    * ``behavior_aliases`` maps alternative labels onto one name, e.g. ``{"body-sniff": "gentle-sniff"}``
    * status must be START or STOP, else ``ValueError``
    * a warning (not an error) is raised for every (subject, behavior) whose rows do not
      alternate START, STOP, START, STOP... in time order, e.g. a missing STOP
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"events table is missing columns: {missing}")
    df = events.loc[:, list(REQUIRED_COLUMNS)].copy()

    df["time_s"] = pd.to_numeric(df["time_s"], errors="raise")
    if df["time_s"].isna().any() or (df["time_s"] < 0).any():
        raise ValueError("time_s must be non-negative numbers without gaps")
    df["subject"] = df["subject"].astype(str).str.strip()
    df["behavior"] = df["behavior"].astype(str).str.strip().str.lower()
    if behavior_aliases:
        aliases = {k.strip().lower(): v.strip().lower() for k, v in behavior_aliases.items()}
        df["behavior"] = df["behavior"].replace(aliases)
    df["status"] = df["status"].astype(str).str.strip().str.upper()
    bad = sorted(set(df["status"]) - set(VALID_STATUS))
    if bad:
        raise ValueError(f"status must be START or STOP, found {bad}")

    df = df.sort_values("time_s", kind="stable").reset_index(drop=True)
    for (subject, behavior), g in df.groupby(["subject", "behavior"]):
        seq = g["status"].tolist()
        if seq != (["START", "STOP"] * len(seq))[:len(seq)] or len(seq) % 2:
            warnings.warn(
                f"{subject} / {behavior}: START and STOP rows do not alternate "
                f"({seq.count('START')} START, {seq.count('STOP')} STOP)"
            )
    return df


def onsets_by_subject_behavior(
    events: pd.DataFrame,
    exclude_behaviors: tuple[str, ...] = (),
    behavior_aliases: dict[str, str] | None = None,
) -> dict[tuple[str, str], np.ndarray]:
    """Return ``{(subject, behavior): sorted START times in seconds}``.

    The table is validated first (see :func:`validate_events`). Behaviors in
    ``exclude_behaviors`` (compared after lower-casing) are left out. Times stay on the
    clock of the table; use ``sync.annotation_to_neural_time`` to move them to the neural clock.
    """
    df = validate_events(events, behavior_aliases)
    excluded = {b.strip().lower() for b in exclude_behaviors}
    starts = df[(df["status"] == "START") & ~df["behavior"].isin(excluded)]
    return {
        (subject, behavior): np.sort(g["time_s"].to_numpy())
        for (subject, behavior), g in starts.groupby(["subject", "behavior"])
    }
