"""Synthetic data with known ground truth, used by the demo and by the tests.

Two generators, kept separate because raw signals are large and spike trains are not:

* :func:`make_raw_snippet`: a few seconds of multichannel raw signal with spikes, independent
  noise and noise shared by all channels. Used to test preprocessing.
* :func:`make_digital_words`: a digital-input stream with trigger pulses at known times.
* :func:`make_session`: spike trains of several units over a full session, plus the behavior
  events table, with some units responding to some events. Used to test responsivity and PSTHs.

Everything is seeded, so the same ``seed`` always gives the same data. Times are in seconds.
"""

import numpy as np
import pandas as pd

BEHAVIORS = ("allo-groom", "nose-to-anogenital", "nose-to-nose", "gentle-sniff", "approaching")
SUBJECTS = ("rat_A", "rat_B")
IGNORED_BEHAVIORS = ("approaching",)  # present in the events table, never given a response

_SLOT_S = 8.0          # events start in separate 8 s slots, so responses never overlap
_SLOT_JITTER_S = 1.5   # onset is placed randomly within the first 1.5 s of its slot
_EDGE_S = 10.0         # no events in the first/last 10 s


# ----------------------------------------------------------------------------------------
# Raw signal
# ----------------------------------------------------------------------------------------

def _spike_template(fs: float, length_ms: float = 2.0, trough_ms: float = 0.5) -> tuple[np.ndarray, int]:
    """Biphasic spike (negative trough, small positive rebound). Returns (template, trough index)."""
    n = int(round(length_ms * 1e-3 * fs))
    trough_idx = int(round(trough_ms * 1e-3 * fs))
    t = (np.arange(n) - trough_idx) / fs
    trough = -np.exp(-0.5 * (t / 0.12e-3) ** 2)
    rebound = 0.35 * np.exp(-0.5 * ((t - 0.35e-3) / 0.25e-3) ** 2)
    return trough + rebound, trough_idx


def make_raw_snippet(
    seed: int = 0,
    duration_s: float = 5.0,
    n_channels: int = 16,
    fs: float = 32000.0,
    spike_rate_hz: float = 10.0,
    noise_sigma: float = 5.0,
    shared_noise_sigma: float = 15.0,
) -> dict:
    """Raw multichannel signal (arbitrary units, think microvolts) with known spikes.

    Each channel contains:
      * its own white noise (``noise_sigma``),
      * spikes that appear on that channel only (trough -60 to -120),
      * noise identical on all channels: broadband white noise (``shared_noise_sigma``) plus
        slow oscillations at 4, 12 and 50 Hz (what the band-pass should remove).

    Returns a dict with ``data`` (n_channels, n_samples), ``fs``, and the ground truth
    ``spike_times_s``, ``spike_channels`` and ``spike_amplitudes`` (trough times, same order).
    """
    rng = np.random.default_rng(seed)
    n_samples = int(round(duration_s * fs))
    t = np.arange(n_samples) / fs

    independent = rng.normal(0.0, noise_sigma, size=(n_channels, n_samples))
    shared = rng.normal(0.0, shared_noise_sigma, size=n_samples)
    slow = (150 * np.sin(2 * np.pi * 4 * t) + 80 * np.sin(2 * np.pi * 12 * t + 1.0)
            + 50 * np.sin(2 * np.pi * 50 * t + 2.0))
    data = independent + shared + slow  # broadcast over channels

    template, trough_idx = _spike_template(fs)
    min_gap = int(round(3e-3 * fs))
    margin = len(template)
    times, channels, amps = [], [], []
    for ch in range(n_channels):
        n = rng.poisson(spike_rate_hz * duration_s)
        idx = np.sort(rng.integers(margin, n_samples - margin, size=n))
        idx = idx[np.concatenate([[True], np.diff(idx) >= min_gap])] if len(idx) else idx
        amp = rng.uniform(60, 120, size=len(idx))
        for i, a in zip(idx, amp):
            data[ch, i - trough_idx:i - trough_idx + len(template)] += a * template
        times.append(idx / fs)
        channels.append(np.full(len(idx), ch))
        amps.append(amp)

    order = np.argsort(np.concatenate(times), kind="stable")
    return {
        "data": data,
        "fs": float(fs),
        "spike_times_s": np.concatenate(times)[order],
        "spike_channels": np.concatenate(channels)[order],
        "spike_amplitudes": np.concatenate(amps)[order],
    }


# ----------------------------------------------------------------------------------------
# Digital input (trigger pulses)
# ----------------------------------------------------------------------------------------

def make_digital_words(
    pulse_times_s,
    duration_s: float,
    fs: float = 32000.0,
    bit: int = 3,
    pulse_width_s: float = 0.01,
    start_high: bool = False,
) -> np.ndarray:
    """A digital-input stream (uint16 words, one per sample) with pulses on one bit.

    The target ``bit`` is high for ``pulse_width_s`` after each time in ``pulse_times_s``.
    A neighbouring bit toggles at 5 Hz the whole time, as a distractor: code that reads the
    word as a whole, instead of one bit, will find the wrong edges. With ``start_high`` the
    target bit is already high at sample 0 (no rising edge there).
    """
    n = int(round(duration_s * fs))
    words = np.zeros(n, dtype=np.uint16)
    for t in np.atleast_1d(pulse_times_s):
        i0, i1 = int(round(t * fs)), int(round((t + pulse_width_s) * fs))
        words[i0:i1] |= np.uint16(1 << bit)
    if start_high:
        words[:int(round(pulse_width_s * fs))] |= np.uint16(1 << bit)
    distractor = (bit + 1) % 16
    toggle = (np.arange(n) // int(fs / 10)) % 2
    words |= (toggle.astype(np.uint16) << np.uint16(distractor))
    return words


# ----------------------------------------------------------------------------------------
# Full session: units + events
# ----------------------------------------------------------------------------------------

def _poisson_train(rng: np.random.Generator, rate_hz: float, duration_s: float) -> np.ndarray:
    return np.sort(rng.uniform(0.0, duration_s, rng.poisson(rate_hz * duration_s)))


def make_session(
    seed: int = 0,
    duration_s: float = 900.0,
    n_units: int = 20,
    n_slow_units: int = 2,
    n_up_units: int = 3,
    n_down_units: int = 3,
    n_events_per_key: int = 8,
    gain_up: float = 4.0,
    gain_down: float = 0.2,
    response_window_s: float = 2.0,
) -> dict:
    """Spike trains and behavior events for one session, with simulated responses.

    Units
      * ``slow`` units fire at 0.3 Hz and never respond.
      * ``up`` units multiply their rate by ``gain_up`` for ``response_window_s`` after every onset
        of one (subject, behavior) pair; ``down`` units multiply it by ``gain_down``
        (spikes are removed at random).
      * all other units are unresponsive.
    Each responsive unit has its own pair; "approaching" events exist but never evoke a response.

    Time
      Spike times are on the *neural* clock. Events are on the *video* clock (seconds from the
      first frame), as in an annotation file. The first video frame happened at
      ``first_frame_neural_s`` on the neural clock, so neural onset = video time + that value.

    Returns a dict:
      ``spike_times_s``       list of arrays, one per unit (neural clock)
      ``events``              DataFrame (time_s, subject, behavior, status in {"START", "STOP"}), video clock
      ``first_frame_neural_s``  float
      ``duration_s``          float
      ``truth``               dict with ``units`` (DataFrame: unit, role, baseline_hz, subject,
                              behavior, gain) and ``onsets_neural_s`` ({(subject, behavior): array})
    """
    rng = np.random.default_rng(seed)
    n_special = n_slow_units + n_up_units + n_down_units
    if n_units < n_special:
        raise ValueError("n_units is smaller than the number of slow + up + down units")
    first_frame_neural_s = float(rng.uniform(10.0, 30.0))

    # --- events: every key gets n_events_per_key onsets, in distinct slots ---
    keys = [(s, b) for s in SUBJECTS for b in BEHAVIORS]
    last_video_s = duration_s - first_frame_neural_s - _EDGE_S
    n_slots = int((last_video_s - _EDGE_S) // _SLOT_S)
    n_events = len(keys) * n_events_per_key
    if n_events > n_slots:
        raise ValueError(f"{n_events} events do not fit in {n_slots} slots; increase duration_s or reduce n_events_per_key")
    slots = rng.choice(n_slots, size=n_events, replace=False)
    key_of_event = rng.permutation(np.repeat(np.arange(len(keys)), n_events_per_key))
    onsets_video = _EDGE_S + slots * _SLOT_S + rng.uniform(0.0, _SLOT_JITTER_S, size=n_events)
    lengths = rng.uniform(1.0, 4.0, size=n_events)

    rows = []
    onsets_neural = {k: [] for k in keys}
    for k_idx, on, ln in zip(key_of_event, onsets_video, lengths):
        subject, behavior = keys[k_idx]
        rows.append((on, subject, behavior, "START"))
        rows.append((on + ln, subject, behavior, "STOP"))
        onsets_neural[keys[k_idx]].append(on + first_frame_neural_s)
    events = pd.DataFrame(rows, columns=["time_s", "subject", "behavior", "status"])
    events = events.sort_values("time_s", kind="stable").reset_index(drop=True)
    onsets_neural = {k: np.sort(np.array(v)) for k, v in onsets_neural.items()}

    # --- units ---
    roles = (["slow"] * n_slow_units + ["up"] * n_up_units + ["down"] * n_down_units
             + ["none"] * (n_units - n_special))
    roles = list(rng.permutation(roles))
    base = np.clip(rng.lognormal(np.log(5.0), 0.7, size=n_units), 1.5, 30.0)
    base = np.where(np.array(roles) == "slow", 0.3, base)
    base = np.where(np.array(roles) == "down", np.maximum(base, 6.0), base)

    eligible = [k for k in keys if k[1] not in IGNORED_BEHAVIORS]
    n_resp = n_up_units + n_down_units
    if n_resp > len(eligible):
        raise ValueError("more responsive units than distinct (subject, behavior) pairs")
    chosen = [eligible[i] for i in rng.choice(len(eligible), size=n_resp, replace=False)]

    spike_trains, truth_rows = [], []
    for u, (role, rate) in enumerate(zip(roles, base)):
        spikes = _poisson_train(rng, rate, duration_s)
        subject = behavior = None
        gain = np.nan
        if role in ("up", "down"):
            subject, behavior = chosen.pop()
            gain = gain_up if role == "up" else gain_down
            for on in onsets_neural[(subject, behavior)]:
                if role == "up":
                    extra = rng.uniform(on, on + response_window_s, rng.poisson(rate * (gain - 1) * response_window_s))
                    spikes = np.concatenate([spikes, extra])
                else:
                    in_win = (spikes >= on) & (spikes < on + response_window_s)
                    spikes = spikes[~(in_win & (rng.random(len(spikes)) > gain))]
            spikes = np.sort(spikes)
        spike_trains.append(spikes)
        truth_rows.append((u, role, float(rate), subject, behavior, gain))

    truth_units = pd.DataFrame(truth_rows, columns=["unit", "role", "baseline_hz", "subject", "behavior", "gain"])
    return {
        "spike_times_s": spike_trains,
        "events": events,
        "first_frame_neural_s": first_frame_neural_s,
        "duration_s": float(duration_s),
        "truth": {"units": truth_units, "onsets_neural_s": onsets_neural},
    }
