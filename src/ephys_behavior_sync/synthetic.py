"""Synthetic recordings with known ground truth, used by the demo and by the tests.

DRAFT: signature only. Planned contents of a generated session:
* a multichannel signal with spikes, noise and a artifact shared across channels;
* digital trigger and frame pulses with a known clock offset;
* an events table;
* units with known baseline rates, some with injected event-locked responses.
"""


def make_session(seed: int = 0, duration_s: float = 600.0, n_channels: int = 16):
    """Return a dict with the synthetic signal, triggers, events, units and the ground truth."""
    raise NotImplementedError
