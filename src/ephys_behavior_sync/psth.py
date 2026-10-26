"""Peri-event time histograms (PSTH) and raster plots.

Defaults follow the associated study: window from -1 s to +2 s around onset, 50 ms bins,
Gaussian smoothing with sigma = 2 bins (100 ms), baseline = mean rate of the bins before onset.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


def align_spikes(
    spike_times_s: np.ndarray,
    onsets_s: np.ndarray,
    pre_s: float = 1.0,
    post_s: float = 2.0,
) -> list[np.ndarray]:
    """For each onset, the spikes in [onset - pre, onset + post], as times relative to onset.

    ``spike_times_s`` must be sorted. Returns one array per onset (empty if no spikes).
    """
    spikes = np.asarray(spike_times_s, dtype=float)
    trials = []
    for t in np.asarray(onsets_s, dtype=float):
        i0 = np.searchsorted(spikes, t - pre_s, side="left")
        i1 = np.searchsorted(spikes, t + post_s, side="right")
        trials.append(spikes[i0:i1] - t)
    return trials


def compute_psth(
    spike_times_s: np.ndarray,
    onsets_s: np.ndarray,
    pre_s: float = 1.0,
    post_s: float = 2.0,
    bin_s: float = 0.05,
    smooth_sigma_bins: float = 2.0,
    subtract_baseline: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean firing rate around onsets.

    Spikes are counted in bins of ``bin_s``, summed over events and divided by
    ``n_events * bin_s`` to give Hz. If ``subtract_baseline``, the mean rate of the bins whose
    centre is before onset is subtracted (so the result is the change from baseline). The curve is then smoothed with a
    Gaussian of ``smooth_sigma_bins`` bins (0 turns smoothing off).

    Returns ``(bin_centers_s, rate_hz)``. ``(pre_s + post_s)`` must be a whole number of bins.
    """
    onsets = np.asarray(onsets_s, dtype=float)
    if len(onsets) == 0:
        raise ValueError("need at least one onset")
    n_bins = int(round((pre_s + post_s) / bin_s))
    if not np.isclose(n_bins * bin_s, pre_s + post_s):
        raise ValueError("pre_s + post_s must be a whole number of bins")

    edges = -pre_s + bin_s * np.arange(n_bins + 1)
    centers = edges[:-1] + bin_s / 2
    counts = np.zeros(n_bins)
    for trial in align_spikes(spike_times_s, onsets, pre_s, post_s):
        counts += np.histogram(trial, bins=edges)[0]
    rate = counts / (len(onsets) * bin_s)

    if subtract_baseline:
        rate = rate - rate[centers < 0].mean()
    if smooth_sigma_bins > 0:
        rate = gaussian_filter1d(rate, smooth_sigma_bins)
    return centers, rate


def plot_raster_psth(
    spike_times_s: np.ndarray,
    onsets_by_behavior: dict[str, np.ndarray],
    pre_s: float = 1.0,
    post_s: float = 2.0,
    bin_s: float = 0.05,
    smooth_sigma_bins: float = 2.0,
    subtract_baseline: bool = True,
    colors: dict[str, str] | None = None,
    title: str | None = None,
):
    """Raster (top) and PSTH (bottom) of one unit, one color per behavior.

    Trials are stacked by behavior in the order of ``onsets_by_behavior``.
    Returns ``(fig, (ax_raster, ax_psth))``.
    """
    behaviors = list(onsets_by_behavior)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {b: (colors or {}).get(b, cycle[i % len(cycle)]) for i, b in enumerate(behaviors)}

    fig, (ax_r, ax_p) = plt.subplots(
        2, 1, sharex=True, figsize=(6, 6), gridspec_kw={"height_ratios": [2, 1]}
    )
    row = 0
    for b in behaviors:
        trials = align_spikes(spike_times_s, onsets_by_behavior[b], pre_s, post_s)
        ax_r.eventplot(trials, lineoffsets=np.arange(row, row + len(trials)), linelengths=0.8,
                       linewidths=0.6, colors=colors[b])
        row += len(trials)
        centers, rate = compute_psth(spike_times_s, onsets_by_behavior[b], pre_s, post_s,
                                     bin_s, smooth_sigma_bins, subtract_baseline)
        ax_p.plot(centers, rate, color=colors[b], label=b)

    ax_r.set_ylim(row - 0.5, -0.5)          # first trial at the top
    ax_r.set_ylabel("Event")
    ax_p.set_ylabel("Rate change (Hz)" if subtract_baseline else "Rate (Hz)")
    ax_p.set_xlabel("Time from onset (s)")
    ax_p.set_xlim(-pre_s, post_s)
    for ax in (ax_r, ax_p):
        ax.axvline(0, color="k", lw=0.8, ls="--")
    ax_p.legend(frameon=False, fontsize=8)
    if title:
        ax_r.set_title(title)
    fig.tight_layout()
    return fig, (ax_r, ax_p)
