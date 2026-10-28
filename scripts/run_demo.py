"""End-to-end demo on synthetic data: preprocessing, synchronization, events, responsivity, PSTH.

Run from the repo folder:

    python scripts/run_demo.py                      # figures and a table go to ./outputs
    python scripts/run_demo.py --out docs/figures   # elsewhere

Every step is checked against the ground truth that the synthetic generator knows.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no windows
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ephys_behavior_sync.events import onsets_by_subject_behavior  # noqa: E402
from ephys_behavior_sync.preprocessing import preprocess  # noqa: E402
from ephys_behavior_sync.psth import plot_raster_psth  # noqa: E402
from ephys_behavior_sync.responsivity import all_units_responsivity  # noqa: E402
from ephys_behavior_sync.sync import annotation_to_neural_time, detect_edges, estimate_clock_offset_s  # noqa: E402
from ephys_behavior_sync.synthetic import IGNORED_BEHAVIORS, make_digital_words, make_raw_snippet, make_session  # noqa: E402

FS = 32000.0


def demo_preprocessing(out: Path) -> None:
    """Raw vs. cleaned signal on one channel, with the true spike times marked."""
    snip = make_raw_snippet(seed=1, duration_s=2.0, n_channels=16, fs=FS)
    clean = preprocess(snip["data"], FS)
    channel = 0
    t = np.arange(snip["data"].shape[1]) / FS
    spikes = snip["spike_times_s"][snip["spike_channels"] == channel]
    t0 = spikes[0] - 0.05
    window = (t >= t0) & (t < t0 + 0.3)
    in_window = spikes[(spikes >= t0) & (spikes < t0 + 0.3)]

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    for ax, signal, title in ((axes[0], snip["data"][channel], "Raw (shared noise and slow oscillations dominate)"),
                              (axes[1], clean[channel], "Band-passed 300-6000 Hz, median-referenced")):
        ax.plot(t[window], signal[window], lw=0.6, color="k")
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel("Amplitude (a.u.)")
    axes[1].plot(in_window, np.full(len(in_window), clean[channel][window].max() * 1.1), "v", color="C3",
                 label="simulated spike times")
    axes[1].legend(frameon=False, loc="lower right")
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out / "preprocessing.png", dpi=110)
    plt.close(fig)


def demo_sync(first_frame_neural_s: float) -> float:
    """Recover the first-frame time on the neural clock from trigger pulses on both clocks."""
    first_frame_logger_s = 3.0          # first video frame, as stamped by the logger clock
    logger_trigger_s = 1.0              # the shared trigger pulse, seen by the logger clock
    neural_trigger_s = logger_trigger_s + (first_frame_neural_s - first_frame_logger_s)  # ... and by the neural clock
    duration_s = neural_trigger_s + 5.0
    neural_words = make_digital_words([neural_trigger_s], duration_s, FS, bit=3)
    logger_words = make_digital_words([logger_trigger_s], duration_s, FS, bit=3)

    neural_pulses, _ = detect_edges(neural_words, bit=3, fs=FS)
    logger_pulses, _ = detect_edges(logger_words, bit=3, fs=FS)
    offset_s = estimate_clock_offset_s(neural_pulses, logger_pulses)   # add to logger times -> neural times
    estimated = first_frame_logger_s + offset_s
    print(f"sync: first frame at {estimated:.4f} s on the neural clock "
          f"(truth {first_frame_neural_s:.4f} s, error {abs(estimated - first_frame_neural_s) * 1e3:.3f} ms)")
    return estimated


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    session = make_session(seed=0)
    truth = session["truth"]["units"]

    demo_preprocessing(out)
    print("preprocessing: figure written")

    first_frame = demo_sync(session["first_frame_neural_s"])
    video_onsets = onsets_by_subject_behavior(session["events"], exclude_behaviors=IGNORED_BEHAVIORS)
    onsets = {key: annotation_to_neural_time(times, first_frame) for key, times in video_onsets.items()}
    print(f"events: {sum(len(v) for v in onsets.values())} onsets in {len(onsets)} (subject, behavior) pairs")

    results = all_units_responsivity(session["spike_times_s"], onsets)
    results.to_csv(out / "responsivity.csv", index=False)
    merged = results.merge(truth, on="unit", how="left", suffixes=("", "_true"))
    is_target = (merged.subject == merged.subject_true) & (merged.behavior == merged.behavior_true)
    found = int(merged[is_target].responsive.sum())
    false_pos = int(merged[~is_target].responsive.sum())
    print(f"responsivity, raw p < 0.05: {found} of {int(is_target.sum())} simulated responses found; "
        f"{false_pos} false positives among {int((~is_target).sum())} tests without a simulated response")
    holm = all_units_responsivity(session["spike_times_s"], onsets, correction="holm")
    holm = holm.merge(truth, on="unit", how="left", suffixes=("", "_true"))
    holm_target = (holm.subject == holm.subject_true) & (holm.behavior == holm.behavior_true)
    print(f"responsivity, Holm-corrected: {int(holm[holm_target].responsive.sum())} of {int(holm_target.sum())} "
            f"simulated responses found; {int(holm[~holm_target].responsive.sum())} false positives")

    # recovered vs. simulated, one point per (unit, subject, behavior)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(merged[~is_target].mean_before_hz, merged[~is_target].mean_after_hz, s=12,
               color="0.7", label="no simulated response")
    ax.scatter(merged[is_target].mean_before_hz, merged[is_target].mean_after_hz, s=40,
               color="C3", label="simulated response")
    ax.plot([0, 40], [0, 40], "k--", lw=0.8)
    ax.set_xlabel("Mean rate before onset (Hz)")
    ax.set_ylabel("Mean rate after onset (Hz)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "responsivity.png", dpi=110)
    plt.close(fig)

    # raster + PSTH for one unit that was simulated to increase its rate after one behavior
    row = truth[truth.role == "up"].iloc[0]
    by_behavior = {b: onsets[(row.subject, b)] for b in ("allo-groom", "nose-to-nose", "gentle-sniff")}
    fig, _ = plot_raster_psth(session["spike_times_s"][int(row.unit)], by_behavior,
                              title=f"Synthetic unit {int(row.unit)}: simulated rate increase after {row.behavior}")
    fig.savefig(out / "psth.png", dpi=110)
    plt.close(fig)
    print(f"done: outputs in {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="folder for figures and tables")
    main(parser.parse_args().out)
