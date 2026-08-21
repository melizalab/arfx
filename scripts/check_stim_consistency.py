#!/usr/bin/env python
# -*- mode: python -*-
"""Cross-check a jpresent-ephys session against the open-ephys recording of it.

jpresent-ephys drives stimulus presentation through a hardware chain that
open-ephys records independently on four ADC lines, alongside jpresent's own
event log (via jrecord). This script checks that the two logs agree on the
trial sequence, and that the expected pulses show up on the ADC lines:

  ADC2  a copy of the stimulus waveform (not checked here)
  ADC3  a TTL cueing the stimulus start/stop
  ADC4  an opto-LED trigger, present on roughly half of trials
  ADC5  the raw positive/negative clicks that a Schmitt trigger turns into ADC3

Run against an ARF file produced by arfx-oephys (not the open-ephys tree
directly) and the jpresent/jrecord ARF file for the same session:

    uv run python scripts/check_stim_consistency.py oeaudio.arf jpresent.arf

This is occasional, by-hand QC tooling for a specific recording setup, not a
general-purpose entry point -- it is not installed as a console script and is
not exercised by the test suite.
"""

import argparse
import logging
import re
import sys
from dataclasses import dataclass

import arf
import h5py as h5
import numpy as np

from arfx.core import entries_by_creation, setup_log

log = logging.getLogger("check-stim-consistency")

MESSAGE_DATASET = "MessageCenter"
CUE_CHANNEL = "ADC3"
CLICK_CHANNEL = "ADC5"
OPTO_CHANNEL = "ADC4"

# How far past a trial's onset to look for the opto pulse. It only needs to
# cover the LED pulse itself, not the whole (possibly much longer) stimulus.
OPTO_SEARCH_WINDOW = 0.3
# How far around a cue edge to look for the click that drives it.
CLICK_SEARCH_WINDOW = 0.02
# A pulse must depart from the local baseline by this many robust standard
# deviations (MAD-based) to count as signal rather than noise. Used only for
# ADC5, the one channel here that isn't a simple digital logic line -- see
# TTL_THRESHOLD_VOLTS for ADC3/ADC4.
SIGNIFICANCE_K = 6.0
# Width of the quiet reference sampled for an ADC5 click search -- see
# quiet_reference.
CUE_EDGE_BASELINE = 0.05
# Decision threshold for ADC3 (cue) and ADC4 (opto), both nominal 0/5V TTL
# lines. These get a fixed voltage threshold rather than a per-window
# statistical one: the whole point of checking them is whether the trigger
# circuit produced a real logic level, and a threshold derived from a
# window's own noise can't tell a genuine partial-swing fault from normal
# trial-to-trial variation -- it would just report whatever the window
# happened to contain as "the pulse", including a full electrical fault.
TTL_THRESHOLD_VOLTS = 2.5
# Margin searched past each message-reported onset/offset for the actual cue
# pulse edge. Generous relative to typical audio-buffer latency (order
# 100-300ms) without assuming a specific value -- inter-trial gaps in this
# recording are on the order of seconds, so this doesn't risk pulling in a
# neighboring trial's pulse.
CUE_SEARCH_PAD = 0.5

_MESSAGE_RX = re.compile(r"^(start|stop) (.+)$")


@dataclass
class Trial:
    name: str
    onset: float
    offset: float
    source: str  # entry name, for diagnostics and ADC lookups


def entry_t0(entry: h5.Group) -> float:
    """Absolute (epoch) start time of an entry."""
    return arf.timestamp_to_float(entry.attrs["timestamp"])


def decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def oephys_trials(fp: h5.File) -> list[Trial]:
    """Trials from the open-ephys MessageCenter 'start <stim>'/'stop <stim>' log.

    Message 'start' fields are raw absolute sample numbers on the acquisition
    board's own counter, not indices into the stored ADC arrays, so they need
    the same offset correction as the continuous datasets -- taken here from
    the cue channel, which is present in every entry this script looks at.
    """
    trials = []
    for _, name, entry in entries_by_creation(fp):
        if MESSAGE_DATASET not in entry or CUE_CHANNEL not in entry:
            continue
        t0 = entry_t0(entry)
        cue = entry[CUE_CHANNEL]
        sampling_rate = cue.attrs["sampling_rate"]
        sample_offset = round(cue.attrs["offset"] * sampling_rate)

        pending: dict[str, float] = {}
        for raw_start, message in entry[MESSAGE_DATASET][:][["start", "message"]]:
            match = _MESSAGE_RX.match(decode(message))
            if match is None:
                continue
            kind, stim = match.groups()
            t = t0 + (int(raw_start) - sample_offset) / sampling_rate
            if kind == "start":
                if stim in pending:
                    log.warning(
                        "%s: 'start %s' at %.3fs supersedes an unmatched start",
                        name,
                        stim,
                        t,
                    )
                pending[stim] = t
            else:
                onset = pending.pop(stim, None)
                if onset is None:
                    log.warning(
                        "%s: 'stop %s' at %.3fs has no matching start", name, stim, t
                    )
                    continue
                trials.append(Trial(stim, onset, t, name))
        for stim, onset in pending.items():
            log.warning(
                "%s: 'start %s' at %.3fs has no matching stop", name, stim, onset
            )
    return trials


def jpresent_trials(fp: h5.File) -> list[Trial]:
    """Trials from jpresent's own 'jstim' log.

    A trial is a run of rows starting at a status==1 row, up to (but not
    including) the next one -- status==1 is jpresent's own trial-start
    marker. Grouping by contiguous message name instead would merge two
    consecutive presentations of the same stimulus into one trial, which is
    common here: there are only a handful of distinct stimulus names across
    thousands of trials, so back-to-back repeats happen by chance often
    enough to matter. A trial missing its status==16 (stop) row is logged
    rather than dropped, since a session can end mid-trial or a stop message
    can be lost -- the observed data has two such cases, matching the two
    'no matching stop' warnings on the open-ephys side exactly.
    """
    trials = []
    for _, name, entry in entries_by_creation(fp):
        if "jstim" not in entry:
            continue
        t0 = entry_t0(entry)
        dset = entry["jstim"]
        sampling_rate = dset.attrs["sampling_rate"]
        rows = dset[:]
        if len(rows) == 0:
            continue
        status = rows["status"]
        starts_idx = np.flatnonzero(status == 1)
        if starts_idx.size == 0:
            log.warning("%s: no status==1 (trial start) rows in jstim", name)
            continue
        boundaries = np.append(starts_idx, len(rows))
        for k in range(starts_idx.size):
            si, end = boundaries[k], boundaries[k + 1]
            segment = rows[si:end]
            message = decode(segment["message"][0])
            if 16 not in segment["status"]:
                # Dropped rather than kept with a best-guess offset: the
                # open-ephys side drops the equivalent case too (an unmatched
                # 'start' with no 'stop' never becomes a trial there either),
                # and keeping one side's guess but not the other's would
                # misalign every trial after it in the positional comparison.
                log.warning(
                    "%s: trial %r starting at row %d has no stop (status 16), dropped",
                    name,
                    message,
                    si,
                )
                continue
            onset = t0 + int(segment["start"][0]) / sampling_rate
            offset = t0 + int(segment["start"][-1]) / sampling_rate
            trials.append(Trial(message, onset, offset, name))
    return trials


def mad(x: np.ndarray) -> float:
    """Median absolute deviation, scaled to be a robust estimate of sigma."""
    med = np.median(x)
    return float(np.median(np.abs(x - med)) * 1.4826)


def to_volts(dataset: h5.Dataset, raw: np.ndarray) -> np.ndarray:
    return raw.astype(np.float64) * dataset.attrs["bit_volts"]


def find_ttl_edges(volts: np.ndarray, threshold: float = TTL_THRESHOLD_VOLTS):
    """First and last index where `volts` is above `threshold`, or None.

    A fixed voltage threshold, not a statistical one: ADC3 and ADC4 are
    meant to be simple 0/5V TTL lines, so the question is literally whether
    the signal crossed a real logic level, and by how much of the window --
    not how it compares to its own local noise, which would treat a
    circuit that only partially swings toward 5V the same as one that
    swings cleanly, as long as either was consistent within its own window.
    """
    high = np.flatnonzero(volts > threshold)
    if high.size == 0:
        return None
    return int(high[0]), int(high[-1])


def detect_pulse(
    window: np.ndarray,
    baseline_region: np.ndarray | None = None,
    k: float = SIGNIFICANCE_K,
):
    """Find the largest pulse in `window`, relative to a baseline.

    Used for ADC5 (the click circuit's analog output) -- ADC3 and ADC4 are
    simple TTL lines checked against a fixed voltage instead (find_ttl_edges),
    since for those the question is whether a real logic level was reached,
    not how the window compares to its own noise.

    Returns (rise_idx, fall_idx, positive) for the contiguous run of samples
    within half the peak deviation of the largest excursion, or None if
    nothing in the window departs from the baseline by more than `k` robust
    standard deviations -- i.e. there is no pulse here, only noise.

    `baseline_region` should be a separate, known-quiet sample -- not a slice
    of `window` itself (see quiet_reference) -- so that a pulse occupying
    most of the window can't pull the window's own median inside it and
    invert peak/baseline.
    """
    if window.size == 0:
        return None
    if baseline_region is None or baseline_region.size == 0:
        baseline_region = window
    baseline = np.median(baseline_region)
    noise = mad(baseline_region)
    dev = window.astype(np.float64) - baseline
    peak_idx = int(np.argmax(np.abs(dev)))
    peak = dev[peak_idx]
    threshold = k * noise if noise > 0 else 0.0
    if abs(peak) <= threshold:
        return None
    half = 0.5 * peak
    above = dev >= half if peak > 0 else dev <= half
    idx = np.flatnonzero(above)
    if idx.size == 0:
        return None
    return int(idx[0]), int(idx[-1]), peak > 0


@dataclass
class AdcCheck:
    trial: Trial
    cue_pulse: tuple | None
    onset_error: float | None
    offset_error: float | None
    click_at_rise: bool
    click_at_fall: bool
    opto_present: bool
    cue_low_volts: float | None
    cue_high_volts: float | None


def quiet_reference(
    dataset: h5.Dataset,
    ref_idx: int,
    sampling_rate: float,
    forward: bool,
    width: float = CUE_EDGE_BASELINE,
    gap: float = CUE_SEARCH_PAD,
) -> np.ndarray:
    """A short, known-quiet sample of `dataset`, offset from `ref_idx`.

    Used as an external baseline reference for detect_pulse -- see its
    docstring for why the search window's own median can't always serve as
    one. `gap` (default CUE_SEARCH_PAD) keeps the reference well clear of the
    cue pulse itself: e.g. a fall edge's forward reference must not land
    inside the *next* trial's cue-high span, and CUE_SEARCH_PAD is already
    established as the margin trials are guaranteed to be separated by.
    """
    n = max(1, round(width * sampling_rate))
    g = round(gap * sampling_rate)
    if forward:
        lo, hi = ref_idx + g, ref_idx + g + n
    else:
        hi = max(0, ref_idx - g)
        lo = max(0, hi - n)
    lo, hi = max(0, lo), min(dataset.shape[0], hi)
    return dataset[lo:hi] if hi > lo else dataset[0:0]


def check_adc_pulses(fp: h5.File, trials: list[Trial]) -> list[AdcCheck]:
    """Locate the cue pulse for each trial, and the click/opto pulses near it.

    The cue-pulse search window is padded by a fixed margin around the
    message-reported onset/offset -- generous enough to catch the actual
    hardware output latency (measured at ~270ms on the example recording,
    but not assumed to be any particular value here) without knowing it in
    advance. Once the cue pulse itself is found, the click and opto searches
    are anchored to *its* edges rather than to the message times, since they
    are all triggered by the same underlying event and the cue pulse is the
    most direct measurement of when it actually happened.

    Opto presence is checked over the same span the cue pulse occupies (plus
    a small trailing margin), rather than a lookback before the cue's rise:
    a lookback there turned out to reliably catch the click circuit's own
    crosstalk on ADC4 (a tight cluster a few ms before the cue rise, close
    to full-scale -- easily mistaken for a real opto pulse by amplitude
    alone) rather than anything opto-related.
    """
    by_entry: dict[str, list[Trial]] = {}
    for t in trials:
        by_entry.setdefault(t.source, []).append(t)

    results = []
    for entry_name, entry_trials in by_entry.items():
        entry = fp[entry_name]
        cue = entry[CUE_CHANNEL]
        click = entry[CLICK_CHANNEL]
        opto = entry[OPTO_CHANNEL]
        sampling_rate = cue.attrs["sampling_rate"]
        t0 = entry_t0(entry)
        nsamples = cue.shape[0]

        def idx_at(
            t: float, t0: float = t0, sampling_rate: float = sampling_rate
        ) -> int:
            return round((t - t0) * sampling_rate)

        pad = round(CUE_SEARCH_PAD * sampling_rate)
        click_span = max(1, round(CLICK_SEARCH_WINDOW * sampling_rate))
        opto_span = round(OPTO_SEARCH_WINDOW * sampling_rate)

        for trial in entry_trials:
            lo = max(0, idx_at(trial.onset) - pad)
            hi = min(nsamples, idx_at(trial.offset) + pad)
            if hi <= lo:
                results.append(
                    AdcCheck(trial, None, None, None, False, False, False, None, None)
                )
                continue

            window_volts = to_volts(cue, cue[lo:hi])
            edges = find_ttl_edges(window_volts)
            onset_error = offset_error = None
            click_at_rise = click_at_fall = False
            opto_present = False
            cue_low_volts = cue_high_volts = None
            if edges is not None:
                low_mask = window_volts <= TTL_THRESHOLD_VOLTS
                if low_mask.any():
                    cue_low_volts = float(np.median(window_volts[low_mask]))
                if (~low_mask).any():
                    cue_high_volts = float(np.median(window_volts[~low_mask]))
                rise_idx, fall_idx = edges
                onset_error = (lo + rise_idx) / sampling_rate - (trial.onset - t0)
                offset_error = (lo + fall_idx) / sampling_rate - (trial.offset - t0)

                rise_abs = lo + rise_idx
                fall_abs = lo + fall_idx
                rise_window = click[
                    max(0, rise_abs - click_span) : rise_abs + click_span
                ]
                fall_window = click[
                    max(0, fall_abs - click_span) : fall_abs + click_span
                ]
                rise_click = detect_pulse(
                    rise_window,
                    baseline_region=quiet_reference(
                        click, rise_abs, sampling_rate, forward=False
                    ),
                )
                fall_click = detect_pulse(
                    fall_window,
                    baseline_region=quiet_reference(
                        click, fall_abs, sampling_rate, forward=True
                    ),
                )
                click_at_rise = rise_click is not None and rise_click[2]
                click_at_fall = fall_click is not None and not fall_click[2]

                opto_hi = min(nsamples, rise_abs + opto_span)
                opto_volts = to_volts(opto, opto[rise_abs:opto_hi])
                opto_present = bool(opto_volts.max() > TTL_THRESHOLD_VOLTS)

            results.append(
                AdcCheck(
                    trial,
                    edges,
                    onset_error,
                    offset_error,
                    click_at_rise,
                    click_at_fall,
                    opto_present,
                    cue_low_volts,
                    cue_high_volts,
                )
            )
    return results


def compare_trial_sequences(oe_trials: list[Trial], jp_trials: list[Trial]):
    """Positional comparison: trial i in each log should be the same trial.

    Both logs are strictly ordered records of one session, so matching by
    position is more informative than matching by nearest time -- it makes a
    dropped or reordered trial visible as a divergence point rather than
    something nearest-time matching could paper over.
    """
    mismatches = []
    n = min(len(oe_trials), len(jp_trials))
    lags = []
    for i in range(n):
        oe, jp = oe_trials[i], jp_trials[i]
        if oe.name != jp.name:
            mismatches.append((i, oe, jp))
        else:
            lags.append(oe.onset - jp.onset)
    return mismatches, np.array(lags)


def robust_tolerance(
    lags: np.ndarray,
    fallback: float,
    k: float = 5.0,
    min_n: int = 10,
    ceiling: float = 1.0,
) -> float:
    """A data-derived timing tolerance, bounded away from both noise and nonsense.

    `ceiling` caps what the estimate can return: message/network latency and
    edge-detection slop are both sub-second in this setup, so a much larger
    value means the trial alignment feeding `lags` is wrong, not that the
    tolerance should be huge -- better to keep the checks strict and let them
    fail loudly than to silently widen the tolerance to match broken input.
    """
    if lags.size < min_n:
        return fallback
    return min(max(k * mad(lags), fallback), ceiling)


def robust_outliers(values: np.ndarray, tolerance: float | None):
    """Median-centered outlier detection.

    Each of the timing comparisons this script makes (jpresent vs.
    open-ephys message lag, message vs. cue-pulse latency for onset and for
    offset) has its own systematic offset -- network latency and hardware
    output latency are different things with different magnitudes -- so
    outliers are flagged relative to that comparison's own median rather
    than relative to zero. Returns (center, tolerance used, outlier indices).
    """
    center = float(np.median(values))
    tol = (
        tolerance if tolerance is not None else robust_tolerance(values, fallback=0.05)
    )
    outliers = np.flatnonzero(np.abs(values - center) > tol)
    return center, tol, outliers


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="check_stim_consistency",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("oephys_arf", help="ARF file produced by arfx-oephys")
    p.add_argument("jpresent_arf", help="ARF file produced by jrecord/jpresent")
    p.add_argument(
        "--tolerance",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "max acceptable deviation from each check's own median latency; "
            "default is derived from the data, separately for each check"
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_log(log, args.verbose)

    with (
        h5.File(args.oephys_arf, "r") as oe_fp,
        h5.File(args.jpresent_arf, "r") as jp_fp,
    ):
        oe_trials = oephys_trials(oe_fp)
        jp_trials = jpresent_trials(jp_fp)

        print(f"open-ephys (MessageCenter): {len(oe_trials)} trials")
        print(f"jpresent (jstim):           {len(jp_trials)} trials")

        mismatches, lags = compare_trial_sequences(oe_trials, jp_trials)

        ok = True
        if len(oe_trials) != len(jp_trials):
            print(f"FAIL: trial count differs ({len(oe_trials)} vs {len(jp_trials)})")
            ok = False
        if mismatches:
            ok = False
            print(f"FAIL: {len(mismatches)} trial(s) with mismatched names:")
            for i, oe, jp in mismatches[:10]:
                print(f"  #{i}: open-ephys={oe.name!r} jpresent={jp.name!r}")
            if len(mismatches) > 10:
                print(f"  ... and {len(mismatches) - 10} more")

        if lags.size:
            center, tol, outliers = robust_outliers(lags, args.tolerance)
            print(
                f"message lag (open-ephys - jpresent): "
                f"median={center * 1000:.1f}ms tolerance=±{tol * 1000:.1f}ms"
            )
            if outliers.size:
                ok = False
                print(f"FAIL: {outliers.size} trial(s) with anomalous message lag")
                for i in outliers[:10]:
                    print(f"  #{i}: {oe_trials[i].name!r} lag={lags[i] * 1000:.1f}ms")

        adc_checks = check_adc_pulses(oe_fp, oe_trials)
        n_cue_found = sum(1 for c in adc_checks if c.cue_pulse is not None)
        n_click_rise = sum(1 for c in adc_checks if c.click_at_rise)
        n_click_fall = sum(1 for c in adc_checks if c.click_at_fall)
        n_opto = sum(1 for c in adc_checks if c.opto_present)
        total = len(adc_checks)

        print(f"\nADC3 (cue) pulse found:      {n_cue_found}/{total}")
        if total and n_cue_found < total:
            ok = False
            print(f"FAIL: {total - n_cue_found} trial(s) with no detectable ADC3 pulse")

        lows = np.array(
            [c.cue_low_volts for c in adc_checks if c.cue_low_volts is not None]
        )
        highs = np.array(
            [c.cue_high_volts for c in adc_checks if c.cue_high_volts is not None]
        )
        if lows.size and highs.size:
            print(
                f"ADC3 logic levels: low={np.median(lows):.2f}V "
                f"high={np.median(highs):.2f}V "
                f"(threshold={TTL_THRESHOLD_VOLTS:.1f}V)"
            )

        for label, attr in (("onset", "onset_error"), ("offset", "offset_error")):
            errors = np.array(
                [getattr(c, attr) for c in adc_checks if getattr(c, attr) is not None]
            )
            if errors.size == 0:
                continue
            center, tol, outliers = robust_outliers(errors, args.tolerance)
            print(
                f"ADC3 {label} latency (pulse - message): "
                f"median={center * 1000:.1f}ms tolerance=±{tol * 1000:.1f}ms "
                f"({errors.size - outliers.size}/{errors.size} within tolerance)"
            )
            if outliers.size:
                ok = False
                print(f"FAIL: {outliers.size} trial(s) with anomalous {label} latency")

        # Denominator is trials with a found cue pulse, not all trials: a
        # click can only be looked for once its cue edge is known, so a
        # trial with no cue pulse is not a second, independent click failure.
        print(f"ADC5 click at rising edge:   {n_click_rise}/{n_cue_found}")
        print(f"ADC5 click at falling edge:  {n_click_fall}/{n_cue_found}")
        opto_fraction = n_opto / total if total else 0.0
        print(f"ADC4 opto pulse present:     {n_opto}/{total} ({opto_fraction:.1%})")

        if n_cue_found and (n_click_rise < n_cue_found or n_click_fall < n_cue_found):
            ok = False
            print("FAIL: some ADC3 edges have no matching ADC5 click")
        if total and not (0.35 <= opto_fraction <= 0.65):
            print(
                f"NOTE: ADC4 presence fraction ({opto_fraction:.1%}) is far from the "
                "expected ~50% split -- reported, not treated as a failure"
            )

        print("\nPASS" if ok else "\nFAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
