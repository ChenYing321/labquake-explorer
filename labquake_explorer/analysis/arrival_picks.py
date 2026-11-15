"""Automatic arrival picks on full-rate records, for the Pick Arrivals view.

Each channel is picked on its own, from a stretch of noise before the waves
arrive (``noise_end``: the samples before it):

* ``difference``: the trace's departure from the noise level,
  ``|x - median(noise)|``;
* ``slope``: the size of the trace's slope, ``|dx/dt|`` after a moving
  average over ``smoothing`` samples, against the slope of the noise;
* ``extreme``: the largest value of the trace (the earlier "Magic" pick: the
  peak, not the onset).

For ``difference`` and ``slope`` the threshold is ``threshold`` times the
noise's spread (``1.4826 * MAD``, so a standard deviation for Gaussian noise).
The pick is not the first sample above it, which an early noise spike would
take, but the onset of the largest change: from the largest value after the
noise window the search goes back to the last quiet stretch, ``quiet``
samples or more below the threshold, and picks the sample after it.  Shorter
dips, such as the zero crossings of an oscillating arrival, do not end the
search, so an oscillation is picked at its start, not at the cycle of its
largest swing.  A channel whose largest change stays below the threshold is not
picked and is flagged (``FLAG_WEAK``); one whose rise goes back to the end of
the noise window is flagged too (``FLAG_AT_NOISE_END``): the waves may
already be in the noise window.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import uniform_filter1d

METHOD_DIFFERENCE = "difference"
METHOD_SLOPE = "slope"
METHOD_EXTREME = "extreme"
METHODS = (METHOD_DIFFERENCE, METHOD_SLOPE, METHOD_EXTREME)
METHOD_LABELS = {METHOD_DIFFERENCE: "difference from the noise", METHOD_SLOPE: "slope",
                 METHOD_EXTREME: "largest value (Magic)"}

DEFAULT_THRESHOLD = 5.0               # times the noise's spread
DEFAULT_SMOOTHING_SAMPLES = 5         # slope: the moving average before the derivative
DEFAULT_QUIET_SAMPLES = 10            # a stretch below the threshold this long ends the search back
MIN_NOISE_SAMPLES = 10
MAD_TO_SIGMA = 1.4826

FLAG_WEAK = "weak onset: the largest change stays below the threshold"
FLAG_AT_NOISE_END = "onset at the end of the noise window"
FLAG_NO_NOISE = "too few samples before the noise window's end"


@dataclass
class Pick:
    """One channel's automatic pick: the sample, or None when it was not
    picked, the measure's noise spread, its largest value and a flag ("" when fine)."""
    index: Optional[int]
    noise: float = float("nan")
    peak: float = float("nan")
    flag: str = ""


def robust_spread(values) -> tuple:
    """``(median, 1.4826 * MAD)`` of the finite values; the spread falls back to
    the standard deviation when the MAD is zero."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    centre = float(np.median(v))
    spread = MAD_TO_SIGMA * float(np.median(np.abs(v - centre)))
    if spread == 0.0:
        spread = float(np.std(v))
    return centre, spread


def onset_of_largest(measure, start: int, threshold: float, quiet: int = DEFAULT_QUIET_SAMPLES) -> tuple:
    """``(index, flag)``: the sample after the last stretch of at least
    ``quiet`` samples at or below ``threshold`` before the largest value of
    ``measure`` at or after ``start``; ``start`` and ``FLAG_AT_NOISE_END``
    without such a stretch; None and ``FLAG_WEAK`` when the largest value stays
    below the threshold."""
    m = np.asarray(measure, dtype=float)
    tail = np.where(np.isfinite(m[start:]), m[start:], -np.inf)
    if tail.size == 0:
        return None, FLAG_WEAK
    peak = start + int(np.argmax(tail))
    if not m[peak] > threshold:
        return None, FLAG_WEAK
    quiet = max(1, int(quiet))
    run = 0
    for k in range(peak - 1, start - 1, -1):              # back from the peak to a quiet stretch
        if m[k] > threshold:
            run = 0
            continue
        run += 1
        if run >= quiet:
            k_end = k + run - 1                          # the stretch's last sample
            nxt = k_end + 1
            while nxt < peak and not m[nxt] > threshold:  # a stretch longer than quiet: its true end
                nxt += 1
            return nxt, ""
    return start, FLAG_AT_NOISE_END


def pick_difference(x, noise_end: int, threshold: float = DEFAULT_THRESHOLD,
                    quiet: int = DEFAULT_QUIET_SAMPLES) -> Pick:
    x = np.asarray(x, dtype=float)
    if noise_end < MIN_NOISE_SAMPLES:
        return Pick(None, flag=FLAG_NO_NOISE)
    centre, spread = robust_spread(x[:noise_end])
    measure = np.abs(x - centre)
    spread = _floor(spread, measure)
    index, flag = onset_of_largest(measure, noise_end, threshold * spread, quiet)
    return Pick(index, spread, float(np.nanmax(measure[noise_end:])), flag)


def pick_slope(x, noise_end: int, threshold: float = DEFAULT_THRESHOLD,
               smoothing: int = DEFAULT_SMOOTHING_SAMPLES, quiet: int = DEFAULT_QUIET_SAMPLES) -> Pick:
    x = np.asarray(x, dtype=float)
    if noise_end < MIN_NOISE_SAMPLES:
        return Pick(None, flag=FLAG_NO_NOISE)
    smooth = uniform_filter1d(x, max(1, int(smoothing)), mode="nearest") if smoothing > 1 else x
    slope = np.gradient(smooth)                            # per sample: the threshold is relative anyway
    centre, spread = robust_spread(slope[:noise_end])
    measure = np.abs(slope - centre)
    spread = _floor(spread, measure)
    index, flag = onset_of_largest(measure, noise_end, threshold * spread, quiet)
    return Pick(index, spread, float(np.nanmax(measure[noise_end:])), flag)


def pick_extreme(x) -> Pick:
    x = np.asarray(x, dtype=float)
    return Pick(int(np.nanargmax(x)), peak=float(np.nanmax(x)))


def _floor(spread: float, measure) -> float:
    """A spread that is not zero: noiseless (synthetic) records get a tiny one."""
    top = float(np.nanmax(measure)) if np.size(measure) else 0.0
    floor = 1e-12 * top if math.isfinite(top) and top > 0 else 1e-300
    return spread if math.isfinite(spread) and spread > floor else floor


def auto_pick(data, method: str, noise_end: int, threshold: float = DEFAULT_THRESHOLD,
              smoothing: int = DEFAULT_SMOOTHING_SAMPLES, quiet: int = DEFAULT_QUIET_SAMPLES) -> list:
    """One :class:`Pick` per row of ``data`` (``(n_channels, n)``)."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; use one of {METHODS}")
    if not (math.isfinite(threshold) and threshold > 0):
        raise ValueError("the threshold must be > 0")
    rows = np.atleast_2d(np.asarray(data, dtype=float))
    noise_end = int(noise_end)
    out = []
    for x in rows:
        if method == METHOD_DIFFERENCE:
            out.append(pick_difference(x, noise_end, threshold, quiet))
        elif method == METHOD_SLOPE:
            out.append(pick_slope(x, noise_end, threshold, smoothing, quiet))
        else:
            out.append(pick_extreme(x))
    return out


__all__ = ["METHOD_DIFFERENCE", "METHOD_SLOPE", "METHOD_EXTREME", "METHODS", "METHOD_LABELS",
           "DEFAULT_THRESHOLD", "DEFAULT_SMOOTHING_SAMPLES", "DEFAULT_QUIET_SAMPLES", "MIN_NOISE_SAMPLES", "FLAG_WEAK",
           "FLAG_AT_NOISE_END", "FLAG_NO_NOISE", "Pick", "robust_spread", "onset_of_largest",
           "pick_difference", "pick_slope", "pick_extreme", "auto_pick"]
