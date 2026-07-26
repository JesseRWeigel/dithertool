"""Radially averaged power spectrum, and a test for whether noise is blue.

This exists because "blue noise" is the most misused term in dithering. A great
deal of code labelled blue noise is `numpy.random.rand`, which is white: flat
power at every frequency. Blue noise has its power pushed toward high
frequencies, so the radial spectrum rises with frequency and the low-frequency
band is close to empty. That is a measurable property, so it gets measured.

The metrics on a mean-removed input, in normalised radial frequency where 1.0 is
Nyquist along an axis:

``low``
    mean power for 0 < f <= 0.2
``high``
    mean power for 0.4 <= f <= 0.8
``ratio``
    high / low. White noise sits near 1. Blue noise is far above it.
``slope``
    least-squares slope of power against f over 0 < f <= 0.6, normalised by mean
    power. Positive means rising.
``peak_to_mean``
    largest single non-DC coefficient over the mean coefficient magnitude.
    Periodic patterns such as a Bayer matrix concentrate into a few spikes and
    score high; aperiodic blue noise stays low. This separates blue noise from
    ordered patterns, which the low/high ratio alone does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def radial_power_spectrum(arr, bins: int = 32):
    """Return ``(centres, power)`` of the radially averaged power spectrum.

    ``centres`` are normalised frequencies in (0, 1]; 1.0 is Nyquist along an
    axis. The DC term is excluded by removing the mean first and dropping the
    zero-frequency bin.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("need a 2-D array, got %r" % (a.shape,))
    h, w = a.shape
    f = np.fft.fftshift(np.fft.fft2(a - a.mean()))
    power = (np.abs(f) ** 2) / (h * w)
    fy = np.fft.fftshift(np.fft.fftfreq(h)) * 2.0   # -1..1, 1 == Nyquist
    fx = np.fft.fftshift(np.fft.fftfreq(w)) * 2.0
    r = np.hypot(*np.meshgrid(fx, fy, indexing="xy")[::-1])
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.digitize(r.ravel(), edges) - 1
    flat = power.ravel()
    keep = (idx >= 0) & (idx < bins)
    sums = np.bincount(idx[keep], weights=flat[keep], minlength=bins)
    counts = np.bincount(idx[keep], minlength=bins).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    centres = 0.5 * (edges[:-1] + edges[1:])
    # first bin contains DC, which we already removed; drop it to avoid a
    # bin whose mean is dragged down by the forced zero.
    return centres[1:], prof[1:]


def tile_to(arr, height: int, width: int) -> np.ndarray:
    """Tile a small mask up to at least ``height`` x ``width``, then crop.

    Needed before spectral analysis of a small matrix: an 8x8 Bayer matrix has no
    frequency below 0.25, so a low-frequency band measured on it directly is
    empty. Tiling measures the pattern as it is actually used.
    """
    a = np.asarray(arr, dtype=np.float64)
    ry = -(-height // a.shape[0])
    rx = -(-width // a.shape[1])
    return np.tile(a, (ry, rx))[:height, :width]


@dataclass
class SpectrumMetrics:
    low: float
    high: float
    ratio: float
    slope: float
    peak_to_mean: float

    def __str__(self) -> str:
        return (
            "low=%.4g high=%.4g ratio=%.2f slope=%.2f peak_to_mean=%.1f"
            % (self.low, self.high, self.ratio, self.slope, self.peak_to_mean)
        )


def spectrum_metrics(arr, bins: int = 32) -> SpectrumMetrics:
    centres, prof = radial_power_spectrum(arr, bins=bins)
    ok = np.isfinite(prof)
    centres, prof = centres[ok], prof[ok]
    low_sel = centres <= 0.2
    high_sel = (centres >= 0.4) & (centres <= 0.8)
    if not low_sel.any() or not high_sel.any():
        # Refusing to return a nan here on purpose: an empty band means the
        # input is too small to measure, which is not the same as a measurement
        # of zero power. Tile it first with spectrum.tile_to.
        raise ValueError(
            "input %r is too small for these bands: %d low bins, %d high bins"
            % (np.shape(arr), int(low_sel.sum()), int(high_sel.sum()))
        )
    low = float(prof[low_sel].mean())
    high = float(prof[high_sel].mean())
    fit_sel = centres <= 0.6
    mean_power = float(prof.mean())
    coeffs = np.polyfit(centres[fit_sel], prof[fit_sel], 1)
    slope = float(coeffs[0] / mean_power) if mean_power > 0 else 0.0

    a = np.asarray(arr, dtype=np.float64)
    mags = np.abs(np.fft.fft2(a - a.mean()))
    mags_flat = mags.ravel().copy()
    mags_flat[0] = 0.0  # DC
    denom = mags_flat.mean()
    peak_to_mean = float(mags_flat.max() / denom) if denom > 0 else float("inf")

    ratio = float(high / low) if low > 0 else float("inf")
    return SpectrumMetrics(low=low, high=high, ratio=ratio, slope=slope,
                           peak_to_mean=peak_to_mean)


def is_blue_noise(
    arr,
    min_ratio: float = 4.0,
    min_slope: float = 0.5,
    max_peak_to_mean: float = 12.0,
) -> tuple[bool, SpectrumMetrics]:
    """Blue-noise test used by the suite, with the metrics it decided on.

    Thresholds are deliberately loose enough that a correct void-and-cluster
    pattern of any reasonable size passes, and tight enough that white noise and
    periodic ordered patterns both fail. The test suite asserts both directions.
    """
    m = spectrum_metrics(arr)
    verdict = (
        m.ratio >= min_ratio
        and m.slope >= min_slope
        and m.peak_to_mean <= max_peak_to_mean
    )
    return verdict, m
