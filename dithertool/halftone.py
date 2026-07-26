"""Rotated-screen AM halftoning.

Two things make a halftone screen correct, and most implementations miss one or
both.

**Rotate the screen, not the image.** The wrong way, which is everywhere, is to
rotate the image by -theta, apply an axis-aligned screen, and rotate the result
back. That resamples the image twice, so fine detail is filtered away, and it
resamples a bilevel pattern, so the dots break up and alias. The right way is to
leave every pixel where it is and transform the *pixel coordinate* into screen
space:

    u = ( (x+0.5) cos t + (y+0.5) sin t ) / P
    v = ( -(x+0.5) sin t + (y+0.5) cos t ) / P

then compare the image value at that pixel against the spot function evaluated at
the fractional part of (u, v). No resampling happens at all. A second, subtler
version of the same mistake is precomputing one P x P threshold tile with the
rotation baked into it and tiling that axis-aligned. Tiling forces the period to
be an integer in x and y, so the screen angle silently snaps to the nearest
rational tangent: ask for 15 degrees and you get 0. ``wrong.py`` has both, and
the tests measure the angle of the output with an FFT and assert the correct
implementation lands within a degree while the tiled one does not.

**The spot function must be area-linear.** For a flat tone ``t``, the fraction of
inked pixels should be ``t``. That holds only if the spot function is uniformly
distributed over the cell, meaning ``spot(u,v)`` is the *fraction of the cell that
is inked before this point is inked*, not a raw distance. Screens built from raw
distance or from ``cos(u)cos(v)`` produce a strongly nonlinear tone curve: a
round dot normalised by the corner radius reproduces a 50% request as 39% ink.
The spot functions here are analytic CDFs on the unit cell, so coverage equals
tone up to pixel quantisation.

Analytic CDFs used, with (u, v) in [-1/2, 1/2] and r = hypot(u, v):

round
    area of the disc of radius r clipped to the unit square:
    ``pi r^2`` for r <= 1/2, and
    ``pi r^2 - 4 (r^2 acos(1/2r) - 1/2 sqrt(r^2 - 1/4))`` for 1/2 < r <= sqrt(1/2).
    That expression is exactly 1 at r = sqrt(1/2), which is the check that the
    four clipped segments were subtracted correctly.
line
    ``2|v|``, a line screen.
diamond
    ``2 s^2`` for s = |u|+|v| <= 1/2, else ``1 - 2(1-s)^2``.
"""

from __future__ import annotations

import numpy as np

#: Traditional CMYK screen angles in degrees, 30 degrees apart where it matters.
CMYK_ANGLES = {"c": 15.0, "m": 75.0, "y": 0.0, "k": 45.0}


def spot_round(u, v):
    r = np.hypot(u, v)
    out = np.pi * r * r
    mid = r > 0.5
    if np.any(mid):
        rm = np.clip(r[mid], 0.5, None)
        seg = rm * rm * np.arccos(np.clip(1.0 / (2.0 * rm), -1.0, 1.0)) - 0.5 * np.sqrt(
            np.maximum(rm * rm - 0.25, 0.0)
        )
        out[mid] = np.pi * rm * rm - 4.0 * seg
    return np.clip(out, 0.0, 1.0)


def spot_line(u, v):
    return np.clip(2.0 * np.abs(v), 0.0, 1.0)


def spot_diamond(u, v):
    s = np.abs(u) + np.abs(v)
    out = np.where(s <= 0.5, 2.0 * s * s, 1.0 - 2.0 * np.square(np.maximum(1.0 - s, 0.0)))
    return np.clip(out, 0.0, 1.0)


SPOTS = {"round": spot_round, "line": spot_line, "diamond": spot_diamond}


def screen_field(shape, period: float, angle_deg: float, spot: str = "round",
                 phase=(0.0, 0.0)) -> np.ndarray:
    """The threshold field of a rotated screen, one value per pixel in [0, 1]."""
    if period <= 1.0:
        raise ValueError("cell period must be > 1 pixel, got %r" % (period,))
    if spot not in SPOTS:
        raise KeyError("unknown spot %r, have %s" % (spot, sorted(SPOTS)))
    h, w = shape
    t = np.deg2rad(angle_deg)
    ct, st = np.cos(t), np.sin(t)
    x = np.arange(w, dtype=np.float64)[None, :] + 0.5 + phase[1]
    y = np.arange(h, dtype=np.float64)[:, None] + 0.5 + phase[0]
    u = (x * ct + y * st) / period
    v = (-x * st + y * ct) / period
    fu = u - np.floor(u) - 0.5
    fv = v - np.floor(v) - 0.5
    return SPOTS[spot](fu, fv)


def halftone(img, period: float = 8.0, angle_deg: float = 45.0, spot: str = "round",
             levels: int = 2, phase=(0.0, 0.0)) -> np.ndarray:
    """Halftone one channel of floats in [0, 1] with a rotated screen."""
    a = np.asarray(img, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("halftone works on one channel, got %r" % (a.shape,))
    if levels < 2:
        raise ValueError("levels must be >= 2")
    field = screen_field(a.shape, period, angle_deg, spot, phase)
    scaled = np.clip(a, 0.0, 1.0) * (levels - 1)
    base = np.floor(scaled)
    frac = scaled - base
    return np.clip((base + (frac > field).astype(np.float64)) / (levels - 1), 0.0, 1.0)


def measure_screen(binary, window: int = 2) -> tuple[float, float]:
    """Recover ``(frequency_in_cycles_per_pixel, angle_deg)`` from a halftone.

    Finds the strongest non-DC peak of the power spectrum and refines it with a
    power-weighted centroid over a small window, which gets well under a degree
    of angular resolution. The returned angle is folded into [0, 90) because a
    square screen lattice has axes at theta and theta+90 and the two are
    indistinguishable in the spectrum.
    """
    a = np.asarray(binary, dtype=np.float64)
    h, w = a.shape
    f = np.fft.fftshift(np.fft.fft2(a - a.mean()))
    power = np.abs(f) ** 2
    cy, cx = h // 2, w // 2
    # suppress the DC neighbourhood so low-frequency content cannot win
    yy, xx = np.mgrid[0:h, 0:w]
    r_bins = np.hypot(yy - cy, xx - cx)
    power = np.where(r_bins < 3, 0.0, power)
    py, px = np.unravel_index(int(np.argmax(power)), power.shape)
    y0, y1 = max(0, py - window), min(h, py + window + 1)
    x0, x1 = max(0, px - window), min(w, px + window + 1)
    sub = power[y0:y1, x0:x1]
    wy = np.arange(y0, y1)[:, None].astype(np.float64)
    wx = np.arange(x0, x1)[None, :].astype(np.float64)
    tot = sub.sum()
    ry = float((sub * wy).sum() / tot)
    rx = float((sub * wx).sum() / tot)
    ky = (ry - cy) / h        # cycles per pixel
    kx = (rx - cx) / w
    freq = float(np.hypot(kx, ky))
    angle = float(np.rad2deg(np.arctan2(ky, kx)) % 90.0)
    return freq, angle


def coverage(binary) -> float:
    """Fraction of pixels that are ink-free (value 1), for tone checks."""
    a = np.asarray(binary, dtype=np.float64)
    return float(a.mean())
