"""Array plumbing, transfer functions, and the synthetic test images.

Two things in here matter for correctness rather than convenience.

**Transfer function.** Error diffusion conserves the mean of whatever numbers you
hand it. If you hand it sRGB code values, it conserves mean code value, and the
printed or displayed result is too dark, because a 50% pattern of black and white
pixels averages to 50% *luminance*, which is sRGB code 188, not 128. Dithering in
linear light fixes that. Both modes are available here and the test suite checks
conservation of the quantity each mode actually operates on.

**Level spacing.** For an ``n``-level output the reproducible values are
``k/(n-1)``, k = 0..n-1, and the nearest-level rule is ``round(v*(n-1))``. Code
that quantizes with ``floor(v*n)/n`` cannot reach white and biases everything
dark by half a level.
"""

from __future__ import annotations

import numpy as np


def to_float(img) -> np.ndarray:
    """uint8 image to float64 in [0, 1]."""
    return np.asarray(img, dtype=np.float64) / 255.0


def to_u8(img) -> np.ndarray:
    """float image in [0, 1] to uint8, rounding half away from zero."""
    a = np.asarray(img, dtype=np.float64)
    return np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)


def srgb_to_linear(v):
    v = np.asarray(v, dtype=np.float64)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(v):
    v = np.asarray(v, dtype=np.float64)
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055)


def rec709_luminance(rgb_linear) -> np.ndarray:
    a = np.asarray(rgb_linear, dtype=np.float64)
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def quantize_levels(v: float, levels: int) -> float:
    """Nearest of ``levels`` evenly spaced values in [0, 1].

    ``v`` may lie outside [0, 1] because accumulated error can push it there.
    The *index* is clamped, the input is not, so the returned error keeps the
    overflow and the diffusion stays lossless.
    """
    k = int(round(v * (levels - 1)))
    if k < 0:
        k = 0
    elif k > levels - 1:
        k = levels - 1
    return k / (levels - 1)


def level_values(levels: int) -> np.ndarray:
    return np.arange(levels, dtype=np.float64) / (levels - 1)


# --------------------------------------------------------------------------
# synthetic sources, so the repo needs no external image
# --------------------------------------------------------------------------

def linear_ramp(w: int = 256, h: int = 64) -> np.ndarray:
    """Horizontal 0..1 ramp, the reference target for tone reproduction."""
    x = np.linspace(0.0, 1.0, w)
    return np.tile(x, (h, 1))


def step_wedge(w: int = 256, h: int = 64, steps: int = 16) -> np.ndarray:
    idx = np.minimum((np.arange(w) * steps) // w, steps - 1)
    row = idx / (steps - 1)
    return np.tile(row, (h, 1))


def zone_plate(size: int = 192, k: float = 0.32) -> np.ndarray:
    """Radial frequency sweep. Any resampling or screen aliasing shows here."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    cx = cy = (size - 1) / 2.0
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    return 0.5 + 0.5 * np.cos(k * r2 * (np.pi / size))


def radial_gradient(w: int = 192, h: int = 192) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r = np.hypot(x - cx, y - cy) / np.hypot(cx, cy)
    return np.clip(1.0 - r, 0.0, 1.0)


def line_pattern(w: int = 256, h: int = 128) -> np.ndarray:
    """One-pixel lines, thin wedges and small solids on a mid field."""
    img = np.full((h, w), 0.5)
    for i, period in enumerate((2, 3, 4, 6, 8, 12)):
        y0 = 4 + i * (h - 8) // 6
        y1 = y0 + (h - 8) // 6 - 4
        cols = np.arange(w // 2)
        img[y0:y1, : w // 2] = np.where((cols % period) == 0, 1.0, 0.0)
    for i in range(6):
        y0 = 4 + i * (h - 8) // 6
        y1 = y0 + (h - 8) // 6 - 4
        img[y0:y1, w // 2 + 8 : w // 2 + 8 + 4 * (i + 1)] = i / 5.0
    img[h - 12 : h - 4, 8:40] = 0.0
    img[h - 12 : h - 4, 48:80] = 1.0
    return img


def color_bars(w: int = 256, h: int = 96) -> np.ndarray:
    """Saturated primaries over a neutral ramp, for per-channel behaviour."""
    img = np.zeros((h, w, 3))
    bars = [
        (1, 1, 1), (1, 1, 0), (0, 1, 1), (0, 1, 0),
        (1, 0, 1), (1, 0, 0), (0, 0, 1), (0.5, 0.5, 0.5),
    ]
    top = h // 2
    for i, c in enumerate(bars):
        x0 = i * w // len(bars)
        x1 = (i + 1) * w // len(bars)
        img[:top, x0:x1] = c
    ramp = np.linspace(0, 1, w)
    img[top:, :, 0] = ramp
    img[top:, :, 1] = ramp[::-1]
    img[top:, :, 2] = 0.5
    return img


def color_wheel(size: int = 192) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    cx = cy = (size - 1) / 2.0
    ang = (np.arctan2(y - cy, x - cx) / (2 * np.pi)) % 1.0
    rad = np.clip(np.hypot(x - cx, y - cy) / (size / 2.0), 0, 1)
    h6 = ang * 6.0
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    s = np.clip(rad, 0, 1)
    v = np.clip(1.2 - 0.4 * rad, 0, 1)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    out = np.stack([r, g, b], axis=-1)
    out[rad > 1.0] = 1.0
    return np.clip(out, 0, 1)


def flat(value: float, w: int = 128, h: int = 128) -> np.ndarray:
    return np.full((h, w), float(value))
