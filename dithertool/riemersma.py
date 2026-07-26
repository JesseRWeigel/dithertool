"""Riemersma dithering: error diffusion along a Hilbert curve.

This is a different family from the kernel methods in `diffusion.py`. Those push error to
fixed spatial neighbours and scan in raster order. Riemersma instead walks a space-filling
curve and carries a short, exponentially decaying memory of recent errors.

A claim worth not making. The usual pitch for Riemersma is that a Hilbert curve has no
preferred direction, so the artifacts come out isotropic. Measured on this implementation
that is false, and measuring it was the point. At level 0.25 the output disagrees across one
diagonal twice as often as across the other (0.404 against 0.201), which is stronger
directional structure than Floyd-Steinberg shows at the same level (0.487 against 0.498).
The Hilbert curve has its own quadrant geometry and the error memory follows it.

What does hold, and is tested, is that Riemersma avoids the degenerate case. On a flat
mid-grey, Floyd-Steinberg collapses into a rigid checkerboard: every horizontal and vertical
neighbour differs and every diagonal neighbour matches, exactly, at every pixel. Serpentine
scanning does not change this. Riemersma produces an aperiodic texture at the same input.

Two details decide whether an implementation is really Riemersma:

The curve must be a genuine Hilbert traversal, visiting every pixel exactly once with each
step adjacent to the previous one. A curve that jumps is a raster scan wearing a costume,
and the error memory then bleeds between spatially distant pixels, which is the whole thing
the method is designed to avoid. `hilbert_points` is tested for both properties.

The error weights must decay geometrically over the queue, not uniformly. Riemersma's
parameter is the ratio between the oldest and newest weight, conventionally 1/16. A uniform
average over the queue is a box blur of the error and loses the locality that makes the
output look good.
"""
from __future__ import annotations

import numpy as np

from .imageops import quantize_levels

# Riemersma's own values from the 1998 article: a 16-entry queue and a 1/16 decay across it.
DEFAULT_QUEUE = 16
DEFAULT_RATIO = 1.0 / 16.0


def _d2xy(order: int, d: int) -> tuple[int, int]:
    """Map a distance along the Hilbert curve to coordinates, per Hacker's Delight.

    `order` is the side length and must be a power of two. Working from the distance rather
    than recursing lets a caller take a prefix of the curve, which is what makes the
    non-power-of-two case below straightforward.
    """
    rx = ry = 0
    x = y = 0
    t = d
    s = 1
    while s < order:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        # Rotate the quadrant so the curve stays connected across the join.
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def hilbert_points(height: int, width: int) -> list[tuple[int, int]]:
    """Every pixel of a height x width image, in Hilbert order.

    The Hilbert curve is only defined on a square of side 2^k, so the curve is generated for
    the smallest enclosing power of two and points outside the image are skipped. Skipping
    preserves the ordering and keeps the traversal connected in the region that matters,
    which is why this is the standard treatment. It does mean a very non-square image has a
    few longer hops where the curve leaves and re-enters the frame; `max_step` in the tests
    measures exactly that rather than pretending it does not happen.
    """
    if height <= 0 or width <= 0:
        raise ValueError("hilbert_points needs a positive shape, got %rx%r" % (height, width))
    order = 1
    while order < max(height, width):
        order *= 2
    pts = []
    for d in range(order * order):
        x, y = _d2xy(order, d)
        if y < height and x < width:
            pts.append((y, x))
    return pts


def queue_weights(size: int = DEFAULT_QUEUE, ratio: float = DEFAULT_RATIO) -> np.ndarray:
    """Geometric weights for the error queue, newest first.

    Index 0 is the most recent error and gets weight 1. Index size-1 is the oldest and gets
    weight `ratio`. The weights are normalised so the whole queue distributes a total of 1,
    which is what keeps mean luminance intact.
    """
    if size < 1:
        raise ValueError("queue size must be at least 1")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    if size == 1:
        return np.array([1.0])
    exponents = np.arange(size) / (size - 1)
    w = ratio ** exponents
    return w / w.sum()


def riemersma(img, levels: int = 2, queue_size: int = DEFAULT_QUEUE,
              ratio: float = DEFAULT_RATIO) -> np.ndarray:
    """Dither a single-channel float image along a Hilbert curve.

    Returns a float array on the same quantisation grid as the kernel methods, so the two
    families are directly comparable.
    """
    a = np.asarray(img, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("riemersma works on one channel, got shape %r" % (a.shape,))
    if levels < 2:
        raise ValueError("levels must be at least 2")

    weights = queue_weights(queue_size, ratio)
    out = np.zeros_like(a)
    # The queue holds recent quantisation errors, newest at index 0.
    errors = np.zeros(queue_size)

    for (r, c) in hilbert_points(a.shape[0], a.shape[1]):
        # Only the correction is clamped into range; the stored error stays unclamped so the
        # accounting below stays exact.
        target = a[r, c] + float(np.dot(weights, errors))
        value = quantize_levels(min(max(target, 0.0), 1.0), levels)
        out[r, c] = value
        errors = np.roll(errors, 1)
        errors[0] = target - value
    return out


def traversal_stats(height: int, width: int) -> dict:
    """Measurable facts about the curve, used by the tests and reported by the CLI."""
    pts = hilbert_points(height, width)
    steps = [abs(pts[i][0] - pts[i - 1][0]) + abs(pts[i][1] - pts[i - 1][1])
             for i in range(1, len(pts))]
    return {
        "points": len(pts),
        "expected": height * width,
        "unique": len(set(pts)),
        "adjacent_steps": sum(1 for s in steps if s == 1),
        "total_steps": len(steps),
        "max_step": max(steps) if steps else 0,
    }
