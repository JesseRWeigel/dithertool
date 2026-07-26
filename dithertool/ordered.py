"""Ordered dithering: Bayer matrix construction and the shared tiling rule.

Specific errors this module avoids, all covered by tests:

1. **The recursion has a fixed orientation.** With ``M`` the order-n matrix,

   ::

       M(2n) = [[ 4M+0, 4M+2 ],
                [ 4M+3, 4M+1 ]]

   Swapping the ``+2`` and ``+3`` quadrants transposes the pattern, which is
   easy to do and hard to see, so the tests compare against the literal
   canonical 4x4 matrix rather than against a property.

2. **Thresholds are cell-centred.** For an order-n matrix the threshold for rank
   ``v`` is ``(v + 0.5) / n^2``, mean exactly 0.5. Dividing by ``n^2 - 1`` puts a
   threshold at 1.0 and clips highlights; dividing ``v`` alone by ``n^2`` gives a
   mean of ``0.5 - 1/(2n^2)`` and biases the whole image light.

3. **Only powers of two.** The recursion produces orders 1, 2, 4, 8, 16, ...
   A "Bayer 3x3" or "Bayer 6x6" is some other matrix wearing the name.

4. **Tiling uses modulo of the pixel coordinate**, so the matrix must be tiled
   at its own size with no scaling. Nearest-neighbour scaling of a Bayer matrix
   to the image size, which appears in a surprising amount of shader code,
   destroys the void-and-cluster balance the recursion establishes.

The multilevel rule is ``v*(L-1)`` split into integer and fractional parts, with
the fraction compared against the threshold. That degenerates to the plain
comparison at ``L = 2``.
"""

from __future__ import annotations

import numpy as np

CANONICAL_BAYER_4 = np.array(
    [
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5],
    ],
    dtype=np.int64,
)


def bayer_ranks(order: int) -> np.ndarray:
    """The order-n Bayer matrix as integer ranks 0..n^2-1.

    ``order`` must be a power of two.
    """
    if order < 1 or (order & (order - 1)) != 0:
        raise ValueError("Bayer order must be a power of two, got %d" % order)
    m = np.zeros((1, 1), dtype=np.int64)
    while m.shape[0] < order:
        m = np.block(
            [
                [4 * m + 0, 4 * m + 2],
                [4 * m + 3, 4 * m + 1],
            ]
        )
    return m


def ranks_to_thresholds(ranks) -> np.ndarray:
    """Cell-centred thresholds in (0, 1) from a rank permutation matrix."""
    r = np.asarray(ranks, dtype=np.float64)
    n = r.size
    uniq = np.unique(r)
    if uniq.size != n:
        raise ValueError(
            "rank matrix must be a permutation of 0..%d, found %d distinct values"
            % (n - 1, uniq.size)
        )
    if uniq[0] != 0 or uniq[-1] != n - 1:
        raise ValueError("rank matrix must run from 0 to %d" % (n - 1))
    return (r + 0.5) / n


def bayer_thresholds(order: int) -> np.ndarray:
    return ranks_to_thresholds(bayer_ranks(order))


def ordered_dither(img, thresholds, levels: int = 2, offset=(0, 0)) -> np.ndarray:
    """Dither one channel against a tiled threshold matrix.

    thresholds
        2-D array of values in (0, 1), tiled by modulo of the pixel coordinate.
    offset
        (y, x) phase shift of the tiling. Useful for giving each colour channel
        a different phase so the channels do not align into visible clumps.
    """
    a = np.asarray(img, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("ordered_dither works on one channel, got %r" % (a.shape,))
    if levels < 2:
        raise ValueError("levels must be >= 2")
    t = np.asarray(thresholds, dtype=np.float64)
    th, tw = t.shape
    h, w = a.shape
    ys = (np.arange(h) + offset[0]) % th
    xs = (np.arange(w) + offset[1]) % tw
    tile = t[np.ix_(ys, xs)]

    scaled = np.clip(a, 0.0, 1.0) * (levels - 1)
    base = np.floor(scaled)
    frac = scaled - base
    step = (frac > tile).astype(np.float64)
    out = (base + step) / (levels - 1)
    return np.clip(out, 0.0, 1.0)
