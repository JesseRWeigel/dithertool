"""Blue-noise threshold masks by void-and-cluster (Ulichney, 1993).

The specific error this module exists to avoid: shipping `random()` and calling
it blue noise. White noise has flat power at all frequencies, so a white-noise
threshold mask produces the clumpy, grainy look people are trying to escape when
they reach for blue noise. Blue noise has its energy pushed to high frequencies.
``spectrum.is_blue_noise`` checks that, and the test suite runs it in both
directions: the mask this module produces must pass, and uniform random noise of
the same size must fail.

Method, on a toroidal domain so the mask tiles seamlessly:

1. Scatter a small fraction of ones at random, then repeatedly move the point
   from the tightest cluster into the largest void until that swap becomes a
   no-op. This gives a well distributed starting pattern that no longer depends
   much on the initial random draw.
2. Rank the starting ones downward by repeatedly removing the tightest cluster.
3. Rank upward by repeatedly filling the largest void until every pixel has a
   rank.

The paper splits step 3 at half density and reverses the roles of ones and
zeros. With a linear filter that split is a no-op: for kernel sum ``K``,
``filter(1 - p) == K - filter(p)`` exactly, so the tightest cluster of zeros and
the largest void of ones are the same pixel. Keeping one loop avoids the
copy-paste bug where the reversed phase forgets to invert its comparison.

Cluster and void are measured with a wrapped Gaussian of sigma 1.9. The filtered
field is maintained incrementally: toggling one pixel adds or subtracts one
shifted copy of the kernel, so the whole construction is O(n^4) element updates
with no repeated convolution.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .ordered import ranks_to_thresholds


def _wrapped_gaussian(n: int, sigma: float) -> np.ndarray:
    """Gaussian kernel on the n x n torus, centred at index (0, 0)."""
    d = np.arange(n)
    d = np.minimum(d, n - d)
    dy2 = (d ** 2)[:, None]
    dx2 = (d ** 2)[None, :]
    return np.exp(-(dy2 + dx2) / (2.0 * sigma * sigma))


def _argmax_masked(field: np.ndarray, mask: np.ndarray) -> int:
    """Index of the largest field value where mask is True. Ties: lowest index."""
    vals = np.where(mask, field, -np.inf)
    return int(np.argmax(vals))


def _argmin_masked(field: np.ndarray, mask: np.ndarray) -> int:
    vals = np.where(mask, field, np.inf)
    return int(np.argmin(vals))


def void_and_cluster_ranks(
    n: int = 64,
    sigma: float = 1.9,
    seed: int = 0,
    initial_density: float = 0.1,
    max_swaps: int = 100000,
) -> np.ndarray:
    """Rank permutation of 0..n^2-1 with blue-noise spatial distribution."""
    if n < 4:
        raise ValueError("n must be at least 4")
    kernel = _wrapped_gaussian(n, sigma).ravel()
    size = n * n

    # Shifting a flat kernel by a linear index is a roll in each axis. Precompute
    # the 2-D view once and reuse it.
    kernel2 = kernel.reshape(n, n)

    def shifted(idx: int) -> np.ndarray:
        y, x = divmod(idx, n)
        return np.roll(np.roll(kernel2, y, axis=0), x, axis=1).ravel()

    rng = np.random.RandomState(seed)   # frozen legacy stream: stable across versions
    ones = max(1, int(round(size * initial_density)))
    pattern = np.zeros(size, dtype=bool)
    pattern[rng.permutation(size)[:ones]] = True

    field = np.zeros(size, dtype=np.float64)
    for idx in np.flatnonzero(pattern):
        field += shifted(int(idx))

    # 1. relax the initial pattern
    for _ in range(max_swaps):
        tight = _argmax_masked(field, pattern)
        pattern[tight] = False
        field -= shifted(tight)
        void = _argmin_masked(field, ~pattern)
        if void == tight:
            pattern[tight] = True
            field += shifted(tight)
            break
        pattern[void] = True
        field += shifted(void)
    else:
        raise RuntimeError("void-and-cluster relaxation did not settle")

    initial = pattern.copy()
    initial_field = field.copy()
    ranks = np.full(size, -1, dtype=np.int64)

    # 2. rank the starting ones downward
    work = pattern.copy()
    field = initial_field.copy()
    for rank in range(ones - 1, -1, -1):
        tight = _argmax_masked(field, work)
        work[tight] = False
        field -= shifted(tight)
        ranks[tight] = rank

    # 3. rank upward from the starting pattern to full
    work = initial.copy()
    field = initial_field.copy()
    for rank in range(ones, size):
        void = _argmin_masked(field, ~work)
        work[void] = True
        field += shifted(void)
        ranks[void] = rank

    if (ranks < 0).any():
        raise RuntimeError("void-and-cluster left %d pixels unranked" % int((ranks < 0).sum()))
    return ranks.reshape(n, n)


@lru_cache(maxsize=8)
def _cached_ranks(n: int, sigma: float, seed: int) -> np.ndarray:
    r = void_and_cluster_ranks(n=n, sigma=sigma, seed=seed)
    r.flags.writeable = False
    return r


def blue_noise_ranks(n: int = 64, sigma: float = 1.9, seed: int = 0) -> np.ndarray:
    return np.array(_cached_ranks(n, sigma, seed))


def blue_noise_thresholds(n: int = 64, sigma: float = 1.9, seed: int = 0) -> np.ndarray:
    """Cell-centred blue-noise threshold mask in (0, 1), tileable."""
    return ranks_to_thresholds(blue_noise_ranks(n=n, sigma=sigma, seed=seed))


def white_noise_thresholds(n: int = 64, seed: int = 0) -> np.ndarray:
    """A white-noise mask of the same construction, for negative controls.

    Present so the test suite can prove the blue-noise test rejects something.
    Do not use this for output you care about.
    """
    rng = np.random.RandomState(seed + 991)
    return ranks_to_thresholds(rng.permutation(n * n).reshape(n, n))
