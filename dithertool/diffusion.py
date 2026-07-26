"""Error diffusion, done so that the error is actually conserved.

The kernel table is the boring part. These are the parts implementations get
wrong, and each one is checked by the test suite:

1. **The working buffer must be floating point.** Adding the error back into a
   uint8 array clips at 0 and 255 and silently destroys it.

2. **Do not clamp the corrected value before quantizing.** The nearest-level
   choice is clamped because the output palette is finite, but the error is
   ``corrected - chosen`` using the *unclamped* corrected value. Implementations
   that clamp first throw away exactly the error that the algorithm exists to
   carry, which lifts shadows and crushes highlights. ``wrong.py`` contains that
   variant and the tests measure its drift.

3. **Weights must be renormalised at the borders, or the loss must be owned.**
   A kernel entry pointing off the image either gets dropped, which leaks error
   and tints the edges, or the in-bounds weights get rescaled so the same
   fraction of error is always distributed. Both are implemented,
   ``edge="renormalize"`` is the default, and the leak of the other one is
   measured rather than assumed to be negligible.

4. **Atkinson distributes only 3/4 of the error on purpose.** Its six neighbours
   each take 1/8, and the remaining 1/4 is discarded. That discard is what gives
   Atkinson its contrast. Rescaling the weights to sum to 1, for example 1/6
   each, is a different and much flatter algorithm that many implementations
   ship under Atkinson's name. ``KERNELS`` records each kernel's intended total
   and the tests assert Atkinson's is 0.75 and that the normalised impostor
   behaves measurably differently.

5. **Serpentine order mirrors the kernel.** Reversing the scan direction without
   mirroring ``dx`` diffuses error backwards into pixels already emitted, which
   quietly turns half your rows into a plain threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .imageops import quantize_levels

# (dx, dy, weight)
KERNELS: dict[str, list[tuple[int, int, float]]] = {
    "floyd-steinberg": [
        (1, 0, 7 / 16), (-1, 1, 3 / 16), (0, 1, 5 / 16), (1, 1, 1 / 16),
    ],
    "atkinson": [
        (1, 0, 1 / 8), (2, 0, 1 / 8),
        (-1, 1, 1 / 8), (0, 1, 1 / 8), (1, 1, 1 / 8),
        (0, 2, 1 / 8),
    ],
    "jarvis": [
        (1, 0, 7 / 48), (2, 0, 5 / 48),
        (-2, 1, 3 / 48), (-1, 1, 5 / 48), (0, 1, 7 / 48), (1, 1, 5 / 48), (2, 1, 3 / 48),
        (-2, 2, 1 / 48), (-1, 2, 3 / 48), (0, 2, 5 / 48), (1, 2, 3 / 48), (2, 2, 1 / 48),
    ],
    "stucki": [
        (1, 0, 8 / 42), (2, 0, 4 / 42),
        (-2, 1, 2 / 42), (-1, 1, 4 / 42), (0, 1, 8 / 42), (1, 1, 4 / 42), (2, 1, 2 / 42),
        (-2, 2, 1 / 42), (-1, 2, 2 / 42), (0, 2, 4 / 42), (1, 2, 2 / 42), (2, 2, 1 / 42),
    ],
    "burkes": [
        (1, 0, 8 / 32), (2, 0, 4 / 32),
        (-2, 1, 2 / 32), (-1, 1, 4 / 32), (0, 1, 8 / 32), (1, 1, 4 / 32), (2, 1, 2 / 32),
    ],
    "sierra3": [
        (1, 0, 5 / 32), (2, 0, 3 / 32),
        (-2, 1, 2 / 32), (-1, 1, 4 / 32), (0, 1, 5 / 32), (1, 1, 4 / 32), (2, 1, 2 / 32),
        (-1, 2, 2 / 32), (0, 2, 3 / 32), (1, 2, 2 / 32),
    ],
}

#: The fraction of each pixel's error a kernel is *meant* to pass on. Every
#: kernel here sums to 1 except Atkinson, which sums to 3/4 by design.
KERNEL_TOTALS: dict[str, float] = {name: sum(w for _, _, w in k) for name, k in KERNELS.items()}


@dataclass
class DiffusionResult:
    """Output plus the error accounting needed to prove nothing leaked."""

    image: np.ndarray
    total_in: float
    total_out: float
    total_error: float          # sum of e_i = corrected_i - emitted_i
    kernel_total: float         # T, the intended distributed fraction
    edge_leak: float            # signed error dropped off the border
    pixels: int
    levels: int
    residual: float = field(default=0.0)

    @property
    def mean_in(self) -> float:
        return self.total_in / self.pixels

    @property
    def mean_out(self) -> float:
        return self.total_out / self.pixels

    def bookkeeping_residual(self) -> float:
        """``sum(in) - sum(out) - (1-T)*sum(e) - edge_leak``, which must be 0.

        This is an identity, not a heuristic. If it is not zero to floating
        point precision, error went missing somewhere in the loop.
        """
        return (
            self.total_in
            - self.total_out
            - (1.0 - self.kernel_total) * self.total_error
            - self.edge_leak
        )


def diffuse(
    img,
    kernel="floyd-steinberg",
    levels: int = 2,
    serpentine: bool = True,
    edge: str = "renormalize",
) -> DiffusionResult:
    """Error-diffuse a single channel of floats in [0, 1].

    kernel
        A name from :data:`KERNELS` or an explicit ``[(dx, dy, weight), ...]``
        list. ``dy`` must be >= 0, and entries with ``dy == 0`` must have
        ``dx > 0``: a kernel may only push error at pixels not yet emitted.
    levels
        Output levels, >= 2. 2 gives a bilevel image.
    edge
        ``"renormalize"`` rescales the in-bounds weights so the same fraction of
        error is distributed everywhere. ``"discard"`` drops out-of-bounds
        weights, the traditional behaviour, and records the loss in
        ``edge_leak``.
    """
    if isinstance(kernel, str):
        if kernel not in KERNELS:
            raise KeyError("unknown kernel %r, have %s" % (kernel, sorted(KERNELS)))
        kname, ker = kernel, KERNELS[kernel]
    else:
        kname, ker = None, list(kernel)
    if levels < 2:
        raise ValueError("levels must be >= 2")
    if edge not in ("renormalize", "discard"):
        raise ValueError("edge must be 'renormalize' or 'discard'")
    for dx, dy, _ in ker:
        if dy < 0 or (dy == 0 and dx <= 0):
            raise ValueError(
                "kernel entry (%d,%d) points at an already-emitted pixel" % (dx, dy)
            )

    a = np.asarray(img, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("diffuse works on one channel, got shape %r" % (a.shape,))
    h, w = a.shape
    ktotal = float(sum(wt for _, _, wt in ker))

    work = a.reshape(-1).tolist()          # python floats: no per-element numpy overhead
    out = [0.0] * (h * w)
    total_error = 0.0
    edge_leak = 0.0

    for y in range(h):
        reverse = serpentine and (y % 2 == 1)
        xs = range(w - 1, -1, -1) if reverse else range(w)
        flip = -1 if reverse else 1
        for x in xs:
            i = y * w + x
            corrected = work[i]
            emitted = quantize_levels(corrected, levels)
            err = corrected - emitted
            out[i] = emitted
            total_error += err

            # which kernel entries land inside the image
            hits = []
            inside_weight = 0.0
            for dx, dy, wt in ker:
                nx = x + dx * flip
                ny = y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    hits.append((ny * w + nx, wt))
                    inside_weight += wt
            if not hits:
                edge_leak += err * ktotal
                continue
            if edge == "renormalize":
                scale = ktotal / inside_weight
                for j, wt in hits:
                    work[j] += err * wt * scale
            else:
                for j, wt in hits:
                    work[j] += err * wt
                edge_leak += err * (ktotal - inside_weight)

    result = np.asarray(out, dtype=np.float64).reshape(h, w)
    res = DiffusionResult(
        image=result,
        total_in=float(a.sum()),
        total_out=float(result.sum()),
        total_error=total_error,
        kernel_total=ktotal,
        edge_leak=edge_leak,
        pixels=h * w,
        levels=levels,
    )
    del kname
    return res


def floyd_steinberg(img, **kw) -> np.ndarray:
    return diffuse(img, "floyd-steinberg", **kw).image


def atkinson(img, **kw) -> np.ndarray:
    return diffuse(img, "atkinson", **kw).image
