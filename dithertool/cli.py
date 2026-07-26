"""Command line interface.

Three jobs: dither an image, print the measured properties of an algorithm, and build the
comparison gallery. The `measure` subcommand exists because the numbers are the point of this
project, and a user should be able to reproduce them without reading the test suite.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import bluenoise, diffusion, halftone, imageops, ordered, pngio, riemersma, spectrum

ALGORITHMS = ["bayer", "blue-noise", "white-noise", "riemersma", "halftone"] + list(diffusion.KERNELS)

PATTERNS = {
    "ramp": lambda: imageops.linear_ramp(512, 128),
    "wedge": lambda: imageops.step_wedge(512, 128, steps=16),
    "radial": lambda: imageops.radial_gradient(256, 256),
    "zone": lambda: imageops.zone_plate(256),
    "midgrey": lambda: imageops.flat(0.5, 256, 256),
    "lines": lambda: imageops.line_pattern(256, 128),
}


def to_gray(a: np.ndarray) -> np.ndarray:
    """Collapse to one channel, because every algorithm here is single-channel.

    Uses Rec.709 luminance in linear light rather than averaging the encoded channels, which
    is the difference between a correct grey and a muddy one.
    """
    a = imageops.to_float(a)
    if a.ndim == 2:
        return a
    if a.ndim == 3 and a.shape[2] >= 3:
        lin = imageops.srgb_to_linear(a[:, :, :3])
        return imageops.linear_to_srgb(imageops.rec709_luminance(lin))
    raise ValueError("cannot interpret shape %r as an image" % (a.shape,))


def apply(name: str, img: np.ndarray, *, levels: int = 2, size: int = 64, order: int = 8,
          period: float = 8.0, angle: float = 45.0, spot: str = "round",
          seed: int = 0) -> np.ndarray:
    """Run one algorithm by name. Every caller in this project goes through here."""
    if name == "bayer":
        return ordered.ordered_dither(img, ordered.bayer_thresholds(order), levels=levels)
    if name == "blue-noise":
        return ordered.ordered_dither(img, bluenoise.blue_noise_thresholds(size, seed=seed),
                                      levels=levels)
    if name == "white-noise":
        return ordered.ordered_dither(img, bluenoise.white_noise_thresholds(size, seed=seed),
                                      levels=levels)
    if name == "riemersma":
        return riemersma.riemersma(img, levels=levels)
    if name == "halftone":
        return halftone.halftone(img, period=period, angle_deg=angle, spot=spot, levels=levels)
    if name in diffusion.KERNELS:
        return diffusion.diffuse(img, kernel=name, levels=levels).image
    raise ValueError("unknown algorithm %r, pick from: %s" % (name, ", ".join(ALGORITHMS)))


def load(path: str, pattern: str | None) -> tuple[np.ndarray, str]:
    if pattern:
        if pattern not in PATTERNS:
            raise ValueError("unknown pattern %r, pick from: %s"
                             % (pattern, ", ".join(PATTERNS)))
        return PATTERNS[pattern](), f"pattern:{pattern}"
    if not path:
        raise ValueError("give an input image or --pattern")
    return to_gray(pngio.read_png(path)), path


def cmd_dither(a) -> int:
    img, source = load(a.input, a.pattern)
    out = apply(a.algorithm, img, levels=a.levels, size=a.size, order=a.order,
                period=a.period, angle=a.angle, spot=a.spot, seed=a.seed)
    pngio.write_png(a.output, imageops.to_u8(out), bitdepth=1 if a.levels == 2 else 8)
    sys.stderr.write(
        f"{a.algorithm}: {source} -> {a.output} "
        f"({out.shape[1]}x{out.shape[0]}, mean in {float(np.mean(img)):.4f}, "
        f"out {float(np.mean(out)):.4f})\n")
    return 0


def cmd_measure(a) -> int:
    """Print the properties the test suite asserts, so they can be checked by hand."""
    print("mask spectra (a dither mask is judged by its radial power spectrum)")
    print(f"  {'mask':<22} {'low':>10} {'high':>10} {'ratio':>9} {'slope':>7} "
          f"{'peak/mean':>10}  blue?")
    masks = [
        ("blue-noise 64", bluenoise.blue_noise_thresholds(64)),
        ("blue-noise 32", bluenoise.blue_noise_thresholds(32)),
        ("white-noise 64", bluenoise.white_noise_thresholds(64, seed=1)),
        ("bayer 8 (tiled)", spectrum.tile_to(ordered.bayer_thresholds(8), 64, 64)),
        ("bayer 4 (tiled)", spectrum.tile_to(ordered.bayer_thresholds(4), 64, 64)),
    ]
    for label, m in masks:
        ok, s = spectrum.is_blue_noise(m)
        print(f"  {label:<22} {s.low:10.3e} {s.high:10.3e} {s.ratio:9.2f} {s.slope:7.2f} "
              f"{s.peak_to_mean:10.2f}  {'yes' if ok else 'no'}")

    print("\ntone response on flat fields (output mean against requested level)")
    algos = ["bayer", "blue-noise", "riemersma", "floyd-steinberg", "atkinson"]
    print(f"  {'level':>6}" + "".join(f"{n:>17}" for n in algos))
    for level in (0.1, 0.25, 0.5, 0.75, 0.9):
        row = f"  {level:6.2f}"
        for name in algos:
            got = float(np.mean(apply(name, imageops.flat(level, 96, 96))))
            row += f"{got:11.4f}{got - level:+7.4f}"[:17].rjust(17)
        print(row)
    print("\n  atkinson deviates by design: it discards a quarter of the error, which expands")
    print("  contrast antisymmetrically about mid-grey. Every other row should sit near zero.")

    print("\nerror accounting (sum(in) - sum(out) - (1-T)*sum(err) - leak, must be 0)")
    img = imageops.step_wedge(96, 64, steps=8)
    for name in diffusion.KERNELS:
        r = diffusion.diffuse(img, kernel=name)
        print(f"  {name:<18} T={r.kernel_total:.4f}  leak={r.edge_leak:+.4e}  "
              f"residual={r.bookkeeping_residual():+.2e}")

    print("\nhilbert traversal (riemersma walks a space-filling curve)")
    for shape in ((64, 64), (48, 64)):
        s = riemersma.traversal_stats(*shape)
        print(f"  {shape[0]}x{shape[1]}: {s['points']}/{s['expected']} pixels, "
              f"{s['unique']} unique, {s['adjacent_steps']}/{s['total_steps']} steps adjacent, "
              f"longest step {s['max_step']}")
    return 0


def cmd_gallery(a) -> int:
    from .gallery import build
    return build(a.outdir)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="dithertool",
        description="Correct Bayer, blue-noise, error-diffusion and Riemersma dithering, "
                    "plus rotated-screen halftoning, with the properties measured rather "
                    "than asserted.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dither", help="dither one image")
    d.add_argument("input", nargs="?", help="input PNG")
    d.add_argument("-o", "--output", required=True)
    d.add_argument("-a", "--algorithm", default="blue-noise", choices=ALGORITHMS)
    d.add_argument("--pattern", choices=list(PATTERNS),
                   help="use a generated test pattern instead of an input file")
    d.add_argument("--levels", type=int, default=2, help="output levels, 2 for 1-bit")
    d.add_argument("--size", type=int, default=64, help="noise mask side")
    d.add_argument("--order", type=int, default=8, help="bayer matrix order")
    d.add_argument("--period", type=float, default=8.0, help="halftone cell period")
    d.add_argument("--angle", type=float, default=45.0, help="halftone screen angle")
    d.add_argument("--spot", default="round", choices=["round", "line", "diamond"])
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(fn=cmd_dither)

    m = sub.add_parser("measure", help="print the measured properties of every algorithm")
    m.set_defaults(fn=cmd_measure)

    g = sub.add_parser("gallery", help="build the 1:1 comparison gallery")
    g.add_argument("--outdir", default="docs")
    g.set_defaults(fn=cmd_gallery)

    a = p.parse_args(argv)
    try:
        return a.fn(a)
    except (ValueError, FileNotFoundError, OSError) as err:
        sys.stderr.write(f"dithertool: {err}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
