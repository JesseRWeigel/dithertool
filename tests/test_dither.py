"""Measurable properties, both directions.

The premise of this project is that dithering implementations are usually wrong in specific,
nameable ways, which obliges the tests to detect those specific wrongnesses rather than confirm
that something ran. Every assertion here targets a property a plausible broken implementation
would violate, and the ones that could pass vacuously carry a negative control that feeds the
same check a deliberately wrong input and requires rejection.
"""
import unittest

import numpy as np

from dithertool.ordered import bayer_ranks, bayer_thresholds, ordered_dither
from dithertool.diffusion import KERNELS, diffuse, atkinson, floyd_steinberg
from dithertool.bluenoise import blue_noise_ranks, blue_noise_thresholds, white_noise_thresholds
from dithertool.spectrum import spectrum_metrics, is_blue_noise, tile_to
from dithertool.halftone import halftone, coverage, measure_screen, screen_field
from dithertool.imageops import (flat, color_bars, step_wedge, linear_ramp,
                                 radial_gradient, to_u8)

# Kernels that distribute the whole error. Atkinson deliberately does not, so it is excluded
# from luminance preservation and gets its own test.
CONSERVING = [k for k, v in KERNELS.items() if abs(sum(w for _, _, w in v) - 1.0) < 1e-9]


class TestBayer(unittest.TestCase):
    def test_order_n_gives_exactly_n_squared_distinct_ranks(self):
        # A transposed or mis-recursed Bayer matrix normally duplicates ranks. Demanding an
        # exact permutation of 0..n^2-1 catches that where a "looks random" check would not.
        for order in (2, 4, 8, 16):
            ranks = bayer_ranks(order)
            self.assertEqual(ranks.shape, (order, order))
            self.assertEqual(sorted(ranks.ravel().tolist()), list(range(order * order)),
                             f"order {order} is not a permutation of 0..n^2-1")

    def test_thresholds_stay_strictly_inside_the_unit_interval(self):
        # A threshold of exactly 0 or 1 makes an extreme level unreachable, which shows up as
        # crushed shadows or clipped highlights. Correct normalisation is (rank + 0.5)/n^2.
        for order in (2, 4, 8):
            th = bayer_thresholds(order)
            self.assertGreater(th.min(), 0.0)
            self.assertLess(th.max(), 1.0)
            self.assertAlmostEqual(float(th.mean()), 0.5, places=9,
                                   msg="a normalised threshold set must average 0.5")

    def test_the_matrix_is_not_its_own_transpose(self):
        # Transposing the Bayer matrix is a common bug that still passes a permutation check.
        # The canonical matrix is genuinely asymmetric, so equality here would prove a swap.
        ranks = bayer_ranks(4)
        self.assertFalse(np.array_equal(ranks, ranks.T),
                         "a symmetric Bayer matrix means the recursion is wrong")

    def test_flat_midgrey_dithers_to_half_coverage(self):
        out = ordered_dither(flat(0.5, 64, 64), bayer_thresholds(8))
        self.assertAlmostEqual(coverage(out), 0.5, delta=0.02)

    def test_a_ramp_dithers_monotonically(self):
        # Coverage must increase with input level. A sign error in the comparison inverts the
        # image, which every mean-based test above would happily accept.
        th = bayer_thresholds(8)
        levels = [coverage(ordered_dither(flat(v, 64, 64), th)) for v in (0.1, 0.3, 0.5, 0.7, 0.9)]
        self.assertEqual(levels, sorted(levels), f"coverage is not monotonic in level: {levels}")


class TestErrorDiffusion(unittest.TestCase):
    def test_output_is_strictly_one_bit(self):
        img = radial_gradient(64, 64)
        for name in KERNELS:
            out = diffuse(img, kernel=name).image
            vals = np.unique(out)
            self.assertTrue(set(vals.tolist()) <= {0.0, 1.0},
                            f"{name} produced non-binary values: {vals[:6]}")

    def test_error_accounting_balances_exactly(self):
        # The strongest available test, and an exact identity rather than a tolerance:
        #   sum(in) - sum(out) - (1-T)*sum(error) - edge_leak == 0
        # Any error silently dropped at a border, double-counted, or lost to a clamp shows up
        # here as a nonzero residual. Every kernel must satisfy it, Atkinson included, because
        # T carries the fraction the kernel intends to distribute.
        img = step_wedge(96, 64, steps=8)
        for name in KERNELS:
            for edge in ("renormalize", "discard"):
                r = diffuse(img, kernel=name, edge=edge)
                self.assertAlmostEqual(
                    r.bookkeeping_residual(), 0.0, places=6,
                    msg=f"{name}/{edge} loses error: residual {r.bookkeeping_residual():.3e}")

    def test_conserving_kernels_preserve_mean_luminance(self):
        for level in (0.25, 0.5, 0.75):
            img = flat(level, 80, 80)
            for name in CONSERVING:
                got = float(diffuse(img, kernel=name).image.mean())
                self.assertAlmostEqual(got, level, delta=0.03,
                                       msg=f"{name} at {level} drifted to {got:.3f}")

    def test_atkinson_distributes_six_eighths_not_eight_eighths(self):
        # THE classic Atkinson bug. It deliberately discards a quarter of the error to gain
        # contrast: six neighbours at 1/8 each. An implementation "fixed" to sum to 1 is just
        # Floyd-Steinberg with a different stencil, and looks wrong in exactly the way
        # Atkinson exists to avoid.
        weights = [w for _, _, w in KERNELS["atkinson"]]
        self.assertEqual(len(weights), 6, "Atkinson has exactly six recipients")
        for w in weights:
            self.assertAlmostEqual(w, 0.125, places=9)
        self.assertAlmostEqual(sum(weights), 0.75, places=9,
                               msg="Atkinson must discard a quarter of the error")

    def test_atkinson_expands_contrast_antisymmetrically(self):
        # The visible consequence of that discard, stated as something measurable. An earlier
        # version of this test asserted Atkinson is "lighter in the midtones" and failed: on a
        # flat mid-grey both algorithms average exactly 0.5, because the discarded error is
        # symmetric there. The real property is contrast expansion. Atkinson pushes shadows
        # toward black and highlights toward white, monotonically and antisymmetrically about
        # 0.5, while a conserving kernel tracks the input level. Measured on this
        # implementation, level 0.1 renders at 0.003 and level 0.9 at 0.997.
        for level in (0.1, 0.2, 0.3, 0.4):
            lo, hi = flat(level, 96, 96), flat(1.0 - level, 96, 96)
            a_lo, a_hi = float(atkinson(lo).mean()), float(atkinson(hi).mean())
            f_lo, f_hi = float(floyd_steinberg(lo).mean()), float(floyd_steinberg(hi).mean())
            self.assertLess(a_lo, f_lo - 0.01, f"shadow {level} was not pushed darker")
            self.assertGreater(a_hi, f_hi + 0.01, f"highlight {1 - level} was not pushed lighter")
            # Antisymmetry: the two deviations must mirror each other about mid-grey.
            self.assertAlmostEqual((a_lo - f_lo), -(a_hi - f_hi), places=6,
                                   msg=f"the tone curve is not symmetric about 0.5 at {level}")

    def test_the_contrast_expansion_grows_toward_the_extremes(self):
        # Monotonicity of the effect, which a single-point check cannot establish.
        deltas = [float(atkinson(flat(v, 96, 96)).mean()) - v for v in (0.4, 0.3, 0.2, 0.1)]
        self.assertEqual(deltas, sorted(deltas, reverse=True),
                         f"the shadow push should deepen toward black, got {deltas}")

    def test_serpentine_changes_the_result(self):
        # If the flag does nothing the scan is not actually alternating direction, and the
        # worm artifacts serpentine exists to break up are still there.
        img = linear_ramp(64, 48)
        self.assertFalse(
            np.array_equal(diffuse(img, serpentine=True).image,
                           diffuse(img, serpentine=False).image),
            "serpentine had no effect, so the scan direction is not alternating")

    def test_an_amplifying_kernel_does_fail_the_luminance_check(self):
        # Negative control for the luminance test: it has to be able to fail. A kernel whose
        # weights sum well above 1 amplifies error and drifts the mean off the input level.
        bad = [(1, 0, 0.9), (0, 1, 0.9), (-1, 1, 0.9)]
        got = float(diffuse(flat(0.5, 80, 80), kernel=bad).image.mean())
        self.assertGreater(abs(got - 0.5), 0.03,
                           "an amplifying kernel did not drift, so the luminance check is inert")


class TestBlueNoise(unittest.TestCase):
    def test_ranks_are_a_permutation(self):
        ranks = blue_noise_ranks(32)
        self.assertEqual(sorted(ranks.ravel().tolist()), list(range(32 * 32)))

    def test_the_generated_mask_is_spectrally_blue(self):
        # The only test that actually distinguishes blue noise from noise: low radial
        # frequencies must be suppressed relative to high ones. Void-and-cluster produces
        # this. A shuffled permutation does not, however random it looks.
        ok, m = is_blue_noise(blue_noise_thresholds(32))
        self.assertTrue(ok, f"the generated mask is not spectrally blue: {m}")
        self.assertGreater(m.ratio, 10.0, f"high/low power ratio only {m.ratio:.2f}")

    def test_white_noise_is_rejected_by_the_same_check(self):
        # The negative control that matters most here. A good deal of published "blue noise"
        # is white noise with a better name, and a check that cannot separate them proves
        # nothing about the check above.
        ok, m = is_blue_noise(white_noise_thresholds(32, seed=20260726))
        self.assertFalse(ok, f"white noise passed the blue-noise test: {m}")
        self.assertLess(m.ratio, 3.0, f"white noise ratio should sit near 1, got {m.ratio:.2f}")

    def test_blue_and_white_are_orders_of_magnitude_apart(self):
        rb = spectrum_metrics(blue_noise_thresholds(32)).ratio
        rw = spectrum_metrics(white_noise_thresholds(32, seed=7)).ratio
        self.assertGreater(rb / rw, 50.0,
                           f"blue {rb:.1f} against white {rw:.1f} is not a clear separation")

    def test_bayer_is_not_blue_and_the_spectrum_says_why(self):
        # Bayer is the third case, and neither blue nor white. Its energy sits in a few sharp
        # peaks, which is the measurable cause of its visible cross-hatch. A 4x4 matrix is too
        # small to band, so it is tiled first, which is also how a dither mask is really used.
        tiled = tile_to(bayer_thresholds(4), 64, 64)
        ok, m = is_blue_noise(tiled)
        self.assertFalse(ok, f"Bayer should not pass a blue-noise test: {m}")
        self.assertGreater(m.peak_to_mean, 12.0,
                           f"Bayer's spectrum should be peaky, got peak_to_mean {m.peak_to_mean:.1f}")

    def test_generation_is_deterministic_for_a_seed(self):
        self.assertTrue(np.array_equal(blue_noise_ranks(16, seed=3), blue_noise_ranks(16, seed=3)))
        self.assertFalse(np.array_equal(blue_noise_ranks(16, seed=3), blue_noise_ranks(16, seed=4)),
                         "different seeds produced identical masks, so the seed is ignored")

    def test_dithering_through_the_blue_mask_preserves_level(self):
        th = blue_noise_thresholds(32)
        for level in (0.25, 0.5, 0.75):
            got = coverage(ordered_dither(flat(level, 64, 64), th))
            self.assertAlmostEqual(got, level, delta=0.03, msg=f"level {level} gave {got:.3f}")


class TestHalftone(unittest.TestCase):
    def test_coverage_tracks_the_requested_level(self):
        for level in (0.2, 0.5, 0.8):
            got = coverage(halftone(flat(level, 96, 96), period=6.0, angle_deg=45.0))
            self.assertAlmostEqual(got, level, delta=0.06, msg=f"level {level} gave {got:.3f}")

    def test_rotating_the_screen_keeps_the_cell_period(self):
        # The classic halftone bug is rotating the sampling grid rather than the screen, which
        # aliases and changes the effective cell size with angle. The measured period has to
        # hold across angles, or the tonal response shifts as you rotate.
        periods = []
        for angle in (0.0, 15.0, 30.0, 45.0, 75.0):
            field = screen_field((120, 120), period=8.0, angle_deg=angle)
            periods.append(measure_screen(field < 0.5)[0])
        spread = max(periods) - min(periods)
        self.assertLess(spread, 2.0,
                        f"period varies by {spread:.2f} across angles, so the grid is rotating")

    def test_all_spot_shapes_are_one_bit_and_track_level(self):
        for spot in ("round", "line", "diamond"):
            out = halftone(flat(0.4, 96, 96), period=8.0, angle_deg=15.0, spot=spot)
            self.assertTrue(set(np.unique(out).tolist()) <= {0.0, 1.0}, f"{spot} is not 1-bit")
            self.assertAlmostEqual(coverage(out), 0.4, delta=0.07, msg=f"{spot} coverage is off")


class TestDeterminismAndIO(unittest.TestCase):
    def test_every_algorithm_is_byte_identical_on_a_rerun(self):
        img = radial_gradient(64, 64)
        th = bayer_thresholds(8)
        for label, fn in (("ordered", lambda i: ordered_dither(i, th)),
                          ("atkinson", lambda i: atkinson(i)),
                          ("floyd-steinberg", lambda i: floyd_steinberg(i)),
                          ("halftone", lambda i: halftone(i, period=6.0))):
            self.assertTrue(np.array_equal(fn(img), fn(img)), f"{label} is not reproducible")

    def test_png_round_trips_within_one_8bit_step(self):
        import pathlib
        import tempfile
        from dithertool.pngio import write_png, read_png, read_header
        img = to_u8(color_bars(48, 32))
        p = pathlib.Path(tempfile.mkdtemp()) / "t.png"
        write_png(str(p), img)
        hdr = read_header(str(p))
        self.assertEqual((hdr["height"], hdr["width"]), img.shape[:2])
        back = read_png(str(p))
        self.assertEqual(back.shape, img.shape)
        self.assertTrue(np.array_equal(back, img), "png did not round-trip exactly")

    def test_a_one_bit_png_really_is_one_bit_on_disk(self):
        import pathlib
        import tempfile
        from dithertool.pngio import write_png, read_header
        out = to_u8(atkinson(radial_gradient(48, 48)))
        p = pathlib.Path(tempfile.mkdtemp()) / "b.png"
        write_png(str(p), out, bitdepth=1)
        self.assertEqual(read_header(str(p))["bitdepth"], 1,
                         "a 1-bit request wrote a wider file, so the size claim is false")


if __name__ == "__main__":
    unittest.main()


class TestPngRejections(unittest.TestCase):
    """Malformed and unsupported files must be refused by name, not by accident.

    A decoder that eventually throws for the wrong reason is still failing closed, so these
    are about the diagnostic being true. A 16-bit PNG used to report "unknown PNG filter type
    128", which is a correct statement about the byte the decoder happened to land on and a
    misleading one about the file.
    """

    @staticmethod
    def _synth(width=4, height=4, depth=8, ctype=0, interlace=0):
        import struct
        import zlib
        def chunk(tag, data):
            return (struct.pack(">I", len(data)) + tag + data
                    + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))
        ihdr = struct.pack(">IIBBBBB", width, height, depth, ctype, 0, 0, interlace)
        raw = b"".join(b"\x00" + b"\x80" * width for _ in range(height))
        comp = zlib.compressobj()
        return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
                + chunk(b"IDAT", comp.compress(raw) + comp.flush()) + chunk(b"IEND", b""))

    def _read(self, blob):
        import pathlib
        import tempfile
        from dithertool.pngio import read_png
        p = pathlib.Path(tempfile.mkdtemp()) / "x.png"
        p.write_bytes(blob)
        return read_png(str(p))

    def test_sixteen_bit_is_refused_by_name(self):
        with self.assertRaises(ValueError) as cm:
            self._read(self._synth(depth=16))
        self.assertIn("bit depth", str(cm.exception).lower(),
                      f"the message blames the wrong thing: {cm.exception}")

    def test_interlaced_is_refused_by_name(self):
        with self.assertRaises(ValueError) as cm:
            self._read(self._synth(interlace=1))
        self.assertIn("interlaced", str(cm.exception).lower())

    def test_a_palette_image_without_a_palette_is_refused(self):
        with self.assertRaises(ValueError):
            self._read(self._synth(ctype=3))

    def test_a_supported_file_still_reads(self):
        # Negative control for the three above: if the synthesiser produced unreadable bytes
        # regardless of the field under test, those tests would pass for the wrong reason.
        out = self._read(self._synth(depth=8, ctype=0))
        self.assertEqual(out.shape, (4, 4))
