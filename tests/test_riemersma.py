"""Riemersma dithering, including the claim about it that turned out to be false.

The Hilbert traversal is the part most likely to be silently wrong, because a broken curve
still produces a plausible-looking dithered image. It is therefore tested as a curve, on its
own terms, rather than only through its output.
"""
import unittest

import numpy as np

from dithertool.riemersma import (riemersma, hilbert_points, queue_weights, traversal_stats,
                                  DEFAULT_QUEUE, DEFAULT_RATIO)
from dithertool.diffusion import floyd_steinberg, diffuse
from dithertool.imageops import flat, radial_gradient, step_wedge
from dithertool.halftone import coverage


def diagonal_disagreement(binary):
    """Disagreement across each diagonal. Equal values mean no diagonal preference."""
    down = float(np.mean(binary[1:, 1:] != binary[:-1, :-1]))
    up = float(np.mean(binary[1:, :-1] != binary[:-1, 1:]))
    return down, up


def axis_disagreement(binary):
    h = float(np.mean(binary[:, 1:] != binary[:, :-1]))
    v = float(np.mean(binary[1:, :] != binary[:-1, :]))
    return h, v


class TestHilbertCurve(unittest.TestCase):
    def test_the_curve_visits_every_pixel_exactly_once(self):
        # The invariant that makes it a space-filling curve. A traversal that repeats or skips
        # pixels still yields a dithered-looking image, so the output cannot catch this.
        for shape in ((64, 64), (32, 32), (48, 64), (50, 37), (1, 1), (3, 9)):
            s = traversal_stats(*shape)
            self.assertEqual(s["points"], s["expected"],
                             f"{shape}: visited {s['points']} of {s['expected']} pixels")
            self.assertEqual(s["unique"], s["expected"], f"{shape}: revisited pixels")

    def test_every_step_is_adjacent_on_a_power_of_two_square(self):
        # The defining property of a Hilbert curve. If steps jump, the error memory bleeds
        # between spatially distant pixels and the method loses its whole justification.
        for side in (2, 4, 8, 16, 64):
            s = traversal_stats(side, side)
            self.assertEqual(s["max_step"], 1,
                             f"{side}x{side}: a step of {s['max_step']} means the curve jumps")
            self.assertEqual(s["adjacent_steps"], s["total_steps"])

    def test_a_non_square_frame_stays_almost_entirely_adjacent(self):
        # Clipping a square curve to a non-square frame necessarily creates a few hops where
        # the curve leaves and re-enters. This bounds them instead of ignoring them.
        s = traversal_stats(48, 64)
        fraction = s["adjacent_steps"] / s["total_steps"]
        self.assertGreater(fraction, 0.99, f"only {fraction:.3f} of steps are adjacent")

    def test_a_shuffled_traversal_would_fail_the_adjacency_check(self):
        # Negative control for the two tests above. If a random ordering also passed, the
        # adjacency check would be measuring nothing.
        rng = np.random.default_rng(11)
        pts = hilbert_points(16, 16)
        shuffled = [pts[i] for i in rng.permutation(len(pts))]
        steps = [abs(shuffled[i][0] - shuffled[i - 1][0]) + abs(shuffled[i][1] - shuffled[i - 1][1])
                 for i in range(1, len(shuffled))]
        self.assertGreater(max(steps), 1, "a shuffled order should contain jumps")
        self.assertLess(sum(1 for s in steps if s == 1) / len(steps), 0.5)

    def test_a_bad_shape_is_refused(self):
        for shape in ((0, 4), (4, 0), (-1, 4)):
            with self.assertRaises(ValueError):
                hilbert_points(*shape)


class TestQueueWeights(unittest.TestCase):
    def test_weights_are_normalised_so_luminance_survives(self):
        for size in (1, 2, 8, 16, 32):
            self.assertAlmostEqual(float(queue_weights(size).sum()), 1.0, places=12)

    def test_the_decay_is_geometric_with_the_requested_ratio(self):
        # Riemersma's parameter is the ratio between the newest and oldest weight. A uniform
        # average over the queue is a box blur of the error and loses the locality that makes
        # the output usable, so the shape of the decay is the thing to check.
        for ratio in (1 / 4, 1 / 16, 1 / 64):
            w = queue_weights(16, ratio)
            self.assertAlmostEqual(w[0] / w[-1], 1 / ratio, places=6)
            self.assertTrue(np.all(np.diff(w) < 0), "weights must decrease monotonically")

    def test_a_ratio_of_one_gives_uniform_weights(self):
        w = queue_weights(8, 1.0)
        self.assertTrue(np.allclose(w, 1 / 8), "ratio 1 should degenerate to a flat average")

    def test_bad_parameters_are_refused(self):
        for bad in (lambda: queue_weights(0), lambda: queue_weights(8, 0.0),
                    lambda: queue_weights(8, 1.5)):
            with self.assertRaises(ValueError):
                bad()


class TestRiemersmaOutput(unittest.TestCase):
    def test_output_is_strictly_one_bit(self):
        out = riemersma(radial_gradient(64, 64))
        self.assertTrue(set(np.unique(out).tolist()) <= {0.0, 1.0})

    def test_luminance_is_preserved_closely(self):
        # The normalised queue means the error is fully redistributed, so this should be tight.
        # Measured within 0.0005 on flat fields, which is much closer than the 0.03 the kernel
        # methods need.
        for level in (0.1, 0.25, 0.5, 0.75, 0.9):
            got = float(riemersma(flat(level, 64, 64)).mean())
            self.assertAlmostEqual(got, level, delta=0.002,
                                   msg=f"level {level} came out at {got:.4f}")

    def test_coverage_is_monotonic_in_level(self):
        levels = [coverage(riemersma(flat(v, 64, 64))) for v in (0.1, 0.3, 0.5, 0.7, 0.9)]
        self.assertEqual(levels, sorted(levels), f"not monotonic: {levels}")

    def test_it_is_deterministic(self):
        img = radial_gradient(48, 48)
        self.assertTrue(np.array_equal(riemersma(img), riemersma(img)))

    def test_it_differs_from_the_kernel_methods(self):
        # If these matched, riemersma would be an alias and the whole module decorative.
        img = step_wedge(64, 64, steps=8)
        self.assertFalse(np.array_equal(riemersma(img), floyd_steinberg(img)))

    def test_the_queue_ratio_actually_changes_the_output(self):
        img = radial_gradient(64, 64)
        self.assertFalse(np.array_equal(riemersma(img, ratio=1 / 64), riemersma(img, ratio=1 / 2)),
                         "the ratio had no effect, so the weights are being ignored")

    def test_bad_parameters_are_refused(self):
        with self.assertRaises(ValueError):
            riemersma(radial_gradient(8, 8), levels=1)
        with self.assertRaises(ValueError):
            riemersma(np.zeros((4, 4, 3)))


class TestTheDegenerateCase(unittest.TestCase):
    """The one measured advantage, and the claim that failed."""

    def test_floyd_steinberg_collapses_to_a_checkerboard_on_flat_midgrey(self):
        # Establishes the baseline for the next test, and is a real property worth pinning:
        # on flat 0.5 every horizontal and vertical neighbour differs and every diagonal
        # neighbour matches, exactly. Serpentine scanning does not change it.
        for serp in (True, False):
            b = diffuse(flat(0.5, 96, 96), kernel="floyd-steinberg", serpentine=serp).image > 0.5
            h, v = axis_disagreement(b)
            down, up = diagonal_disagreement(b)
            self.assertAlmostEqual(h, 1.0, places=6, msg=f"serpentine={serp}")
            self.assertAlmostEqual(v, 1.0, places=6, msg=f"serpentine={serp}")
            self.assertAlmostEqual(down, 0.0, places=6, msg=f"serpentine={serp}")
            self.assertAlmostEqual(up, 0.0, places=6, msg=f"serpentine={serp}")

    def test_riemersma_does_not_collapse_on_flat_midgrey(self):
        b = riemersma(flat(0.5, 96, 96)) > 0.5
        h, v = axis_disagreement(b)
        self.assertLess(h, 0.9, f"horizontal disagreement {h:.3f} is checkerboard-like")
        self.assertLess(v, 0.9, f"vertical disagreement {v:.3f} is checkerboard-like")
        down, up = diagonal_disagreement(b)
        self.assertGreater(down, 0.05, "a perfectly matching diagonal means a checkerboard")

    def test_riemersma_is_not_isotropic_which_is_the_opposite_of_the_usual_claim(self):
        # Documented here because it is a negative result, and leaving it out of the tests
        # would let the claim creep back into the README later. Riemersma is usually sold on
        # the Hilbert curve having no preferred direction. At level 0.25 this implementation
        # disagrees across one diagonal about twice as often as the other, which is stronger
        # directional structure than Floyd-Steinberg shows at the same level.
        r_down, r_up = diagonal_disagreement(riemersma(flat(0.25, 128, 128)) > 0.5)
        f_down, f_up = diagonal_disagreement(floyd_steinberg(flat(0.25, 128, 128)) > 0.5)
        self.assertGreater(abs(r_down - r_up), 0.1,
                           f"riemersma diagonals {r_down:.3f}/{r_up:.3f} look isotropic here, "
                           "so the README's negative result needs re-measuring")
        self.assertGreater(abs(r_down - r_up), abs(f_down - f_up),
                           "riemersma should show MORE diagonal structure, per the measurement")


if __name__ == "__main__":
    unittest.main()
