# dithertool

Nine dithering algorithms in pure Python and NumPy, with the properties that are usually
claimed measured instead.

Bayer, void-and-cluster blue noise, five error-diffusion kernels, Riemersma along a Hilbert
curve, and rotated-screen halftoning. No image libraries: the PNG encoder and decoder are
about 150 lines in `pngio.py`.

**[See the comparison gallery](https://jesserweigel.github.io/dithertool/)**

Catalog task `ART-009`. Part of [722 things to build](https://github.com/JesseRWeigel/722-things-to-build).

## The point

Dithering implementations are easy to get subtly wrong and hard to check by eye, which is why
the same handful of bugs keeps shipping. A transposed Bayer matrix still looks like Bayer. A
"blue noise" mask made by shuffling a permutation still looks like noise. An error-diffusion
kernel that drops the error at the image border still produces a plausible picture, slightly
too light. In every case the output looks fine and the numbers do not.

So each claim here has a measurement attached, and the measurement is in the test suite:

```
$ python3 -m dithertool.cli measure

mask spectra (a dither mask is judged by its radial power spectrum)
  mask                          low       high     ratio   slope  peak/mean  blue?
  blue-noise 64           4.428e-06  7.214e-02  16290.48    1.17       3.59  yes
  blue-noise 32           9.623e-06  8.049e-02   8365.02    0.98       3.81  yes
  white-noise 64          8.341e-02  8.880e-02      1.06    0.33       3.45  no
  bayer 8 (tiled)         0.000e+00  1.239e-02       inf    3.28    1673.27  no
```

Blue noise suppresses low spatial frequencies by a factor of 16,000. White noise, built the
same way with the spectral shaping removed, comes in at 1.06. That gap is the whole difference
between the two, and it is invisible in a side-by-side screenshot.

## Three things worth reading

### Bayer has an infinite ratio, and a ratio test would call it perfect blue noise

A perfectly periodic matrix has exactly zero energy in the low band, so its high-to-low ratio
is not merely large, it is infinite. A blue-noise check written the obvious way, "the ratio
must be high", ranks Bayer above every real blue-noise mask ever generated.

What rejects it is the peak-to-mean column. Bayer's energy sits in a few sharp spectral spikes
at roughly 1,700 times the mean, and those spikes are the cross-hatch you can see in the
gallery. `is_blue_noise` therefore tests three properties, high-to-low ratio, spectral slope,
and peak-to-mean, and Bayer fails on the third while passing the first two spectacularly.

### Riemersma is not isotropic, which is the opposite of the usual claim

The standard pitch for Riemersma dithering is that a Hilbert curve has no preferred direction,
so the artifacts come out isotropic. Measured on this implementation that is false. At input
level 0.25 the output disagrees across one diagonal about twice as often as the other, 0.404
against 0.201, which is *stronger* directional structure than Floyd-Steinberg shows at the same
level, 0.487 against 0.498. The Hilbert curve has its own quadrant geometry and the error
memory follows it.

This is recorded as a passing test, `test_riemersma_is_not_isotropic_which_is_the_opposite_of_the_usual_claim`,
so that if someone later "fixes" the code to be isotropic the suite fails and forces a
re-measurement rather than an assumption.

What Riemersma does demonstrably do is avoid the degenerate case. On a flat mid-grey,
Floyd-Steinberg collapses into a rigid checkerboard: every horizontal and vertical neighbour
differs and every diagonal neighbour matches, exactly, at every pixel. Serpentine scanning does
not change it. Riemersma produces an aperiodic texture at the same input. Both facts are
asserted to six decimal places.

### Atkinson is supposed to fail the luminance test

Atkinson distributes six neighbours at 1/8 each, summing to 6/8, deliberately discarding a
quarter of the error to gain contrast. An implementation "corrected" so the weights sum to 1
is Floyd-Steinberg with an unusual stencil.

The consequence is measurable and antisymmetric:

```
tone response on flat fields (output mean against requested level)
   level            bayer       blue-noise        riemersma  floyd-steinberg         atkinson
    0.10     0.0938-0.006     0.1008+0.000     0.0998-0.000     0.1000+0.000     0.0035-0.096
    0.25     0.2500+0.000     0.2495-0.000     0.2499-0.000     0.2500+0.000     0.1751-0.074
    0.50     0.5000+0.000     0.4991-0.000     0.5000+0.000     0.5000+0.000     0.5000+0.000
    0.75     0.7500+0.000     0.7490-0.001     0.7501+0.000     0.7500+0.000     0.8249+0.074
    0.90     0.9062+0.006     0.8994-0.000     0.9002+0.000     0.9000-0.000     0.9965+0.096
```

Level 0.1 renders at 0.003 and level 0.9 at 0.997, mirrored about mid-grey to six decimal
places. An earlier version of the test asserted Atkinson is "lighter in the midtones" and
failed, because at exactly 0.5 the discarded error is symmetric and both algorithms average
0.5000. The property is contrast expansion, not lightening, and the test now says so.

Every other kernel is held to an exact conservation identity instead:

```
sum(in) - sum(out) - (1 - T) * sum(error) - edge_leak == 0
```

where `T` is the fraction the kernel intends to distribute. Worst residual across six kernels
and both edge modes is 9.59e-14, which is floating point exact. Any error dropped at a border,
double-counted, or lost to a clamp shows up here.

## Using it

```bash
python3 -m dithertool.cli dither in.png -o out.png -a blue-noise
python3 -m dithertool.cli dither --pattern zone -a atkinson -o zone.png
python3 -m dithertool.cli dither --pattern wedge -a halftone --period 6 --angle 15 -o h.png
python3 -m dithertool.cli measure          # every number in this README
python3 -m dithertool.cli gallery          # rebuild docs/index.html
```

Algorithms: `bayer`, `blue-noise`, `white-noise`, `riemersma`, `halftone`,
`floyd-steinberg`, `atkinson`, `jarvis`, `stucki`, `burkes`, `sierra3`.

Test patterns, for when you have no input handy: `ramp`, `wedge`, `radial`, `zone`,
`midgrey`, `lines`.

Only NumPy is required. Exit `0` on success, `2` on bad usage.

Installing it puts a `dithertool` command on the path, so `dithertool measure` works in place
of the `python3 -m` form:

```bash
pip install -e .
dithertool dither --pattern radial -a riemersma -o r.png
```

## The gallery renders at 1:1, and that took a second pass

A dithered image viewed at any zoom other than 100% shows moire from the browser's scaler
rather than from the algorithm, so a gallery that scales its images is comparing scaler bugs.
Every panel is emitted with explicit `width` and `height` and `image-rendering: pixelated`.

The first version still got this wrong. It used the ordinary responsive idiom,
`max-width: 100%; height: auto`, which on a 390px viewport scaled the 512px wide panels down to
301px. Checked in a real browser, 20 of 53 images were being resampled, and the page body
scrolled sideways as well. Images now keep their intrinsic size and each figure is its own
horizontal scroll container, so 1:1 holds at every viewport. Verify check 7 asserts every image
sits inside a scroll container and that no `max-width` has crept back onto `figure img`.

## Status

Verified 2026-07-26.

```
$ bash scripts/verify.sh
1. unit suite
  ok    Ran 49 tests passed

2. the CLI runs end to end and writes a real file
  ok    wrote a 3451 byte PNG
  decoded 128x512, bitdepth 1, 2 distinct values
  ok    the 1-bit claim holds on disk, not just in memory

3. blue noise is spectrally blue, and white noise is not
  blue  : ratio    8365.02 slope  0.98 peak/mean     3.81 -> True
  white : ratio       0.79 slope -0.59 peak/mean     3.04 -> False
  bayer : ratio        inf slope  3.28 peak/mean  1673.27 -> False
  ok    blue passes, white is rejected, and Bayer is rejected despite an infinite ratio

4. error accounting balances exactly for every kernel
  worst residual across 6 kernels x 2 edge modes: 9.59e-14
  ok    no kernel loses or invents error

5. the Hilbert curve really is a Hilbert curve
  64x64: 4096/4096 pixels, 4096 unique, longest step 1
  32x32: 1024/1024 pixels, 1024 unique, longest step 1
  16x16: 256/256 pixels, 256 unique, longest step 1
  ok    every pixel visited once, every step adjacent

6. the measure subcommand produces the documented numbers
  ok    measure printed 30 lines of measurements

7. the gallery builds and every image is inline at 1:1
  ok    53 inline images, all sized, all in scroll containers, no remote assets

8. no secrets and no personal paths in anything tracked
  ok    no credential-shaped or personal strings found

9 passed, 0 failed
VERIFY OK
```

### The verify script was attacked, and caught all three

A verify command that passes on broken code is worse than none, so it was tested by breaking
the implementation on purpose and confirming it noticed.

| Sabotage | Result |
|---|---|
| `void_and_cluster_ranks` replaced with a shuffled permutation | Check 3 failed, ratio fell from 8365 to 0.93, plus 1 unit failure. Exit 1 |
| The Hilbert traversal replaced with raster order | Check 5 failed, longest step 64 instead of 1, plus 3 unit failures. Exit 1 |
| `edge_leak` no longer recorded in `diffuse` | Check 4 failed, worst residual 6.49e-02 instead of 1e-14, plus 1 unit failure. Exit 1 |

Each sabotage was reverted and the suite returns to 9 passed, 0 failed.

The suite also carries its own negative controls, because a check that cannot fail proves
nothing: white noise must be *rejected* by the blue-noise test, a shuffled traversal must fail
the adjacency test, and a kernel whose weights sum above 1 must fail the luminance test.

## Limitations

- **Single channel only.** Every algorithm operates on one channel. Colour input is converted
  to Rec.709 luminance in linear light. There is no per-channel colour dithering and no
  palette quantisation, so this will not reduce an image to an arbitrary 16-colour palette.
- **Pure Python speed.** Error diffusion and Riemersma are per-pixel Python loops, so they run
  in seconds on a 512x512 image and are not suitable for video. The ordered methods are
  vectorised and fast. Void-and-cluster generation of a 64x64 mask takes a few seconds and is
  cached.
- **PNG only, and a subset of it.** `pngio` writes 1-bit and 8-bit grayscale plus 8-bit RGB,
  and reads non-interlaced files. Interlaced PNGs, 16-bit samples, palette images and colour
  profiles are refused rather than guessed at.
- **The halftone screen is measured on periodicity, not on registration.** Cell period holds
  across angles to within 2 pixels, which is what the test asserts. Multi-channel screen angle
  sets and true moire-free CMYK registration are out of scope.
- **`spectrum_metrics` refuses inputs below roughly 8x8** because there are not enough radial
  bins to form two bands. Small masks like a 4x4 Bayer matrix must be tiled first with
  `tile_to`, which is also how a mask is used in practice.
- **The isotropy result is for this implementation at the levels tested.** Different queue
  sizes and decay ratios may behave differently, and the finding above should not be read as a
  general statement about all Riemersma variants.

## License

MIT.
