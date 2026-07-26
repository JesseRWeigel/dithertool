#!/usr/bin/env bash
# Verification for dithertool.
#
# The design rule here: every check must be capable of failing on a plausible break. The
# obvious wrong version of this script runs the test suite and reports success, which passes
# just as happily when the blue noise generator has been replaced by random(). So the two
# claims that matter most, that the blue noise is spectrally blue and that a 1-bit file really
# is 1 bit on disk, are additionally asserted here at the shell level against the actual
# artifacts, and the discrimination is checked in both directions.
#
# Attacked on 2026-07-26 by replacing void_and_cluster_ranks with a shuffled permutation.
# Check 3 failed with exit 1 and the script exited nonzero. See README.
set -euo pipefail
cd "$(dirname "$0")/.."

pass=0
fail=0
ok()   { printf '  ok    %s\n' "$1"; pass=$((pass + 1)); }
bad()  { printf '  FAIL  %s\n' "$1"; fail=$((fail + 1)); }
have() { command -v "$1" >/dev/null 2>&1; }

PY=python3
$PY -c 'import numpy' 2>/dev/null || { echo "numpy is required"; exit 2; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "1. unit suite"
if out=$($PY -m unittest discover -s tests -t . 2>&1); then
  ok "$(printf '%s' "$out" | grep -oE 'Ran [0-9]+ tests' | head -1) passed"
else
  printf '%s\n' "$out" | tail -30
  bad "unit suite"
fi

echo
echo "2. the CLI runs end to end and writes a real file"
if $PY -m dithertool.cli dither --pattern wedge -a blue-noise -o "$work/wedge.png" 2>"$work/e1"; then
  bytes=$(wc -c < "$work/wedge.png")
  if [ "$bytes" -gt 100 ]; then ok "wrote a $bytes byte PNG"; else bad "PNG is only $bytes bytes"; fi
else
  cat "$work/e1"; bad "the dither subcommand failed"
fi

# A file the tool wrote must be readable by the tool, and by something that is not the tool.
if $PY -c "
from dithertool.pngio import read_png, read_header
import sys
h = read_header('$work/wedge.png')
a = read_png('$work/wedge.png')
assert h['bitdepth'] == 1, 'a 2-level output claimed %r bits on disk' % h['bitdepth']
# read_png decodes to the 8-bit scale, so a 1-bit file must come back as exactly {0, 255}.
# Checking the count as well as the membership catches a file that is uniformly one value,
# which would satisfy the subset test while carrying no image at all.
vals = set(a.ravel().tolist())
assert vals <= {0, 255}, 'decoded values are not binary: %r' % sorted(vals)[:5]
assert len(vals) == 2, 'the decoded image has %d distinct value(s), so it is blank' % len(vals)
print('  decoded %dx%d, bitdepth %d, %d distinct values' % (h['height'], h['width'], h['bitdepth'], len(vals)))
" 2>"$work/e2"; then
  ok "the 1-bit claim holds on disk, not just in memory"
else
  cat "$work/e2"; bad "the written PNG does not match its claim"
fi

echo
echo "3. blue noise is spectrally blue, and white noise is not"
# The core claim of the project. Asserted here as well as in the suite because it is the one
# thing most likely to be quietly broken by a refactor, and because a positive-only check
# would pass on white noise relabelled as blue.
if $PY -c "
from dithertool.bluenoise import blue_noise_thresholds, white_noise_thresholds
from dithertool.spectrum import is_blue_noise, spectrum_metrics, tile_to
from dithertool.ordered import bayer_thresholds
import sys

blue_ok, b = is_blue_noise(blue_noise_thresholds(32))
white_ok, w = is_blue_noise(white_noise_thresholds(32, seed=5))
bayer_ok, y = is_blue_noise(tile_to(bayer_thresholds(8), 64, 64))

print('  blue  : ratio %10.2f slope %5.2f peak/mean %8.2f -> %s' % (b.ratio, b.slope, b.peak_to_mean, blue_ok))
print('  white : ratio %10.2f slope %5.2f peak/mean %8.2f -> %s' % (w.ratio, w.slope, w.peak_to_mean, white_ok))
print('  bayer : ratio %10s slope %5.2f peak/mean %8.2f -> %s' % ('inf' if y.ratio == float('inf') else '%.2f' % y.ratio, y.slope, y.peak_to_mean, bayer_ok))

fails = []
if not blue_ok:  fails.append('the generated blue noise is NOT spectrally blue')
if white_ok:     fails.append('white noise PASSED the blue-noise test, so the test is inert')
if bayer_ok:     fails.append('Bayer passed the blue-noise test; its infinite ratio needs the peak/mean guard')
if b.ratio < 10: fails.append('blue/high-low ratio only %.2f' % b.ratio)
if w.ratio > 3:  fails.append('white noise ratio %.2f is too high to be white' % w.ratio)
if fails:
    for f in fails: print('    ' + f)
    sys.exit(1)
" 2>"$work/e3"; then
  ok "blue passes, white is rejected, and Bayer is rejected despite an infinite ratio"
else
  cat "$work/e3" 2>/dev/null; bad "the blue-noise discrimination is broken"
fi

echo
echo "4. error accounting balances exactly for every kernel"
if $PY -c "
from dithertool.diffusion import KERNELS, diffuse
from dithertool.imageops import step_wedge
import sys
img = step_wedge(96, 64, steps=8)
worst = 0.0; bad_ones = []
for name in KERNELS:
    for edge in ('renormalize', 'discard'):
        r = diffuse(img, kernel=name, edge=edge)
        res = abs(r.bookkeeping_residual())
        worst = max(worst, res)
        if res > 1e-6: bad_ones.append('%s/%s residual %.3e' % (name, edge, res))
print('  worst residual across %d kernels x 2 edge modes: %.2e' % (len(KERNELS), worst))
if bad_ones:
    for b in bad_ones: print('    ' + b)
    sys.exit(1)
" 2>"$work/e4"; then
  ok "no kernel loses or invents error"
else
  cat "$work/e4"; bad "error accounting does not balance"
fi

echo
echo "5. the Hilbert curve really is a Hilbert curve"
if $PY -c "
from dithertool.riemersma import traversal_stats
import sys
fails = []
for shape in ((64, 64), (32, 32), (16, 16)):
    s = traversal_stats(*shape)
    print('  %dx%d: %d/%d pixels, %d unique, longest step %d' % (
        shape[0], shape[1], s['points'], s['expected'], s['unique'], s['max_step']))
    if s['points'] != s['expected']: fails.append('%r does not visit every pixel' % (shape,))
    if s['unique'] != s['expected']: fails.append('%r revisits pixels' % (shape,))
    if s['max_step'] != 1: fails.append('%r has a step of %d, so the curve jumps' % (shape, s['max_step']))
if fails:
    for f in fails: print('    ' + f)
    sys.exit(1)
" 2>"$work/e5"; then
  ok "every pixel visited once, every step adjacent"
else
  cat "$work/e5"; bad "the Hilbert traversal is broken"
fi

echo
echo "6. the measure subcommand produces the documented numbers"
if $PY -m dithertool.cli measure > "$work/measure.txt" 2>&1; then
  if grep -q "blue-noise 64" "$work/measure.txt" && grep -q "hilbert traversal" "$work/measure.txt"; then
    ok "measure printed $(wc -l < "$work/measure.txt") lines of measurements"
  else
    bad "measure output is missing expected sections"
  fi
else
  tail -20 "$work/measure.txt"; bad "measure crashed"
fi

echo
echo "7. the gallery builds and every image is inline at 1:1"
if $PY -m dithertool.cli gallery --outdir "$work/docs" > "$work/g.txt" 2>&1; then
  page="$work/docs/index.html"
  imgs=$(grep -o 'data:image/png;base64,' "$page" | wc -l)
  sized=$(grep -o '<img src="data:image/png;base64,[^"]*" width="[0-9]*" height="[0-9]*"' "$page" | wc -l)
  remote=$(grep -oE 'src="https?://|href="https?://[^"]*\.(css|js)' "$page" | wc -l || true)
  if [ "$imgs" -lt 40 ]; then bad "only $imgs inline images, expected at least 40"
  elif [ "$imgs" -ne "$sized" ]; then bad "$imgs images but only $sized carry explicit width and height, so some will be scaled"
  elif [ "$remote" -ne 0 ]; then bad "$remote remote asset references, the page must be self-contained"
  elif ! grep -q 'image-rendering: pixelated' "$page"; then bad "no pixelated rendering rule, so the browser will smooth the output"
  else
    # Every image must sit inside a .viewport scroll container. Checked because the ordinary
    # responsive idiom, max-width:100% on the image, silently scaled 20 of these panels down
    # on a 390px viewport, which is exactly the resampled output this page claims not to show.
    wrapped=$(grep -o '<div class="viewport"><img src="data:image/png;base64,' "$page" | wc -l)
    if [ "$wrapped" -ne "$imgs" ]; then
      bad "$imgs images but only $wrapped are in a scroll container, so some will be scaled on a narrow screen"
    elif grep -qE 'figure img[^}]*max-width' "$page"; then
      bad "figure img still carries a max-width, which breaks 1:1 on narrow screens"
    else
      ok "$imgs inline images, all sized, all in scroll containers, no remote assets"
    fi
  fi
else
  cat "$work/g.txt"; bad "the gallery failed to build"
fi

echo
echo "8. no secrets and no personal paths in anything tracked"
# AGENTS.md requires this of every project. The pattern list is deliberately broad.
if leaks=$(grep -rIEn 'sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|/home/[a-z]+/(Projects|Documents)|BEGIN [A-Z ]*PRIVATE KEY' \
      --include='*.py' --include='*.md' --include='*.sh' --include='*.html' --include='*.toml' \
      . 2>/dev/null); then
  printf '%s\n' "$leaks" | head -10
  bad "found credential-shaped or personal strings"
else
  ok "no credential-shaped or personal strings found"
fi

echo
printf '%d passed, %d failed\n' "$pass" "$fail"
if [ "$fail" -ne 0 ]; then echo "VERIFY FAILED"; exit 1; fi
echo "VERIFY OK"
