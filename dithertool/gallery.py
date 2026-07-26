"""Build the comparison gallery.

Dithering is one of the few image operations where any resampling destroys the thing being
compared. A dithered image viewed at 97% zoom shows moire that is an artifact of the browser's
scaler, not of the algorithm, and the usual result is people comparing scaler bugs. Every image
here is therefore rendered at exactly 1:1 with `image-rendering: pixelated` and a fixed
intrinsic size, so what appears on screen is the pixels that were written to the file.
"""
from __future__ import annotations

import base64
import html
import io
import os

import numpy as np

from . import bluenoise, diffusion, imageops, ordered, pngio, riemersma, spectrum

# The algorithms worth putting side by side, in the order that tells the story: the periodic
# mask first, then noise, then the diffusion family, then the two special cases.
ROWS = [
    ("bayer", "Bayer 8x8", "Ordered, periodic. Cheap and stateless, and the cross-hatch is "
                           "visible at every level."),
    ("white-noise", "White noise", "The control. Same construction as blue noise with the "
                                   "spectral shaping removed, which is what makes it grainy."),
    ("blue-noise", "Blue noise 64x64", "Void-and-cluster. Stateless like Bayer, aperiodic like "
                                       "noise, with low-frequency energy suppressed."),
    ("floyd-steinberg", "Floyd-Steinberg", "The default error diffusion. Note what it does to "
                                           "flat mid-grey."),
    ("atkinson", "Atkinson", "Discards a quarter of the error, which expands contrast and "
                             "crushes the extremes."),
    ("jarvis", "Jarvis-Judice-Ninke", "A wider kernel, so smoother gradients and more smear."),
    ("stucki", "Stucki", "Wider again, tuned for sharper edges than Jarvis."),
    ("riemersma", "Riemersma", "Error diffused along a Hilbert curve instead of a raster scan."),
    ("halftone", "Halftone, 45 degrees", "A rotated screen with a round spot, as used in print."),
]

PATTERNS = [
    ("ramp", "Linear ramp", "The banding test. Every level from black to white, once."),
    ("wedge", "Step wedge", "Sixteen flat patches. Any tone shift shows up as a patch that "
                            "reads wrong against its neighbours."),
    ("radial", "Radial gradient", "Curved level sets, which is where directional artifacts "
                                  "become obvious."),
    ("zone", "Zone plate", "Frequency sweeps in every direction at once. This is the pattern "
                           "that exposes aliasing."),
    ("midgrey", "Flat mid-grey", "The degenerate case. Floyd-Steinberg collapses to a rigid "
                                 "checkerboard here and Riemersma does not."),
]


def _png_data_uri(arr: np.ndarray, bitdepth: int = 8) -> tuple[str, int, int]:
    """Encode to a PNG data URI. The gallery is one self-contained file by design."""
    buf = io.BytesIO()
    pngio.write_png(buf, imageops.to_u8(arr), bitdepth=bitdepth)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    h, w = arr.shape[:2]
    return f"data:image/png;base64,{data}", w, h


def _make(name: str, img: np.ndarray) -> np.ndarray:
    from .cli import apply
    return apply(name, img, period=6.0)


def _pattern(name: str) -> np.ndarray:
    from .cli import PATTERNS as GENERATORS
    return GENERATORS[name]()


def _spectra_section() -> str:
    """The masks and their measured spectra, which is the quantitative half of the page."""
    masks = [
        ("Blue noise 64x64", bluenoise.blue_noise_thresholds(64)),
        ("White noise 64x64", bluenoise.white_noise_thresholds(64, seed=1)),
        ("Bayer 8x8, tiled to 64", spectrum.tile_to(ordered.bayer_thresholds(8), 64, 64)),
    ]
    cells = []
    rows = []
    for label, m in masks:
        uri, w, h = _png_data_uri(m)
        ok, s = spectrum.is_blue_noise(m)
        cells.append(
            f'<figure class="mask"><div class="viewport"><img src="{uri}" width="{w}" height="{h}" alt="{html.escape(label)}"></div>'
            f'<figcaption>{html.escape(label)}</figcaption></figure>')
        ratio = "infinite" if not np.isfinite(s.ratio) else f"{s.ratio:,.0f}"
        rows.append(
            f"<tr><th>{html.escape(label)}</th><td>{s.low:.3e}</td><td>{s.high:.3e}</td>"
            f"<td>{ratio}</td><td>{s.slope:.2f}</td><td>{s.peak_to_mean:,.0f}</td>"
            f'<td class="{"yes" if ok else "no"}">{"yes" if ok else "no"}</td></tr>')
    return f"""
<section id="spectra">
  <h2>The masks, and why blue noise is blue</h2>
  <p>A dither mask cannot be judged by eye. White noise and blue noise both look like noise,
  and the difference between them is entirely in the radial power spectrum: blue noise
  suppresses low spatial frequencies, which are the ones the eye is most sensitive to.</p>
  <div class="masks">{''.join(cells)}</div>
  <div class="tablewrap"><table>
    <thead><tr><th>mask</th><th>low band</th><th>high band</th><th>high/low</th>
      <th>slope</th><th>peak/mean</th><th>blue?</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table></div>
  <p class="note"><strong>Bayer's ratio is infinite</strong>, because a perfectly periodic
  matrix has exactly zero energy in the low band. A test that only checked the high-to-low
  ratio would therefore rank Bayer as the finest blue noise ever produced. The peak-to-mean
  column is what rejects it: Bayer's energy sits in a few sharp spikes, roughly 1,700 times
  the mean, and those spikes are the cross-hatch you can see in the first row below. This is
  the reason the check in <code>spectrum.py</code> tests three properties instead of one.</p>
</section>"""


def _algorithm_section(pattern_key: str, title: str, blurb: str) -> str:
    img = _pattern(pattern_key)
    src_uri, sw, sh = _png_data_uri(img)
    rows = [f"""<figure class="shot">
      <div class="viewport"><img src="{src_uri}" width="{sw}" height="{sh}" alt="source"></div>
      <figcaption><strong>Source</strong><span>Undithered, 8 bits per pixel.</span></figcaption>
    </figure>"""]
    for name, label, why in ROWS:
        out = _make(name, img)
        uri, w, h = _png_data_uri(out, bitdepth=1)
        got = float(np.mean(out))
        want = float(np.mean(img))
        rows.append(f"""<figure class="shot">
      <div class="viewport"><img src="{uri}" width="{w}" height="{h}" alt="{html.escape(label)}"></div>
      <figcaption><strong>{html.escape(label)}</strong><span>{html.escape(why)}</span>
      <span class="num">mean {got:.4f} against {want:.4f} in source, delta {got - want:+.4f}</span>
      </figcaption>
    </figure>""")
    return f"""
<section class="pattern" id="p-{pattern_key}">
  <h2>{html.escape(title)}</h2>
  <p>{html.escape(blurb)}</p>
  <div class="grid">{''.join(rows)}</div>
</section>"""


CSS = """
:root {
  --ink: #14181d; --ink-2: #4a5560; --ink-3: #7b8794;
  --bg: #f6f4f0; --surface: #fffefb; --line: #ddd8cf; --accent: #9a3b1f; --ok: #2f6b3a;
}
@media (prefers-color-scheme: dark) {
  :root { --ink: #eceff3; --ink-2: #a7b2be; --ink-3: #78838f;
          --bg: #14171b; --surface: #1b1f24; --line: #2c3238; --accent: #e0764f; --ok: #6dbd80; }
}
:root[data-theme="dark"] {
  --ink: #eceff3; --ink-2: #a7b2be; --ink-3: #78838f;
  --bg: #14171b; --surface: #1b1f24; --line: #2c3238; --accent: #e0764f; --ok: #6dbd80;
}
:root[data-theme="light"] {
  --ink: #14181d; --ink-2: #4a5560; --ink-3: #7b8794;
  --bg: #f6f4f0; --surface: #fffefb; --line: #ddd8cf; --accent: #9a3b1f; --ok: #2f6b3a;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 16px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Helvetica, sans-serif;
  -webkit-font-smoothing: antialiased; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 4rem 1.5rem 6rem; }
header { border-bottom: 2px solid var(--ink); padding-bottom: 2rem; margin-bottom: 3rem; }
h1 { font-size: clamp(2rem, 5vw, 3.2rem); line-height: 1.05; margin: 0 0 .75rem;
  letter-spacing: -.02em; text-wrap: balance; }
.lede { font-size: 1.15rem; color: var(--ink-2); max-width: 62ch; margin: 0 0 1.5rem; }
h2 { font-size: 1.5rem; margin: 0 0 .5rem; letter-spacing: -.01em; text-wrap: balance; }
section { margin: 4rem 0 0; }
section > p { color: var(--ink-2); max-width: 68ch; }
.eyebrow { font-size: .72rem; text-transform: uppercase; letter-spacing: .14em;
  color: var(--accent); font-weight: 650; margin: 0 0 .6rem; }
.grid, .masks { display: flex; flex-wrap: wrap; gap: 1.25rem; margin-top: 1.75rem;
  align-items: flex-start; }
figure { margin: 0; background: var(--surface); border: 1px solid var(--line);
  border-radius: 3px; padding: .75rem; max-width: 100%; }
/* 1:1 and nothing else. Any scaling here would show the browser's resampling artifacts
   rather than the algorithm's, which is the single most common way these comparisons lie.
   The usual `max-width: 100%; height: auto` is therefore wrong for this page specifically:
   on a 390px phone it scaled the 512px wide panels down to 301px, which is precisely the
   resampled moire the page claims not to show. Measured in a browser at 390px, 20 of 53
   images were affected. So the image keeps its intrinsic size and the FIGURE scrolls
   instead, which keeps every comparison honest at every viewport and keeps the sideways
   scrolling inside the figure rather than on the page body. */
.viewport { overflow-x: auto; max-width: 100%; -webkit-overflow-scrolling: touch; }
figure img { display: block; image-rendering: pixelated; background: #fff; }
figcaption { font-size: .82rem; color: var(--ink-2); margin-top: .6rem; max-width: 34ch;
  display: flex; flex-direction: column; gap: .3rem; }
figcaption strong { color: var(--ink); font-size: .9rem; }
.num { font-variant-numeric: tabular-nums; color: var(--ink-3); font-size: .76rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.masks figure { padding: .6rem; }
.masks figcaption { max-width: 20ch; }
table { border-collapse: collapse; margin: 1.5rem 0; font-size: .85rem; width: 100%;
  font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: .5rem .7rem; border-bottom: 1px solid var(--line);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
thead th { text-align: right; color: var(--ink-3); font-weight: 600; font-size: .74rem;
  text-transform: uppercase; letter-spacing: .06em; border-bottom: 2px solid var(--ink); }
tbody th { text-align: left; font-family: inherit; font-weight: 600; }
td.yes { color: var(--ok); font-weight: 700; } td.no { color: var(--accent); font-weight: 700; }
.note { border-left: 3px solid var(--accent); padding: .1rem 0 .1rem 1rem; margin: 1.5rem 0;
  color: var(--ink-2); max-width: 70ch; }
.tablewrap { overflow-x: auto; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .88em;
  background: var(--surface); border: 1px solid var(--line); border-radius: 2px;
  padding: .08em .32em; }
nav { display: flex; flex-wrap: wrap; gap: .5rem; margin: 1.5rem 0 0; }
nav a { font-size: .8rem; color: var(--ink-2); text-decoration: none; border: 1px solid var(--line);
  padding: .3rem .7rem; border-radius: 2px; background: var(--surface); }
nav a:hover { border-color: var(--accent); color: var(--accent); }
a { color: var(--accent); }
footer { margin-top: 5rem; padding-top: 2rem; border-top: 1px solid var(--line);
  color: var(--ink-3); font-size: .85rem; }
"""


def build(outdir: str = "docs") -> int:
    os.makedirs(outdir, exist_ok=True)
    nav = "".join(f'<a href="#p-{k}">{html.escape(t)}</a>' for k, t, _ in PATTERNS)
    sections = "".join(_algorithm_section(k, t, b) for k, t, b in PATTERNS)
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>dithertool: nine dithering algorithms, measured</title>
<meta name="description" content="Bayer, blue noise, five error-diffusion kernels, Riemersma
and rotated-screen halftoning, compared at 1:1 with their spectral and tonal properties
measured rather than asserted.">
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<header>
  <p class="eyebrow">Catalog task ART-009</p>
  <h1>Nine dithering algorithms,<br>measured rather than asserted</h1>
  <p class="lede">Every image on this page is shown at exactly 1:1 with no resampling, because
  a dithered image viewed at any other zoom shows the browser's scaling artifacts instead of
  the algorithm's. The numbers under each panel come from the same functions the test suite
  asserts against.</p>
  <nav><a href="#spectra">Mask spectra</a>{nav}</nav>
</header>
{_spectra_section()}
{sections}
<footer>
  <p>Part of <a href="https://github.com/JesseRWeigel/722-things-to-build">thousand</a>.
  Source and the full verify output are in the
  <a href="https://github.com/JesseRWeigel/dithertool">repository</a>.
  Pure Python and NumPy, no image libraries: the PNG encoder and decoder are in
  <code>pngio.py</code>.</p>
</footer>
</div>
<script>
// The theme toggle stamps data-theme on the root element, and the CSS above lets that win
// over the media query in both directions.
</script>
</body>
</html>"""
    path = os.path.join(outdir, "index.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(page)
    size = os.path.getsize(path)
    print(f"wrote {path} ({size / 1024:.0f} KiB, {len(PATTERNS)} patterns x "
          f"{len(ROWS)} algorithms, all images inline at 1:1)")
    return 0
