"""Minimal, dependency-free PNG reader and writer.

Only what this toolkit needs, but correct for that subset:

writer
    1-bit grayscale, 8-bit grayscale, 8-bit RGB. Non-interlaced, filter type 0
    (None) on every scanline. Deterministic output: zlib level and strategy are
    pinned so the same array always produces the same bytes.

reader
    Non-interlaced PNG with bit depth 1, 2, 4, 8 or 16 and color type 0 (gray),
    2 (RGB), 3 (palette), 4 (gray+alpha) or 6 (RGBA). All five scanline filters
    are implemented. 16-bit samples are reduced to 8 bits. Alpha is composited
    over white so a read always yields gray or RGB uint8.

The point of writing this by hand is that the test suite can assert the actual
IHDR bit depth of a file we claim is 1-bit, without trusting an image library
to have kept it that way.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def write_png(path, array, bitdepth: int = 8) -> None:
    """Write ``array`` to ``path``.

    array
        (h, w) uint8 for grayscale, or (h, w, 3) uint8 for RGB.
    bitdepth
        8, or 1 for grayscale arrays whose only values are 0 and 255.
    """
    a = np.asarray(array)
    if a.dtype != np.uint8:
        raise ValueError("write_png needs a uint8 array, got %s" % a.dtype)
    if a.ndim == 2:
        color_type = 0
        channels = 1
    elif a.ndim == 3 and a.shape[2] == 3:
        color_type = 2
        channels = 3
    else:
        raise ValueError("write_png needs (h,w) or (h,w,3), got %r" % (a.shape,))

    h, w = a.shape[0], a.shape[1]

    if bitdepth == 1:
        if color_type != 0:
            raise ValueError("1-bit output is grayscale only")
        vals = np.unique(a)
        if not np.isin(vals, (0, 255)).all():
            raise ValueError(
                "1-bit output needs values in {0,255}, saw %s" % (vals[:8],)
            )
        bits = (a > 0).astype(np.uint8)
        packed = np.packbits(bits, axis=1)
        raw = b"".join(b"\x00" + packed[y].tobytes() for y in range(h))
    elif bitdepth == 8:
        raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))
    else:
        raise ValueError("bitdepth must be 1 or 8")

    ihdr = struct.pack(">IIBBBBB", w, h, bitdepth, color_type, 0, 0, 0)
    comp = zlib.compressobj(level=9, wbits=15, strategy=zlib.Z_DEFAULT_STRATEGY)
    idat = comp.compress(raw) + comp.flush()
    blob = (
        PNG_MAGIC
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )
    # Accept an already-open binary stream as well as a path, so the gallery can encode
    # straight to memory for its data URIs without a temporary file on disk.
    if hasattr(path, "write"):
        path.write(blob)
    else:
        with open(path, "wb") as fh:
            fh.write(blob)
    del channels


def read_header(path) -> dict:
    """Return the IHDR fields of a PNG without decoding pixels."""
    with open(path, "rb") as fh:
        magic = fh.read(8)
        if magic != PNG_MAGIC:
            raise ValueError("not a PNG: %s" % path)
        length = struct.unpack(">I", fh.read(4))[0]
        tag = fh.read(4)
        if tag != b"IHDR" or length != 13:
            raise ValueError("first chunk is not a 13-byte IHDR")
        w, h, depth, ctype, comp, filt, interlace = struct.unpack(">IIBBBBB", fh.read(13))
    return {
        "width": w,
        "height": h,
        "bitdepth": depth,
        "color_type": ctype,
        "compression": comp,
        "filter": filt,
        "interlace": interlace,
    }


_BPP_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def _unfilter(raw: bytes, h: int, stride: int, bpp: int) -> np.ndarray:
    out = np.zeros((h, stride), dtype=np.uint8)
    pos = 0
    prev = np.zeros(stride, dtype=np.int32)
    for y in range(h):
        ftype = raw[pos]
        pos += 1
        line = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=pos).astype(np.int32)
        pos += stride
        cur = line.copy()
        if ftype == 0:
            pass
        elif ftype == 1:
            for i in range(bpp, stride):
                cur[i] = (cur[i] + cur[i - bpp]) & 0xFF
        elif ftype == 2:
            cur = (cur + prev) & 0xFF
        elif ftype == 3:
            for i in range(stride):
                left = cur[i - bpp] if i >= bpp else 0
                cur[i] = (cur[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:
            for i in range(stride):
                a = cur[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                if pa <= pb and pa <= pc:
                    pred = a
                elif pb <= pc:
                    pred = b
                else:
                    pred = c
                cur[i] = (cur[i] + pred) & 0xFF
        else:
            raise ValueError("unknown PNG filter type %d" % ftype)
        out[y] = cur.astype(np.uint8)
        prev = cur
    return out


def read_png(path) -> np.ndarray:
    """Read a PNG into a uint8 array, (h,w) gray or (h,w,3) RGB."""
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:8] != PNG_MAGIC:
        raise ValueError("not a PNG: %s" % path)
    pos = 8
    idat = []
    palette = None
    hdr = None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            hdr = struct.unpack(">IIBBBBB", payload)
        elif tag == b"PLTE":
            palette = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3)
        elif tag == b"IDAT":
            idat.append(payload)
        elif tag == b"IEND":
            break
    if hdr is None:
        raise ValueError("no IHDR")
    w, h, depth, ctype, comp, filt, interlace = hdr
    if interlace:
        raise ValueError("interlaced PNG is not supported")
    if ctype not in _BPP_CHANNELS:
        raise ValueError("unsupported color type %d" % ctype)
    # Reject an unsupported bit depth here, by name. Without this the file still fails, but
    # much later and for the wrong reason: a 16-bit image has a stride twice what the decoder
    # expects, so the row loop reads a sample byte where a filter byte should be and reports
    # "unknown PNG filter type 128". That is a true statement about the byte it happened to
    # read and a useless one about the actual problem.
    if depth not in (1, 2, 4, 8):
        raise ValueError("unsupported bit depth %d, this decoder handles 1, 2, 4 and 8" % depth)
    nch = _BPP_CHANNELS[ctype]
    raw = zlib.decompress(b"".join(idat))
    bits_per_pixel = nch * depth
    stride = (w * bits_per_pixel + 7) // 8
    bpp = max(1, bits_per_pixel // 8)
    rows = _unfilter(raw, h, stride, bpp)

    if depth == 8:
        samples = rows[:, : w * nch].reshape(h, w, nch)
    elif depth == 16:
        wide = rows[:, : w * nch * 2].reshape(h, w, nch, 2)
        samples = wide[..., 0]  # high byte, a fair 16->8 reduction
    else:
        unpacked = np.unpackbits(rows, axis=1)
        per = depth
        vals = np.zeros((h, stride * 8 // per), dtype=np.uint8)
        for b in range(per):
            vals |= (unpacked[:, b::per] << (per - 1 - b)).astype(np.uint8)
        vals = vals[:, : w * nch]
        if ctype == 3:
            samples = vals.reshape(h, w, nch)
        else:
            scale = 255 // ((1 << per) - 1)
            samples = (vals * scale).reshape(h, w, nch)

    if ctype == 3:
        if palette is None:
            raise ValueError("palette image with no PLTE")
        return palette[samples[:, :, 0]]
    if ctype == 0:
        return samples[:, :, 0].copy()
    if ctype == 2:
        return samples.copy()
    if ctype == 4:
        gray = samples[:, :, 0].astype(np.float64)
        alpha = samples[:, :, 1].astype(np.float64) / 255.0
        return np.clip(gray * alpha + 255.0 * (1 - alpha), 0, 255).astype(np.uint8)
    rgb = samples[:, :, :3].astype(np.float64)
    alpha = samples[:, :, 3].astype(np.float64)[:, :, None] / 255.0
    return np.clip(rgb * alpha + 255.0 * (1 - alpha), 0, 255).astype(np.uint8)
