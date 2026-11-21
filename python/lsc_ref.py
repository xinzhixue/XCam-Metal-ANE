"""Golden reference for lens shading correction (LSC) on HDR+ raw frames.

Author: Xue -c
Date: 2026-11-08

Conventions (every Metal / Core ML implementation must match these):
  * Planes are packed in the HDR+ gain-map order [R, Gr, Gb, B]
    (hdrplusdata.org: "[R, Gred, Gblue, B]"; Gr = green in red-containing rows).
    So plane k is always multiplied by map channel k, for any CFA layout.
  * The gain map (Ny x Nx x 4, e.g. 13 x 17) has its corner samples ON the
    corners of the image (Camera2 definition). For a plane of size Hp x Wp,
    plane pixel (y, x) samples the grid at
        gx = x * (Nx - 1) / (Wp - 1),   gy = y * (Ny - 1) / (Hp - 1).
  * Output is normalized linear signal: (raw - black) / (white - black) * gain.
    No clipping, so values > 1 are kept (needed for exact comparison).
"""
import numpy as np
import rawpy
import tifffile

CHANNELS = ("R", "Gr", "Gb", "B")


def load_frame(dng_path, map_path):
    """Return dict with raw (H,W uint16), cfa offsets, levels, wb, gain map."""
    with rawpy.imread(dng_path) as raw:
        img = raw.raw_image_visible.copy()
        pattern = raw.raw_pattern.copy()          # 2x2 of color indices
        desc = raw.color_desc.decode()            # e.g. "RGBG"
        black = np.array(raw.black_level_per_channel, dtype=np.float32)
        white = float(raw.white_level)
        wb = np.array(raw.camera_whitebalance[:3], dtype=np.float32)  # = 1/AsShotNeutral
    gain = tifffile.imread(map_path).astype(np.float32)               # Ny, Nx, 4
    # (row, col) offset inside the 2x2 CFA tile for each plane in CHANNELS order.
    # A green is Gr if its row also holds the red pixel, else Gb.
    names = [[desc[pattern[r, c]] for c in range(2)] for r in range(2)]
    offsets = {}
    for r in range(2):
        for c in range(2):
            name = names[r][c]
            if name in ("R", "B"):
                offsets[name] = (r, c)
            else:
                offsets["Gr" if "R" in names[r] else "Gb"] = (r, c)
    assert len(offsets) == 4, f"unexpected CFA {names}"
    off = [offsets[k] for k in CHANNELS]
    # rawpy indexes black_level_per_channel by CFA color index (the values in raw_pattern).
    blk = np.array([black[pattern[r, c]] for r, c in off], dtype=np.float32)
    return dict(raw=img, offsets=off, black=blk, white=white, wb=wb, gain=gain)


def pack(raw, offsets):
    """H x W Bayer -> 4 x H/2 x W/2 planes in CHANNELS order."""
    return np.stack([raw[r::2, c::2] for r, c in offsets])


def unpack(planes, offsets):
    _, hp, wp = planes.shape
    out = np.empty((hp * 2, wp * 2), dtype=planes.dtype)
    for k, (r, c) in enumerate(offsets):
        out[r::2, c::2] = planes[k]
    return out


def upsample_gain(gain, hp, wp):
    """Bilinear upsample Ny x Nx x 4 grid to 4 x hp x wp (corner-aligned), float64 math."""
    ny, nx, _ = gain.shape
    gy = np.arange(hp, dtype=np.float64) * (ny - 1) / (hp - 1)
    gx = np.arange(wp, dtype=np.float64) * (nx - 1) / (wp - 1)
    y0 = np.minimum(np.floor(gy).astype(int), ny - 2)
    x0 = np.minimum(np.floor(gx).astype(int), nx - 2)
    fy = (gy - y0)[:, None, None]
    fx = (gx - x0)[None, :, None]
    g = gain.astype(np.float64)
    top = g[y0][:, x0] * (1 - fx) + g[y0][:, x0 + 1] * fx
    bot = g[y0 + 1][:, x0] * (1 - fx) + g[y0 + 1][:, x0 + 1] * fx
    up = top * (1 - fy) + bot * fy                    # hp, wp, 4
    return np.ascontiguousarray(up.transpose(2, 0, 1)).astype(np.float32)


def apply_lsc(planes, black, white, gain):
    """Golden LSC: normalized (planes - black) * upsampled gain. Returns float32 4 x hp x wp."""
    _, hp, wp = planes.shape
    g = upsample_gain(gain, hp, wp)
    norm = (planes.astype(np.float32) - black[:, None, None]) / (white - black[:, None, None])
    return norm * g


def normalize_only(planes, black, white):
    return (planes.astype(np.float32) - black[:, None, None]) / (white - black[:, None, None])


def to_srgb_preview(planes, wb, exposure=None):
    """Quick half-res preview: R, mean(G), B -> WB -> auto exposure -> gamma. For eyes only."""
    rgb = np.stack([planes[0], 0.5 * (planes[1] + planes[2]), planes[3]], axis=-1)
    rgb = np.clip(rgb, 0, None) * (wb / wb[1])
    if exposure is None:
        exposure = 1.0 / max(np.percentile(rgb[..., 1], 99), 1e-6)
    rgb = np.clip(rgb * exposure, 0, 1)
    return rgb ** (1 / 2.2), exposure
