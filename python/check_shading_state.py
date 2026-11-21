"""Diagnostic: are HDR+ DNGs uncorrected (vignetted) or already shading-corrected?

Author: Xue -c
Date: 2026-11-13

Average log-brightness over many scenes on the gain-map grid. Scene content averages
out; lens vignetting does not. If the DNG is uncorrected, the mean raw falloff should
track 1/gain (corner/center ~ 1/3.7). If already corrected, the mean raw is ~flat and
applying the gain again makes corners ~3.7x too bright.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import lsc_ref as L  # noqa: E402
from step1_reference import DATA, complete_bursts  # noqa: E402


def block_mean_log(plane, ny, nx, sat):
    """Mean of log(signal) in ny x nx blocks, ignoring dark/saturated pixels."""
    h, w = plane.shape
    out = np.full((ny, nx), np.nan)
    ys = np.linspace(0, h, ny + 1).astype(int)
    xs = np.linspace(0, w, nx + 1).astype(int)
    for i in range(ny):
        for j in range(nx):
            b = plane[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            v = b[(b > 0.01) & (b < sat)]
            if v.size > 100:
                out[i, j] = np.log(v).mean()
    return out


def center_ref(a):
    return a - a[a.shape[0] // 2, a.shape[1] // 2]


def corners(a):
    return float(np.exp(np.nanmean([a[0, 0], a[0, -1], a[-1, 0], a[-1, -1]])))


def edges_lr(a):
    """Left/right edge midpoints: avoids the sky-on-top / ground-on-bottom scene bias."""
    m = a.shape[0] // 2
    return float(np.exp(np.nanmean([a[m, 0], a[m, -1]])))


if __name__ == "__main__":
    groups = {}
    for b in complete_bursts():
        d = os.path.join(DATA, b)
        f = L.load_frame(os.path.join(d, "payload_N000.dng"), os.path.join(d, "lens_shading_map_N000.tiff"))
        p = L.normalize_only(L.pack(f["raw"], f["offsets"]), f["black"], f["white"])
        g = 0.5 * (p[1] + p[2])
        ny, nx = f["gain"].shape[:2]
        acc, maps = groups.setdefault((f["raw"].shape, f["gain"].shape), ([], []))
        acc.append(block_mean_log(g, ny, nx, 0.95))
        maps.append(np.log(0.5 * (f["gain"][..., 1] + f["gain"][..., 2])))

    np.set_printoptions(precision=2, suppress=True, linewidth=160)
    print("brightness relative to center, green, averaged over scenes")
    print(f"{'raw shape':>14} {'grid':>12} {'n':>4} | {'corner: raw':>11} {'1/gain':>7} {'raw*gain':>8} "
          f"| {'L/R edge: raw':>13} {'1/gain':>7} {'raw*gain':>8}")
    for (rs, gs), (acc, maps) in sorted(groups.items(), key=lambda kv: -len(kv[1][0])):
        mr = center_ref(np.nanmean(np.array(acc), axis=0))
        mm = center_ref(np.array(maps).mean(axis=0))
        print(f"{str(rs):>14} {str(gs[:2]):>12} {len(acc):>4} | {corners(mr):>11.2f} {corners(-mm):>7.2f} "
              f"{corners(mr + mm):>8.2f} | {edges_lr(mr):>13.2f} {edges_lr(-mm):>7.2f} {edges_lr(mr + mm):>8.2f}")
    print("\nUncorrected DNG => raw ~ 1/gain and raw*gain ~ 1.  Pre-corrected DNG => raw ~ 1, raw*gain ~ gain.")
