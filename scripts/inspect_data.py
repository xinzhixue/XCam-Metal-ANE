"""Print DNG + gain-map metadata for downloaded HDR+ bursts.

Author: Xue -c
Date: 2026-11-06

Usage: python scripts/inspect_data.py [n_bursts]
"""
import collections
import glob
import os
import sys

import numpy as np
import rawpy
import tifffile

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "hdrplus")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    dirs = sorted(d for d in glob.glob(os.path.join(DATA, "*"))
                  if os.path.exists(os.path.join(d, "lens_shading_map_N000.tiff")))
    summary = collections.Counter()
    for i, d in enumerate(dirs):
        with rawpy.imread(os.path.join(d, "payload_N000.dng")) as raw:
            shape = raw.raw_image_visible.shape
            pat = "".join(raw.color_desc.decode()[c] for c in raw.raw_pattern.flatten())
            bl, wl = raw.black_level_per_channel, raw.white_level
            wb = raw.camera_whitebalance
            sizes = raw.sizes
        m = tifffile.imread(os.path.join(d, "lens_shading_map_N000.tiff"))
        summary[(shape, pat, m.shape)] += 1
        if i < n:
            print(os.path.basename(d))
            print(f"  raw {shape} {raw.raw_image_visible.dtype if False else ''}pattern={pat} "
                  f"black={bl} white={wl} wb={np.round(wb, 3)}")
            print(f"  sizes: raw={sizes.raw_width}x{sizes.raw_height} "
                  f"visible={sizes.width}x{sizes.height} margins=({sizes.left_margin},{sizes.top_margin}) "
                  f"flip={sizes.flip}")
            print(f"  map {m.shape} {m.dtype} min={m.min():.3f} max={m.max():.3f} "
                  f"center={np.round(m[m.shape[0] // 2, m.shape[1] // 2], 3)} "
                  f"corner={np.round(m[0, 0], 3)}")
    print("\n(raw shape, CFA, map shape) -> count")
    for k, v in summary.most_common():
        print(" ", k, v)
