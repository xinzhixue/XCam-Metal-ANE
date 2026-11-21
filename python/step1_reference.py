"""Step 1: run golden LSC on HDR+ frames, save before/after previews and golden binaries.

Author: Xue -c
Date: 2026-11-11

Usage: python python/step1_reference.py [burst_name ...]
  Default: first 6 downloaded bursts.
Outputs:
  out/step1/<burst>_before_after.png   preview (before | after | gain map)
  out/golden/<burst>.{raw_u16,gain_f32,out_f32}.bin + .json   for Metal checks
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import lsc_ref as L  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data", "hdrplus")
OUT = os.path.join(ROOT, "out")


def complete_bursts():
    ok = []
    for d in sorted(glob.glob(os.path.join(DATA, "*"))):
        if all(os.path.exists(os.path.join(d, f))
               for f in ("payload_N000.dng", "lens_shading_map_N000.tiff")):
            ok.append(os.path.basename(d))
    return ok


def corner_center_ratio(planes):
    """Mean of 4 corner patches / center patch, per channel. Flat scene -> 1.0."""
    _, h, w = planes.shape
    p = max(h, w) // 20
    c = planes[:, h // 2 - p:h // 2 + p, w // 2 - p:w // 2 + p].mean(axis=(1, 2))
    corners = np.mean([planes[:, :p, :p].mean(axis=(1, 2)), planes[:, :p, -p:].mean(axis=(1, 2)),
                       planes[:, -p:, :p].mean(axis=(1, 2)), planes[:, -p:, -p:].mean(axis=(1, 2))], axis=0)
    return corners / np.maximum(c, 1e-6)


def run(burst, save_golden):
    d = os.path.join(DATA, burst)
    f = L.load_frame(os.path.join(d, "payload_N000.dng"), os.path.join(d, "lens_shading_map_N000.tiff"))
    planes = L.pack(f["raw"], f["offsets"])
    assert np.array_equal(L.unpack(planes, f["offsets"]), f["raw"]), "pack/unpack mismatch"
    before = L.normalize_only(planes, f["black"], f["white"])
    after = L.apply_lsc(planes, f["black"], f["white"], f["gain"])

    # One shared exposure from the center half of the frame (gain ~1 there), so the
    # corner brightening is visible instead of being hidden by per-image auto exposure.
    _, h, w = before.shape
    _, exp = L.to_srgb_preview(before[:, h // 4:3 * h // 4, w // 4:3 * w // 4], f["wb"])
    pb, _ = L.to_srgb_preview(before, f["wb"], exp)
    pa, _ = L.to_srgb_preview(after, f["wb"], exp)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw=dict(width_ratios=[4, 4, 3]))
    ax[0].imshow(pb); ax[0].set_title("before (raw, no LSC)")
    ax[1].imshow(pa); ax[1].set_title("after golden LSC")
    im = ax[2].imshow(f["gain"][..., 1], cmap="magma"); ax[2].set_title("gain map Gr")
    plt.colorbar(im, ax=ax[2])
    for a in ax:
        a.axis("off")
    os.makedirs(os.path.join(OUT, "step1"), exist_ok=True)
    fig.savefig(os.path.join(OUT, "step1", f"{burst}_before_after.png"), dpi=80, bbox_inches="tight")
    plt.close(fig)

    rb, ra = corner_center_ratio(before), corner_center_ratio(after)
    print(f"{burst}: corner/center before={np.round(rb, 2)} after={np.round(ra, 2)} "
          f"gain max={f['gain'].max():.2f}")

    if save_golden:
        g = os.path.join(OUT, "golden")
        os.makedirs(g, exist_ok=True)
        f["raw"].astype(np.uint16).tofile(os.path.join(g, f"{burst}.raw_u16.bin"))
        f["gain"].astype(np.float32).tofile(os.path.join(g, f"{burst}.gain_f32.bin"))  # Ny,Nx,4
        after.astype(np.float32).tofile(os.path.join(g, f"{burst}.out_f32.bin"))       # 4,Hp,Wp
        meta = dict(height=int(f["raw"].shape[0]), width=int(f["raw"].shape[1]),
                    grid_h=int(f["gain"].shape[0]), grid_w=int(f["gain"].shape[1]),
                    offsets=[list(map(int, o)) for o in f["offsets"]],
                    black=f["black"].tolist(), white=f["white"], channels=list(L.CHANNELS))
        with open(os.path.join(g, f"{burst}.json"), "w") as fh:
            json.dump(meta, fh, indent=1)


if __name__ == "__main__":
    bursts = sys.argv[1:] or complete_bursts()[:6]
    for i, b in enumerate(bursts):
        run(b, save_golden=(i == 0))
    print("previews ->", os.path.join(OUT, "step1"))
