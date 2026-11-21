"""Download reference frame (N000) + its lens shading map for each HDR+ burst.

Author: Xue -c
Date: 2026-11-03

Usage: python scripts/download_subset.py [max_bursts]
Writes to data/hdrplus/<burst>/ . Skips files already present.
"""
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from list_bucket import ROOT, gcs_list  # noqa: E402

BASE = "https://storage.googleapis.com/hdrplusdata/"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "hdrplus")
FILES = ["payload_N000.dng", "lens_shading_map_N000.tiff", "rgb2rgb.txt"]


def fetch(job, retries=4):
    url, dst = job
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r, open(tmp, "wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            os.replace(tmp, dst)
            return os.path.getsize(dst)
        except OSError as e:
            print(f"  retry {attempt + 1}/{retries} {os.path.basename(os.path.dirname(dst))}: {e}", flush=True)
    raise RuntimeError(f"failed: {url}")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    _, bursts = gcs_list(ROOT, "/")
    bursts = [b[len(ROOT):-1] for b in bursts][:limit]
    jobs = [(BASE + ROOT + b + "/" + f, os.path.join(OUT, b, f))
            for b in bursts for f in FILES]
    with ThreadPoolExecutor(4) as ex:
        total = 0
        futures = [ex.submit(fetch, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            total += fut.result()
            if i % 15 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} files, {total / 1e6:.0f} MB new", flush=True)
    print("done:", len(bursts), "bursts ->", os.path.abspath(OUT))
