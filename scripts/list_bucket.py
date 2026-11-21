"""List objects in the public HDR+ GCS bucket over plain HTTPS (no gsutil).

Author: Xue -c
Date: 2026-11-01

Usage:
  python scripts/list_bucket.py                 # list burst folder names
  python scripts/list_bucket.py <burst_name>    # list files in one burst
"""
import json
import sys
import urllib.parse
import urllib.request

API = "https://storage.googleapis.com/storage/v1/b/hdrplusdata/o"
ROOT = "20171106_subset/bursts/"


def gcs_list(prefix, delimiter=None):
    items, prefixes, token = [], [], None
    while True:
        q = {"prefix": prefix, "maxResults": "1000"}
        if delimiter:
            q["delimiter"] = delimiter
        if token:
            q["pageToken"] = token
        with urllib.request.urlopen(API + "?" + urllib.parse.urlencode(q)) as r:
            d = json.load(r)
        items += d.get("items", [])
        prefixes += d.get("prefixes", [])
        token = d.get("nextPageToken")
        if not token:
            return items, prefixes


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _, bursts = gcs_list(ROOT, "/")
        print(len(bursts), "bursts")
        for b in bursts[:10]:
            print(" ", b[len(ROOT):-1])
    else:
        items, _ = gcs_list(ROOT + sys.argv[1] + "/")
        for i in items:
            print(f"{int(i['size']):>10}  {i['name'][len(ROOT):]}")
