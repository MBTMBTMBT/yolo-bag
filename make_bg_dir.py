#!/usr/bin/env python3
"""
make_bg_dir.py  –  Unsplash Research ➜ bg_dir/  (robot backgrounds)

 * Handles files named photos.csv000 / photos.tsv000 … inside ./backgrounds/
 * Auto-detects separator and URL column name.
 * Downloads images in parallel, resizes to WIDTH×HEIGHT (default 640×480).
 * Skips any JPEG that already exists in bg_dir/.

Requires: pandas, requests, opencv-python, numpy
"""

import argparse, concurrent.futures as cf, os, re, sys, cv2, numpy as np
import pandas as pd, requests
from pathlib import Path
from time import time

# ───────────────────── helpers ──────────────────────
def detect_sep(path):
    with open(path, "r", encoding="utf8") as f:
        return "\t" if "\t" in f.readline() else ","

URL_CANDIDATES = ["url_o", "photo_image_url", "photo_url",
                  "url", "download_url"]

def load_urls(csv_glob):
    parts = []
    for fp in sorted(Path(".").glob(csv_glob)):
        sep = detect_sep(fp)
        df  = pd.read_csv(fp, sep=sep, low_memory=False)
        url_col = next((c for c in URL_CANDIDATES if c in df.columns), None)
        if url_col is None:
            raise ValueError(f"{fp} has none of the expected URL columns")
        df = df.loc[:, ["photo_id", url_col]].rename(columns={url_col: "url_o"})
        parts.append(df)
    out = pd.concat(parts, ignore_index=True).drop_duplicates("photo_id")
    out = out[out.url_o.notnull() & out.url_o.astype(str).str.startswith("http")]
    return out.reset_index(drop=True)

def parse_wh(s):
    m = re.fullmatch(r"(\d+)x(\d+)", s)
    if not m: raise ValueError("resize must look like 640x480")
    return int(m.group(1)), int(m.group(2))

def dl_resize(row, out_dir, wh):
    jpg_path = out_dir / f"{row.photo_id}.jpg"
    if jpg_path.exists(): return True
    try:
        r = requests.get(row.url_o, timeout=10); r.raise_for_status()
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        if img is None: raise ValueError
        img = cv2.resize(img, wh, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(jpg_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return True
    except Exception: return False

# ───────────────────── main ──────────────────────
def main(a):
    t0 = time()
    out_dir = Path(a.output_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    df = load_urls(a.csv_glob)
    if a.sample != -1 and a.sample < len(df):
        df = df.sample(a.sample, random_state=42).reset_index(drop=True)
    wh = parse_wh(a.resize)
    print(f"[INFO] downloading {len(df):,} images → {out_dir} at {wh[0]}×{wh[1]} …")
    ok = 0
    with cf.ThreadPoolExecutor(max_workers=a.threads) as ex:
        for res in cf.as_completed([ex.submit(dl_resize, r, out_dir, wh) for r in df.itertuples()]):
            ok += res.result()
    print(f"[✔] {ok:,}/{len(df):,} images ready  ({time()-t0:.1f}s)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv_glob", default="backgrounds/photos.csv*", help="glob for Unsplash files")
    p.add_argument("--output_dir", default="bg_dir", help="destination folder")
    p.add_argument("--sample", type=int, default=5000, help="-1 = all rows")
    p.add_argument("--resize", default="640x480", help="WIDTHxHEIGHT")
    p.add_argument("--threads", type=int, default=16, help="parallel downloads")
    main(p.parse_args())
