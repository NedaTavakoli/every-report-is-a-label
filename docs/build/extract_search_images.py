#!/usr/bin/env python3
"""
Images for the text-search figure: for three queries, the top-4 test examinations returned by the report-pretrained model
(outputs/lge_clip/search_demo.json, ranked by image-query similarity, chosen by rank, not by eye), middle short-axis slice from
the training cache (or the first available view). Writes docs/figures/images/search/<query>_<rank>.png (local only;
patient images) and gray placeholders in images_placeholder/search/. Prints counts only.
"""
import os, json
import numpy as np, h5py
from PIL import Image, ImageDraw
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
Q = ["sarcoid", "midwall", "no_lge"]
r = json.load(open(os.path.join(ROOT, "outputs", "lge_clip", "search_demo.json")))
d = os.path.join(ROOT, "docs", "figures", "images", "search"); dp = os.path.join(ROOT, "docs", "figures", "images_placeholder", "search")
os.makedirs(d, exist_ok=True); os.makedirs(dp, exist_ok=True); n = 0
for q in Q:
    k = 0
    for acc in r["trained"][q]["top"]:          # next rank fills in when an exam has no cached image
        if k == 4: break
        acc = acc.split("_")[0]; f = os.path.join(ROOT, "data", "cache_h5", acc + ".h5")
        if not os.path.exists(f): continue
        with h5py.File(f, "r") as h:
            keys = sorted(h, key=lambda s: (0 if "SA" in s.upper() else 1, s)); v = h[keys[0]][()]
        a = v[len(v) // 2]; H, W = a.shape; m = min(H, W); a = a[(H - m) // 2:(H - m) // 2 + m, (W - m) // 2:(W - m) // 2 + m]
        Image.fromarray(a).resize((400, 400), Image.BICUBIC).save(os.path.join(d, f"{q}_{k}.png")); n += 1
        g = Image.new("L", (400, 400), 90); ImageDraw.Draw(g).text((20, 190), f"{q} {k + 1}", fill=230); g.save(os.path.join(dp, f"{q}_{k}.png")); k += 1
print("images", n)
