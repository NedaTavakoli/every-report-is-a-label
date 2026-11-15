#!/usr/bin/env python3
"""
Example LGE slices for the figures, chosen by objective criteria only (nobody picked them by eye):
test-set examinations whose own report the pretrained model ranked first, with all four views; sorted by anonymized
accession, first three. One middle slice per view. Writes docs/figures/images/<anon_acc>_<view>.png (patient images:
local only, never to chat) and same-size gray placeholders docs/figures/images_placeholder/ for layout checks.
"""
import os, json
import numpy as np, pandas as pd, h5py
from PIL import Image, ImageDraw

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIEWS = ["sa", "2ch", "3ch", "4ch"]
rank = json.load(open(os.path.join(ROOT, "outputs", "lge_clip", "retrieval_ci.json")))["rank_trained"]
ser = pd.read_csv(os.path.join(ROOT, "data", "manifest", "series.csv"), dtype=str)
top = sorted(a for a, r in rank.items() if r == 1)
pick = []
for a in top:
    s = ser[ser.anon_acc == a]
    p = os.path.join(ROOT, "data", "cache_h5", a + ".h5")
    if not os.path.exists(p): continue
    with h5py.File(p, "r") as h:
        got = {}
        for v in VIEWS:
            keys = [k for k in s[s.view == v].series.str.replace("/", "_") if k in h]
            if keys:
                k = max(keys, key=lambda k: h[k].shape[0])          # the series with most slices
                got[v] = h[k][h[k].shape[0] // 2]                    # middle slice
        if len(got) == 4: pick.append((a, got))
    if len(pick) == 3: break
out, ph = os.path.join(ROOT, "docs", "figures", "images"), os.path.join(ROOT, "docs", "figures", "images_placeholder")
os.makedirs(out, exist_ok=True); os.makedirs(ph, exist_ok=True)
for a, got in pick:
    for v, im in got.items():
        Image.fromarray(im).resize((512, 512), Image.BICUBIC).save(os.path.join(out, f"{a}_{v}.png"))
        g = Image.new("L", (512, 512), 90); d = ImageDraw.Draw(g); d.text((20, 240), f"{a} {v}", fill=230); g.save(os.path.join(ph, f"{a}_{v}.png"))
json.dump([a for a, _ in pick], open(os.path.join(out, "examples.json"), "w"))
print("examples (report IDs):", [a for a, _ in pick], "| rank-1 exams:", len(top))
