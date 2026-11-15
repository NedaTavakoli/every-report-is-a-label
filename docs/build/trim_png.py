#!/usr/bin/env python3
"""Crop white margins from rendered figures (keeps a small border) so that every figure uses the full printed width.
Usage: trim_png.py file.png [...] (in place)."""
import sys
import numpy as np
from PIL import Image
for f in sys.argv[1:]:
    im = Image.open(f).convert("RGB"); a = np.asarray(im)
    ys, xs = np.where((a < 245).any(-1))
    if len(ys) == 0: continue
    pad = int(0.012 * im.width)
    box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, im.width), min(ys.max() + pad, im.height))
    im.crop(box).save(f)
