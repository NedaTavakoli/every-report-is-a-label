#!/usr/bin/env python3
"""
External test set: EMIDEC (MICCAI 2020; Dijon, France; CC BY-NC-SA 4.0, non-commercial research; cite the EMIDEC paper).
100 labeled short-axis LGE exams: 33 normal (N), 67 myocardial infarction (P), with expert contours
(1 LV cavity, 2 myocardium, 3 infarct, 4 no-reflow) and a clinical text file per case.
Packs each stack exactly as cache_lge.py does for our exams (per-stack 1-99 percentile to uint8, shorter side 256, slices in
order) into data/emidec/emidec.h5 and writes data/emidec/labels.csv (case, mi, infarct % of LV myocardium, clinical fields).
NIfTI arrays are (x, y, z); each slice is transposed to (rows, cols) as DICOM stores it.
"""
import os, re, glob
import numpy as np, pandas as pd, h5py, nibabel as nib
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "data", "emidec-dataset-1.0.1")
OUT = os.path.join(ROOT, "data", "emidec")
SHORTER = 256


def to_uint8(stack):
    lo, hi = np.percentile(stack, [1, 99]); out = []
    for a in stack:
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        h, w = a.shape; r = SHORTER / min(h, w)
        out.append(np.asarray(Image.fromarray(a.astype(np.uint8)).resize((round(w * r), round(h * r)), Image.BILINEAR)))
    return np.stack(out)


def clinical(path):
    d = {}
    for line in open(path, encoding="latin-1"):
        if ":" not in line: continue
        k, v = line.split(":", 1); k = re.sub(r"[^A-Za-z0-9+ ]", "", k).strip().lower().replace(" ", "_"); v = v.strip()
        if k and v: d["clin_" + k] = v
    return d


def main():
    os.makedirs(OUT, exist_ok=True); rows = []
    with h5py.File(os.path.join(OUT, "emidec.h5"), "w") as h:
        for d in sorted(glob.glob(os.path.join(SRC, "Case_*"))):
            n = os.path.basename(d)
            im = np.asarray(nib.load(f"{d}/Images/{n}.nii.gz").dataobj, dtype=np.float32)
            m = np.asarray(nib.load(f"{d}/Contours/{n}.nii.gz").dataobj).astype(np.uint8)
            stack = np.stack([im[:, :, k].T for k in range(im.shape[2])])
            h.create_dataset(n, data=to_uint8(stack), compression="lzf")
            myo = int(np.isin(m, [2, 3, 4]).sum()); inf = int(np.isin(m, [3, 4]).sum())
            r = {"case": n, "mi": int(n.split("_")[1][0] == "P"), "infarct_pct": 100 * inf / max(myo, 1),
                 "noreflow": int((m == 4).any()), "slices": im.shape[2]}
            t = os.path.join(SRC, n.replace("_", " ") + ".txt")
            if os.path.exists(t): r.update(clinical(t))
            rows.append(r)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "labels.csv"), index=False)
    print(len(df), "cases; MI", int(df.mi.sum()), "; infarct % median (MI cases)", round(df[df.mi == 1].infarct_pct.median(), 1),
          "; normals with any infarct label", int(((df.mi == 0) & (df.infarct_pct > 0)).sum()), "; columns", list(df.columns))


if __name__ == "__main__":
    main()
