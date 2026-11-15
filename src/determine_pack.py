#!/usr/bin/env python3
"""
External multi-center test set from the DETERMINE trial (ScarNet's packed slices, data/determine/determine_slices.npz from
src/scarnet/pack_determine.py): short-axis LGE magnitude slices with expert contours (blood pool, myocardium, scar), patient
index and trial site. Slices duplicated between ScarNet's Training and Testing folders are removed (identical pixels).
Per patient: the remaining slices in stored order (adjacent-slice correlation 0.73 vs 0.47 for random order, so the order
is anatomical), rescaled to uint8 by the stack's 1-99 percentiles and resized to a 256-px shorter side, as cache_lge.py does.
Writes data/determine/determine_stacks.h5 (one dataset per patient index) and data/determine/determine_labels.csv
(pid, site, n_slices, infarct_pct = scar / (scar + myocardium) over all contoured slices, scar_any). No names are stored.
"""
import os, hashlib
import numpy as np, pandas as pd, h5py
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    z = np.load(os.path.join(ROOT, "data", "determine", "determine_slices.npz"))
    I, M, pid, site = z["image"], z["mask"], z["pid"], z["site"]
    h = np.array([hashlib.md5(I[i].tobytes()).hexdigest() for i in range(len(I))])
    _, first = np.unique(h, return_index=True); keep = np.zeros(len(I), bool); keep[first] = True
    rows = []
    with h5py.File(os.path.join(ROOT, "data", "determine", "determine_stacks.h5"), "w") as f:
        for p in np.unique(pid):
            ix = np.where(keep & (pid == p))[0]
            st = I[ix].astype(np.float32); lo, hi = np.percentile(st, [1, 99])
            st = (np.clip((st - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
            st = np.stack([np.asarray(Image.fromarray(a).resize((256, 256), Image.BILINEAR)) for a in st])
            f.create_dataset(str(int(p)), data=st, compression="lzf")
            scar, myo = int((M[ix] == 2).sum()), int((M[ix] == 3).sum())
            rows.append({"pid": int(p), "site": int(site[ix[0]]), "n_slices": len(ix), "infarct_pct": 100 * scar / max(scar + myo, 1),
                         "scar_any": int(scar > 0)})
    d = pd.DataFrame(rows); d.to_csv(os.path.join(ROOT, "data", "determine", "determine_labels.csv"), index=False)
    print(len(d), "patients,", d.site.nunique(), "sites; slices kept", int(keep.sum()), "of", len(I),
          "; infarct % median", round(d.infarct_pct.median(), 1), "; no scar", int((d.scar_any == 0).sum()))


if __name__ == "__main__":
    main()
