#!/usr/bin/env python3
"""
Converts each LGE series to pixels in HDF5 so that training does not read DICOM over the network. One file per exam,
data/cache_h5/<anon_acc>.h5, one dataset per series: uint8 (slices, H, W), shorter side fixed, ordered by InstanceNumber.
Only pixels and the slice count are stored: no DICOM header goes into the cache. Finished exams are skipped on restart.
"""
import os, argparse, concurrent.futures as cf
import numpy as np, pandas as pd, h5py, pydicom
from PIL import Image
from common import load_config


def load_series(folder, shorter):
    items = []
    for f in sorted(os.listdir(folder)):
        if f.startswith("."): continue
        d = pydicom.dcmread(os.path.join(folder, f))
        a = d.pixel_array.astype(np.float32)
        if a.ndim != 2: continue    # enhanced or colour objects are not LGE magnitude slices
        if getattr(d, "PhotometricInterpretation", "") == "MONOCHROME1": a = a.max() - a
        items.append((int(getattr(d, "InstanceNumber", 0) or 0), a))
    if not items: return None
    items.sort(key=lambda x: x[0])
    stack = [a for _, a in items]
    lo, hi = np.percentile(np.concatenate([a.ravel() for a in stack]), [1, 99])   # per series, so slices stay comparable
    out = []
    for a in stack:
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        h, w = a.shape; r = shorter / min(h, w)
        im = Image.fromarray(a.astype(np.uint8)).resize((round(w * r), round(h * r)), Image.BILINEAR)
        out.append(np.asarray(im))
    if len({x.shape for x in out}) > 1:     # mixed matrix within a series: use the most common shape
        shapes = [x.shape for x in out]; keep = max(set(shapes), key=shapes.count)
        out = [x for x in out if x.shape == keep]
    return np.stack(out)


def cache_exam(acc, group, root, out_dir, shorter):
    path = os.path.join(out_dir, f"{acc}.h5")
    if os.path.exists(path): return acc, "skipped", 0
    tmp, n = path + ".tmp", 0
    try:
        with h5py.File(tmp, "w") as h:
            for s in group.itertuples():
                try:
                    v = load_series(os.path.join(root, s.rel_path), shorter)
                except Exception as ex:
                    print(f"  {acc} series skipped: {type(ex).__name__}: {str(ex)[:120]}", flush=True); v = None
                if v is None: continue
                h.create_dataset(s.series.replace("/", "_"), data=v, compression="lzf"); n += 1
        os.replace(tmp, path)
        return acc, "ok", n
    except OSError as ex:      # network timeout: leave no partial file, a rerun retries this exam
        if os.path.exists(tmp): os.remove(tmp)
        return acc, f"failed {type(ex).__name__}", 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lge_clip.yaml")
    ap.add_argument("--limit", type=int, default=None, help="only the first N exams (test)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(args.config); P = cfg["paths"]
    s = pd.read_csv(os.path.join(P["manifest_dir"], "series.csv"), dtype=str)
    os.makedirs(P["cache_dir"], exist_ok=True)
    groups = list(s.groupby("anon_acc"))[:args.limit]
    res = {}
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(cache_exam, a, g, P["lge_root"], P["cache_dir"], cfg["data"]["shorter_side"]) for a, g in groups]
        for i, f in enumerate(cf.as_completed(futs)):
            _, st, _ = f.result(); res[st] = res.get(st, 0) + 1
            if i % 100 == 0: print(f"{i + 1}/{len(groups)} {res}", flush=True)
    print("done", res)


if __name__ == "__main__":
    main()
