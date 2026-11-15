#!/usr/bin/env python3
"""
Full DETERMINE external test set with CORE-LAB labels (run where the trial data are stored).
  images: DETERMINE-PROC exam folders (flat DICOM folders; file names start with the series number). Per patient, the LGE
          magnitude short-axis series that was contoured (DICOM_Analysis_Report_Filtered.xlsx: Is_Contour_Empty == "No"; the
          series with the most images; if that has < 4 images the acquisition was slice by slice, and all contoured
          series of the patient are combined and ordered by slice location). Long-axis series are skipped.
          Each stack is cached exactly as cache_lge.py does (1-99 percentile to uint8, shorter side 256).
  labels: DETERMINE core-lab export (DETERMINE_PREDETERMINE_Final_Export_with_MRDATE_FULLforNeda.xlsx), accepted analyses:
          infarct % and mass, LV mass, LVEF, EDV, ESV, infarct territory (LAD / LCX / RCA, any segment), microvascular
          obstruction score > 0, transmurality score, aneurysm.
Writes data/determine_full/{stacks.h5, labels.csv} keyed by an anonymous index (no trial ID or initials) and a private
crosswalk data/determine_full/crosswalk_PRIVATE.csv (index -> trial ID; keep private, never share or print).
"""
import os, re, hashlib, concurrent.futures as cf
import numpy as np, pandas as pd, h5py, pydicom
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P = os.environ.get("DETERMINE_TABLES", "/path/to/DETERMINE/tables/")      # core-lab export + per-series analysis report
S = os.environ.get("DETERMINE_DICOM", "/path/to/DETERMINE/dicom")         # one flat DICOM folder per exam
OUT = os.path.join(ROOT, "data", "determine_full")
LAX = re.compile(r"(2|3|4)\s*-?\s*ch|lax|long|vla|hla|lvot", re.I)
X_ = "FinalUseAnalysis::"; G_ = "FinalUseSegements::"


def num(s): return pd.to_numeric(s, errors="coerce")


def labels():
    x = pd.read_excel(P + "DETERMINE_PREDETERMINE_Final_Export_with_MRDATE_FULLforNeda.xlsx")
    x = x[x.AcceptOrReject == "Accepted"].copy(); x["key"] = x.Subject_ID.astype(str).str[:9]
    o = pd.DataFrame({"key": x.key, "site": x.key.str[:4].astype(int), "infarct_pct": num(x[X_ + "Infarct%_calc"]),
                      "infarct_g": num(x[X_ + "INFARCT"]), "lv_mass_g": num(x[X_ + "MRLVMSS"]), "lvef": num(x[X_ + "LVEF_calc"]),
                      "edv": num(x[X_ + "EDV"]), "esv": num(x[X_ + "ESV"])})
    for t in ("LAD", "LCX", "RCA"): o[f"territory_{t.lower()}"] = (num(x[G_ + f"InfLocLAD".replace("LAD", t)]) > 0).astype(float).where(num(x[G_ + "InfLocLAD".replace("LAD", t)]).notna())
    o["mvo"] = (num(x[G_ + "SMicroObstSCORE"]) > 0).astype(float).where(num(x[G_ + "SMicroObstSCORE"]).notna())
    o["transmurality_score"] = num(x[G_ + "STranSCORE"]); o["aneurysm"] = num(x[G_ + "Aneurysm Present"])
    return o.drop_duplicates("key")


def sernum(s): return int(re.sub(r"\D", "", str(s)))     # the report stores "SER1701"


def read_stack(folder, series_list):
    series_list = [sernum(s) for s in series_list]
    files = [f for f in os.listdir(folder) if not f.startswith(".") and f.split("_")[0].isdigit() and int(f.split("_")[0]) in set(series_list)]
    items = []
    for f in files:
        try: h = pydicom.dcmread(os.path.join(folder, f))
        except Exception: continue
        if int(getattr(h, "SeriesNumber", -1)) not in set(series_list): continue
        a = h.pixel_array.astype(np.float32)
        if a.ndim != 2: continue
        if getattr(h, "PhotometricInterpretation", "") == "MONOCHROME1": a = a.max() - a
        loc = float(getattr(h, "SliceLocation", 0) or 0); inst = int(getattr(h, "InstanceNumber", 0) or 0)
        items.append(((loc, inst) if len(series_list) > 1 else (inst, loc), a))
    if not items: return None
    items.sort(key=lambda t: t[0]); shapes = [a.shape for _, a in items]; keep = max(set(shapes), key=shapes.count)
    st = [a for _, a in items if a.shape == keep]
    lo, hi = np.percentile(np.concatenate([a.ravel() for a in st]), [1, 99]); out = []
    for a in st:
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255; h_, w_ = a.shape; r = 256 / min(h_, w_)
        out.append(np.asarray(Image.fromarray(a.astype(np.uint8)).resize((round(w_ * r), round(h_ * r)), Image.BILINEAR)))
    return np.stack(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    lab = labels()
    f = pd.read_excel(P + "DICOM_Analysis_Report_Filtered.xlsx"); f["key"] = f.Patient_ID.astype(str).str[:9]
    f = f[(f.Is_Contour_Empty.astype(str) == "No") & ~f.Series_Description.astype(str).str.contains(LAX)]
    folders = {}
    for e in os.listdir(S):
        if not e.startswith("."): folders.setdefault(e[:9], []).append(os.path.join(S, e))
    keys = sorted(set(lab.key) & set(f.key) & set(folders))
    parts = os.path.join(OUT, "_parts"); os.makedirs(parts, exist_ok=True)      # resumable: one file per finished patient
    pname = lambda k: os.path.join(parts, hashlib.sha1(k.encode()).hexdigest()[:16] + ".npz")
    if os.path.exists(os.path.join(OUT, "crosswalk_PRIVATE.csv")) and os.path.exists(os.path.join(OUT, "stacks.h5")):
        old = pd.read_csv(os.path.join(OUT, "crosswalk_PRIVATE.csv")); oldl = pd.read_csv(os.path.join(OUT, "labels.csv")) if os.path.exists(os.path.join(OUT, "labels.csv")) else None
        with h5py.File(os.path.join(OUT, "stacks.h5"), "r") as h:
            for r in old.itertuples():
                if not os.path.exists(pname(r.trial_key)) and str(r.idx) in h:
                    comb = int(oldl.set_index("idx").combined_series.get(r.idx, 0)) if oldl is not None else 0
                    np.savez(pname(r.trial_key), st=h[str(r.idx)][()], combined=comb)
    def job0(k):
        if os.path.exists(pname(k)):
            z = np.load(pname(k)); return k, z["st"], bool(z["combined"])
        k, st, comb = job(k)
        if st is not None: np.savez(pname(k), st=st, combined=int(comb))
        return k, st, comb
    def job(k):
        rows = f[f.key == k].sort_values("Number_of_DICOM_Images", ascending=False)
        top = rows.iloc[0]; series = [top.MAG_Serie] if top.Number_of_DICOM_Images >= 4 else rows.MAG_Serie.tolist()
        for fo in folders[k]:
            try:
                st = read_stack(fo, series)
                if st is not None and len(st) >= 3: return k, st, len(series) > 1
            except Exception:
                continue
        return k, None, False
    res = {}
    with cf.ThreadPoolExecutor(8) as ex:
        for i, (k, st, combined) in enumerate(ex.map(job0, keys)):
            res[k] = (st, combined)
            if i % 100 == 0: print(f"{i + 1}/{len(keys)}", flush=True)
    ok = [k for k in keys if res[k][0] is not None]
    cw, rows = [], []
    with h5py.File(os.path.join(OUT, "stacks.h5"), "w") as h:
        for i, k in enumerate(ok):
            h.create_dataset(str(i), data=res[k][0], compression="lzf"); cw.append({"idx": i, "trial_key": k})
            r = lab[lab.key == k].iloc[0].drop("key").to_dict(); r.update(idx=i, n_slices=len(res[k][0]), combined_series=int(res[k][1])); rows.append(r)
    pd.DataFrame(cw).to_csv(os.path.join(OUT, "crosswalk_PRIVATE.csv"), index=False)
    d = pd.DataFrame(rows); d = d[["idx"] + [c for c in d.columns if c != "idx"]]; d.to_csv(os.path.join(OUT, "labels.csv"), index=False)
    print("patients", len(d), "of", len(keys), "| sites", d.site.nunique(), "| slices median", d.n_slices.median(), "| combined", int(d.combined_series.sum()),
          "| infarct % median", round(d.infarct_pct.median(), 1), "| LVEF median", round(d.lvef.median(), 1),
          "| LAD/LCX/RCA", [int(d[f'territory_{t}'].sum()) for t in ('lad', 'lcx', 'rca')], "| MVO", int(d.mvo.sum()), flush=True)


if __name__ == "__main__":
    main()
