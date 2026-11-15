#!/usr/bin/env python3
"""
Scanner metadata per exam for robustness analyses: reads ONE DICOM header (no pixels) of the first LGE series of each exam
and keeps only Manufacturer, ManufacturerModelName, MagneticFieldStrength and the study year.
Writes data/manifest/scanner.csv keyed by anon_acc (no MRN, no dates beyond the year). locally (share mounted).
"""
import os, concurrent.futures as cf
import pandas as pd, pydicom
from common import load_config, ROOT


def one(acc, rel, root):
    d = os.path.join(root, rel)
    for f in sorted(os.listdir(d)):
        if f.startswith("."): continue
        try:
            h = pydicom.dcmread(os.path.join(d, f), stop_before_pixels=True,
                                specific_tags=["Manufacturer", "ManufacturerModelName", "MagneticFieldStrength", "StudyDate"])
        except Exception:
            continue
        fs = getattr(h, "MagneticFieldStrength", "")
        try: fs = round(float(fs), 1)
        except (TypeError, ValueError): fs = None
        sd = str(getattr(h, "StudyDate", "") or "")
        return {"anon_acc": acc, "vendor": str(getattr(h, "Manufacturer", "") or "").strip(),
                "model": str(getattr(h, "ManufacturerModelName", "") or "").strip(), "field_t": fs,
                "year": int(sd[:4]) if sd[:4].isdigit() else None}
    return {"anon_acc": acc}


def main():
    cfg = load_config(os.path.join(ROOT, "configs", "lge_clip.yaml")); root = cfg["paths"]["lge_root"]
    s = pd.read_csv(os.path.join(ROOT, "data", "manifest", "series.csv"), dtype=str).groupby("anon_acc").rel_path.first()
    with cf.ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(lambda a: one(a, s[a], root), s.index))
    d = pd.DataFrame(rows).drop(columns="year")     # DICOM dates are shifted by de-identification: use the EHR exam year
    c = pd.read_csv(os.path.join(ROOT, "data", "manifest", "clinical.csv"), usecols=["anon_acc", "exam_year"])
    d = d.merge(c.rename(columns={"exam_year": "year"}), on="anon_acc", how="left"); d.to_csv(os.path.join(ROOT, "data", "manifest", "scanner.csv"), index=False)
    print(len(d), "exams; vendor:", d.vendor.str.upper().str.split().str[0].value_counts().to_dict(),
          "; field:", d.field_t.value_counts().to_dict(), "; missing:", int(d.vendor.isna().sum()))


if __name__ == "__main__":
    main()
