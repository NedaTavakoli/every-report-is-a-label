#!/usr/bin/env python3
"""
Builds the training manifest from the Exams sheet of the exam table (paths.workbook; format in docs/DATA_FORMAT.md). Only anonymized image IDs, the series
folders, the report text and the expert label are read: no MRN, no accession number and no NLP class.

Writes to data/manifest/
  exams.csv   one row per exam: anon_mrn, anon_acc, split, text, text_source, expert_label, vital_status
  series.csv  one row per LGE series: anon_acc, series, view, rel_path
Splits are by patient. Every patient with an expert-labeled exam goes to "expert" and is never used for pretraining.
"""
import os, re, argparse
import numpy as np, pandas as pd
from common import load_config

COLS = ["anon_mrn", "anon_acc", "lge_mag_folder", "lge_mag_series", "report_text", "expert_label_R1", "n_mrns", "vital_status"]
FIND = re.compile(r"\bFINDINGS\b\s*:", re.I)
SIGN = re.compile(r"\b(electronically signed|dictated by|signed by|i have personally|attending radiologist|attestation)\b.*", re.I | re.S)


def view_of(name):
    """SA, 2-, 3- or 4-chamber from a series description; other for anything else"""
    n = name.lower()
    if re.search(r"(^|[^a-z0-9])(sa|sax)([^a-z0-9]|$)|short[ _-]?axis", n): return "sa"
    for v in "234":
        if re.search(rf"(^|[^0-9a-z]){v}\s*-?\s*(c|ch|chamber)([^a-z]|$)", n): return v + "ch"
    if "lvot" in n: return "3ch"
    return "other"


def clean_report(text):
    """FINDINGS onward (FINDINGS and IMPRESSION describe the images; HISTORY and TECHNIQUE do not), without signature lines"""
    t = str(text)
    m = FIND.search(t)
    source = "findings_impression" if m else "full_report"
    if m: t = t[m.start():]
    return SIGN.sub("", t).strip(), source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lge_clip.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config); P, D = cfg["paths"], cfg["data"]
    e = pd.read_excel(P["workbook"], sheet_name="Exams", dtype=str, usecols=COLS)
    assert not {"nlp_class", "scar_type"} & set(e.columns), "NLP labels must not enter this project"
    n0 = len(e)
    e = e[e.lge_mag_folder.notna() & e.anon_acc.notna() & e.anon_mrn.notna() & e.report_text.notna()].copy()
    print(f"{len(e):,} of {n0:,} exams have LGE series and report text")
    bad = pd.to_numeric(e.n_mrns, errors="coerce").fillna(1) > 1
    e = e[~bad]; print(f"dropped {int(bad.sum())} exam(s) whose accession is listed under more than one MRN")
    cleaned = e.report_text.map(clean_report)
    e["text"], e["text_source"] = cleaned.str[0], cleaned.str[1]
    e["expert_label"] = e.expert_label_R1.fillna("").str.split(";").str[0]

    # patient split, expert patients held out
    expert = set(e[e.expert_label != ""].anon_mrn)
    rest = np.array(sorted(set(e.anon_mrn) - expert))
    np.random.default_rng(D["split"]["seed"]).shuffle(rest)
    n_tr, n_va = int(len(rest) * D["split"]["train"]), int(len(rest) * D["split"]["val"])
    split = {**{m: "train" for m in rest[:n_tr]}, **{m: "val" for m in rest[n_tr:n_tr + n_va]},
             **{m: "test" for m in rest[n_tr + n_va:]}, **{m: "expert" for m in expert}}
    e["split"] = e.anon_mrn.map(split)

    # series
    excl = re.compile(D["exclude_series_regex"], re.I)
    rows, dropped = [], {"excluded": 0, "other_view": 0}
    for acc, folder, names in zip(e.anon_acc, e.lge_mag_folder, e.lge_mag_series):
        rel = os.path.relpath(folder, P["lge_root"])
        for s in str(names).split(";"):
            s = s.strip()
            if not s: continue
            v = view_of(s)
            if excl.search(s): dropped["excluded"] += 1
            elif v == "other" or v not in D["views"]: dropped["other_view"] += 1
            else: rows.append((acc, s, v, os.path.join(rel, s)))
    s = pd.DataFrame(rows, columns=["anon_acc", "series", "view", "rel_path"])
    e = e[e.anon_acc.isin(s.anon_acc)]

    os.makedirs(P["manifest_dir"], exist_ok=True)
    e[["anon_mrn", "anon_acc", "split", "text", "text_source", "expert_label", "vital_status"]].to_csv(os.path.join(P["manifest_dir"], "exams.csv"), index=False)
    s.to_csv(os.path.join(P["manifest_dir"], "series.csv"), index=False)
    print("series kept by view:", s.view.value_counts().to_dict(), "| dropped:", dropped)
    print("exams by split:", e.split.value_counts().to_dict())
    print("patients by split:", e.groupby("split").anon_mrn.nunique().to_dict())
    print("series by split:", s.merge(e[["anon_acc", "split"]], on="anon_acc").split.value_counts().to_dict())
    print("text source:", e.text_source.value_counts().to_dict(), "| median characters:", int(e.text.str.len().median()))
    print("deceased exams by split:", e[e.vital_status == "deceased"].split.value_counts().to_dict(), "| no vital status:", int(e.vital_status.isna().sum()))
    print("expert exams by label:", e[e.expert_label != ""].expert_label.value_counts().to_dict())


if __name__ == "__main__":
    main()
