#!/usr/bin/env python3
"""
Full verbatim report (FINDINGS and IMPRESSION) of the whole-examination figure's example (FIG_EXAMPLE_ACC or the first rank-1 test example).
Safeguards: dates, years and long numbers masked; physician names / signatures (Dr X, X MD, signed by ...) masked.
Writes docs/figures/images/whole_exam/report_full.json (local only, never printed) and a same-length neutral placeholder in
docs/figures/images_placeholder/whole_exam/. Prints only lengths and mask counts. The author checks the text.
"""
import os, re, json
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ACC = os.environ.get("FIG_EXAMPLE_ACC") or json.load(open(os.path.join(ROOT, "docs", "figures", "images", "examples.json")))[0]
MASKS = [(re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"), "[date]"), (re.compile(r"\b(19|20)\d{2}\b"), "[year]"),
         (re.compile(r"\b\d{5,}\b"), "[number]"),
         (re.compile(r"\b(Dr\.?|Doctor)\s+[A-Z][A-Za-z'-]+(\s+[A-Z][A-Za-z'-]+)?"), "[physician]"),
         (re.compile(r"\b[A-Z][A-Za-z'-]+(\s+[A-Z]\.)?(\s+[A-Z][A-Za-z'-]+)?,?\s+(MD|M\.D\.|DO|PhD)\b"), "[physician]"),
         (re.compile(r"(electronically\s+)?signed\s+by\b.*", re.I), "[signature]")]
t = pd.read_csv(os.path.join(ROOT, "data", "manifest", "exams.csv"), dtype=str, keep_default_na=False).set_index("anon_acc").loc[ACC, "text"]
t = re.sub(r"[ \t]+", " ", t).strip(); n = {}
for rx, rep in MASKS:
    t, k = rx.subn(rep, t); n[rep] = n.get(rep, 0) + k
m = re.split(r"\bIMPRESSION\s*:\s*", t, maxsplit=1, flags=re.I)
fnd = re.sub(r"^\s*FINDINGS\s*:\s*", "", m[0], flags=re.I).strip(); imp = m[1].strip() if len(m) > 1 else ""
fnd, imp = re.sub(r"\s*\n\s*", " ", fnd), re.sub(r"\s*\n\s*", " ", imp)
d = os.path.join(ROOT, "docs", "figures", "images", "whole_exam"); pd_ = os.path.join(ROOT, "docs", "figures", "images_placeholder", "whole_exam")
json.dump({"findings": fnd, "impression": imp}, open(os.path.join(d, "report_full.json"), "w"))
filler = lambda k: ("Placeholder report text for layout checks only. " * 60)[:k]
json.dump({"findings": filler(len(fnd)), "impression": filler(len(imp))}, open(os.path.join(pd_, "report_full.json"), "w"))
print("findings chars", len(fnd), "| impression chars", len(imp), "| masked", n)
