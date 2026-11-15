#!/usr/bin/env python3
"""
Verbatim report excerpts for the figure examples (docs/figures/images/examples.json): from each examination's report
(FINDINGS onward, as used in training), the sentences that mention enhancement or scar, up to MAXC characters, in report order.
Dates and long numbers are masked as a safeguard; the author must still check the excerpts before publication.
Writes docs/figures/images/report_excerpts.json (local only; never printed). Prints only lengths.
Also writes docs/figures/images_placeholder/report_excerpts.json: same lengths, neutral filler text (layout checks).
"""
import os, re, json
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MAXC = 260
KEY = re.compile(r"enhanc|\bLGE\b|scar|fibros|infarct|gadolinium|transmural", re.I)
DATE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(19|20)\d{2}\b|\b\d{5,}\b")
ex = json.load(open(os.path.join(ROOT, "docs", "figures", "images", "examples.json")))
e = pd.read_csv(os.path.join(ROOT, "data", "manifest", "exams.csv"), dtype=str, keep_default_na=False).set_index("anon_acc")
out, ph = {}, {}
for a in ex:
    t = re.sub(r"\s+", " ", e.loc[a, "text"]).strip()
    t = re.sub(r"^\s*FINDINGS\s*:\s*", "", t, flags=re.I)
    sents = [s.strip() for s in re.split(r"(?<=[.;])\s+(?=[A-Z])", t) if s.strip()]
    pick, n = [], 0
    for s in sents:
        if KEY.search(s) and n + len(s) <= MAXC:
            pick.append(DATE.sub("[date]", s)); n += len(s) + 1
    if not pick:
        pick = [DATE.sub("[date]", sents[0])[:MAXC]] if sents else ["[no text]"]
    out[a] = " ".join(pick)
    ph[a] = ("Placeholder report excerpt of the same length " * 10)[:len(out[a])]
json.dump(out, open(os.path.join(ROOT, "docs", "figures", "images", "report_excerpts.json"), "w"), indent=1)
json.dump(ph, open(os.path.join(ROOT, "docs", "figures", "images_placeholder", "report_excerpts.json"), "w"), indent=1)
print({a: len(v) for a, v in out.items()}, "| keyword match:", {a: bool(KEY.search(v)) for a, v in out.items()})
