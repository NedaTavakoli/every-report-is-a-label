#!/usr/bin/env python3
"""
LV ejection fraction from the report text by a fixed pattern (no NLP or LLM model): the first percentage within 40
characters after "LVEF", "LV ejection fraction", "left ventricular ejection fraction" or "LV EF" (fallback: "ejection
fraction" or "EF" not preceded by "right ventricular" / "RV"); ranges ("45-50%") take the
midpoint; values outside 5-90% are discarded. RV ejection fraction is not matched. Writes data/manifest/lvef.csv
(anon_acc, lvef). Prints coverage only.
"""
import os, re
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAT = re.compile(r"(?:\bLVEF\b|\bLV\s*EF\b|\bLV\s+ejection\s+fraction|left\s+ventricular\s+ejection\s+fraction)[^%]{0,40}?(\d{1,2}(?:\.\d)?)\s*(?:(?:-|–|to)\s*(\d{1,2}(?:\.\d)?))?\s*%", re.I)


FALL = re.compile(r"(?<![Rr]ight ventricular )(?<!RV )(?<!\bRV)\b(?:ejection\s+fraction|EF)\b[^%]{0,40}?(\d{1,2}(?:\.\d)?)\s*(?:(?:-|–|to)\s*(\d{1,2}(?:\.\d)?))?\s*%", re.I)


def lvef(t):
    m = PAT.search(str(t))
    if not m:      # fallback: "ejection fraction" / "EF" not preceded by right ventricular / RV (first such mention; usually the LV)
        m = FALL.search(str(t))
    if not m: return np.nan
    v = float(m.group(1)) if not m.group(2) else (float(m.group(1)) + float(m.group(2))) / 2
    return v if 5 <= v <= 90 else np.nan


e = pd.read_csv(os.path.join(ROOT, "data", "manifest", "exams.csv"), dtype=str, keep_default_na=False)
e["lvef"] = e.text.map(lvef)
e[["anon_acc", "lvef"]].to_csv(os.path.join(ROOT, "data", "manifest", "lvef.csv"), index=False)
print("exams", len(e), "| with LVEF", int(e.lvef.notna().sum()), f"({100 * e.lvef.notna().mean():.0f}%)", "| median", e.lvef.median(),
      "| IQR", e.lvef.quantile([.25, .75]).tolist(), "| <35%:", int((e.lvef < 35).sum()), "| by split", e.groupby("split").lvef.apply(lambda s: round(s.notna().mean(), 2)).to_dict())
