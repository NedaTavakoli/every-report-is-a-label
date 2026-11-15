#!/usr/bin/env python3
"""
Does the LGE image score find hidden risk among patients who look low-risk clinically?
Two out-of-fold risk scores over all non-expert exams with vital status (grouped 5-fold CV by patient; penalty by inner
grouped CV, probe_clinical.fit):
  clinical score  age, sex, examination year, BMI, diabetes, hypertension, hyperlipidemia, CKD, eGFR < 60, smoking and LVEF
                  (median imputation + missingness indicators refit within each training fold)
  image score     the frozen report-pretrained LGE image features alone (as risk_groups.py)
Each score is cut into thirds over all exams; the 3 x 3 grid gives n, deaths and all-cause mortality (95% Wilson CI) per
cell. Also: within each clinical third, AUROC of the image score. Writes outputs/lge_clip/risk_grid.json.
"""
import os, json
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import probe_clinical as PC
from risk_groups import wilson
from common import load_config, seed_everything, device, ROOT


def main():
    cfg = load_config(os.path.join(ROOT, "configs", "lge_clip.yaml")); seed_everything(42); dev = device(); P = cfg["paths"]
    e = pd.read_csv(os.path.join(P["manifest_dir"], "exams.csv"), dtype=str, keep_default_na=False)[["anon_acc", "split", "vital_status"]]
    c = pd.read_csv(os.path.join(P["manifest_dir"], "clinical.csv")).merge(pd.read_csv(os.path.join(P["manifest_dir"], "lvef.csv")), on="anon_acc", how="left")
    c["smoke_former"] = (c.smoking == "former").astype(float).where(c.smoking.notna()); c["smoke_current"] = (c.smoking == "current").astype(float).where(c.smoking.notna())
    ids, V, _ = PC.embeddings(cfg, "trained", dev, P["out_dir"])
    d = ids.merge(e, on="anon_acc", how="left").merge(c, on="anon_acc", how="left")
    m = (d.vital_status.isin(["alive", "deceased"]) & (d.split != "expert")).values
    d = d[m].reset_index(drop=True); V = V[m]; y = (d.vital_status == "deceased").astype(int).values; g = d.anon_mrn.values
    base = list(PC.CLIN); PC.CLIN = base + ["lvef"]
    s_img, s_cli = np.zeros(len(d)), np.zeros(len(d))
    for tr, te in GroupKFold(5).split(V, groups=g):
        rows = np.zeros(len(d), bool); rows[tr] = True; Xc = PC.clinical_matrix(d, rows)
        s_cli[te] = PC.predict(PC.fit(Xc[tr], y[tr], g[tr], True), Xc[te], True)
        s_img[te] = PC.predict(PC.fit(V[tr], y[tr], g[tr], True), V[te], True)
    PC.CLIN = base
    T = ["low", "middle", "high"]
    d["cli"] = pd.qcut(s_cli, 3, labels=T).astype(str); d["img"] = pd.qcut(s_img, 3, labels=T).astype(str); d["dead"] = y
    out = {"n": int(len(d)), "deaths": int(y.sum()), "auroc_clinical_score": float(roc_auc_score(y, s_cli)), "auroc_image_score": float(roc_auc_score(y, s_img)),
           "grid": {}, "image_auroc_within_clinical": {}}
    for ct in T:
        out["grid"][ct] = {}
        for it in T:
            q = d[(d.cli == ct) & (d.img == it)]
            out["grid"][ct][it] = {"n": int(len(q)), "deaths": int(q.dead.sum()), "mortality": wilson(int(q.dead.sum()), int(len(q)))}
        q = d[d.cli == ct]
        out["image_auroc_within_clinical"][ct] = float(roc_auc_score(q.dead, s_img[q.index])) if q.dead.nunique() > 1 else None
    json.dump(out, open(os.path.join(P["out_dir"], "risk_grid.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "grid"}, indent=1))
    for ct in T: print("clinical", ct, {it: (v["n"], v["deaths"], round(100 * v["mortality"][0], 1)) for it, v in out["grid"][ct].items()})


if __name__ == "__main__":
    main()
