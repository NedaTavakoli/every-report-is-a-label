#!/usr/bin/env python3
"""
Clinically readable vital-status analyses (all-cause death at data extraction; no follow-up time):
  image risk score  out-of-fold probability from a logistic model on the report-pretrained image features alone
                    (grouped 5-fold CV over all non-expert exams, penalty by inner grouped CV; probe_clinical.fit)
  1 tertiles        death rate (95% Wilson CI) in low / middle / high image-risk tertiles, overall and within LVEF strata
                    (<= 35%, 36-50%, > 50%; LVEF from the report, extract_lvef.py)
  2 adjusted OR     logistic regression of death on the standardized image score + age, sex, exam year, BMI, diabetes,
                    hypertension, hyperlipidemia, CKD, eGFR < 60, smoking and LVEF (median-imputed); OR per SD with 95% CI
                    by bootstrap of patients (1,000); overall and in LVEF > 35%
  3 discrimination  AUROC of the image score within each LVEF stratum, and of clinical+LVEF vs clinical+LVEF+image within
                    LVEF > 35% (out-of-fold predictions from mortality_lvef.py, mortality_lvef_oof.npz)
Writes outputs/lge_clip/risk_groups.json.
"""
import os, json
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import probe_clinical as PC
from common import load_config, seed_everything, device, ROOT

COV = ["age", "male", "exam_year", "bmi", "diabetes", "hypertension", "hyperlipidemia", "ckd", "egfr_lt60", "smoke_former", "smoke_current", "lvef"]


def wilson(k, n, z=1.96):
    if n == 0: return [np.nan, np.nan, np.nan]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(p), float(c - h), float(c + h)]


def adj_or(df, rng, B=1000):
    X = df[COV].astype(float); X = X.fillna(X.median()); X = (X - X.mean()) / X.std().replace(0, 1)
    s = (df.score - df.score.mean()) / df.score.std(); A = np.column_stack([s.values, X.values]); y = df.dead.values
    fit = lambda i: LogisticRegression(C=1e6, max_iter=5000).fit(A[i], y[i]).coef_[0][0]
    est = fit(np.arange(len(y))); pats = df.anon_mrn.values; up = np.unique(pats); idx = {p: np.where(pats == p)[0] for p in up}; bs = []
    for _ in range(B):
        i = np.concatenate([idx[p] for p in rng.choice(up, len(up))])
        if y[i].sum() > 5: bs.append(fit(i))
    return {"or_per_sd": float(np.exp(est)), "ci95": [float(np.exp(np.percentile(bs, 2.5))), float(np.exp(np.percentile(bs, 97.5)))], "n": int(len(y)), "deaths": int(y.sum())}


def main():
    cfg = load_config(os.path.join(ROOT, "configs", "lge_clip.yaml")); seed_everything(42); dev = device(); P = cfg["paths"]; rng = np.random.default_rng(42)
    e = pd.read_csv(os.path.join(P["manifest_dir"], "exams.csv"), dtype=str, keep_default_na=False)[["anon_acc", "split", "vital_status"]]
    c = pd.read_csv(os.path.join(P["manifest_dir"], "clinical.csv")).merge(pd.read_csv(os.path.join(P["manifest_dir"], "lvef.csv")), on="anon_acc", how="left")
    c["smoke_former"] = (c.smoking == "former").astype(float).where(c.smoking.notna()); c["smoke_current"] = (c.smoking == "current").astype(float).where(c.smoking.notna())
    ids, V, _ = PC.embeddings(cfg, "trained", dev, P["out_dir"])
    d = ids.merge(e, on="anon_acc", how="left").merge(c, on="anon_acc", how="left")
    m = (d.vital_status.isin(["alive", "deceased"]) & (d.split != "expert")).values
    d = d[m].reset_index(drop=True); V = V[m]; d["dead"] = (d.vital_status == "deceased").astype(int); g = d.anon_mrn.values; y = d.dead.values
    oof = np.zeros(len(d))
    for tr, te in GroupKFold(5).split(V, groups=g):
        oof[te] = PC.predict(PC.fit(V[tr], y[tr], g[tr], True), V[te], True)
    d["score"] = oof; d["tertile"] = pd.qcut(d.score, 3, labels=["low", "middle", "high"]).astype(str)
    d["lvef_group"] = pd.cut(d.lvef, [0, 35, 50, 100], labels=["<=35", "36-50", ">50"]).astype(str)
    out = {"n": int(len(d)), "deaths": int(y.sum()), "lvef_missing": int(d.lvef.isna().sum()), "auroc_image_score_all": float(roc_auc_score(y, oof))}
    out["tertiles_all"] = {t: {"n": int((d.tertile == t).sum()), "death_rate": wilson(int(d.dead[d.tertile == t].sum()), int((d.tertile == t).sum()))} for t in ["low", "middle", "high"]}
    out["by_lvef"] = {}
    for lg in ["<=35", "36-50", ">50"]:
        s = d[d.lvef_group == lg]
        out["by_lvef"][lg] = {"n": int(len(s)), "deaths": int(s.dead.sum()), "auroc_image_score": float(roc_auc_score(s.dead, s.score)) if s.dead.nunique() > 1 else None,
                              "tertiles": {t: {"n": int((s.tertile == t).sum()), "death_rate": wilson(int(s.dead[s.tertile == t].sum()), int((s.tertile == t).sum()))} for t in ["low", "middle", "high"]}}
    out["adjusted_or_all"] = adj_or(d, rng); out["adjusted_or_lvef_gt35"] = adj_or(d[d.lvef > 35].reset_index(drop=True), rng)
    fo = os.path.join(P["out_dir"], "mortality_lvef_oof.npz")
    if os.path.exists(fo):     # same exam order as mortality_lvef.py (non-expert exams with vital status, emb_trained order)
        z = np.load(fo); lv = d.lvef.values; sel = lv > 35
        if len(z["y"]) == len(d) and (z["y"] == y).all():
            out["lvef_gt35_auroc"] = {k: float(roc_auc_score(y[sel], z[k][sel])) for k in ["clinical_lvef", "clinical_lvef_image"] if k in z}
    json.dump(out, open(os.path.join(P["out_dir"], "risk_groups.json"), "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
