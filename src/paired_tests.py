#!/usr/bin/env python3
"""
Paired comparisons between encoders on the external tests (same cases, same folds, same bootstrap samples):
  EMIDEC     MI vs normal AUROC (as emidec_eval.py: 20 x 5-fold CV, averaged out-of-fold scores)
  DETERMINE  core lab (as determine_full_eval.py; cached embeddings): infarct %, LVEF, ESV (Spearman) and large infarct (AUROC),
             random CV and site-held-out CV
Comparisons: report-pretrained ("trained") vs image-only ("imgssl"), and same-vendor batches ("samevendor") vs "trained".
Difference = A - B with a 95% CI from 2,000 paired bootstrap samples of cases. Writes outputs/lge_clip/paired_tests.json.
"""
import os, json
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, GroupKFold
from common import load_config, seed_everything, device, ROOT
import emidec_eval as EE
import determine_eval as DE

PAIRS = [("trained", "imgssl"), ("samevendor", "trained"), ("trained", "dinov2_generic")]
SPEC = {"trained": "trained", "imgssl": "imgssl", "samevendor": "samevendor=outputs/lge_clip_samevendor"}


def paired(y, a, b, f, B=2000, seed=0):
    rng = np.random.default_rng(seed); d = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) > 1: d.append(f(y[i], a[i]) - f(y[i], b[i]))
    return {"a": float(f(y, a)), "b": float(f(y, b)), "diff": float(f(y, a) - f(y, b)),
            "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_two_sided": float(min(1, 2 * min(np.mean(np.array(d) <= 0), np.mean(np.array(d) >= 0))))}


def main():
    cfg = load_config(os.path.join(ROOT, "configs", "lge_clip.yaml")); seed_everything(42); dev = device(); out = {}
    # ---- EMIDEC
    lab = pd.read_csv(os.path.join(ROOT, "data", "emidec", "labels.csv")); y = lab.mi.values; S = {}
    for n, sp in SPEC.items():
        _, m = DE.model_for(cfg, sp, dev); S[n] = EE.cv_scores(EE.embed(m, lab.case.tolist(), cfg, dev), y); del m; torch.cuda.empty_cache()
    out["emidec_mi_auroc"] = {f"{a}_vs_{b}": paired(y, S[a], S[b], roc_auc_score) for a, b in PAIRS if a in S and b in S}
    print("emidec", json.dumps(out["emidec_mi_auroc"]), flush=True)
    # ---- DETERMINE core lab (cached embeddings from determine_full_eval.py)
    dl = pd.read_csv(os.path.join(ROOT, "data", "determine_full", "labels.csv")); dl["large_infarct"] = (dl.infarct_pct > dl.infarct_pct.median()).astype(float)
    X = {n: np.load(os.path.join(cfg["paths"]["out_dir"], f"determine_full_emb_{n}.npy")) for n in ["trained", "imgssl", "samevendor", "dinov2_generic"]}
    rho = lambda u, v: spearmanr(u, v)[0]; out["determine"] = {}
    for t in ["infarct_pct", "lvef", "esv", "large_infarct"]:
        m = dl[t].notna().values; yy = dl[t].values[m]; reg = t != "large_infarct"; f = rho if reg else roc_auc_score
        for cv, sp in (("random_cv", [list(KFold(5, shuffle=True, random_state=k).split(yy)) for k in range(10)]),
                       ("site_held_out", [list(GroupKFold(5).split(yy, groups=dl.site.values[m]))])):
            P = {n: DE.oof(X[n][m], yy, sp, reg) for n in X}
            out["determine"][f"{t}|{cv}"] = {f"{a}_vs_{b}": paired(yy, P[a], P[b], f) for a, b in PAIRS}
            print(t, cv, {k: (round(v["diff"], 3), [round(x, 3) for x in v["ci"]]) for k, v in out["determine"][f"{t}|{cv}"].items()}, flush=True)
    json.dump(out, open(os.path.join(cfg["paths"]["out_dir"], "paired_tests.json"), "w"), indent=1); print("saved")


if __name__ == "__main__":
    main()
