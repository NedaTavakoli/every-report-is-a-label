#!/usr/bin/env python3
"""
How much does each encoder encode the scanner vendor (site signature)?
  1 vendor decodability: AUROC of a logistic probe predicting GE vs Siemens from the frozen exam embedding (grouped 5-fold CV
    by patient, fixed C = 1, standardized), for every encoder with cached exam embeddings
  2 2-D maps: t-SNE (perplexity 30, seed 0) of all exam embeddings for the report-pretrained and the same-vendor model,
    saved with vendor, LVEF and exam year for plotting (anonymous point index only; no exam IDs)
Writes outputs/lge_clip/embed_map.json.
"""
import os, json
import numpy as np, pandas as pd
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from common import ROOT

O = os.path.join(ROOT, "outputs")
SRC = {"random": f"{O}/lge_clip/emb_random.npz", "cine": f"{O}/lge_clip/emb_cine.npz", "dinov2_generic": f"{O}/lge_clip/emb_dinov2.npz",
       "imgssl": f"{O}/lge_clip/emb_imgssl.npz", "trained": f"{O}/lge_clip/emb_trained.npz",
       "samevendor": f"{O}/lge_clip_samevendor/emb_eval.npz", "dinov2clip": f"{O}/lge_clip_dinov2/emb_eval.npz"}


def main():
    sc = pd.read_csv(os.path.join(ROOT, "data", "manifest", "scanner.csv")); sc["is_ge"] = (sc.vendor.str.upper().str.split().str[0] == "GE").astype(int)
    lv = pd.read_csv(os.path.join(ROOT, "data", "manifest", "lvef.csv")); out = {"vendor_auroc": {}, "maps": {}}
    for n, f in SRC.items():
        if not os.path.exists(f): continue
        z = np.load(f, allow_pickle=True)
        d = pd.DataFrame({"anon_acc": z["acc"], "anon_mrn": z["mrn"]}).merge(sc[["anon_acc", "is_ge", "year"]], on="anon_acc", how="left").merge(lv, on="anon_acc", how="left")
        V = z["V"]; y = d.is_ge.values; p = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(V, groups=d.anon_mrn.values):
            p[te] = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(V[tr], y[tr]).decision_function(V[te])
        out["vendor_auroc"][n] = float(roc_auc_score(y, p)); print(n, "vendor AUROC", round(out["vendor_auroc"][n], 3), flush=True)
        if n in ("trained", "samevendor"):
            xy = TSNE(2, perplexity=30, random_state=0, init="pca").fit_transform(StandardScaler().fit_transform(V))
            out["maps"][n] = {"x": xy[:, 0].round(3).tolist(), "y": xy[:, 1].round(3).tolist(), "ge": y.tolist(),
                              "lvef": d.lvef.round(0).fillna(-1).tolist(), "year": d.year.fillna(-1).astype(int).tolist()}
    json.dump(out, open(os.path.join(O, "lge_clip", "embed_map.json"), "w")); print("saved")


if __name__ == "__main__":
    main()
