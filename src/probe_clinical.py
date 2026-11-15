#!/usr/bin/env python3
"""
Clinical targets and mortality from frozen embeddings, for each image-encoder weight set.

1. Embeds every exam once per weight set (random, cine, trained): image = mean of series embeddings; report = trained
   text encoder (full text). Cached in outputs/.../emb_<weights>.npz (anonymized exam IDs and vectors only).
2. Biomarker probes: train split -> test split. Logistic (binary) or ridge (continuous) regression on standardized
   embeddings; the penalty is chosen by 5-fold cross-validation on the training split, grouped by patient.
3. Mortality (vital status at extraction; no follow-up time): clinical model (age, sex, exam year, BMI, comorbidities,
   eGFR < 60, smoking) alone and with image and/or report embeddings. Test AUROC with patient-level bootstrap CIs,
   and the paired difference against the clinical model. A grouped 5-fold cross-validation over all non-expert exams is
   a sensitivity analysis with more events.
Nothing here selects the pretrained model. Writes outputs/.../clinical_probes.json and mortality.json.
"""
import os, json, argparse
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error
from common import load_config, seed_everything, device
from data import SeriesText
from models import ReportImageCLIP
from evaluate import embed

SPLITS = ["train", "val", "test", "expert"]
BINARY = ["male", "diabetes", "ckd", "egfr_lt60", "ever_smoker"]
CONTINUOUS = ["age", "bmi", "ldl", "hdl"]
CLIN = ["age", "male", "exam_year", "bmi", "diabetes", "hypertension", "hyperlipidemia", "ckd", "egfr_lt60", "smoke_former", "smoke_current"]
B = 2000


def embeddings(cfg, w, dev, out_dir):
    f = os.path.join(out_dir, f"emb_{w}.npz")
    if not os.path.exists(f):
        if w == "imgssl":      # image-only control (train_imgssl.py); emidec_eval.load_model strips the checkpointing wrapper
            import emidec_eval; model = emidec_eval.load_model(cfg, "imgssl", dev)
        else:
            model = ReportImageCLIP(cfg, load_vision_weights=w == "cine").to(dev).eval()
        if w == "trained":
            model.load_state_dict(torch.load(os.path.join(out_dir, "last.pt"), map_location="cpu")["model"])
        info, V, T = embed(model, SeriesText(cfg, SPLITS, False), cfg, dev)
        np.savez(f, acc=info.acc.values, mrn=info.mrn.values, V=V, T=T)
        del model; torch.cuda.empty_cache()
    z = np.load(f, allow_pickle=True)
    return pd.DataFrame({"anon_acc": z["acc"], "anon_mrn": z["mrn"]}), z["V"], z["T"]


def fit(X, y, g, binary):
    """penalty from grouped 5-fold CV on the training data only"""
    if binary:
        est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
        grid, score = {"logisticregression__C": [1e-3, 1e-2, 1e-1, 1, 10]}, "roc_auc"
    else:
        est = make_pipeline(StandardScaler(), Ridge())
        grid, score = {"ridge__alpha": [1, 10, 100, 1000, 10000]}, "r2"
    return GridSearchCV(est, grid, cv=GroupKFold(5), scoring=score).fit(X, y, groups=g).best_estimator_


def predict(m, X, binary): return m.predict_proba(X)[:, 1] if binary else m.predict(X)


def boot(y, preds, g, metric, rng):
    """patient-level bootstrap: point estimate and 95% CI per prediction, and of each difference from the first"""
    pats = np.unique(g); idx = {p: np.where(g == p)[0] for p in pats}
    pt = [metric(y, p) for p in preds]; bs = []
    for _ in range(B):
        i = np.concatenate([idx[p] for p in rng.choice(pats, len(pats))])
        if len(np.unique(y[i])) < 2 and metric is roc_auc_score: continue
        bs.append([metric(y[i], p[i]) for p in preds])
    bs = np.array(bs); ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return [{"estimate": float(pt[k]), "ci95": ci(bs[:, k]),
             **({} if k == 0 else {"diff_vs_first": float(pt[k] - pt[0]), "diff_ci95": ci(bs[:, k] - bs[:, 0])})} for k in range(len(preds))]


def clinical_matrix(c, fit_on):
    """median imputation (from the training rows) plus a missingness indicator per variable"""
    X = c[CLIN].astype(float).copy(); med = X[fit_on].median()
    miss = X.isna().astype(float).add_suffix("_missing"); miss = miss.loc[:, miss[fit_on].sum() > 0]
    return pd.concat([X.fillna(med), miss], axis=1).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lge_clip.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config); seed_everything(42); dev = device(); P = cfg["paths"]; rng = np.random.default_rng(42)
    e = pd.read_csv(os.path.join(P["manifest_dir"], "exams.csv"), dtype=str, keep_default_na=False)[["anon_acc", "split", "vital_status"]]
    c = pd.read_csv(os.path.join(P["manifest_dir"], "clinical.csv"))
    c["ever_smoker"] = c.smoking.map({"never": 0, "former": 1, "current": 1})
    c["smoke_former"] = (c.smoking == "former").astype(float).where(c.smoking.notna())
    c["smoke_current"] = (c.smoking == "current").astype(float).where(c.smoking.notna())
    c["deceased"] = e.set_index("anon_acc").vital_status.reindex(c.anon_acc).map({"alive": 0, "deceased": 1}).values

    emb = {w: embeddings(cfg, w, dev, P["out_dir"]) for w in ["random", "cine", "trained"]}
    ids = emb["trained"][0]
    for w in emb: assert (emb[w][0].anon_acc.values == ids.anon_acc.values).all()
    d = ids.merge(e, on="anon_acc", how="left").merge(c, on="anon_acc", how="left")
    tr, te = (d.split == "train").values, (d.split == "test").values
    feats = {f"image_{w}": emb[w][1] for w in emb}; feats["report_trained"] = emb["trained"][2]

    # ---- biomarker probes
    res = {}
    for t in BINARY + CONTINUOUS:
        binary = t in BINARY; y = d[t].values.astype(float); a, b = tr & ~np.isnan(y), te & ~np.isnan(y)
        preds = {k: predict(fit(X[a], y[a], d.anon_mrn.values[a], binary), X[b], binary) for k, X in feats.items()}
        metric = roc_auc_score if binary else r2_score
        names = list(preds); r = boot(y[b], [preds[k] for k in names], d.anon_mrn.values[b], metric, rng)
        res[t] = {"metric": "AUROC" if binary else "R2", "n_train": int(a.sum()), "n_test": int(b.sum()),
                  **({"prevalence_test": float(y[b].mean())} if binary else {"mae": {k: float(mean_absolute_error(y[b], preds[k])) for k in names}}),
                  **{k: v for k, v in zip(names, r)}}
        print(t, {k: round(v["estimate"], 3) for k, v in zip(names, r)}, flush=True)
    json.dump(res, open(os.path.join(P["out_dir"], "clinical_probes.json"), "w"), indent=1)

    # ---- mortality
    m = d.deceased.notna().values & (d.split != "expert").values
    y = d.deceased.values.astype(float); g = d.anon_mrn.values
    Xc = clinical_matrix(d, tr & m)
    sets = {"clinical": Xc, "image": feats["image_trained"], "report": feats["report_trained"],
            "clinical+image": np.hstack([Xc, feats["image_trained"]]), "clinical+report": np.hstack([Xc, feats["report_trained"]]),
            "clinical+image+report": np.hstack([Xc, feats["image_trained"], feats["report_trained"]]),
            "clinical+image_cine": np.hstack([Xc, feats["image_cine"]]), "clinical+image_random": np.hstack([Xc, feats["image_random"]])}
    a, b = tr & m, te & m
    pr = {k: predict(fit(X[a], y[a], g[a], True), X[b], True) for k, X in sets.items()}
    names = list(pr); r = boot(y[b], [pr[k] for k in names], g[b], roc_auc_score, rng)
    out = {"definition": "vital status at extraction; no follow-up time", "n_train": int(a.sum()), "events_train": int(y[a].sum()),
           "n_test": int(b.sum()), "events_test": int(y[b].sum()), "test": dict(zip(names, r))}
    # sensitivity: grouped 5-fold CV over all non-expert exams (the clinical imputation is refit in each fold)
    idx = np.where(m)[0]; oof = {k: np.zeros(len(idx)) for k in sets}
    for f_tr, f_te in GroupKFold(5).split(idx, groups=g[idx]):
        fit_rows = np.zeros(len(d), bool); fit_rows[idx[f_tr]] = True
        Xc_f = clinical_matrix(d, fit_rows)
        S = {k: (Xc_f if k == "clinical" else np.hstack([Xc_f, *[X[:, Xc.shape[1]:]]]) if k.startswith("clinical+") else X) for k, X in sets.items()}
        for k, X in S.items():
            oof[k][f_te] = predict(fit(X[idx[f_tr]], y[idx[f_tr]], g[idx[f_tr]], True), X[idx[f_te]], True)
    r = boot(y[idx], [oof[k] for k in names], g[idx], roc_auc_score, rng)
    out.update(n_cv=int(len(idx)), events_cv=int(y[idx].sum()), cv=dict(zip(names, r)))
    json.dump(out, open(os.path.join(P["out_dir"], "mortality.json"), "w"), indent=1)
    for part in ["test", "cv"]:
        print(part, {k: round(v["estimate"], 3) for k, v in out[part].items()})


if __name__ == "__main__":
    main()
