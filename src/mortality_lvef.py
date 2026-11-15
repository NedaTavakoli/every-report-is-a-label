#!/usr/bin/env python3
"""
Vital status beyond LVEF: does the LGE image embedding add to a clinical model that already includes the LV ejection
fraction from the report (data/manifest/lvef.csv, extract_lvef.py)? Same design as probe_clinical.py's mortality analysis
(grouped 5-fold CV over all non-expert exams, imputation refit per fold, penalty by inner grouped CV; and train -> test),
with models: clinical, clinical+LVEF, clinical+LVEF+image (report-pretrained / cine / random / DINOv2 / image-only / same-vendor / DINOv2-backbone CLIP);
also clinical-variable probes for each image feature set, clinical+LVEF+report.
Differences vs clinical+LVEF by patient bootstrap. Writes outputs/lge_clip/mortality_lvef.json.
"""
import os, json
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import probe_clinical as PC
from common import load_config, seed_everything, device, ROOT


def main():
    cfg = load_config(os.path.join(ROOT, "configs", "lge_clip.yaml")); seed_everything(42); dev = device(); P = cfg["paths"]
    rng = np.random.default_rng(42)
    e = pd.read_csv(os.path.join(P["manifest_dir"], "exams.csv"), dtype=str, keep_default_na=False)[["anon_acc", "split", "vital_status"]]
    c = pd.read_csv(os.path.join(P["manifest_dir"], "clinical.csv")).merge(pd.read_csv(os.path.join(P["manifest_dir"], "lvef.csv")), on="anon_acc", how="left")
    c["smoke_former"] = (c.smoking == "former").astype(float).where(c.smoking.notna())
    c["smoke_current"] = (c.smoking == "current").astype(float).where(c.smoking.notna())
    c["deceased"] = e.set_index("anon_acc").vital_status.reindex(c.anon_acc).map({"alive": 0, "deceased": 1}).values
    emb = {w: PC.embeddings(cfg, w, dev, P["out_dir"]) for w in ["random", "cine", "trained"]}
    ids = emb["trained"][0]
    d = ids.merge(e, on="anon_acc", how="left").merge(c, on="anon_acc", how="left")
    m = d.deceased.notna().values & (d.split != "expert").values
    y = d.deceased.values.astype(float); g = d.anon_mrn.values; tr, te = (d.split == "train").values, (d.split == "test").values
    base = list(PC.CLIN)

    def cm(rows, lv):
        PC.CLIN = base + (["lvef"] if lv else []); X = PC.clinical_matrix(d, rows); PC.CLIN = base; return X

    img = {"image": emb["trained"][1], "image_cine": emb["cine"][1], "image_random": emb["random"][1], "report": emb["trained"][2]}
    if os.path.exists(os.path.join(ROOT, "outputs", "lge_imgssl", "last.pt")):      # image-only control
        e2 = PC.embeddings(cfg, "imgssl", dev, P["out_dir"]); pos = {a: i for i, a in enumerate(e2[0].anon_acc)}
        img["image_imgssl"] = e2[1][[pos[a] for a in d.anon_acc]]
    for nm, rd in (("samevendor", "outputs/lge_clip_samevendor"), ("dinov2clip", "outputs/lge_clip_dinov2")):   # eval_runs.py embeddings
        fe = os.path.join(ROOT, rd, "emb_eval.npz")
        if os.path.exists(fe):
            ze = np.load(fe, allow_pickle=True); pos = {a: i for i, a in enumerate(ze["acc"])}; img[f"image_{nm}"] = ze["V"][[pos[a] for a in d.anon_acc]]
    fd = os.path.join(P["out_dir"], "emb_dinov2.npz")      # general vision foundation model benchmark (bench_dinov2.py)
    if os.path.exists(fd):
        zd = np.load(fd, allow_pickle=True); pos = {a: i for i, a in enumerate(zd["acc"])}
        img["image_dinov2"] = zd["V"][[pos[a] for a in d.anon_acc]]
    def sets(rows):
        X0, X1 = cm(rows, False), cm(rows, True)
        return {"clinical": X0, "clinical+lvef": X1, **{f"clinical+lvef+{k}": np.hstack([X1, v]) for k, v in img.items()}}
    names = list(sets(tr & m))
    out = {"lvef_coverage": float(d.lvef[m].notna().mean()), "lvef_median": float(d.lvef[m].median())}
    S = sets(tr & m); a, b = tr & m, te & m
    pr = [PC.predict(PC.fit(S[k][a], y[a], g[a], True), S[k][b], True) for k in names]
    r = PC.boot(y[b], [pr[1]] + [p for i, p in enumerate(pr) if i != 1], g[b], roc_auc_score, rng)
    order = [names[1]] + [k for i, k in enumerate(names) if i != 1]
    out.update(n_test=int(b.sum()), events_test=int(y[b].sum()), test=dict(zip(order, r)))
    idx = np.where(m)[0]; oof = {k: np.zeros(len(idx)) for k in names}
    for f_tr, f_te in GroupKFold(5).split(idx, groups=g[idx]):
        rows = np.zeros(len(d), bool); rows[idx[f_tr]] = True; Sf = sets(rows)
        for k in names: oof[k][f_te] = PC.predict(PC.fit(Sf[k][idx[f_tr]], y[idx[f_tr]], g[idx[f_tr]], True), Sf[k][idx[f_te]], True)
    r = PC.boot(y[idx], [oof[k] for k in order], g[idx], roc_auc_score, rng)
    out.update(n_cv=int(len(idx)), events_cv=int(y[idx].sum()), cv=dict(zip(order, r)), reference="clinical+lvef (differences are vs this model)")
    # paired comparisons against the report-pretrained image features (same folds, same bootstrap samples)
    o2 = ["clinical+lvef+image"] + [k for k in order if k != "clinical+lvef+image"]
    out["cv_vs_report_image"] = dict(zip(o2, PC.boot(y[idx], [oof[k] for k in o2], g[idx], roc_auc_score, np.random.default_rng(7))))
    np.savez(os.path.join(P["out_dir"], "mortality_lvef_oof.npz"), y=y[idx], g=g[idx], **{k.replace("+", "_"): v for k, v in oof.items()})
    print("cv_vs_report_image", {k: (round(v["estimate"], 3), [round(x, 3) for x in v.get("diff_ci95", [])]) for k, v in out["cv_vs_report_image"].items()}, flush=True)
    # clinical-variable probes (train -> test) for every image feature set, as eval_runs.py
    from sklearn.metrics import r2_score
    d["ever_smoker"] = d.smoking.map({"never": 0, "former": 1, "current": 1}); out["clinical_probes"] = {}
    for k, V in img.items():
        if k == "report": continue
        out["clinical_probes"][k] = {}
        for t in PC.BINARY + PC.CONTINUOUS:
            binary = t in PC.BINARY; yy = d[t].values.astype(float); A, Bm = tr & ~np.isnan(yy), te & ~np.isnan(yy)
            pp = PC.predict(PC.fit(V[A], yy[A], g[A], True if binary else False), V[Bm], binary)
            out["clinical_probes"][k][t] = float(roc_auc_score(yy[Bm], pp) if binary else r2_score(yy[Bm], pp))
        print("clinical", k, {t: round(v, 3) for t, v in out["clinical_probes"][k].items()}, flush=True)
    json.dump(out, open(os.path.join(P["out_dir"], "mortality_lvef.json"), "w"), indent=1)
    for part in ("test", "cv"):
        print(part, {k: (round(v["estimate"], 3), [round(x, 3) for x in v.get("diff_ci95", [])]) for k, v in out[part].items()}, flush=True)


if __name__ == "__main__":
    main()
