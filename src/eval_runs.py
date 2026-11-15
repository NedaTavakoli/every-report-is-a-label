#!/usr/bin/env python3
"""
One evaluation for any report-image pretraining run (data-scaling fractions, same-vendor batches, ...), so runs can be
compared on equal terms with the main model:
  retrieval   test split: recall@1/@5, median rank; 95% CI by patient bootstrap
  clinical    frozen image features -> clinical variables (probe_clinical.py's fit: penalty by grouped CV on the train
              split, evaluated on the test split); AUROC or R2
  shortcut    share of top-1 reports from the same vendor / within 2 years (as robustness.py)
  emidec      MI vs normal AUROC (20 x 5-fold CV, original orientation), infarct % Spearman (as emidec_eval.py)
Usage: python eval_runs.py --runs full=outputs/lge_clip frac25=outputs/lge_clip_frac25 ...   (each dir holds last.pt)
Embeddings are cached in <dir>/emb_eval.npz. Writes outputs/lge_clip/eval_runs.json (merged with earlier entries).
"""
import os, json, argparse
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, r2_score
from common import load_config, seed_everything, device, ROOT
from data import SeriesText
from models import ReportImageCLIP
from evaluate import embed
from probe_clinical import fit, predict, BINARY, CONTINUOUS, SPLITS
from robustness import ranks, boot as rboot
import emidec_eval as EE


def run_dir(p): return p if os.path.isabs(p) else os.path.join(ROOT, p)


def get_model(cfg, d, dev):
    m = ReportImageCLIP(cfg, load_vision_weights=False)
    m.load_state_dict(torch.load(os.path.join(d, "last.pt"), map_location="cpu")["model"]); return m.to(dev).eval()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/lge_clip.yaml"); ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args(); cfg = load_config(os.path.join(ROOT, a.config) if not os.path.isabs(a.config) else a.config)
    seed_everything(42); dev = device()
    e = pd.read_csv(os.path.join(ROOT, "data", "manifest", "exams.csv"), dtype=str, keep_default_na=False)[["anon_acc", "split"]]
    c = pd.read_csv(os.path.join(ROOT, "data", "manifest", "clinical.csv"))
    c["ever_smoker"] = c.smoking.map({"never": 0, "former": 1, "current": 1})      # as probe_clinical.py
    sc = pd.read_csv(os.path.join(ROOT, "data", "manifest", "scanner.csv")); sc["vendor"] = sc.vendor.str.upper().str.split().str[0]
    lab = pd.read_csv(os.path.join(ROOT, "data", "emidec", "labels.csv")); y_em = lab.mi.values
    outp = os.path.join(cfg["paths"]["out_dir"], "eval_runs.json")
    res = json.load(open(outp)) if os.path.exists(outp) else {}
    for spec in a.runs:
        name, d = spec.split("="); d = run_dir(d); model = None
        f = os.path.join(d, "emb_eval.npz")
        if not os.path.exists(f):
            model = get_model(cfg, d, dev)
            info, V, T = embed(model, SeriesText(cfg, SPLITS, False), cfg, dev)
            np.savez(f, acc=info.acc.values, mrn=info.mrn.values, V=V, T=T)
        z = np.load(f, allow_pickle=True)
        D = pd.DataFrame({"anon_acc": z["acc"], "anon_mrn": z["mrn"]}).merge(e, on="anon_acc", how="left").merge(c, on="anon_acc", how="left").merge(sc[["anon_acc", "vendor", "year"]], on="anon_acc", how="left")
        V, T = z["V"], z["T"]; tr, te = (D.split == "train").values, (D.split == "test").values
        out = {"dir": os.path.relpath(d, ROOT)}
        r, top = ranks(V[te], T[te]); out["retrieval"] = {**rboot(r, D.anon_mrn.values[te]), "median_rank": float(np.median(r) + 1), "n": int(te.sum())}
        vv, yy = D.vendor.values[te], D.year.values[te]
        out["shortcut"] = {"top1_same_vendor": float((vv[top] == vv).mean()), "top1_within_2y": float((np.abs(yy[top] - yy) <= 2).mean())}
        out["clinical"] = {}
        for t in BINARY + CONTINUOUS:
            if t not in D: continue
            binary = t in BINARY; y = D[t].values.astype(float); A, B = tr & ~np.isnan(y), te & ~np.isnan(y)
            p = predict(fit(V[A], y[A], D.anon_mrn.values[A], binary), V[B], binary)
            out["clinical"][t] = float(roc_auc_score(y[B], p) if binary else r2_score(y[B], p))
        if model is None: model = get_model(cfg, d, dev)
        X = EE.embed(model, lab.case.tolist(), cfg, dev); sc_ = EE.cv_scores(X, y_em)
        out["emidec"] = {**EE.boot_auc(y_em, sc_), "infarct_spearman": float(spearmanr(EE.infarct(X, lab.infarct_pct.values), lab.infarct_pct.values)[0])}
        del model; torch.cuda.empty_cache()
        res[name] = out; print(name, json.dumps({k: out[k] for k in ("retrieval", "shortcut", "emidec")}), flush=True)
        print("  clinical", {k: round(v, 3) for k, v in out["clinical"].items()}, flush=True)
        json.dump(res, open(outp, "w"), indent=1)
    print("saved", outp)


if __name__ == "__main__":
    main()
