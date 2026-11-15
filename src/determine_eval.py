#!/usr/bin/env python3
"""
Multi-center external test on DETERMINE (data/determine from determine_pack.py; 353 patients with prior infarction,
43 trial sites, expert scar contours). Frozen embeddings of each patient's short-axis LGE stack (16 frames, center crop 224)
for every encoder; nothing is tuned on DETERMINE (fixed alpha = 1 / C = 1):
  infarct size   ridge regression of expert infarct % (scar / LV myocardium); Spearman rho of out-of-fold predictions
  large infarct  infarct % above the cohort median vs below; AUROC of out-of-fold logistic scores
  Two cross-validation schemes: (a) random 5-fold x 20 repeats; (b) leave-sites-out: GroupKFold(5) by trial site, so every
  prediction is for a site the probe never saw. 95% CIs by bootstrap of patients (2,000).
Weights: random, cine, trained (outputs/lge_clip), imgssl (outputs/lge_imgssl) and any run dir: name=dir.
Writes outputs/lge_clip/determine_eval.json (merged with earlier entries).
"""
import os, json, argparse
import numpy as np, pandas as pd, h5py, torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from common import load_config, seed_everything, device, ROOT
from data import to_frames, center
from models import ReportImageCLIP
import emidec_eval as EE


def model_for(cfg, spec, dev):
    if "=" in spec:
        name, d = spec.split("="); d = d if os.path.isabs(d) else os.path.join(ROOT, d)
        m = ReportImageCLIP(cfg, load_vision_weights=False)
        m.load_state_dict(torch.load(os.path.join(d, "last.pt"), map_location="cpu")["model"]); return name, m.to(dev).eval()
    return spec, EE.load_model(cfg, spec, dev)


@torch.no_grad()
def embed(model, pids, cfg, dev):
    D = cfg["data"]; amp = dev.type == "cuda"; out = []
    with h5py.File(os.path.join(ROOT, "data", "determine", "determine_stacks.h5"), "r") as h:
        for p in pids:
            x = center(to_frames(h[str(p)][()], D["frames"]), D["crop"], D["shorter_side"]).unsqueeze(0).repeat(3, 1, 1, 1)[None]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out.append(model.encode_video(x.to(dev)).float().cpu().numpy()[0])
    return np.stack(out)


def oof(X, y, splits, reg):
    P = np.zeros((len(splits), len(y)))
    for r, sp in enumerate(splits):
        for tr, te in sp:
            m = make_pipeline(StandardScaler(), Ridge(alpha=1.0) if reg else LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"))
            m.fit(X[tr], y[tr]); P[r, te] = m.predict(X[te]) if reg else m.decision_function(X[te])
    return P.mean(0)


def boot(y, s, f, B=2000, seed=0):
    rng = np.random.default_rng(seed); v = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) > 1: v.append(f(y[i], s[i]))
    return [float(f(y, s)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/lge_clip.yaml")
    ap.add_argument("--weights", nargs="+", default=["random", "cine", "trained"]); a = ap.parse_args()
    cfg = load_config(a.config if os.path.isabs(a.config) else os.path.join(ROOT, a.config)); seed_everything(42); dev = device()
    lab = pd.read_csv(os.path.join(ROOT, "data", "determine", "determine_labels.csv"))
    y = lab.infarct_pct.values; big = (y > np.median(y)).astype(int); site = lab.site.values
    rand = [list(KFold(5, shuffle=True, random_state=r).split(y)) for r in range(20)]
    bysite = [list(GroupKFold(5).split(y, groups=site))]
    rho = lambda a, b: spearmanr(a, b)[0]
    outp = os.path.join(cfg["paths"]["out_dir"], "determine_eval.json")
    res = json.load(open(outp)) if os.path.exists(outp) else {}
    res["_cohort"] = {"patients": int(len(y)), "sites": int(lab.site.nunique()), "infarct_pct_median": float(np.median(y)),
                      "infarct_pct_iqr": [float(np.percentile(y, 25)), float(np.percentile(y, 75))], "no_scar": int((lab.scar_any == 0).sum())}
    for spec in a.weights:
        name, model = model_for(cfg, spec, dev); X = embed(model, lab.pid.tolist(), cfg, dev); del model; torch.cuda.empty_cache()
        r = {}
        for cv, sp in (("random_cv", rand), ("site_held_out", bysite)):
            r[cv] = {"infarct_spearman": boot(y, oof(X, y, sp, True), rho), "large_infarct_auroc": boot(big, oof(X, big, sp, False), roc_auc_score)}
        res[name] = r; print(name, json.dumps(r), flush=True)
        json.dump(res, open(outp, "w"), indent=1)
    print("saved", outp)


if __name__ == "__main__":
    main()
