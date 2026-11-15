#!/usr/bin/env python3
"""
External evaluation on EMIDEC (data/emidec from emidec_pack.py; another center, another country, expert labels).
Frozen image embeddings of the short-axis LGE stack (same preprocessing as our test exams: 16 frames, shorter side 256,
center crop 224), for each weight set:
  random   no pretraining            cine    cmr_c0.1 only (Shad et al.)
  trained  report-image pretraining (outputs/lge_clip/last.pt)
  imgssl   image-only pretraining, same everything but no reports (outputs/lge_imgssl/last.pt; train_imgssl.py)
Tasks (no hyperparameter is tuned on EMIDEC; probe = standardize + logistic / ridge regression, fixed C=1 / alpha=1):
  1 MI vs normal       repeated (20x) stratified 5-fold CV; AUROC of repeat-averaged out-of-fold scores, 95% CI by
                       bootstrap of cases (2,000); paired bootstrap differences vs 'trained'
  2 label efficiency   train on n labeled cases (n = 6, 10, 20, 40, 70; class-stratified), test on the rest; 200 draws;
                       mean AUROC and 2.5-97.5 percentiles
  3 infarct size       ridge on embeddings, 5-fold x 20; Spearman rho with expert infarct % (all 100 cases and MI only)
  4 zero-shot (trained only, needs the text encoder): similarity to hand-written prompts, no EMIDEC label used
Writes outputs/lge_clip/emidec_eval.json (case-level numbers only, no images).
"""
import os, json, argparse
import numpy as np, pandas as pd, h5py, torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from common import load_config, seed_everything, device, ROOT
from data import to_frames, center
from models import ReportImageCLIP

PROMPTS = {  # written by hand before any EMIDEC result; not tuned
    "mi": "Subendocardial or transmural late gadolinium enhancement in a coronary territory, consistent with myocardial infarction.",
    "normal": "No late gadolinium enhancement. No myocardial scar or fibrosis."}


def load_model(cfg, w, dev):
    P = cfg["paths"]
    m = ReportImageCLIP(cfg, load_vision_weights=w == "cine")
    if w == "trained":
        m.load_state_dict(torch.load(os.path.join(P["out_dir"], "last.pt"), map_location="cpu")["model"])
    elif w == "imgssl":
        sd = torch.load(os.path.join(ROOT, "outputs", "lge_imgssl", "last.pt"), map_location="cpu")["model"]
        sd = {k.replace(".blk.", "."): v for k, v in sd.items() if k.startswith(("vision.", "video_proj."))}
        r = m.load_state_dict(sd, strict=False)
        miss = [k for k in r.missing_keys if k.startswith(("vision.", "video_proj."))]
        if miss or r.unexpected_keys: raise RuntimeError(f"imgssl weights do not fit: {miss[:3]} {r.unexpected_keys[:3]}")
    return m.to(dev).eval()


@torch.no_grad()
def embed(model, cases, cfg, dev):
    D = cfg["data"]; amp = dev.type == "cuda"; out = []
    with h5py.File(os.path.join(ROOT, "data", "emidec", "emidec.h5"), "r") as h:
        for c in cases:
            x = center(to_frames(h[c][()], D["frames"]), D["crop"], D["shorter_side"]).unsqueeze(0).repeat(3, 1, 1, 1)[None]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out.append(model.encode_video(x.to(dev)).float().cpu().numpy()[0])
    return np.stack(out)


def clf(): return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"))


def cv_scores(X, y, reps=20):
    S = np.zeros((reps, len(y)))
    for r in range(reps):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
            S[r, te] = clf().fit(X[tr], y[tr]).decision_function(X[te])
    return S.mean(0)


def boot_auc(y, s, ref=None, B=2000, seed=0):
    rng = np.random.default_rng(seed); a, d = [], []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2: continue
        a.append(roc_auc_score(y[i], s[i]))
        if ref is not None: d.append(a[-1] - roc_auc_score(y[i], ref[i]))
    o = {"auroc": float(roc_auc_score(y, s)), "ci": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]}
    if ref is not None:
        o["diff_vs_trained"] = float(roc_auc_score(y, s) - roc_auc_score(y, ref))
        o["diff_ci"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    return o


def label_eff(X, y, ns=(6, 10, 20, 40, 70), draws=200):
    out = {}
    for n in ns:
        rng = np.random.default_rng(n); a = []
        for _ in range(draws):
            k1 = max(1, round(n * y.mean())); k0 = n - k1
            tr = np.concatenate([rng.choice(np.where(y == 1)[0], k1, False), rng.choice(np.where(y == 0)[0], k0, False)])
            te = np.setdiff1d(np.arange(len(y)), tr)
            a.append(roc_auc_score(y[te], clf().fit(X[tr], y[tr]).decision_function(X[te])))
        out[str(n)] = {"mean": float(np.mean(a)), "p2.5": float(np.percentile(a, 2.5)), "p97.5": float(np.percentile(a, 97.5))}
    return out


def infarct(X, t, reps=20):
    P = np.zeros((reps, len(t)))
    for r in range(reps):
        for tr, te in KFold(5, shuffle=True, random_state=r).split(X):
            P[r, te] = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[tr], t[tr]).predict(X[te])
    return P.mean(0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/lge_clip.yaml")
    ap.add_argument("--weights", nargs="+", default=["random", "cine", "trained", "imgssl"]); a = ap.parse_args()
    cfg = load_config(a.config); seed_everything(42); dev = device()
    lab = pd.read_csv(os.path.join(ROOT, "data", "emidec", "labels.csv")); y = lab.mi.values; t = lab.infarct_pct.values
    emb, res = {}, {"n": int(len(y)), "n_mi": int(y.sum()), "weights": {}}
    for w in a.weights:
        if w == "imgssl" and not os.path.exists(os.path.join(ROOT, "outputs", "lge_imgssl", "last.pt")):
            print("imgssl weights not found, skipped"); continue
        model = load_model(cfg, w, dev); emb[w] = embed(model, lab.case.tolist(), cfg, dev)
        if w == "trained":
            with torch.no_grad(): T = model.encode_text(list(PROMPTS.values()), dev).float().cpu().numpy()
            s = emb[w] @ T.T
            res["zero_shot_trained"] = boot_auc(y, s[:, 0] - s[:, 1])
        del model; torch.cuda.empty_cache()
    sc = {w: cv_scores(emb[w], y) for w in emb}
    for w in emb:
        pi = infarct(emb[w], t)
        res["weights"][w] = {"mi_vs_normal": boot_auc(y, sc[w], sc.get("trained") if w != "trained" else None),
                             "label_efficiency": label_eff(emb[w], y),
                             "infarct_pct_spearman_all": float(spearmanr(pi, t)[0]),
                             "infarct_pct_spearman_mi": float(spearmanr(pi[y == 1], t[y == 1])[0])}
        print(w, json.dumps(res["weights"][w]["mi_vs_normal"]), flush=True)
    path = os.path.join(cfg["paths"]["out_dir"], "emidec_eval.json")
    json.dump(res, open(path, "w"), indent=1); print(json.dumps(res, indent=1)); print("saved", path)


if __name__ == "__main__":
    main()
