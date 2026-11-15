#!/usr/bin/env python3
"""
Multi-center external test against the DETERMINE CORE LAB (data/determine_full from determine_full_pack.py).
Frozen embeddings of each patient's LGE short-axis stack (same input as for our exams), for every encoder; nothing tuned on
DETERMINE (ridge alpha = 1, logistic C = 1, standardized features):
  continuous  infarct %, infarct mass, LV mass, LVEF, EDV, ESV, transmurality score -> Spearman rho of out-of-fold predictions
  binary      large infarct (> median %), infarct in LAD / LCX / RCA territory, microvascular obstruction -> AUROC
  CV: (a) random 5-fold x 10 repeats, (b) leave-sites-out GroupKFold(5) by trial site. 95% CI: bootstrap of patients (2,000).
Encoders: random | cine | trained | imgssl | dinov2_generic (DINOv2 ViT-B/14, frozen, no report training) | name=run_dir.
Embeddings cached in outputs/lge_clip/determine_full_emb_<name>.npy. Writes outputs/lge_clip/determine_full_eval.json.
"""
import os, json, argparse
import numpy as np, pandas as pd, h5py, torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, GroupKFold
from transformers import AutoModel
from common import load_config, seed_everything, device, ROOT
from data import to_frames, center
import determine_eval as DE
import bench_dinov2 as BD

CONT = ["infarct_pct", "infarct_g", "lv_mass_g", "lvef", "edv", "esv", "transmurality_score"]
BIN = ["large_infarct", "territory_lad", "territory_lcx", "territory_rca", "mvo"]


@torch.no_grad()
def embed(spec, cfg, dev, idx):
    D = cfg["data"]; out = []
    if spec == "dinov2_generic":
        net = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval(); f = lambda x: BD.enc(net, x, dev).numpy()[0]
    else:
        _, model = DE.model_for(cfg, spec, dev); amp = dev.type == "cuda"
        def f(x):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp): return model.encode_video(x.to(dev)).float().cpu().numpy()[0]
    with h5py.File(os.path.join(ROOT, "data", "determine_full", "stacks.h5"), "r") as h:
        for i in idx: out.append(f(center(to_frames(h[str(i)][()], D["frames"]), D["crop"], D["shorter_side"]).unsqueeze(0).repeat(3, 1, 1, 1)[None]))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/lge_clip.yaml")
    ap.add_argument("--weights", nargs="+", default=["random", "cine", "trained", "dinov2_generic"]); a = ap.parse_args()
    cfg = load_config(a.config if os.path.isabs(a.config) else os.path.join(ROOT, a.config)); seed_everything(42); dev = device()
    lab = pd.read_csv(os.path.join(ROOT, "data", "determine_full", "labels.csv"))
    lab["large_infarct"] = (lab.infarct_pct > lab.infarct_pct.median()).astype(float)
    site = lab.site.values; rho = lambda u, v: spearmanr(u, v)[0]
    outp = os.path.join(cfg["paths"]["out_dir"], "determine_full_eval.json")
    res = json.load(open(outp)) if os.path.exists(outp) else {}
    res["_cohort"] = {"patients": int(len(lab)), "sites": int(lab.site.nunique()), **{f"median_{c}": float(lab[c].median()) for c in CONT},
                      **{f"n_{b}": int(lab[b].sum()) for b in BIN}}
    for spec in a.weights:
        name = spec.split("=")[0]; fe = os.path.join(cfg["paths"]["out_dir"], f"determine_full_emb_{name}.npy")
        X = np.load(fe) if os.path.exists(fe) else embed(spec, cfg, dev, lab.idx.tolist())
        np.save(fe, X); torch.cuda.empty_cache(); r = {}
        for t in CONT + BIN:
            m = lab[t].notna().values; y = lab[t].values[m]; Xm, s = X[m], site[m]; reg = t in CONT
            rand = [list(KFold(5, shuffle=True, random_state=k).split(y)) for k in range(10)]; bys = [list(GroupKFold(5).split(y, groups=s))]
            f = rho if reg else roc_auc_score
            r[t] = {"n": int(m.sum()), "random_cv": DE.boot(y, DE.oof(Xm, y, rand, reg), f), "site_held_out": DE.boot(y, DE.oof(Xm, y, bys, reg), f)}
        res[name] = r; print(name, {t: (round(v["random_cv"][0], 2), round(v["site_held_out"][0], 2)) for t, v in r.items()}, flush=True)
        json.dump(res, open(outp, "w"), indent=1)
    print("saved", outp)


if __name__ == "__main__":
    main()
