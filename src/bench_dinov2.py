#!/usr/bin/env python3
"""
Benchmark: a general-purpose vision foundation model (DINOv2 ViT-B/14, facebook/dinov2-base; natural images, no medical or
report training) as a frozen feature extractor on the same tasks as our encoders. Each slice (the same 16 frames and
224-px center crop as our encoder's input) -> CLS embedding; mean over slices, then over an exam's series.
Tasks (identical code paths and settings to eval_runs.py / emidec_eval.py / determine_eval.py):
  clinical variables (train -> test), EMIDEC MI vs normal + infarct size, DETERMINE infarct size + large infarct (random and
  site-held-out CV). Report retrieval is not possible (no text tower). Writes outputs/lge_clip/bench_dinov2.json.
"""
import os, json
import numpy as np, pandas as pd, h5py, torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import KFold, GroupKFold
from transformers import AutoModel
from common import load_config, seed_everything, device, ROOT
from data import SeriesText, to_frames, center
from probe_clinical import fit, predict, BINARY, CONTINUOUS, SPLITS
import emidec_eval as EE
import determine_eval as DE

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def enc(net, video, dev):
    """video: (N, 3, F, H, W) in [0, 1] -> (N, D): mean of per-frame CLS tokens"""
    n, c, f, h, w = video.shape
    x = video.permute(0, 2, 1, 3, 4).reshape(n * f, c, h, w)
    x = ((x - MEAN) / STD).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        z = net(pixel_values=x).last_hidden_state[:, 0]
    return z.float().reshape(n, f, -1).mean(1).cpu()


def stack_video(vol, D):
    return center(to_frames(vol, D["frames"]), D["crop"], D["shorter_side"]).unsqueeze(0).repeat(3, 1, 1, 1)[None]


def main():
    cfg = load_config(os.path.join(ROOT, "configs", "lge_clip.yaml")); seed_everything(42); dev = device(); D = cfg["data"]
    net = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()
    out = {"model": "facebook/dinov2-base"}
    # ---- our cohort: exam embeddings
    f = os.path.join(cfg["paths"]["out_dir"], "emb_dinov2.npz")
    if not os.path.exists(f):
        ex = {}
        for b in DataLoader(SeriesText(cfg, SPLITS, False), batch_size=16, num_workers=D["num_workers"]):
            z = enc(net, b["video"], dev)
            for a, m, v in zip(b["anon_acc"], b["anon_mrn"], z.numpy()): ex.setdefault(a, [m, []])[1].append(v)
        keys = list(ex); np.savez(f, acc=np.array(keys), mrn=np.array([ex[k][0] for k in keys]), V=np.stack([np.mean(ex[k][1], 0) for k in keys]))
    z = np.load(f, allow_pickle=True)
    e = pd.read_csv(os.path.join(ROOT, "data", "manifest", "exams.csv"), dtype=str, keep_default_na=False)[["anon_acc", "split"]]
    c = pd.read_csv(os.path.join(ROOT, "data", "manifest", "clinical.csv")); c["ever_smoker"] = c.smoking.map({"never": 0, "former": 1, "current": 1})
    d = pd.DataFrame({"anon_acc": z["acc"], "anon_mrn": z["mrn"]}).merge(e, on="anon_acc", how="left").merge(c, on="anon_acc", how="left")
    V = z["V"]; tr, te = (d.split == "train").values, (d.split == "test").values; out["clinical"] = {}
    for t in BINARY + CONTINUOUS:
        binary = t in BINARY; y = d[t].values.astype(float); A, B = tr & ~np.isnan(y), te & ~np.isnan(y)
        p = predict(fit(V[A], y[A], d.anon_mrn.values[A], binary), V[B], binary)
        out["clinical"][t] = float(roc_auc_score(y[B], p) if binary else r2_score(y[B], p))
    print("clinical", {k: round(v, 3) for k, v in out["clinical"].items()}, flush=True)
    # ---- EMIDEC
    lab = pd.read_csv(os.path.join(ROOT, "data", "emidec", "labels.csv"))
    with h5py.File(os.path.join(ROOT, "data", "emidec", "emidec.h5"), "r") as h:
        X = torch.cat([enc(net, stack_video(h[k][()], D), dev) for k in lab.case]).numpy()
    out["emidec"] = {**EE.boot_auc(lab.mi.values, EE.cv_scores(X, lab.mi.values)),
                     "infarct_spearman": float(spearmanr(EE.infarct(X, lab.infarct_pct.values), lab.infarct_pct.values)[0])}
    print("emidec", out["emidec"], flush=True)
    # ---- DETERMINE
    dl = pd.read_csv(os.path.join(ROOT, "data", "determine", "determine_labels.csv")); y = dl.infarct_pct.values; big = (y > np.median(y)).astype(int)
    with h5py.File(os.path.join(ROOT, "data", "determine", "determine_stacks.h5"), "r") as h:
        X = torch.cat([enc(net, stack_video(h[str(p)][()], D), dev) for p in dl.pid]).numpy()
    rand = [list(KFold(5, shuffle=True, random_state=r).split(y)) for r in range(20)]; bysite = [list(GroupKFold(5).split(y, groups=dl.site.values))]
    rho = lambda a, b: spearmanr(a, b)[0]
    out["determine"] = {cv: {"infarct_spearman": DE.boot(y, DE.oof(X, y, sp, True), rho), "large_infarct_auroc": DE.boot(big, DE.oof(X, big, sp, False), roc_auc_score)}
                        for cv, sp in (("random_cv", rand), ("site_held_out", bysite))}
    print("determine", out["determine"], flush=True)
    json.dump(out, open(os.path.join(cfg["paths"]["out_dir"], "bench_dinov2.json"), "w"), indent=1); print("saved")


if __name__ == "__main__":
    main()
