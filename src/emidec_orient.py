#!/usr/bin/env python3
"""
Orientation check for the EMIDEC external test. NIfTI and DICOM may store the in-plane axes differently, and our encoder
never saw flips in training, so EMIDEC stacks are embedded in all 8 in-plane orientations (4 rotations x mirror).
For each orientation and weight set it reports:
  match      label-free: mean over EMIDEC cases of the highest cosine similarity to our own test short-axis stacks
             (same encoder). Chooses the orientation without any EMIDEC label.
  probe      MI vs normal AUROC (as emidec_eval.py: 20 x 5-fold CV, fixed C=1)
  zero_shot  trained only: MI prompt minus no-LGE prompt similarity, AUROC
Also 'avg8': the mean embedding over the 8 orientations (orientation-free). Writes outputs/lge_clip/emidec_orient.json.
"""
import os, json, argparse
import numpy as np, pandas as pd, h5py, torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from common import load_config, seed_everything, device, ROOT
from data import to_frames, center, SeriesText
from emidec_eval import load_model, cv_scores, PROMPTS, boot_auc

ORIENT = [(k, f) for f in (False, True) for k in range(4)]   # rot90 k times, then optional left-right mirror


@torch.no_grad()
def embed_emidec(model, cases, cfg, dev):
    D = cfg["data"]; amp = dev.type == "cuda"; E = np.zeros((8, len(cases), cfg["model"]["proj_dim"]), np.float32)
    with h5py.File(os.path.join(ROOT, "data", "emidec", "emidec.h5"), "r") as h:
        for i, c in enumerate(cases):
            base = center(to_frames(h[c][()], D["frames"]), D["crop"], D["shorter_side"])   # F, 224, 224
            xs = []
            for k, f in ORIENT:
                x = torch.rot90(base, k, (1, 2)); x = torch.flip(x, (2,)) if f else x
                xs.append(x.unsqueeze(0).repeat(3, 1, 1, 1))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                E[:, i] = model.encode_video(torch.stack(xs).to(dev)).float().cpu().numpy()
    return E


@torch.no_grad()
def embed_ours_sa(model, cfg, dev):
    ds = SeriesText(cfg, ["test"], False); ds.d = ds.d[ds.d["view"] == "sa"].reset_index(drop=True); ds.patient = ds.d.anon_mrn.values
    amp = dev.type == "cuda"; out = []
    for b in DataLoader(ds, batch_size=cfg["train"]["batch_size"], num_workers=cfg["data"]["num_workers"]):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out.append(model.encode_video(b["video"].to(dev)).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/lge_clip.yaml")
    ap.add_argument("--weights", nargs="+", default=["random", "cine", "trained"]); a = ap.parse_args()
    cfg = load_config(a.config); seed_everything(42); dev = device()
    lab = pd.read_csv(os.path.join(ROOT, "data", "emidec", "labels.csv")); y = lab.mi.values
    avg_scores = {}
    res = {"orientations": [f"rot{90 * k}{'+mirror' if f else ''}" for k, f in ORIENT] + ["avg8"], "weights": {}}
    for w in a.weights:
        model = load_model(cfg, w, dev)
        E = embed_emidec(model, lab.case.tolist(), cfg, dev); ours = embed_ours_sa(model, cfg, dev)
        with torch.no_grad(): T = model.encode_text(list(PROMPTS.values()), dev).float().cpu().numpy() if w == "trained" else None
        avg = E.mean(0); avg /= np.linalg.norm(avg, axis=1, keepdims=True)
        rows = []; avg_scores[w] = cv_scores(avg, y)
        for name, X in zip(res["orientations"], list(E) + [avg]):
            r = {"orientation": name, "match": float((X @ ours.T).max(1).mean()), "probe_auroc": float(roc_auc_score(y, cv_scores(X, y)))}
            if T is not None: s = X @ T.T; r["zero_shot_auroc"] = float(roc_auc_score(y, s[:, 0] - s[:, 1]))
            rows.append(r); print(w, json.dumps(r), flush=True)
        best = max(rows[:8], key=lambda r: r["match"])
        res["weights"][w] = {"n_our_sa_stacks": int(len(ours)), "rows": rows, "label_free_choice": best["orientation"]}
        del model; torch.cuda.empty_cache()
    res["avg8_ci"] = {w: boot_auc(y, sc, avg_scores.get("trained") if w != "trained" else None) for w, sc in avg_scores.items()}
    print(json.dumps(res["avg8_ci"]), flush=True)
    path = os.path.join(cfg["paths"]["out_dir"], "emidec_orient.json"); json.dump(res, open(path, "w"), indent=1); print("saved", path)


if __name__ == "__main__":
    main()
