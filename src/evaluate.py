#!/usr/bin/env python3
"""
Evaluates one set of image weights on the held-out sets:
  retrieval    test split: right report among all test reports, against chance
  linear probe expert exams: logistic regression on frozen image embeddings, cross-validated by patient, Reader 1 labels
  zero-shot    expert exams: nearest hand-written class description (configs: eval.class_prompts)
--weights: trained (last.pt, default) | cine (cmr_c0.1 only, before report training) | random (no pretraining)
Comparing the three shows what the cine checkpoint and the LGE report training each add.
"""
import os, json, argparse
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import cohen_kappa_score
from common import load_config, seed_everything, device
from data import SeriesText
from models import ReportImageCLIP


@torch.no_grad()
def embed(model, ds, cfg, dev):
    amp = dev.type == "cuda"; rec = []
    for b in DataLoader(ds, batch_size=cfg["train"]["batch_size"], num_workers=cfg["data"]["num_workers"]):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            v = model.encode_video(b["video"].to(dev)); t = model.encode_text(b["text"], dev)
        for i in range(len(v)):
            rec.append((b["anon_acc"][i], b["anon_mrn"][i], b["expert_label"][i], v[i].float().cpu().numpy(), t[i].float().cpu().numpy()))
    ex = {}
    for a, m, l, vi, ti in rec: ex.setdefault(a, dict(mrn=m, label=l, v=[], t=ti))["v"].append(vi)
    keys = list(ex)
    V = np.stack([np.mean(ex[k]["v"], 0) for k in keys]); V /= np.linalg.norm(V, axis=1, keepdims=True)
    return pd.DataFrame({"acc": keys, "mrn": [ex[k]["mrn"] for k in keys], "label": [ex[k]["label"] for k in keys]}), V, np.stack([ex[k]["t"] for k in keys])


def retrieval(V, T):
    r = (np.argsort(-(V @ T.T), axis=1) == np.arange(len(V))[:, None]).argmax(1)
    n = len(V)
    return {"n_exams": n, "recall@1": float((r < 1).mean()), "chance@1": 1 / n, "recall@5": float((r < 5).mean()),
            "chance@5": min(5 / n, 1), "median_rank": float(np.median(r) + 1)}


def probe(info, V):
    """cross-validated by patient; classes with too few patients to appear in every fold are left out and reported"""
    lab = (info.label != "").values; info, V = info[lab], V[lab]     # other exams of expert patients have no label
    y, g = info.label.values, info.mrn.values
    per = info.groupby("label").mrn.nunique(); k = 5
    keep = per[per >= k].index.tolist(); drop = per[per < k].to_dict()
    m = np.isin(y, keep)
    if m.sum() < 20 or len(keep) < 2:
        return {"n": int(len(y)), "status": "too few expert exams with images for a cross-validated probe", "by_class": per.to_dict()}
    pred = np.empty(m.sum(), dtype=object)
    for tr, te in StratifiedGroupKFold(k, shuffle=True, random_state=42).split(V[m], y[m], g[m]):
        pred[te] = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")).fit(V[m][tr], y[m][tr]).predict(V[m][te])
    return {"n": int(m.sum()), "classes": keep, "left_out_classes": drop, "accuracy": float((pred == y[m]).mean()),
            "kappa": float(cohen_kappa_score(y[m], pred)), "most_common_class_accuracy": float(pd.Series(y[m]).value_counts(normalize=True).iloc[0])}


@torch.no_grad()
def zero_shot(model, info, V, prompts, dev):
    names = list(prompts)
    t = model.encode_text([prompts[n] for n in names], dev).float().cpu().numpy()
    pred = np.array(names)[(V @ t.T).argmax(1)]; y = info.label.values
    m = np.isin(y, names)
    return {"n": int(m.sum()), "excluded_labels": sorted(set(y) - set(names)), "accuracy": float((pred[m] == y[m]).mean()),
            "kappa": float(cohen_kappa_score(y[m], pred[m]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lge_clip.yaml")
    ap.add_argument("--weights", choices=["trained", "cine", "random"], default="trained")
    args = ap.parse_args()
    cfg = load_config(args.config); seed_everything(42); dev = device(); P = cfg["paths"]
    model = ReportImageCLIP(cfg, load_vision_weights=args.weights == "cine").to(dev).eval()
    if args.weights == "trained":
        model.load_state_dict(torch.load(os.path.join(P["out_dir"], "last.pt"), map_location="cpu")["model"])
    out = {"weights": args.weights}
    info, V, T = embed(model, SeriesText(cfg, ["test"], False), cfg, dev)
    out["retrieval_test"] = retrieval(V, T)
    try:
        info, V, _ = embed(model, SeriesText(cfg, ["expert"], False), cfg, dev)
        out["linear_probe_expert"] = probe(info, V)
        out["zero_shot_expert"] = zero_shot(model, info, V, cfg["eval"]["class_prompts"], dev)
    except RuntimeError as ex:
        out["expert"] = str(ex)
    os.makedirs(P["out_dir"], exist_ok=True)
    path = os.path.join(P["out_dir"], f"eval_{args.weights}.json")
    json.dump(out, open(path, "w"), indent=1); print(json.dumps(out, indent=1)); print("saved", path)


if __name__ == "__main__":
    main()
