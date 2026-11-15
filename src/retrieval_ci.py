#!/usr/bin/env python3
"""
95% CIs for test retrieval (top-1, top-5 recall, median rank), and paired differences of the pretrained model against
the random and cine encoders. Ranks are computed once in the full test pool (as in evaluate.py); the per-examination
results are then resampled by patient (2,000 bootstrap samples). The test set is embedded exactly as in evaluate.py
(seed 42 set just before each model is built), because the untrained text projection of the random and cine models depends
on the seed. Writes outputs/.../retrieval_ci.json.
"""
import os, json, argparse
import numpy as np, pandas as pd, torch
from common import load_config, seed_everything, device
from data import SeriesText
from models import ReportImageCLIP
from evaluate import embed, retrieval

B = 2000
W = ["random", "cine", "trained"]


def ranks(V, T):
    return (np.argsort(-(V @ T.T), axis=1) == np.arange(len(V))[:, None]).argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lge_clip.yaml")
    cfg = load_config(ap.parse_args().config); P = cfg["paths"]
    dev = device(); R, g = {}, None
    for w in W:
        seed_everything(42)
        model = ReportImageCLIP(cfg, load_vision_weights=w == "cine").to(dev).eval()
        if w == "trained":
            model.load_state_dict(torch.load(os.path.join(P["out_dir"], "last.pt"), map_location="cpu")["model"])
        info, V, T = embed(model, SeriesText(cfg, ["test"], False), cfg, dev)
        if g is None: g, acc0 = info.mrn.values, info.acc.values
        assert (info.acc.values == acc0).all()
        R[w] = ranks(V, T)
        ref = json.load(open(os.path.join(P["out_dir"], f"eval_{w}.json")))["retrieval_test"]
        assert abs(retrieval(V, T)["recall@5"] - ref["recall@5"]) < 1e-9, (w, retrieval(V, T), ref)
        del model; torch.cuda.empty_cache()
    n = len(g)
    stats = {"recall@1": lambda r: (r < 1).mean(), "recall@5": lambda r: (r < 5).mean(), "median_rank": lambda r: np.median(r) + 1}
    pats = np.unique(g); idx = {p: np.where(g == p)[0] for p in pats}; rng = np.random.default_rng(42)
    bs = {w: {s: [] for s in stats} for w in W}
    for _ in range(B):
        i = np.concatenate([idx[p] for p in rng.choice(pats, len(pats))])
        for w in W:
            for s, f in stats.items(): bs[w][s].append(f(R[w][i]))
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    out = {"n_exams": n, "n_patients": len(pats), "bootstrap": B, "chance@1": 1 / n, "chance@5": 5 / n}
    for w in W:
        out[w] = {s: {"estimate": float(f(R[w])), "ci95": ci(bs[w][s])} for s, f in stats.items()}
    for w in W:
        if w != "trained":
            for s, f in stats.items():
                d = np.array(bs["trained"][s]) - np.array(bs[w][s])
                out["trained"][s][f"diff_vs_{w}"] = {"estimate": float(f(R["trained"]) - f(R[w])), "ci95": ci(d)}
    out["rank_trained"] = {str(a): int(r) + 1 for a, r in zip(acc0, R["trained"])}   # anonymized accession -> rank of its own report (figure examples)
    out["recall_curve"] = {w: [float((R[w] < k).mean()) for k in range(1, 51)] for w in W}   # recall@k, k = 1..50 (Figure 3)
    json.dump(out, open(os.path.join(P["out_dir"], "retrieval_ci.json"), "w"), indent=1)
    for w in W: print(w, {s: (round(v["estimate"], 3), [round(x, 3) for x in v["ci95"]]) for s, v in out[w].items()})


if __name__ == "__main__":
    main()
