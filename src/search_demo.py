#!/usr/bin/env python3
"""
Text-to-image and image-to-image search on the test set (Figure: searching the LGE archive).

1. Text -> image: each clinical query sentence is embedded by the text encoder; test examinations are ranked by the cosine
   similarity of their image embedding to the query. Top-k examination IDs are saved for the figure.
2. Objective check (no labels were created for training or model choice): an examination counts as matching a query when
   its own report contains the corresponding keyword pattern. For each query we report the AUROC of the image-query
   similarity for that report mention over all test examinations (patient bootstrap 95% CI), precision among the top 10,
   and the base rate. The cine checkpoint is the reference (its text and image towers were never aligned).
3. Image -> image: for each example examination, the most similar test examinations of other patients.
Writes outputs/.../search_demo.json (anonymized IDs and numbers only; no report text).
"""
import os, re, json, argparse
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from common import load_config, seed_everything, device
from data import SeriesText
from models import ReportImageCLIP
from evaluate import embed

QUERIES = {
    "no_lge": ("No late gadolinium enhancement to suggest myocardial scar or fibrosis.",
               r"\bno (evidence of )?((abnormal|myocardial|late|delayed|gadolinium|significant)\s+)*(enhancement|scar|fibrosis|lge)\b"),
    "infarct": ("Subendocardial late gadolinium enhancement in a coronary distribution consistent with prior myocardial infarction.",
                r"infarct"),
    "midwall": ("Mid-wall late gadolinium enhancement of the septum in a non-ischemic pattern.",
                r"mid[- ]?(wall|myocardial)"),
    "rv_insertion": ("Late gadolinium enhancement at the right ventricular insertion points.",
                     r"insertion point|hinge point"),
    "amyloid": ("Diffuse subendocardial enhancement with abnormal myocardial nulling, suggestive of cardiac amyloidosis.",
                r"amyloid"),
    "sarcoid": ("Patchy multifocal late gadolinium enhancement suggestive of cardiac sarcoidosis.",
                r"sarcoid"),
}
K, B = 10, 2000


def boot_auc(y, s, g, rng):
    pats = np.unique(g); idx = {p: np.where(g == p)[0] for p in pats}; out = []
    for _ in range(B):
        i = np.concatenate([idx[p] for p in rng.choice(pats, len(pats))])
        if 0 < y[i].sum() < len(i): out.append(roc_auc_score(y[i], s[i]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lge_clip.yaml")
    ap.add_argument("--examples", default="", help="comma-separated anonymized accessions for image-to-image search")
    args = ap.parse_args()
    cfg = load_config(args.config); dev = device(); P = cfg["paths"]
    text = pd.read_csv(os.path.join(P["manifest_dir"], "exams.csv"), dtype=str, keep_default_na=False).set_index("anon_acc").text
    out = {"queries": {k: v[0] for k, v in QUERIES.items()}, "k": K}
    for w in ["trained", "cine"]:
        seed_everything(42)
        model = ReportImageCLIP(cfg, load_vision_weights=w == "cine").to(dev).eval()
        if w == "trained":
            model.load_state_dict(torch.load(os.path.join(P["out_dir"], "last.pt"), map_location="cpu")["model"])
        info, V, _ = embed(model, SeriesText(cfg, ["test"], False), cfg, dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            Q = model.encode_text([q for q, _ in QUERIES.values()], dev).float().cpu().numpy()
        S = V @ Q.T; acc, g = info.acc.values, info.mrn.values; rng = np.random.default_rng(42); res = {}
        for j, (name, (_, pat)) in enumerate(QUERIES.items()):
            y = np.array([bool(re.search(pat, text[a], re.I)) for a in acc]).astype(int); order = np.argsort(-S[:, j])
            r = {"n_mention": int(y.sum()), "base_rate": float(y.mean()), f"precision@{K}": float(y[order[:K]].mean())}
            if 0 < y.sum() < len(y):
                r["auroc"] = float(roc_auc_score(y, S[:, j])); r["auroc_ci95"] = boot_auc(y, S[:, j], g, rng)
            if w == "trained": r["top"] = [str(a) for a in acc[order[:K]]]
            res[name] = r
        out[w] = res
        if w == "trained" and args.examples:
            ex = args.examples.split(","); nn = {}
            for e in ex:
                i = int(np.where(acc == e)[0][0]); sim = V @ V[i]; sim[g == g[i]] = -np.inf
                nn[e] = [str(a) for a in acc[np.argsort(-sim)[:4]]]
            out["similar"] = nn
        del model; torch.cuda.empty_cache()
    json.dump(out, open(os.path.join(P["out_dir"], "search_demo.json"), "w"), indent=1)
    for w in ["trained", "cine"]:
        print(w, {k: (v["n_mention"], round(v.get("auroc", float("nan")), 3), [round(x, 2) for x in v.get("auroc_ci95", [])], round(v[f"precision@{K}"], 2), round(v["base_rate"], 3)) for k, v in out[w].items()})


if __name__ == "__main__":
    main()
