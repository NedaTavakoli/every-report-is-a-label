#!/usr/bin/env python3
"""
Robustness of report retrieval by scanner and era (test split; embeddings from probe_clinical.py, emb_<weights>.npz).
  1 recall@1 / @5 by vendor, field strength and exam-year group, ranks computed in the FULL test pool (as the main
    analysis), 95% CI by bootstrap of patients
  2 recall@1 within each subgroup's own pool (only reports from the same subgroup are candidates), against its chance
  3 shortcut check: for test exams, how often the top-ranked report comes from the same vendor / within 2 years,
    against the rate expected from a random report. A model that matched on scanner or era rather than findings would
    show a large excess here.
Writes outputs/lge_clip/robustness.json.
"""
import os, json
import numpy as np, pandas as pd
from common import ROOT

OUT = os.path.join(ROOT, "outputs", "lge_clip")


def ranks(V, T):
    S = V @ T.T
    return (np.argsort(-S, axis=1) == np.arange(len(V))[:, None]).argmax(1), S.argmax(1)


def boot(r, pat, B=2000, seed=0):
    rng = np.random.default_rng(seed); up = np.unique(pat); idx = {p: np.where(pat == p)[0] for p in up}; o = {1: [], 5: []}
    for _ in range(B):
        i = np.concatenate([idx[p] for p in rng.choice(up, len(up))])
        for k in o: o[k].append((r[i] < k).mean())
    return {f"recall@{k}": [float((r < k).mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in o.items()}


def main():
    e = pd.read_csv(os.path.join(ROOT, "data", "manifest", "exams.csv"), dtype=str, usecols=["anon_acc", "split"])
    sc = pd.read_csv(os.path.join(ROOT, "data", "manifest", "scanner.csv"))
    sc["vendor"] = sc.vendor.str.upper().str.split().str[0].replace({"GE": "GE", "SIEMENS": "Siemens"})
    sc["era"] = pd.cut(sc.year, [0, 2018, 3000], labels=["2006-2018", "2019-2025"]).astype(str)
    sc["field"] = sc.field_t.map(lambda x: f"{x:.1f} T")
    res = {}
    for w in ["trained", "cine", "random"]:
        z = np.load(os.path.join(OUT, f"emb_{w}.npz"), allow_pickle=True)
        d = pd.DataFrame({"anon_acc": z["acc"], "anon_mrn": z["mrn"]}).merge(e, on="anon_acc", how="left")
        m = (d.split == "test").values
        V, T = z["V"][m], z["T"][m]; info = d[m].reset_index(drop=True).merge(sc, on="anon_acc", how="left")
        r, top = ranks(V, T); pat = info.anon_mrn.values
        out = {"n": int(len(r)), "all": boot(r, pat)}
        for col in ["vendor", "field", "era"]:
            out[col] = {}
            for g, sub in info.groupby(col):
                i = sub.index.values; rw, _ = ranks(V[i], T[i])
                out[col][g] = {"n": int(len(i)), **boot(r[i], pat[i]),
                               "within_pool_recall@1": float((rw < 1).mean()), "within_pool_chance@1": 1 / len(i)}
        wrong = r > 0
        same_v = (info.vendor.values[top] == info.vendor.values)
        near_y = np.abs(info.year.values[top] - info.year.values) <= 2
        vv = info.vendor.values; yy = info.year.values
        exp_v = np.mean([(vv == v).mean() for v in vv]); exp_y = np.mean([(np.abs(yy - y) <= 2).mean() for y in yy])
        out["shortcut"] = {"top1_same_vendor": float(same_v.mean()), "top1_same_vendor_when_wrong": float(same_v[wrong].mean()),
                           "expected_same_vendor": float(exp_v), "top1_within_2y": float(near_y.mean()),
                           "top1_within_2y_when_wrong": float(near_y[wrong].mean()), "expected_within_2y": float(exp_y)}
        res[w] = out; print(w, json.dumps({k: out[k] for k in ["n", "all", "shortcut"]}), flush=True)
    json.dump(res, open(os.path.join(OUT, "robustness.json"), "w"), indent=1); print("saved")


if __name__ == "__main__":
    main()
