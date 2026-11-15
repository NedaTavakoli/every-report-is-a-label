#!/usr/bin/env python3
"""
Where does the encoder look? Saliency on DETERMINE patients with EXPERT masks (353 patients; data/determine from
determine_pack.py; masks 0 background, 1 blood pool, 2 scar, 3 myocardium).
For each encoder, a ridge probe of expert infarct % is fit on all 353 embeddings (for saliency only, not evaluation); the
saliency of a patient's stack is SmoothGrad (8 noisy copies, sigma 0.05) of |d(probe output)/d(input)|, summed over channels.
The mask stack gets exactly the image's preprocessing (resize to 256, the same 16-frame index, center crop 224).
Enrichment of a region = (share of total saliency inside it) / (share of the image area it covers); 1 = no preference.
Reported per encoder (mean and 95% bootstrap CI over patients) for scar, myocardium (incl. scar), blood pool and the rest.
Three example patients (median, 75th and 90th percentile of infarct %; chosen by rule) are saved for the figure.
Writes outputs/lge_clip/saliency_determine.json and outputs/lge_clip/saliency_examples.npz .
"""
import os, json, hashlib
import numpy as np, pandas as pd, h5py, torch, torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from common import load_config, seed_everything, device, ROOT
from data import to_frames, center
import determine_eval as DE

ENC = {"trained": "trained", "samevendor": "samevendor=outputs/lge_clip_samevendor", "imgssl": "imgssl", "cine": "cine", "random": "random"}


def masks_per_patient():
    z = np.load(os.path.join(ROOT, "data", "determine", "determine_slices.npz")); I, M, pid = z["image"], z["mask"], z["pid"]
    h = np.array([hashlib.md5(I[i].tobytes()).hexdigest() for i in range(len(I))])
    _, first = np.unique(h, return_index=True); keep = np.zeros(len(I), bool); keep[first] = True
    return {int(p): M[np.where(keep & (pid == p))[0]] for p in np.unique(pid)}


def prep(vol, D, nearest=False):
    idx = np.linspace(0, len(vol) - 1, D["frames"]).round().astype(int)
    if nearest:
        t = torch.from_numpy(vol[idx].astype(np.float32))[:, None]
        t = F.interpolate(t, size=(256, 256), mode="nearest")[:, 0]
        o = (256 - D["crop"]) // 2; return t[:, o:o + D["crop"], o:o + D["crop"]].numpy().astype(np.uint8)
    return center(to_frames(vol, D["frames"]), D["crop"], D["shorter_side"])


def main():
    cfg = load_config(os.path.join(ROOT, "configs", "lge_clip.yaml")); seed_everything(0); dev = device(); D = cfg["data"]
    lab = pd.read_csv(os.path.join(ROOT, "data", "determine", "determine_labels.csv")); MK = masks_per_patient()
    with h5py.File(os.path.join(ROOT, "data", "determine", "determine_stacks.h5"), "r") as h: ST = {int(k): h[k][()] for k in h}
    q = lab.infarct_pct.quantile([0.5, 0.75, 0.9]).values
    ex_ids = [int(lab.pid.iloc[(lab.infarct_pct - v).abs().argmin()]) for v in q]
    res, examples = {}, {}
    for name, spec in ENC.items():
        _, model = DE.model_for(cfg, spec, dev); model.eval()
        for p_ in model.parameters(): p_.requires_grad_(False)
        X = DE.embed(model, lab.pid.tolist(), cfg, dev)
        sc = StandardScaler().fit(X); rg = Ridge(alpha=1.0).fit(sc.transform(X), lab.infarct_pct.values)
        w = torch.tensor(rg.coef_ / sc.scale_, dtype=torch.float32, device=dev)
        enr = {k: [] for k in ("scar", "myocardium", "blood_pool", "other")}
        for pid in lab.pid:
            x0 = prep(ST[int(pid)], D).unsqueeze(0).repeat(3, 1, 1, 1)[None].to(dev); m = prep(MK[int(pid)], D, nearest=True)
            g = torch.zeros_like(x0)
            for _ in range(8):
                x = (x0 + 0.05 * torch.randn_like(x0)).clamp(0, 1).requires_grad_(True)
                (model.encode_video(x)[0] @ w).backward(); g += x.grad.abs()
            s = g[0].sum(0).cpu().numpy()                      # (16, 224, 224)
            fr = (np.isin(m, [2, 3])).reshape(len(m), -1).any(1)
            if not fr.any(): continue
            s, mm = s[fr], m[fr]; tot = s.sum(); A = mm.size
            for k, sel in (("scar", mm == 2), ("myocardium", np.isin(mm, [2, 3])), ("blood_pool", mm == 1), ("other", mm == 0)):
                if sel.sum() > 0: enr[k].append(float((s[sel].sum() / tot) / (sel.sum() / A)))
            if int(pid) in ex_ids and name in ("trained", "imgssl", "random"):
                examples[f"{name}_{ex_ids.index(int(pid))}"] = s.astype(np.float16)
                if name == "trained":
                    examples[f"img_{ex_ids.index(int(pid))}"] = x0[0, 0].cpu().numpy()[fr].astype(np.float16); examples[f"mask_{ex_ids.index(int(pid))}"] = mm
        boot = lambda v: [float(np.mean(v)), *[float(x) for x in np.percentile([np.mean(np.random.default_rng(i).choice(v, len(v))) for i in range(1000)], [2.5, 97.5])]]
        res[name] = {k: {"enrichment": boot(v), "n": len(v)} for k, v in enr.items() if v}
        print(name, {k: [round(x, 2) for x in v["enrichment"]] for k, v in res[name].items()}, flush=True)
        del model; torch.cuda.empty_cache()
    res["_examples"] = {"infarct_pct": [float(lab.infarct_pct[lab.pid == i].iloc[0]) for i in ex_ids], "rule": "patients closest to the 50th, 75th, 90th percentile of infarct %"}
    json.dump(res, open(os.path.join(cfg["paths"]["out_dir"], "saliency_determine.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(cfg["paths"]["out_dir"], "saliency_examples.npz"), **examples); print("saved")


if __name__ == "__main__":
    main()
