#!/usr/bin/env python3
"""Checks on real data: loads the cmr_c0.1 checkpoint in safe mode, embeds cached LGE series, runs one training step."""
import os, sys, glob, time, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import h5py, torch, torch.nn as nn
from torch.utils.data import DataLoader
from common import load_config, ROOT
from data import to_frames, center, SeriesText, OnePerPatientBatches
from models import ReportImageCLIP, _cmr_mvit, _load_vision_checkpoint, clip_loss

cfg = load_config(os.path.join(ROOT, "configs", "lge_clip.yaml")); p = os.path.join(ROOT, cfg["model"]["vision_checkpoint"])
sd = torch.load(p, map_location="cpu", weights_only=True); print("top keys", list(sd)[:8]); s = sd.get("state_dict", sd)
print("tensors", len(s), "prefixes", collections.Counter(".".join(k.split(".")[:2]) for k in s).most_common(8))
print("projection shapes", {k: tuple(v.shape) for k, v in s.items() if "proj" in k and "encoder" not in k})
m, d = _cmr_mvit(); before = list(m.parameters())[0].clone(); pr = nn.Linear(d, 512)
_load_vision_checkpoint(m, pr, p, False); print("weights changed:", not torch.equal(before, list(m.parameters())[0]))
m.eval(); E = []
for f in sorted(glob.glob(cfg["paths"]["cache_dir"] + "/*.h5"))[:8]:
    with h5py.File(f) as h:
        for k in list(h)[:4]:
            v = h[k][()]
            if len(v) >= 8:
                x = center(to_frames(v, 16), 224, 256).unsqueeze(0).repeat(3, 1, 1, 1).unsqueeze(0)
                with torch.no_grad(): E.append(nn.functional.normalize(pr(m(x)), dim=-1)[0])
E = torch.stack(E); c = E @ E.T
print("real series", len(E), "mean cosine between different series", round(((c.sum() - len(E)) / (len(E) * (len(E) - 1))).item(), 3), "finite", bool(torch.isfinite(E).all()))
cfg["data"]["num_workers"] = 0
M = ReportImageCLIP(cfg); ds = SeriesText(cfg, ["train", "val", "test", "expert"], True)
print("series", len(ds), "patients", len(set(ds.patient)))
b = next(iter(DataLoader(ds, batch_sampler=OnePerPatientBatches(ds.patient, 4, 0))))
tr = [q for q in M.parameters() if q.requires_grad]
print("batch", tuple(b["video"].shape), "trainable M", round(sum(q.numel() for q in tr) / 1e6, 1), "of", round(sum(q.numel() for q in M.parameters()) / 1e6, 1))
o = torch.optim.AdamW(tr, lr=4.8e-5); t = time.time(); M.train()
v, tx, temp = M(b["video"], b["text"]); l = clip_loss(v, tx, temp, 0.05); o.zero_grad(); l.backward(); o.step()
print("loss", round(l.item(), 3), "(chance", round(float(torch.log(torch.tensor(4.0))), 3), ") seconds", round(time.time() - t, 1))
print("OK")
