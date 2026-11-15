#!/usr/bin/env python3
"""
Image-only control for the report-image pretraining: the same encoder, initialization (cine checkpoint cmr_c0.1), cohort,
train split, batches (one patient per batch), optimizer, schedule, epochs, temperature and flooding as train_clip.py,
but with NO report text. Contrastive (SimCLR-style) loss between two LGE stacks of the same exam: an augmented stack and
another view of the same exam (another augmented copy of the same stack if the exam has a single series).
Comparing these weights with train_clip.py's answers "what do the reports add beyond the images?".
Writes outputs/lge_imgssl/{last.pt, epochNNNN.pt, train_log.csv}. The last checkpoint is the one used downstream.
"""
import os, csv, time, json, random, argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.utils.checkpoint
from torch.utils.data import DataLoader
from common import load_config, seed_everything, device, ROOT
from data import SeriesText, OnePerPatientBatches, to_frames, augment
from models import build_vision, clip_loss
import h5py


class SeriesPairs(SeriesText):
    """item = (augmented stack, augmented other stack of the same exam)"""
    def __init__(self, cfg, splits):
        super().__init__(cfg, splits, True)
        self.by_exam = self.d.groupby("anon_acc").indices

    def _load(self, j):
        r = self.d.iloc[j]
        with h5py.File(os.path.join(self.P["cache_dir"], r.anon_acc + ".h5"), "r") as h:
            vol = h[r.series.replace("/", "_")][()]
        x = augment(to_frames(vol, self.D["frames"]), self.D["crop"])
        return x.unsqueeze(0).repeat(3, 1, 1, 1)

    def __getitem__(self, i):
        same = [j for j in self.by_exam[self.d.anon_acc.iloc[i]] if j != i]
        return {"a": self._load(i), "b": self._load(random.choice(same) if same else i)}


class Ckpt(nn.Module):
    """activation checkpointing around one encoder block: same math, less memory (two stacks per pair need ~2x memory)"""
    def __init__(self, blk):
        super().__init__(); self.blk = blk

    def forward(self, x, thw, *a):
        if self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(self.blk, x, thw, *a, use_reentrant=False)
        return self.blk(x, thw, *a)


class ImageSSL(nn.Module):
    """vision encoder + projection, named as in ReportImageCLIP so downstream code can load vision.* / video_proj.*"""
    def __init__(self, cfg):
        super().__init__()
        self.vision, self.video_proj = build_vision(cfg, True)
        if hasattr(self.vision, "blocks"):
            self.vision.blocks = nn.ModuleList(Ckpt(b) for b in self.vision.blocks)
        self.log_temp = nn.Parameter(torch.log(torch.tensor(float(cfg["model"]["init_temperature"]))))

    def encode_video(self, x): return F.normalize(self.video_proj(self.vision(x)), dim=-1)

    def export_state_dict(self):
        """state dict without the checkpointing wrapper, so keys match ReportImageCLIP (vision.blocks.N.*)"""
        return {k.replace(".blk.", "."): v for k, v in self.state_dict().items()}

    def forward(self, a, b):
        return self.encode_video(a), self.encode_video(b), self.log_temp.clamp(min=-4.6, max=0.0).exp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lge_clip.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default="outputs/lge_imgssl")
    ap.add_argument("--max_batches", type=int, default=0, help="smoke test only")
    args = ap.parse_args()
    cfg = load_config(args.config); T = cfg["train"]
    if args.epochs: T["epochs"] = args.epochs
    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out); os.makedirs(out, exist_ok=True)
    seed_everything(T["seed"]); dev = device(); amp = dev.type == "cuda"
    json.dump(cfg, open(os.path.join(out, "config_used.json"), "w"), indent=1)

    tr = SeriesPairs(cfg, ["train"])
    samp = OnePerPatientBatches(tr.patient, T["batch_size"], T["seed"])
    nw = cfg["data"]["num_workers"]
    trl = DataLoader(tr, batch_sampler=samp, num_workers=nw, pin_memory=amp, persistent_workers=nw > 0)
    print(f"train {len(tr):,} series, {len(set(tr.patient)):,} patients, {len(samp)} batches per epoch", flush=True)

    model = ImageSSL(cfg).to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=T["lr"], weight_decay=T["weight_decay"], eps=T["eps"])
    sched = torch.optim.lr_scheduler.StepLR(opt, T["lr_drop_epoch"], T["lr_drop_factor"])
    start, last = 0, os.path.join(out, "last.pt")
    if os.path.exists(last) and not args.max_batches:
        c = torch.load(last, map_location="cpu"); model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"]); start = c["epoch"] + 1; print("resumed at epoch", start, flush=True)

    logf = os.path.join(out, "train_log.csv"); new = not os.path.exists(logf)
    with open(logf, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["epoch", "loss", "temperature", "lr", "minutes"])
        for ep in range(start, T["epochs"]):
            model.train(); samp.set_epoch(ep); t0, tot, n = time.time(), 0.0, 0
            for b in trl:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    va, vb, temp = model(b["a"].to(dev, non_blocking=True), b["b"].to(dev, non_blocking=True))
                    loss = clip_loss(va, vb, temp, T["flood"])
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), T["grad_clip"]); opt.step()
                tot += loss.item(); n += 1
                if args.max_batches and n >= args.max_batches: break
            sched.step()
            row = [ep, tot / max(n, 1), float(temp), opt.param_groups[0]["lr"], (time.time() - t0) / 60]
            print(row, flush=True)
            if args.max_batches:
                print("smoke test ok; max GPU memory GB", round(torch.cuda.max_memory_allocated() / 1e9, 1) if amp else 0); return
            w.writerow(row); f.flush()
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep}, last)
            if (ep + 1) % T["save_every"] == 0:
                torch.save({"model": model.export_state_dict(), "epoch": ep}, os.path.join(out, f"epoch{ep + 1:04d}.pt"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
