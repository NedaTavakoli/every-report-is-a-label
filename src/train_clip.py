#!/usr/bin/env python3
"""
Report-image contrastive pretraining. Uses the train split; the val split only reports progress. The last checkpoint is
the one to use downstream, so no result on the held-out sets selects a model. Resumes from last.pt if present.
"""
import os, json, csv, time, argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # batch 32 runs out of memory on an A40 without it
import numpy as np, torch
from torch.utils.data import DataLoader
from common import load_config, seed_everything, device
from data import SeriesText, OnePerPatientBatches, SameGroupBatches
from models import ReportImageCLIP, clip_loss


@torch.no_grad()
def retrieval(model, loader, dev, amp):
    """exam-level report retrieval: is the right report near the top for each exam's images?"""
    model.eval(); V, T, A = {}, {}, []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            v = model.encode_video(b["video"].to(dev)); t = model.encode_text(b["text"], dev)
        for a, vi, ti in zip(b["anon_acc"], v.float().cpu(), t.float().cpu()):
            V.setdefault(a, []).append(vi); T[a] = ti
    keys = list(V)
    vm = torch.nn.functional.normalize(torch.stack([torch.stack(V[k]).mean(0) for k in keys]), dim=-1)
    sim = vm @ torch.stack([T[k] for k in keys]).T
    ranks = (sim.argsort(1, descending=True) == torch.arange(len(keys)).unsqueeze(1)).float().argmax(1).numpy()
    return {"n_exams": len(keys), "recall@1": float((ranks < 1).mean()), "recall@5": float((ranks < 5).mean()),
            "median_rank": float(np.median(ranks) + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lge_clip.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--frac", type=float, default=1.0, help="data-scaling runs: random fraction of training PATIENTS (seeded)")
    ap.add_argument("--same_vendor_batches", action="store_true", help="each batch from one scanner vendor (data/manifest/scanner.csv)")
    ap.add_argument("--out", default=None, help="output folder (default: paths.out_dir)")
    args = ap.parse_args()
    cfg = load_config(args.config); T = cfg["train"]; P = cfg["paths"]
    if args.epochs: T["epochs"] = args.epochs
    if args.out: P["out_dir"] = args.out if os.path.isabs(args.out) else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.out)
    seed_everything(T["seed"]); dev = device(); amp = dev.type == "cuda"
    os.makedirs(P["out_dir"], exist_ok=True)
    json.dump(cfg, open(os.path.join(P["out_dir"], "config_used.json"), "w"), indent=1)

    tr = SeriesText(cfg, ["train"], True); va = SeriesText(cfg, ["val"], False)
    if args.frac < 1:
        pats = np.array(sorted(set(tr.patient))); keep = set(np.random.default_rng(T["seed"]).permutation(pats)[:round(args.frac * len(pats))])
        tr.d = tr.d[tr.d.anon_mrn.isin(keep)].reset_index(drop=True); tr.patient = tr.d.anon_mrn.values
    if args.same_vendor_batches:
        import pandas as pd
        sc = pd.read_csv(os.path.join(os.path.dirname(P["manifest_dir"]), "manifest", "scanner.csv")).set_index("anon_acc").vendor
        vend = tr.d.anon_acc.map(sc).fillna("unknown").str.upper().str.split().str[0].values
        samp = SameGroupBatches(tr.patient, vend, T["batch_size"], T["seed"])
        print("same-vendor batches:", {g: len(i) for g, i in samp.groups.items()}, flush=True)
    else:
        samp = OnePerPatientBatches(tr.patient, T["batch_size"], T["seed"])
    nw = cfg["data"]["num_workers"]
    trl = DataLoader(tr, batch_sampler=samp, num_workers=nw, pin_memory=amp, persistent_workers=nw > 0)
    val = DataLoader(va, batch_size=T["batch_size"], num_workers=nw)
    print(f"train {len(tr):,} series, {len(set(tr.patient)):,} patients, {len(samp)} batches per epoch; val {len(va):,} series", flush=True)

    model = ReportImageCLIP(cfg).to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=T["lr"], weight_decay=T["weight_decay"], eps=T["eps"])
    sched = torch.optim.lr_scheduler.StepLR(opt, T["lr_drop_epoch"], T["lr_drop_factor"])
    start, last = 0, os.path.join(P["out_dir"], "last.pt")
    if os.path.exists(last):
        c = torch.load(last, map_location="cpu"); model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"]); start = c["epoch"] + 1; print("resumed at epoch", start, flush=True)

    logf = os.path.join(P["out_dir"], "train_log.csv"); new = not os.path.exists(logf)
    with open(logf, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["epoch", "loss", "temperature", "lr", "minutes", "val_recall@1", "val_recall@5", "val_median_rank"])
        for ep in range(start, T["epochs"]):
            model.train(); samp.set_epoch(ep); t0, tot = time.time(), 0.0
            for b in trl:
                x = b["video"].to(dev, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    v, t, temp = model(x, b["text"])
                    loss = clip_loss(v, t, temp, T["flood"])
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), T["grad_clip"]); opt.step()
                tot += loss.item()
            sched.step()
            r = {}
            if (ep + 1) % T["val_every"] == 0 or ep + 1 == T["epochs"]: r = retrieval(model, val, dev, amp)
            row = [ep, tot / max(len(samp), 1), float(temp), opt.param_groups[0]["lr"], (time.time() - t0) / 60,
                   r.get("recall@1", ""), r.get("recall@5", ""), r.get("median_rank", "")]
            w.writerow(row); f.flush(); print(row, flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep}, last)
            if (ep + 1) % T["save_every"] == 0:
                torch.save({"model": model.state_dict(), "epoch": ep}, os.path.join(P["out_dir"], f"epoch{ep + 1:04d}.pt"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
