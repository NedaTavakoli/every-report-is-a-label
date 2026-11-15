#!/usr/bin/env python3
"""
Downstream training on top of an image encoder. Modes, as defined by Shad et al.:
  frozen     without fine-tuning: encoder and projection fixed, only the classifier head trains
  finetune   projection layer and head train; the encoder stays fixed
  transfer   encoder, projection and head all train
Weights: trained (report-image pretraining, last.pt), cine (cmr_c0.1 only), random.
Tasks:
  etiology   Reader 1 scar class on the expert exams, 5-fold cross-validation by patient. Very few labels.
  deceased   vital status; train split -> test split. Trains on far more labels, but there is no follow-up time:
             recent exams have had less time for an event, so this is not a survival analysis.
All settings are fixed in the config; no held-out result selects anything. Writes outputs/.../finetune_<task>_<weights>_<mode>.json/.csv
"""
import os, json, argparse
import numpy as np, pandas as pd, h5py, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import cohen_kappa_score, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from common import load_config, seed_everything, device
from data import to_frames, augment, center
from models import build_vision


class ExamBags(Dataset):
    """all series of an exam (at most max_series), with one exam label"""
    def __init__(self, cfg, exams, train):
        self.cfg, self.train = cfg, train
        s = pd.read_csv(os.path.join(cfg["paths"]["manifest_dir"], "series.csv"), dtype=str)
        self.series = {}     # only series with a cached dataset (cache_lge.py skips series that fail to decode)
        for a, g in s[s.anon_acc.isin(exams.anon_acc)].groupby("anon_acc"):
            with h5py.File(os.path.join(cfg["paths"]["cache_dir"], a + ".h5"), "r") as h:
                self.series[a] = [n for n in g.series if n.replace("/", "_") in h]
        self.e = exams[[len(self.series.get(a, [])) > 0 for a in exams.anon_acc]].reset_index(drop=True)
        if len(self.e) < len(exams): print(f"  {len(exams) - len(self.e)} exam(s) without a decodable series left out", flush=True)

    def __len__(self): return len(self.e)

    def __getitem__(self, i):
        D = self.cfg["data"]; r = self.e.iloc[i]; names = self.series[r.anon_acc]
        if len(names) > self.cfg["downstream"]["max_series"]:
            names = list(np.random.choice(names, self.cfg["downstream"]["max_series"], replace=False))
        out = []
        with h5py.File(os.path.join(self.cfg["paths"]["cache_dir"], r.anon_acc + ".h5"), "r") as h:
            for n in names:
                k = n.replace("/", "_")
                if k in h:
                    x = to_frames(h[k][()], D["frames"])
                    x = augment(x, D["crop"]) if self.train else center(x, D["crop"], D["shorter_side"])
                    out.append(x.unsqueeze(0).repeat(3, 1, 1, 1))
        return torch.stack(out), int(r.y), r.anon_acc


def collate(b): return [x[0] for x in b], torch.tensor([x[1] for x in b]), [x[2] for x in b]


class AttentionHead(nn.Module):
    """gated attention over the series of an exam (Shad et al.), then a linear classifier"""
    def __init__(self, dim, n_classes, hidden=256):
        super().__init__()
        self.norm = nn.LayerNorm(dim); self.V = nn.Linear(dim, hidden); self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1); self.fc = nn.Linear(dim, n_classes)

    def forward(self, h):
        h = self.norm(h)
        a = torch.softmax(self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))), dim=0)
        return self.fc((a * h).sum(0))


class Net(nn.Module):
    def __init__(self, cfg, weights, mode, n_classes):
        super().__init__()
        self.vision, self.proj = build_vision(cfg, load_weights=weights == "cine")
        if weights == "trained":
            sd = torch.load(os.path.join(cfg["paths"]["out_dir"], "last.pt"), map_location="cpu")["model"]
            self.vision.load_state_dict({k[len("vision."):]: v for k, v in sd.items() if k.startswith("vision.")})
            self.proj.load_state_dict({k[len("video_proj."):]: v for k, v in sd.items() if k.startswith("video_proj.")})
        self.mode = mode
        for p in self.vision.parameters(): p.requires_grad = mode == "transfer"
        for p in self.proj.parameters(): p.requires_grad = mode != "frozen"
        self.head = AttentionHead(self.proj.out_features, n_classes)

    def train(self, m=True):     # fixed parts stay in evaluation mode
        super().train(m)
        if self.mode != "transfer": self.vision.eval()
        if self.mode == "frozen": self.proj.eval()
        return self

    def forward(self, bags, dev):
        sizes = [len(b) for b in bags]; x = torch.cat(bags).to(dev)
        with torch.set_grad_enabled(self.mode == "transfer"):
            h = self.vision(x)
        h = self.proj(h)
        return torch.stack([self.head(p) for p in h.split(sizes)])


def fit_predict(cfg, tr, te, weights, mode, n_classes, dev):
    S = cfg["downstream"]; amp = dev.type == "cuda"
    net = Net(cfg, weights, mode, n_classes).to(dev)
    groups = [{"params": list(net.head.parameters()) + ([] if mode == "frozen" else list(net.proj.parameters())), "lr": S["lr"]}]
    if mode == "transfer": groups.append({"params": list(net.vision.parameters()), "lr": S["lr_encoder_transfer"]})
    opt = torch.optim.AdamW(groups, weight_decay=S["weight_decay"])
    cnt = np.bincount(tr.y.astype(int), minlength=n_classes).astype(float)
    cw = torch.tensor(cnt.sum() / np.maximum(cnt, 1) / n_classes, dtype=torch.float32).to(dev)
    crit = nn.CrossEntropyLoss(weight=cw, reduction="sum")
    # full transfer does not fit 8 exams x 6 series with encoder gradients on a 46 GB GPU: accumulate over micro-batches;
    # the summed loss divided by the batch's total class weight equals the weighted mean over the whole batch
    micro = S.get("transfer_micro_batch", 2) if mode == "transfer" else S["batch_size"]
    nw = cfg["data"]["num_workers"]
    trl = DataLoader(ExamBags(cfg, tr, True), batch_size=S["batch_size"], shuffle=True, collate_fn=collate, num_workers=nw, drop_last=len(tr) > S["batch_size"])
    for ep in range(S["epochs"]):
        net.train(); tot = 0.0
        for bags, y, _ in trl:
            y = y.to(dev); W = cw[y].sum(); opt.zero_grad(set_to_none=True)
            for k in range(0, len(bags), micro):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    loss = crit(net(bags[k:k + micro], dev), y[k:k + micro]) / W
                loss.backward(); tot += loss.item()
            opt.step()
        print(f"  epoch {ep + 1}/{S['epochs']} loss {tot / max(len(trl), 1):.3f}", flush=True)
    net.eval(); P, A = [], []
    with torch.no_grad():
        for bags, _, ids in DataLoader(ExamBags(cfg, te, False), batch_size=S["batch_size"], collate_fn=collate, num_workers=nw):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                P.append(torch.softmax(net(bags, dev).float(), 1).cpu().numpy())
            A += ids
    return np.concatenate(P), A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lge_clip.yaml")
    ap.add_argument("--task", choices=["etiology", "deceased"], required=True)
    ap.add_argument("--weights", choices=["trained", "cine", "random"], default="trained")
    ap.add_argument("--mode", choices=["frozen", "finetune", "transfer"], default="finetune")
    args = ap.parse_args()
    cfg = load_config(args.config); seed_everything(cfg["train"]["seed"]); dev = device(); P = cfg["paths"]
    e = pd.read_csv(os.path.join(P["manifest_dir"], "exams.csv"), dtype=str, keep_default_na=False)
    e = e[[os.path.exists(os.path.join(P["cache_dir"], a + ".h5")) for a in e.anon_acc]]
    out = dict(task=args.task, weights=args.weights, mode=args.mode)
    if args.task == "deceased":
        e = e[e.vital_status.isin(["alive", "deceased"])].copy(); e["y"] = (e.vital_status == "deceased").astype(int)
        tr, te = e[e.split == "train"], e[e.split == "test"]
        p, ids = fit_predict(cfg, tr, te, args.weights, args.mode, 2, dev)
        y = te.set_index("anon_acc").y.reindex(ids).values
        out.update(n_train=len(tr), n_test=len(te), events_train=int(tr.y.sum()), events_test=int(te.y.sum()), auroc=float(roc_auc_score(y, p[:, 1])),
                   balanced_accuracy=float(balanced_accuracy_score(y, p.argmax(1))))
        pred = pd.DataFrame({"anon_acc": ids, "y": y, "p_deceased": p[:, 1]})
    else:
        e = e[e.expert_label != ""].copy()
        per = e.groupby("expert_label").anon_mrn.nunique(); k = cfg["downstream"]["cv_folds"]
        keep = sorted(per[per >= k].index); out["left_out_classes"] = per[per < k].to_dict()
        e = e[e.expert_label.isin(keep)].copy(); e["y"] = e.expert_label.map({c: i for i, c in enumerate(keep)})
        if len(keep) < 2: raise SystemExit(f"too few expert labels: {per.to_dict()}")
        frames = []
        for f, (a, b) in enumerate(StratifiedGroupKFold(k, shuffle=True, random_state=42).split(e, e.y, e.anon_mrn)):
            print("fold", f, flush=True)
            p, ids = fit_predict(cfg, e.iloc[a], e.iloc[b], args.weights, args.mode, len(keep), dev)
            frames.append(pd.DataFrame({"anon_acc": ids, "pred": np.array(keep)[p.argmax(1)]}))
        pred = pd.concat(frames).merge(e[["anon_acc", "expert_label"]], on="anon_acc")
        out.update(n=len(pred), classes=keep, accuracy=float((pred.pred == pred.expert_label).mean()),
                   kappa=float(cohen_kappa_score(pred.expert_label, pred.pred)), most_common_class_accuracy=float(pred.expert_label.value_counts(normalize=True).iloc[0]))
    name = f"finetune_{args.task}_{args.weights}_{args.mode}"
    os.makedirs(P["out_dir"], exist_ok=True)
    pred.to_csv(os.path.join(P["out_dir"], name + ".csv"), index=False)
    json.dump(out, open(os.path.join(P["out_dir"], name + ".json"), "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
