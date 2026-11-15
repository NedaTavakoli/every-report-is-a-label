"""Series-report pairs. Each item is one LGE stack shaped as the encoder's video input, paired with its exam's report."""
import os, re, random
import numpy as np, pandas as pd, h5py, torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, Sampler

SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def sample_sentences(text, k, train):
    """k random sentences, in report order, per step (Shad et al.); the whole text at evaluation"""
    if not train: return text
    s = [x.strip() for x in SENT.split(str(text)) if x and x.strip()]
    if len(s) <= k: return " ".join(s)
    return " ".join(s[i] for i in sorted(random.sample(range(len(s)), k)))


def to_frames(vol, frames):
    """uniform subsampling of the slices; short stacks repeat slices"""
    idx = np.linspace(0, len(vol) - 1, frames).round().astype(int)
    return torch.from_numpy(vol[idx]).float() / 255.0     # F, H, W in [0, 1], as the encoder's Div255


def augment(x, crop):
    """same transform for all slices of a stack; no flips, which would mirror the anatomy"""
    x = TF.rotate(x, random.uniform(-15, 15), interpolation=TF.InterpolationMode.BILINEAR)
    h, w = x.shape[-2:]
    side = int(min(h, w) * random.uniform(0.75, 1.0))
    t, l = random.randint(0, h - side), random.randint(0, w - side)
    x = TF.resized_crop(x, t, l, side, side, [crop, crop], antialias=True)
    x = torch.clamp(x * random.uniform(0.85, 1.15) + random.uniform(-0.05, 0.05), 0, 1)
    return x


def center(x, crop, shorter):
    x = TF.resize(x, [shorter], antialias=True)
    return TF.center_crop(x, [crop, crop])


class SeriesText(Dataset):
    def __init__(self, cfg, splits, train):
        P, D = cfg["paths"], cfg["data"]
        e = pd.read_csv(os.path.join(P["manifest_dir"], "exams.csv"), dtype=str, keep_default_na=False)
        s = pd.read_csv(os.path.join(P["manifest_dir"], "series.csv"), dtype=str)
        d = s.merge(e[e.split.isin(splits)], on="anon_acc")
        keys = {}     # series whose DICOMs failed to decode were skipped by cache_lge.py and have no dataset
        for a in d.anon_acc.unique():
            p = os.path.join(P["cache_dir"], a + ".h5")
            if os.path.exists(p):
                with h5py.File(p, "r") as h: keys[a] = set(h)
            else: keys[a] = set()
        d = d[[r.replace("/", "_") in keys[a] for a, r in zip(d.anon_acc, d.series)]].reset_index(drop=True)
        if len(d) == 0: raise RuntimeError(f"no cached series for {splits}: run cache_lge.py first")
        self.d, self.P, self.D, self.train = d, P, D, train
        self.patient = d.anon_mrn.values

    def __len__(self): return len(self.d)

    def __getitem__(self, i):
        r = self.d.iloc[i]
        with h5py.File(os.path.join(self.P["cache_dir"], r.anon_acc + ".h5"), "r") as h:
            vol = h[r.series.replace("/", "_")][()]
        x = to_frames(vol, self.D["frames"])
        x = augment(x, self.D["crop"]) if self.train else center(x, self.D["crop"], self.D["shorter_side"])
        x = x.unsqueeze(0).repeat(3, 1, 1, 1)     # C, F, H, W
        return {"video": x, "text": sample_sentences(r.text, self.D["sentences_per_step"], self.train),
                "anon_acc": r.anon_acc, "anon_mrn": r.anon_mrn, "view": r["view"], "expert_label": r.expert_label}


class OnePerPatientBatches(Sampler):
    """A patient appears at most once per batch: another scan of the same patient must not count as a wrong match."""
    def __init__(self, patients, batch_size, seed=0, drop_last=True):
        self.pat, self.bs, self.seed, self.drop = list(patients), batch_size, seed, drop_last
        self.epoch = 0

    def set_epoch(self, e): self.epoch = e

    def _batches(self):
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.pat))); rng.shuffle(idx)
        out, cur, seen, wait = [], [], set(), []
        for i in idx:
            if self.pat[i] in seen: wait.append(i); continue
            cur.append(i); seen.add(self.pat[i])
            if len(cur) == self.bs:
                out.append(cur); cur, seen = [], set()
                still = []
                for j in wait:
                    if self.pat[j] in seen: still.append(j)
                    else:
                        cur.append(j); seen.add(self.pat[j])
                wait = still
        if cur and not self.drop: out.append(cur)
        return out

    def __iter__(self): return iter(self._batches())
    def __len__(self): return len(self._batches())


class SameGroupBatches(Sampler):
    """Every batch comes from ONE group (e.g. scanner vendor), still one patient per batch: within-batch negatives then share
    the group, so group signature (site / scanner style) cannot help tell the right report from the wrong ones."""
    def __init__(self, patients, groups, batch_size, seed=0):
        self.groups = {g: np.where(np.asarray(groups) == g)[0] for g in sorted(set(groups))}
        self.sub = {g: OnePerPatientBatches(np.asarray(patients)[i], batch_size, seed) for g, i in self.groups.items()}
        self.seed, self.epoch = seed, 0

    def set_epoch(self, e):
        self.epoch = e
        for s in self.sub.values(): s.set_epoch(e)

    def _batches(self):
        out = [[int(self.groups[g][j]) for j in b] for g, s in self.sub.items() for b in s._batches()]
        random.Random(self.seed + 10_000 + self.epoch).shuffle(out)
        return out

    def __iter__(self): return iter(self._batches())
    def __len__(self): return len(self._batches())
