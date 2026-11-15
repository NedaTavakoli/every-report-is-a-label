"""Shared helpers."""
import os, random
import numpy as np, torch, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["ischemic", "non-ischemic", "mixed", "no-scar", "not-specified"]


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg["paths"].items():
        if k != "lge_root" and not os.path.isabs(v):
            cfg["paths"][k] = os.path.join(ROOT, v)
    return cfg


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
