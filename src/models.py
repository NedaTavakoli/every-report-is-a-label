"""
Vision encoder: multiscale vision transformer of Shad et al. (Nat Biomed Eng 2026), cine-pretrained checkpoint cmr_c0.1,
code from external/cmr_transformer. Text encoder: any Hugging Face encoder. Trained with a symmetric InfoNCE loss with
flooding, as in that paper.
"""
import os, sys
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from common import ROOT


def _cmr_mvit():
    sys.path.insert(0, os.path.join(ROOT, "external", "cmr_transformer"))
    from arch import mvit      # model_factory.py is not imported: it requires the authors' local_config.yaml
    m = mvit.mvit_base_16x4(pretrained=False)
    dim = m.head.proj.in_features
    m.head.proj = nn.Identity()
    return m, dim


class Tiny3D(nn.Module):
    """small network so the pipeline can be tested without the real encoder"""
    def __init__(self, dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(3, dim, 3, stride=2, padding=1), nn.ReLU(), nn.AdaptiveAvgPool3d(1), nn.Flatten())

    def forward(self, x): return self.net(x)


def _load_vision_checkpoint(encoder, proj, path, allow_pickle):
    """loads video_encoder.* and, if its shape fits, video_proj.* ; reports what was loaded, never silently skips"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: download it first (README, step 2), or set model.vision_checkpoint to null")
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        if not allow_pickle:
            raise RuntimeError("this checkpoint needs pickle loading, which can run code from the file. Set "
                               "model.allow_pickle_checkpoint: true if you accept its source.")
        sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
    enc = {k[len("video_encoder."):]: v for k, v in sd.items() if k.startswith("video_encoder.")}
    res = encoder.load_state_dict(enc, strict=False)
    if not enc or res.missing_keys:
        raise RuntimeError(f"checkpoint does not fit the encoder: {len(enc)} tensors, {len(res.missing_keys)} missing, "
                           f"first keys {list(sd)[:5]}")
    pr = {k[len("video_proj."):]: v for k, v in sd.items() if k.startswith("video_proj.")}
    if pr and pr.get("weight", torch.empty(0)).shape == proj.weight.shape:
        proj.load_state_dict(pr); print("  projection head loaded from the checkpoint")
    print(f"  loaded {len(enc)} encoder tensors from {os.path.basename(path)}")


class TextEncoder(nn.Module):
    def __init__(self, name, init=None, freeze_layers=0):
        super().__init__()
        src = init or name
        self.tok = AutoTokenizer.from_pretrained(src)
        self.net = AutoModel.from_pretrained(src)
        self.dim = self.net.config.hidden_size
        layers = None
        for path in ("encoder.layer", "layers"):
            o = self.net
            try:
                for p in path.split("."): o = getattr(o, p)
                layers = o; break
            except AttributeError:
                continue
        if layers is None and freeze_layers: raise RuntimeError("cannot find the text encoder's layers to freeze")
        for l in list(layers or [])[:freeze_layers]:
            for q in l.parameters(): q.requires_grad = False

    def forward(self, texts, max_len, device):
        enc = self.tok(list(texts), truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
        h = self.net(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1)


class DinoVideo(nn.Module):
    """DINOv2 ViT-B/14 (natural-image self-supervised) applied per frame; CLS tokens averaged over the frames of a stack.
    The first `freeze` of its 12 blocks stay frozen; the rest are trained with activation checkpointing (memory)."""
    def __init__(self, name="facebook/dinov2-base", freeze=6):
        super().__init__()
        self.net = AutoModel.from_pretrained(name); self.net.gradient_checkpointing_enable()
        for p in self.net.embeddings.parameters(): p.requires_grad = False
        for l in self.net.encoder.layer[:freeze]:
            for p in l.parameters(): p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):                       # x: N, 3, F, H, W in [0, 1]
        n, c, f, h, w = x.shape
        z = x.permute(0, 2, 1, 3, 4).reshape(n * f, c, h, w)
        z = self.net(pixel_values=(z - self.mean) / self.std).last_hidden_state[:, 0]
        return z.reshape(n, f, -1).mean(1)


def build_vision(cfg, load_weights=True):
    """encoder and projection layer, optionally starting from the checkpoint in the config"""
    M = cfg["model"]
    if M["vision_backend"] == "cmr_mvit": vision, vdim = _cmr_mvit()
    elif M["vision_backend"] == "tiny3d": vision, vdim = Tiny3D(), 32
    elif M["vision_backend"] == "dinov2": vision = DinoVideo(M.get("dinov2_name", "facebook/dinov2-base"), M.get("dinov2_freeze", 6)); vdim = vision.net.config.hidden_size
    else: raise ValueError(M["vision_backend"])
    proj = nn.Linear(vdim, M["proj_dim"])
    if load_weights and M.get("vision_checkpoint"):
        path = M["vision_checkpoint"]
        path = path if os.path.isabs(path) else os.path.join(ROOT, path)
        _load_vision_checkpoint(vision, proj, path, M.get("allow_pickle_checkpoint", False))
    return vision, proj


class ReportImageCLIP(nn.Module):
    def __init__(self, cfg, load_vision_weights=True):
        super().__init__()
        M = cfg["model"]; self.max_len = cfg["data"]["max_text_tokens"]
        self.vision, self.video_proj = build_vision(cfg, load_vision_weights)
        self.text = TextEncoder(M["text_model"], M.get("text_init"), M.get("freeze_text_layers", 0))
        self.text_proj = nn.Linear(self.text.dim, M["proj_dim"])
        self.log_temp = nn.Parameter(torch.log(torch.tensor(float(M["init_temperature"]))))

    def encode_video(self, x): return F.normalize(self.video_proj(self.vision(x)), dim=-1)

    def encode_text(self, texts, device):
        return F.normalize(self.text_proj(self.text(texts, self.max_len, device)), dim=-1)

    def forward(self, video, texts):
        v, t = self.encode_video(video), self.encode_text(texts, video.device)
        return v, t, self.log_temp.clamp(min=-4.6, max=0.0).exp()     # temperature 0.01 to 1


def clip_loss(v, t, temp, flood=0.0):
    logits = v @ t.T / temp
    y = torch.arange(len(v), device=v.device)
    loss = (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y)) / 2
    return (loss - flood).abs() + flood if flood else loss
