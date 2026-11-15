#!/usr/bin/env python3
"""
Download the public pretrained weights and code this project builds on.

  cine CMR checkpoint   rohanshad/cmr_c0.1 "epoch=599-step=124200.ckpt" -> weights/   (gated; CC BY-NC 4.0)
  video transformer     github.com/rohanshad/cmr_transformer (branch publicrelease) -> external/cmr_transformer
  text encoder          thomas-sounack/BioClinical-ModernBERT-base  (Hugging Face cache)
  DINOv2 (comparisons)  facebook/dinov2-base                       (Hugging Face cache; --dinov2)

The cine checkpoint is gated: first accept its terms at https://huggingface.co/rohanshad/cmr_c0.1, then run
`huggingface-cli login` (or set HF_TOKEN) with a token that can read gated repositories.

Usage (from the repository root):
  python scripts/download_checkpoints.py            # checkpoint, code and text encoder
  python scripts/download_checkpoints.py --dinov2   # also DINOv2
  python scripts/download_checkpoints.py --only cine
"""
import argparse, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CINE_REPO, CINE_FILE = "rohanshad/cmr_c0.1", "epoch=599-step=124200.ckpt"
CODE_URL, CODE_BRANCH = "https://github.com/rohanshad/cmr_transformer", "publicrelease"
TEXT_MODEL, DINO_MODEL = "thomas-sounack/BioClinical-ModernBERT-base", "facebook/dinov2-base"


def cine():
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    dst = os.path.join(ROOT, "weights")
    if os.path.exists(os.path.join(dst, CINE_FILE)):
        print("cine checkpoint: already in weights/"); return True
    try:
        p = hf_hub_download(CINE_REPO, CINE_FILE, local_dir=dst)
    except (GatedRepoError, RepositoryNotFoundError):
        print(f"cine checkpoint: access denied. Accept the terms at https://huggingface.co/{CINE_REPO}, "
              "then run `huggingface-cli login` and try again.", file=sys.stderr); return False
    print("cine checkpoint:", os.path.relpath(p, ROOT)); return True


def code():
    dst = os.path.join(ROOT, "external", "cmr_transformer")
    if os.path.isdir(dst):
        print("cmr_transformer: already in external/"); return True
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    r = subprocess.run(["git", "clone", "--depth", "1", "-b", CODE_BRANCH, CODE_URL, dst])
    print("cmr_transformer:", "external/cmr_transformer" if r.returncode == 0 else "clone failed"); return r.returncode == 0


def hub(name):
    from huggingface_hub import snapshot_download
    p = snapshot_download(name, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    print(f"{name}: cached at {p}"); return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["cine", "code", "text", "dinov2"], help="download one item only")
    ap.add_argument("--dinov2", action="store_true", help="also download DINOv2 (comparison experiments)")
    a = ap.parse_args()
    steps = {"cine": cine, "code": code, "text": lambda: hub(TEXT_MODEL), "dinov2": lambda: hub(DINO_MODEL)}
    todo = [a.only] if a.only else ["cine", "code", "text"] + (["dinov2"] if a.dinov2 else [])
    ok = all([steps[k]() for k in todo])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
