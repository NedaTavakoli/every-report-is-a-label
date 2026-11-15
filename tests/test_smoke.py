#!/usr/bin/env python3
"""End-to-end test on synthetic DICOMs and reports: workbook -> manifest -> cache -> training -> evaluation.
No patient data and no downloaded weights. Run: python tests/test_smoke.py [--mvit]"""
import os, sys, json, shutil, subprocess, tempfile, argparse
import numpy as np, pandas as pd, yaml, pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from transformers import BertConfig, BertModel, BertTokenizerFast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_dicom(path, arr, inst):
    m = FileMetaDataset(); m.TransferSyntaxUID = ExplicitVRLittleEndian
    m.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"; m.MediaStorageSOPInstanceUID = generate_uid()
    d = FileDataset(path, {}, file_meta=m, preamble=b"\0" * 128)
    d.SOPClassUID, d.SOPInstanceUID = m.MediaStorageSOPClassUID, m.MediaStorageSOPInstanceUID
    d.Rows, d.Columns = arr.shape; d.InstanceNumber = inst; d.SamplesPerPixel = 1; d.PhotometricInterpretation = "MONOCHROME2"
    d.BitsAllocated = 16; d.BitsStored = 16; d.HighBit = 15; d.PixelRepresentation = 0; d.PixelData = arr.astype(np.uint16).tobytes()
    d.save_as(path)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mvit", action="store_true", help="use the real encoder (random weights)")
    args = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="ssl_smoke_"); rng = np.random.default_rng(0)
    root = os.path.join(tmp, "lge"); rows = []
    series_names = ["PSIR TRUFI SA_MAG", "TRUFI PSIR 2C_MAG", "MAG_3ch PSMDE BH ARDL", "TRUFI PSIR 4c_MAG", "LONG TI TRUFI PSIR 2C_MAG"]
    labels = ["ischemic", "non-ischemic", "no-scar", "mixed"]
    for i in range(48):
        m, a = f"MRN{i:06d}", f"ACC{i:06d}"
        for s in series_names:
            d = os.path.join(root, m, a, s); os.makedirs(d)
            for k in range(int(rng.integers(8, 18))):
                write_dicom(os.path.join(d, f"{k}.dcm"), rng.integers(0, 2000, (96, 80)) + 300 * (i % 4), k + 1)
        rows.append(dict(anon_mrn=m, anon_acc=a, lge_mag_folder=os.path.join(root, m, a), lge_mag_series="; ".join(series_names),
                         report_text=f"HISTORY: chest pain.\nFINDINGS: finding {i % 4}. Normal size. Pattern {i}.\nIMPRESSION: result {i % 4}. Electronically signed by nobody.",
                         expert_label_R1=labels[i % 4] if i < 24 else None, n_mrns="1", epic_mrn="x", nlp_class="no-scar", vital_status="deceased" if i % 3 == 0 else "alive"))
    pd.DataFrame(rows).to_excel(os.path.join(tmp, "wb.xlsx"), sheet_name="Exams", index=False)
    # tiny local text encoder
    tdir = os.path.join(tmp, "bert"); os.makedirs(tdir)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [chr(c) for c in range(97, 123)] + [str(i) for i in range(10)] + ["."]
    open(os.path.join(tdir, "vocab.txt"), "w").write("\n".join(vocab))
    BertTokenizerFast(vocab_file=os.path.join(tdir, "vocab.txt")).save_pretrained(tdir)
    BertModel(BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64)).save_pretrained(tdir)

    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "lge_clip.yaml")))
    cfg["paths"] = dict(workbook=os.path.join(tmp, "wb.xlsx"), lge_root=root, manifest_dir=os.path.join(tmp, "m"),
                        cache_dir=os.path.join(tmp, "c"), out_dir=os.path.join(tmp, "o"))
    cfg["data"]["num_workers"] = 0; cfg["data"]["split"] = {"train": 0.5, "val": 0.25, "test": 0.25, "seed": 1}
    cfg["model"].update(vision_backend="cmr_mvit" if args.mvit else "tiny3d", vision_checkpoint=None, text_model=tdir, freeze_text_layers=1)
    cfg["train"].update(batch_size=4, epochs=2, val_every=1, save_every=1); cfg["downstream"].update(epochs=2, batch_size=4)
    cp = os.path.join(tmp, "cfg.yaml"); yaml.safe_dump(cfg, open(cp, "w"))
    src = os.path.join(ROOT, "src")
    for cmd in (["prepare_manifest.py"], ["cache_lge.py"], ["train_clip.py"], ["evaluate.py"], ["evaluate.py", "--weights", "random"],
                ["finetune.py", "--task", "deceased", "--mode", "frozen", "--weights", "trained"],
                ["finetune.py", "--task", "deceased", "--mode", "finetune", "--weights", "trained"],
                ["finetune.py", "--task", "deceased", "--mode", "transfer", "--weights", "trained"],
                ["finetune.py", "--task", "etiology", "--mode", "finetune", "--weights", "random"]):
        print("==", cmd[0], flush=True)
        subprocess.run([sys.executable, "-W", "ignore", cmd[0]] + cmd[1:] + ["--config", cp], cwd=src, check=True)

    # checks
    e = pd.read_csv(os.path.join(tmp, "m", "exams.csv"), dtype=str); s = pd.read_csv(os.path.join(tmp, "m", "series.csv"), dtype=str)
    assert not {"epic_mrn", "nlp_class"} & (set(e.columns) | set(s.columns)), "identifier or NLP label leaked into the manifest"
    assert not s.series.str.contains("LONG TI").any(), "long-TI series not excluded"
    assert set(s["view"]) == {"sa", "2ch", "3ch", "4ch"}, set(s["view"])
    assert not e.text.str.contains("HISTORY|signed", case=False).any(), "report cleaning failed"
    assert e[e.split == "expert"].anon_mrn.nunique() == 24 and not set(e[e.split == "expert"].anon_mrn) & set(e[e.split != "expert"].anon_mrn)
    assert os.path.exists(os.path.join(tmp, "o", "last.pt"))
    r = json.load(open(os.path.join(tmp, "o", "eval_trained.json"))); assert "linear_probe_expert" in r, r
    for m in ("frozen", "finetune", "transfer"):
        assert "auroc" in json.load(open(os.path.join(tmp, "o", f"finetune_deceased_trained_{m}.json")))
    assert "kappa" in json.load(open(os.path.join(tmp, "o", "finetune_etiology_random_finetune.json")))
    print("ALL CHECKS PASSED"); shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
