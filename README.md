# Every Radiology Report Is a Label

**A multimodal image–report foundation model for late gadolinium enhancement (LGE) CMR that predicts mortality beyond ejection fraction. It learns from routine LGE images paired with the radiology reports written for them, without manual labels.**

Every clinical LGE examination already comes with a written report. This repository trains an LGE image encoder by contrastive
report–image pretraining (CLIP-style): each examination is matched to its own report, so no manual labels are needed. The frozen
image features are then evaluated for:

- **all-cause mortality beyond clinical variables and LVEF** (risk tertiles, adjusted odds ratios, paired comparisons), including
  **hidden risk**: patients who look low-risk on age, risk factors and LVEF but have a high image score (clinical × image risk grid);
- **external, multicenter transfer** against a trial core laboratory (infarct size, LVEF, volumes, LV mass), with whole trial
  sites held out, and on the public EMIDEC dataset;
- **report retrieval and text-to-image search** (the training task);
- **a site shortcut and its fix**: contrastive training can learn which scanner/site an examination came from; drawing every
  training batch from one scanner vendor (`--same_vendor_batches`) removes this reward.

Comparators: the same encoder untrained, the cine CMR foundation model of Shad et al. (`cmr_c0.1`), a general-purpose vision
foundation model (DINOv2), image-only contrastive pretraining on the same data, and a DINOv2 backbone trained with reports.

> No patient data, model weights or results are included. The code runs on your own data laid out as in
> [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md); a synthetic smoke test checks the pipeline end to end.

## Repository

`every-report-is-a-label` — code for the paper of the same title.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# pretrained weights and code: accept the terms at https://huggingface.co/rohanshad/cmr_c0.1 (CC BY-NC 4.0) first
huggingface-cli login                       # token with read access to gated repositories
python scripts/download_checkpoints.py      # add --dinov2 for the comparison experiments
```

`scripts/download_checkpoints.py` downloads:

| Item | Source | Saved to |
|---|---|---|
| Cine CMR checkpoint `cmr_c0.1` (Shad et al.), starting point of the image encoder | Hugging Face `rohanshad/cmr_c0.1`, file `epoch=599-step=124200.ckpt` (gated) | `weights/` |
| Video transformer code (Shad et al.) | `github.com/rohanshad/cmr_transformer`, branch `publicrelease` | `external/cmr_transformer` |
| Text encoder | Hugging Face `thomas-sounack/BioClinical-ModernBERT-base` | Hugging Face cache |
| DINOv2 ViT-B/14 (comparisons only, `--dinov2`) | Hugging Face `facebook/dinov2-base` | Hugging Face cache |

Use `--only cine|code|text|dinov2` to fetch one item. The cine checkpoint loads in PyTorch safe mode (tensors only),
so `model.allow_pickle_checkpoint` can stay `false`. The weights of the report-pretrained model in the paper are not
included in this repository.

## Quick check (synthetic data, no GPU needed)

```bash
python tests/test_smoke.py           # tiny 3D network, synthetic DICOMs and reports
python tests/test_smoke.py --mvit    # same with the real video transformer (needs external/cmr_transformer)
```

## Pipeline

All scripts run from `src/` and read `configs/lge_clip.yaml` (set `paths.workbook` and `paths.lge_root` first).

```bash
cd src
# 1 data
python prepare_manifest.py --config ../configs/lge_clip.yaml   # exams/series manifest, patient-level splits
python cache_lge.py --config ../configs/lge_clip.yaml          # DICOM -> HDF5 cache (resumable)
python scanner_meta.py                                         # vendor / model / field strength per exam
python extract_lvef.py                                         # LVEF from report text (fixed pattern)

# 2 pretraining
python train_clip.py --config ../configs/lge_clip.yaml                                   # report-supervised (main model)
python train_clip.py --config ../configs/lge_clip.yaml --same_vendor_batches \
       --out outputs/lge_clip_samevendor                                                 # site-shortcut fix
python train_clip.py --config ../configs/lge_clip.yaml --frac 0.25 --out outputs/lge_clip_frac25   # data scaling
python train_imgssl.py --config ../configs/lge_clip.yaml                                 # image-only control
python train_clip.py --config ../configs/lge_clip_dinov2.yaml                            # DINOv2 backbone + reports

# 3 evaluation (internal)
python evaluate.py --weights trained         # retrieval (also: random | cine)
python retrieval_ci.py                       # retrieval with patient-bootstrap CIs
python probe_clinical.py                     # clinical-variable probes and vital-status models
python mortality_lvef.py                     # all-cause mortality beyond clinical variables + LVEF, all encoders
python risk_groups.py                        # risk tertiles, within LVEF strata, adjusted odds ratios
python risk_grid.py                          # clinical-risk x image-risk grid (hidden risk)
python robustness.py                         # retrieval by vendor / field / era; site-shortcut check
python embed_map.py                          # vendor decodability, t-SNE maps
python search_demo.py                        # text-to-image search
python bench_dinov2.py                       # general-purpose foundation model benchmark
python eval_runs.py --config configs/lge_clip.yaml --runs full=outputs/lge_clip frac25=outputs/lge_clip_frac25

# 4 external validation
python emidec_pack.py                        # EMIDEC (public, CC BY-NC-SA 4.0) -> cache
python emidec_eval.py --weights random cine trained imgssl
python emidec_orient.py                      # in-plane orientation sensitivity
python determine_full_pack.py                # DETERMINE core lab (restricted data; set DETERMINE_TABLES, DETERMINE_DICOM)
python determine_full_eval.py --weights random cine trained imgssl dinov2_generic samevendor=outputs/lge_clip_samevendor
python paired_tests.py                       # paired encoder comparisons (same cases, folds, bootstrap samples)
python saliency_determine.py                 # saliency enrichment in expert-contoured scar / myocardium
```

Design choices that matter for honest evaluation: patients (not examinations) are split; the expert-labeled patients are held
out of pretraining; the last checkpoint is always used (no model selection on held-out data); probes use fixed or
training-set-tuned regularization; external data are never used for tuning; encoders are compared on the same cases, folds and
bootstrap samples.


## Repository layout

```
configs/      training configurations (main model; DINOv2 backbone)
src/          data preparation, training, evaluation and external-validation scripts
scripts/      download_checkpoints.py (pretrained weights and code)
tests/        synthetic smoke test; checkpoint loading test
docs/         expected input formats (DATA_FORMAT.md)
docs/build/   figure generation (PowerPoint figures, renders, TIFF export, example-image extraction)
```

## Data and licenses

- Code: MIT (see `LICENSE`).
- `cmr_c0.1` checkpoint and `cmr_transformer` code: see their own licenses (CC BY-NC 4.0 for the checkpoint).
- DINOv2 (`facebook/dinov2-base`): Apache 2.0. BioClinical ModernBERT: see its model card.
- EMIDEC: CC BY-NC-SA 4.0, non-commercial research; cite the EMIDEC data paper.
- DETERMINE: available from the trial investigators under a data-use agreement.
- Institutional images and reports are protected health information and are not shared.

## Citation

If you use this code, please cite the accompanying paper (in preparation): *Every Radiology Report Is a Label: a multimodal image–report
foundation model for late gadolinium enhancement CMR that predicts mortality beyond ejection fraction.*
