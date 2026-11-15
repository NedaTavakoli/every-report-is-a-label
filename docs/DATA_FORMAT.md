# Input data formats

The code never needs names, medical record numbers or dates of birth. Use anonymized identifiers throughout.

## 1 Exam table (`paths.workbook`, sheet `Exams`): one row per examination

| column | meaning |
|---|---|
| `anon_mrn` | anonymized patient ID (used for patient-level splits and bootstrap) |
| `anon_acc` | anonymized examination ID |
| `lge_mag_folder` | folder of the examination under `paths.lge_root` |
| `lge_mag_series` | `;`-separated LGE magnitude series folder names |
| `report_text` | report text (FINDINGS onward is used) |
| `expert_label_R1` | optional expert class; patients with a label are held out of pretraining |
| `n_mrns` | number of patients sharing this accession (exams with > 1 are excluded) |
| `vital_status` | `alive` / `deceased` at data extraction (optional outcome) |

`prepare_manifest.py` turns this into `data/manifest/exams.csv` and `series.csv`; `cache_lge.py` reads the DICOM series.

## 2 Clinical variables (`data/manifest/clinical.csv`, optional): one row per `anon_acc`

`age, male, exam_year, bmi, sbp, dbp, pulse, ldl, hdl, egfr_lt60, diabetes, hypertension, hyperlipidemia, ckd, smoking`
(`smoking` = never / former / current; binary columns 0/1; missing allowed).

## 3 Derived per-exam files

- `data/manifest/scanner.csv` (`scanner_meta.py`): vendor, model, field strength; `year` is taken from `clinical.csv`
  (`exam_year`) because de-identified DICOM dates are often shifted.
- `data/manifest/lvef.csv` (`extract_lvef.py`): LVEF parsed from the report text.

## 4 External data

- **EMIDEC** (public): unzip `emidec-dataset-1.0.1` into `data/`; `emidec_pack.py` builds `data/emidec/`.
- **DETERMINE** (restricted): `determine_full_pack.py` expects the core-laboratory export and a per-series analysis report in
  `$DETERMINE_TABLES` and one flat DICOM folder per examination in `$DETERMINE_DICOM`; it writes anonymized stacks and labels to
  `data/determine_full/` and a private crosswalk that must not be shared.
- **Saliency** (`saliency_determine.py`) needs expert masks as slices: `data/determine/determine_slices.npz` with arrays
  `image` (N, 224, 224), `mask` (N, 224, 224; 0 background, 1 blood pool, 2 scar, 3 myocardium), `pid`, `site`, `is_test`;
  `determine_pack.py` converts it to per-patient stacks.
