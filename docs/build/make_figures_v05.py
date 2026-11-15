#!/usr/bin/env python3
"""
New figures for manuscript v05 (native PowerPoint shapes / charts, editable; same style as make_figures.py):
  Figure_9_scorecard          every encoder x every task, colored within each column
  Figure_10_external          DETERMINE core lab (sites held out), EMIDEC, vital status beyond LVEF, for 7 encoders
  Figure_11_shortcut          t-SNE maps colored by vendor (report vs same-vendor), vendor decodability, the fix
  Figure_12_risk              all-cause mortality by image-risk tertile, overall and within LVEF strata
  Figure_13_saliency          saliency enrichment in expert scar / myocardium + example overlays (patient images: local only;
                              layout checked on FIG_PLACEHOLDER=1 copies)
All numbers are read from outputs/*.json; a figure whose inputs are missing is skipped.
"""
import os, json
import numpy as np
from PIL import Image
import make_figures as M
from make_figures import (deck, text, foot, line, panel, bars, style_chart, C, OUT, ROOT, PLACEHOLDER, Inches, Pt, PP_ALIGN, MSO_ANCHOR,
                          MSO_SHAPE, RGBColor, XyChartData, XL_CHART_TYPE)

O = os.path.join(ROOT, "outputs")
def J(*p):
    f = os.path.join(O, *p); return json.load(open(f)) if os.path.exists(f) else None


C.update({"dinov2": RGBColor(0x8E, 0x6C, 0xB0), "dinov2clip": RGBColor(0xC3, 0xB1, 0xE1), "imgssl": RGBColor(0x56, 0xB4, 0xE9),
          "samevendor": RGBColor(0x00, 0x80, 0x80), "ge": RGBColor(0xE6, 0x9F, 0x00), "siemens": RGBColor(0x3B, 0x6E, 0x9E),
          "low": RGBColor(0xC6, 0xDB, 0xEF), "mid": RGBColor(0x6B, 0xAE, 0xD6), "high": RGBColor(0x08, 0x51, 0x9C)})
ENC = ["random", "cine", "dinov2", "imgssl", "trained", "samevendor"]
LAB = {"random": "Random", "cine": "Cine\n(Shad)", "dinov2": "DINOv2\n(generic)", "dinov2clip": "DINOv2\n+ reports", "imgssl": "Image-\nonly",
       "trained": "Reports", "samevendor": "Reports +\nsame-vendor"}
LAB1 = {k: v.replace("-\n", "-").replace("\n", " ") for k, v in LAB.items()}

RC, EM, ER, ERD, BD = J("lge_clip", "retrieval_ci.json"), J("lge_clip", "emidec_eval.json"), J("lge_clip", "eval_runs.json"), J("lge_clip_dinov2", "eval_runs.json"), J("lge_clip", "bench_dinov2.json")
RGRID = None
DF, DFD, ML, EMAP, RG, SAL, PT = (J("lge_clip", "determine_full_eval.json"), J("lge_clip_dinov2", "determine_full_eval.json"), J("lge_clip", "mortality_lvef.json"),
                                  J("lge_clip", "embed_map.json"), J("lge_clip", "risk_groups.json"), J("lge_clip", "saliency_determine.json"), J("lge_clip", "paired_tests.json"))


def metric(enc, m):
    """(estimate, lo, hi) or None"""
    try:
        if m == "r5":
            if enc in ("random", "cine", "trained"): r = RC[enc]["recall@5"]; return (r["estimate"], *r["ci95"])
            src = {"samevendor": ER and ER.get("samevendor"), "dinov2clip": ERD and ERD.get("dinov2clip")}.get(enc)
            return tuple(src["retrieval"]["recall@5"]) if src else None
        if m in ("emidec", "emidec_rho"):
            if enc in ("random", "cine", "trained", "imgssl") and enc in EM["weights"]:
                w = EM["weights"][enc]; return (w["mi_vs_normal"]["auroc"], *w["mi_vs_normal"]["ci"]) if m == "emidec" else (w["infarct_pct_spearman_all"], None, None)
            src = {"samevendor": ER and ER.get("samevendor"), "dinov2clip": ERD and ERD.get("dinov2clip"), "dinov2": BD}.get(enc)
            if not src: return None
            e = src["emidec"]; return (e["auroc"], *e["ci"]) if m == "emidec" else (e["infarct_spearman"], None, None)
        if m.startswith("det_"):
            t = m[4:]; key = {"dinov2": "dinov2_generic"}.get(enc, enc)
            src = DFD if enc == "dinov2clip" else DF
            return tuple(src[key][t]["site_held_out"]) if src and key in src else None
        if m == "mort":
            k = {"trained": "clinical+lvef+image"}.get(enc, f"clinical+lvef+image_{enc}")
            r = ML["cv"].get(k); return (r["diff_vs_first"], *r["diff_ci95"]) if r else None
        if m == "vendor":
            k = {"dinov2": "dinov2_generic"}.get(enc, enc); v = EMAP["vendor_auroc"].get(k) if EMAP else None
            return (v, None, None) if v is not None else None
    except (KeyError, TypeError):
        return None


RGRID = J("lge_clip", "risk_grid.json")


# ---------------------------------------------------------------- Figure 9: scorecard
def figure9():
    cols = [("r5", "Report\nretrieval\n(top-5)", True, lambda v: f"{100 * v:.0f}%"), ("emidec", "EMIDEC\nMI vs normal\n(AUROC)", True, lambda v: f"{v:.2f}"),
            ("emidec_rho", "EMIDEC\ninfarct size\n(ρ)", True, lambda v: f"{v:.2f}"), ("det_infarct_pct", "DETERMINE\ninfarct size\n(ρ, new sites)", True, lambda v: f"{v:.2f}"),
            ("det_lvef", "DETERMINE\nLVEF\n(ρ, new sites)", True, lambda v: f"{v:.2f}"), ("det_large_infarct", "DETERMINE\nlarge infarct\n(AUROC, new sites)", True, lambda v: f"{v:.2f}"),
            ("mort", "Mortality beyond\nclinical + LVEF\n(ΔAUROC)", True, lambda v: f"{v:+.3f}".replace("-", "−"))]
    p, s = deck(7.5, 3.6); x0, y0, rw, rh, lw = 1.3, 0.72, 0.86, 0.36, 1.2
    for j, (_, t, _, _) in enumerate(cols): text(s, x0 + j * rw, 0.08, rw, 0.6, t, size=6.3, bold=True)
    hi, lo = C["trained"], RGBColor(0xFF, 0xFF, 0xFF)
    for j, (m, _, better_high, fmt) in enumerate(cols):
        vals = {e: metric(e, m) for e in ENC}; xs = [v[0] for v in vals.values() if v]
        mn, mx = (min(xs), max(xs)) if xs else (0, 1)
        for i, e in enumerate(ENC):
            v = vals[e]; x, y = x0 + j * rw, y0 + i * rh
            b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x + 0.02), Inches(y + 0.02), Inches(rw - 0.04), Inches(rh - 0.04))
            b.line.fill.background(); b.shadow.inherit = False; b.fill.solid()
            if v is None:
                b.fill.fore_color.rgb = C["pale"]; text(s, x, y, rw, rh, "n/a", size=6.5, color="line", italic=True); continue
            f = (v[0] - mn) / (mx - mn) if mx > mn else 0.5; f = f if better_high else 1 - f
            b.fill.fore_color.rgb = RGBColor(*[int(lo[k] + (hi[k] - lo[k]) * f) for k in range(3)])
            text(s, x, y, rw, rh, fmt(v[0]), size=7.5, bold=e in ("trained", "samevendor"), color="white" if f > 0.6 else "ink")
    for i, e in enumerate(ENC):
        text(s, 0.05, y0 + i * rh, lw, rh, LAB1[e], size=7, bold=e in ("trained", "samevendor"), align=PP_ALIGN.RIGHT, color=e if e not in ("random", "dinov2clip") else "line")
    line(s, x0, y0 + 4 * rh, x0 + len(cols) * rw, y0 + 4 * rh, color="ink", width=0.75, dash=True)
    foot(s, 0.1, 2.95, 7.3, 0.55, "Frozen image encoders (rows) on every test (columns). Shading runs from the worst (white) to the best (dark blue) encoder within each column; "
         "Below the dashed line: report-supervised LGE encoders. n/a: not possible (no text tower). Image-only: same encoder, data and training without reports. "
         "New sites: probe trained and tested on different DETERMINE trial sites.", size=5.8, italic=True, align=PP_ALIGN.LEFT, color="line")
    p.save(os.path.join(OUT, "Figure_9_scorecard.pptx"))


# ---------------------------------------------------------------- Figure 10: external validation and controls
def figure10():
    p, s = deck(7.5, 5.6); W, H = 2.75, 1.45
    pos = [(0.65, 0.62), (4.35, 0.62), (0.65, 3.25), (4.35, 3.25)]
    heads = [("A", "DETERMINE: infarct size", "core lab, Spearman ρ, sites held out", "det_infarct_pct", -0.1, 0.3, (0, 0.1, 0.2, 0.3), lambda v: f"{v:.2f}", 0),
             ("B", "DETERMINE: LVEF", "core lab, Spearman ρ, sites held out", "det_lvef", -0.1, 0.5, (0, 0.2, 0.4), lambda v: f"{v:.2f}", 0),
             ("C", "EMIDEC: infarction vs normal", "AUROC, 100 examinations", "emidec", 0.4, 0.95, (0.5, 0.7, 0.9), lambda v: f"{v:.2f}", 0.5),
             ("D", "All-cause mortality beyond clinical + LVEF", "ΔAUROC added by image features (CV)", "mort", -0.03, 0.08, (0, 0.04, 0.08), lambda v: f"{v:+.3f}".replace("-", "−"), 0)]
    TF = {"D": lambda t: f"{t:+.2f}".replace("+0.00", "0")}
    for (x, y), (L, t1, t2, m, lo, hi, ticks, fmt, ref) in zip(pos, heads):
        panel(s, x - 0.6, y - 0.6, L); text(s, x - 0.3, y - 0.58, W + 0.4, 0.22, t1, size=8, bold=True); text(s, x - 0.3, y - 0.37, W + 0.4, 0.2, t2, size=6, italic=True, color="line")
        v = [metric(e, m) for e in ENC]
        bars(s, x, y, W, H, [a[0] if a else None for a in v], lo, hi, fmt, ci=[(a[1], a[2]) if a and a[1] is not None else (a[0], a[0]) if a else (0, 0) for a in v],
             ref=ref, reflabel="chance" if ref == 0.5 else None, ticks=ticks, tfmt=TF.get(L), keys=ENC, labels=LAB, labsize=5.3, valsize=6.5)
    foot(s, 0.1, 5.25, 7.3, 0.32, "Frozen encoders with a linear probe. DETERMINE: 634 trial patients from 56 sites, probe trained on some sites and tested on others "
         "(GroupKFold by site). D: grouped cross-validation over 3,170 examinations (369 deaths); reference = clinical variables + LVEF. Whiskers: 95% CI (bootstrap).",
         size=5.8, italic=True, align=PP_ALIGN.LEFT, color="line")
    p.save(os.path.join(OUT, "Figure_10_external.pptx"))


# ---------------------------------------------------------------- Figure 11: site shortcut and fix
def scatter(s, x, y, w, h, mp, title, by="ge"):
    """t-SNE map as a native XY scatter; by = 'ge' (vendor) or 'lvef' (LVEF from the report, 3 bands)"""
    cd = XyChartData()
    if by == "ge": groups = [("Siemens", "siemens", lambda i: mp["ge"][i] == 0), ("GE", "ge", lambda i: mp["ge"][i] == 1)]
    else: groups = [("LVEF > 50%", "low", lambda i: mp["lvef"][i] > 50), ("36-50%", "mid", lambda i: 35 < mp["lvef"][i] <= 50),
                    ("≤ 35%", "scar", lambda i: 0 <= mp["lvef"][i] <= 35)]
    for nm, _, sel in groups:
        sr = cd.add_series(nm)
        for i, (a, b) in enumerate(zip(mp["x"], mp["y"])):
            if sel(i): sr.add_data_point(a, b)
    ch = s.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER, Inches(x), Inches(y), Inches(w), Inches(h), cd).chart
    style_chart(ch, size=6.5, legend=True)
    from pptx.enum.chart import XL_TICK_LABEL_POSITION, XL_TICK_MARK
    for ax in (ch.value_axis, ch.category_axis):
        ax.tick_label_position = XL_TICK_LABEL_POSITION.NONE; ax.major_tick_mark = XL_TICK_MARK.NONE; ax.format.line.fill.background()
    ch.value_axis.has_major_gridlines = False
    for sr, (_, col, _) in zip(ch.plots[0].series, groups):
        sr.marker.size = 2; sr.marker.format.fill.solid(); sr.marker.format.fill.fore_color.rgb = C[col]; sr.marker.format.line.fill.background(); sr.format.line.fill.background()
    text(s, x, y - 0.22, w, 0.2, title, size=7.5, bold=True)


def figure11():
    if not (EMAP and EMAP.get("maps")): return
    p, s = deck(7.5, 5.3)
    panel(s, 0.05, 0.02, "A"); scatter(s, 0.2, 0.45, 2.3, 2.2, EMAP["maps"]["trained"], "Reports (standard batches)")
    panel(s, 2.55, 0.02, "B"); scatter(s, 2.7, 0.45, 2.3, 2.2, EMAP["maps"]["samevendor"], "Reports + same-vendor batches")
    panel(s, 5.05, 0.02, "C"); scatter(s, 5.2, 0.45, 2.3, 2.2, EMAP["maps"]["samevendor"], "Same map, colored by LVEF", by="lvef")
    # D: what the fix changes
    sv, tr = (ER or {}).get("samevendor"), (ER or {}).get("full")
    rows = [("Top-ranked report from same vendor", "lower = less shortcut", tr and tr["shortcut"]["top1_same_vendor"], sv and sv["shortcut"]["top1_same_vendor"], 0.5, 1.1, (0.5, 0.75, 1.0), lambda u: f"{100 * u:.0f}%", RG and None),
            ("Test retrieval, top-5", "same center", tr and tr["retrieval"]["recall@5"][0], sv and sv["retrieval"]["recall@5"][0], 0, 0.4, (0, 0.2, 0.4), lambda u: f"{100 * u:.0f}%", None),
            ("EMIDEC infarction vs normal", "AUROC, other country", metric("trained", "emidec")[0], metric("samevendor", "emidec")[0] if metric("samevendor", "emidec") else None, 0.5, 0.9, (0.5, 0.7, 0.9), lambda u: f"{u:.2f}", None),
            ("DETERMINE large infarct", "AUROC, sites held out", metric("trained", "det_large_infarct")[0] if metric("trained", "det_large_infarct") else None,
             metric("samevendor", "det_large_infarct")[0] if metric("samevendor", "det_large_infarct") else None, 0.5, 0.75, (0.5, 0.6, 0.7), lambda u: f"{u:.2f}", None)]
    panel(s, 0.05, 3.0, "D"); text(s, 0.3, 3.03, 7.0, 0.22, "What same-vendor batches change", size=8, bold=True)
    for k, (t1, t2, a, b, lo, hi, ticks, fmt, _) in enumerate(rows):
        x = 0.55 + k * 1.8
        text(s, x - 0.35, 3.28, 1.75, 0.2, t1, size=6.3, bold=True); text(s, x - 0.35, 3.45, 1.75, 0.18, t2, size=5.6, italic=True, color="line")
        bars(s, x + 0.05, 3.75, 1.15, 1.0, [a, b], lo, hi, fmt, ref=0.6953 if k == 0 else (0.5 if k >= 2 else None), reflabel="chance" if k == 0 else None,
             ticks=ticks, keys=("trained", "samevendor"), labels={"trained": "Standard", "samevendor": "Same-\nvendor"}, main="samevendor", labsize=5.6, valsize=7)
    foot(s, 0.1, 5.0, 7.3, 0.28, "A, B: t-SNE of the frozen embeddings of all 3,221 examinations, colored by scanner vendor (not used in training; vendor is "
         "decodable from every encoder's features, AUROC ≈ 1.0). C: map B colored by LVEF from the report (never used in training). "
         "Same-vendor batches: every training batch drawn from one vendor, so vendor cannot help tell the right report from the wrong ones.",
         size=5.8, italic=True, align=PP_ALIGN.LEFT, color="line")
    p.save(os.path.join(OUT, "Figure_11_shortcut.pptx"))


# ---------------------------------------------------------------- Figure 12: clinical risk groups
def figure12():
    if not RG: return
    p, s = deck(7.5, 6.35 if RGRID else 3.3); T3 = ["low", "middle", "high"]; K3 = ("low", "mid", "high"); L3 = {"low": "Low", "mid": "Middle", "high": "High"}
    pc_ = lambda u: f"{100 * u:.0f}%"
    groups = [("All examinations", RG["tertiles_all"], RG["n"], RG["deaths"], RG["auroc_image_score_all"])] + \
             [(f"LVEF {g.replace('<=', '≤ ').replace('>', '> ')}%", RG["by_lvef"][g]["tertiles"], RG["by_lvef"][g]["n"], RG["by_lvef"][g]["deaths"], RG["by_lvef"][g]["auroc_image_score"]) for g in ("<=35", "36-50", ">50")]
    top = max(max(t[k]["death_rate"][2] for k in T3) for _, t, *_ in groups); hi = min(1.0, np.ceil(top * 10) / 10 + 0.05)
    for k, (t1, tt, n_, d_, au) in enumerate(groups):
        x = 0.55 + k * 1.8
        if k == 0: panel(s, 0.05, 0.02, "A")
        if k == 1: panel(s, x - 0.5, 0.02, "B")
        text(s, x - 0.35, 0.05, 1.75, 0.22, t1, size=7.5, bold=True)
        text(s, x - 0.35, 0.27, 1.75, 0.2, f"{d_} deaths / {n_:,}; AUROC {au:.2f}", size=5.8, italic=True, color="line")
        if k == 0:
            yl_ = text(s, -0.62, 1.45, 1.6, 0.25, "All-cause mortality", size=8, bold=True, color="ink"); yl_.rotation = 270
        bars(s, x + 0.05, 0.75, 1.3, 1.8, [tt[t]["death_rate"][0] for t in T3], 0, hi, pc_, ci=[tt[t]["death_rate"][1:] for t in T3],
             ticks=tuple(np.round(np.arange(0, hi + 1e-9, 0.1), 2)), tfmt=pc_, keys=K3, labels=L3, main="high", labsize=6, valsize=7)
    foot(s, 0.1, 2.95, 7.3, 0.3, "Observed all-cause mortality at data extraction (95% Wilson CI) by tertile of the image risk score (out-of-fold probability from the frozen "
         "report-pretrained LGE features alone). B: within LVEF strata (LVEF from the report). No follow-up time: exposure differs by examination year.",
         size=5.8, italic=True, align=PP_ALIGN.LEFT, color="line")
    if RGRID:   # C: clinical-risk third x image-risk third
        y0, cw, ch, x0 = 3.45, 1.35, 0.72, 1.95
        line(s, 0.05, 3.05, 7.45, 3.05, color="grid")
        panel(s, 0.05, 3.12, "C"); text(s, 0.35, 3.13, 7, 0.26, "Hidden risk: all-cause mortality by clinical risk and image risk", size=9, bold=True, align=PP_ALIGN.LEFT)
        text(s, x0, y0, 3 * cw, 0.22, "Image risk (LGE features)", size=7.5, bold=True, color="trained")
        for j, it in enumerate(T3): text(s, x0 + j * cw, y0 + 0.24, cw, 0.2, L3[{"low": "low", "middle": "mid", "high": "high"}[it]], size=7.5, bold=True)
        yc = text(s, 0.05, y0 + 0.5 + 1.5 * ch - 0.15, 1.2, 0.3, "Clinical risk\n(age, risk factors, LVEF)", size=7.5, bold=True, color="clinical"); 
        hi_m = max(RGRID["grid"][ct][it]["mortality"][0] for ct in T3 for it in T3)
        for i, ct in enumerate(T3):
            text(s, x0 - 0.65, y0 + 0.5 + i * ch, 0.6, ch, L3[{"low": "low", "middle": "mid", "high": "high"}[ct]], size=7.5, bold=True, align=PP_ALIGN.RIGHT)
            for j, it in enumerate(T3):
                c_ = RGRID["grid"][ct][it]; m_ = c_["mortality"][0]; f = m_ / hi_m if hi_m else 0
                b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x0 + j * cw + 0.02), Inches(y0 + 0.5 + i * ch + 0.02), Inches(cw - 0.04), Inches(ch - 0.04))
                b.fill.solid(); b.fill.fore_color.rgb = RGBColor(255, int(255 - 200 * f), int(255 - 210 * f)); b.line.fill.background(); b.shadow.inherit = False
                text(s, x0 + j * cw, y0 + 0.5 + i * ch + 0.06, cw, 0.32, f"{100 * m_:.0f}%", size=11, bold=True, color="white" if f > 0.6 else "ink")
                text(s, x0 + j * cw, y0 + 0.5 + i * ch + 0.38, cw, 0.25, f"{c_['deaths']}/{c_['n']}", size=6.5, color="white" if f > 0.6 else "line")
        au = RGRID["image_auroc_within_clinical"]
        text(s, x0 + 3 * cw + 0.12, y0 + 0.5, 1.3, 3 * ch, "Image-score AUROC\nwithin clinical risk:\n" + "\n".join(f"{L3[{'low': 'low', 'middle': 'mid', 'high': 'high'}[ct]]}: {au[ct]:.2f}" for ct in T3 if au.get(ct)),
             size=7, color="line", align=PP_ALIGN.LEFT)
    p.save(os.path.join(OUT, "Figure_12_risk.pptx"))


# ---------------------------------------------------------------- Figure 13: saliency
def overlays():
    """example panels local: the LGE slice with the most expert scar, cropped to the heart (myocardium bounding box + margin),
    upsampled smoothly to 900 px; saliency smoothed (Gaussian, sigma 2.5 px at input resolution) and shown with the inferno
    colormap, opacity rising with saliency; expert scar outline in green. Same crop and scaling for every encoder."""
    from scipy import ndimage
    from matplotlib import cm
    d = os.path.join(ROOT, "docs", "figures", "images", "saliency"); dp = os.path.join(ROOT, "docs", "figures", "images_placeholder", "saliency")
    os.makedirs(d, exist_ok=True); os.makedirs(dp, exist_ok=True)
    f = os.path.join(O, "lge_clip", "saliency_examples.npz")
    if not os.path.exists(f): return False
    z = np.load(f); PX = 900
    for i in range(3):
        if f"img_{i}" not in z: continue
        im, mk = z[f"img_{i}"].astype(np.float32), z[f"mask_{i}"]; k = int(np.argmax((mk == 2).reshape(len(mk), -1).sum(1)))
        # square crop around the WHOLE heart: bounding box of blood pool + myocardium + scar, plus a 30% margin on every side;
        # the image is zero-padded first, so the crop never cuts the heart at the field-of-view edge
        heart = np.argwhere(np.isin(mk[k], [1, 2, 3])); (ya, xa), (yb, xb) = heart.min(0), heart.max(0)
        cy, cx = (ya + yb) / 2, (xa + xb) / 2; half = int(max(yb - ya, xb - xa) * 0.5 * 1.6) + 6; P_ = half + 2
        y0, x0 = int(round(cy)) - half + P_, int(round(cx)) - half + P_
        crop = lambda a_: np.pad(a_, P_)[y0:y0 + 2 * half, x0:x0 + 2 * half]
        base = crop(np.clip(im[k], 0, 1)); lo_, hi_ = np.percentile(base, [1, 99.5]); base = np.clip((base - lo_) / (hi_ - lo_ + 1e-6), 0, 1)
        big = np.asarray(Image.fromarray((base * 255).astype(np.uint8)).resize((PX, PX), Image.LANCZOS)).astype(np.float32) / 255
        scar = np.asarray(Image.fromarray((crop(mk[k]) == 2).astype(np.uint8) * 255).resize((PX, PX), Image.NEAREST)) > 127
        edge = scar & ~ndimage.binary_erosion(scar, iterations=3)
        gray = np.stack([big] * 3, -1)
        g = gray.copy(); g[edge] = [0.25, 1.0, 0.35]
        Image.fromarray((g * 255).astype(np.uint8)).save(os.path.join(d, f"ex{i}_lge.png"))
        for name in ("trained", "imgssl", "random"):
            if f"{name}_{i}" not in z: continue
            sal = ndimage.gaussian_filter(crop(z[f"{name}_{i}"].astype(np.float32)[k]), 2.5)
            sal = np.clip(sal / (np.percentile(sal, 99) + 1e-9), 0, 1)
            sal = np.asarray(Image.fromarray((sal * 255).astype(np.uint8)).resize((PX, PX), Image.BICUBIC)).astype(np.float32) / 255
            alpha = (sal ** 1.2 * 0.75)[..., None]; heat = cm.inferno(sal)[..., :3]
            o = gray * 0.85 * (1 - alpha) + heat * alpha; o[edge] = [0.25, 1.0, 0.35]
            Image.fromarray((np.clip(o, 0, 1) * 255).astype(np.uint8)).save(os.path.join(d, f"ex{i}_{name}.png"))
        for nm in ("lge", "trained", "imgssl", "random"):
            Image.new("RGB", (PX, PX), (90, 90, 90)).save(os.path.join(dp, f"ex{i}_{nm}.png"))
    return True


def figure13():
    if not SAL: return
    overlays()
    p, s = deck(7.5, 7.5); E5 = ["random", "cine", "imgssl", "trained", "samevendor"]
    for k, (reg, t1) in enumerate((("scar", "Expert-contoured scar"), ("myocardium", "LV myocardium (incl. scar)"))):
        x = 0.75 + k * 3.75
        panel(s, x - 0.7, 0.02, "AB"[k]); text(s, x - 0.3, 0.05, 3.2, 0.25, t1, size=10, bold=True)
        text(s, x - 0.3, 0.3, 3.2, 0.22, "saliency enrichment (1 = no preference)", size=8, italic=True, color="line")
        v = [SAL[e][reg]["enrichment"] if e in SAL and reg in SAL[e] else None for e in E5]; top = max(a[2] for a in v if a)
        hi = float(np.ceil(top)); bars(s, x, 0.75, 2.85, 1.65, [a[0] if a else None for a in v], 0, hi, lambda u: f"{u:.1f}", ci=[(a[1], a[2]) if a else (0, 0) for a in v],
                                      ref=1, ticks=tuple(range(0, int(hi) + 1)), keys=E5, labels=LAB, labsize=7.5, valsize=9)
    panel(s, 0.05, 2.95, "C"); text(s, 0.35, 2.97, 6.8, 0.25, "Examples: DETERMINE patients, expert scar outlined in green", size=10, bold=True)
    IMGD = os.path.join(M.IMG, "saliency"); w = 1.22; gx = 0.08; x0 = 0.75
    cols = [("lge", "LGE"), ("trained", "Reports"), ("imgssl", "Image-only"), ("random", "Random")]
    for j, (nm, lab) in enumerate(cols): text(s, x0 + j * (w + gx), 3.25, w, 0.22, lab, size=9, bold=True)
    for i in range(3):
        y = 3.5 + i * (w + gx)
        for j, (nm, _) in enumerate(cols):
            f = os.path.join(IMGD, f"ex{i}_{nm}.png")
            if os.path.exists(f): s.shapes.add_picture(f, Inches(x0 + j * (w + gx)), Inches(y), Inches(w), Inches(w))
        q = SAL["_examples"]["infarct_pct"][i]; text(s, x0 + 4 * (w + gx) + 0.05, y, 1.3, w, f"Infarct size\n{q:.0f}% of LV", size=9, color="line", align=PP_ALIGN.LEFT)
    p.save(os.path.join(OUT, "Figure_13_saliency.pptx"))


# ---------------------------------------------------------------- Figure 14: head-to-head forest plot
def figure14():
    if not PT: return
    ML2 = ML.get("cv_vs_report_image") if ML else None
    comps = [("trained_vs_dinov2_generic", "Reports − DINOv2 (generic)", "dinov2"), ("samevendor_vs_trained", "Same-vendor − standard batches", "samevendor")]
    tasks = [("infarct_pct|site_held_out", "DETERMINE infarct size (ρ)"), ("lvef|site_held_out", "DETERMINE LVEF (ρ)"), ("esv|site_held_out", "DETERMINE ESV (ρ)"),
             ("large_infarct|site_held_out", "DETERMINE large infarct (AUROC)"), ("emidec", "EMIDEC infarction (AUROC)"), ("mort", "Mortality beyond clinical + LVEF (AUROC)")]
    rows = []
    for tk, tl in tasks:
        for ck, cl, col in comps:
            if tk == "emidec": r = PT["emidec_mi_auroc"].get(ck)
            elif tk == "mort":
                r = None
                if ML2 and ck == "trained_vs_imgssl" and "clinical+lvef+image_imgssl" in ML2:
                    q = ML2["clinical+lvef+image_imgssl"]; r = {"diff": -q["diff_vs_first"], "ci": [-q["diff_ci95"][1], -q["diff_ci95"][0]]}
                if ML2 and ck == "trained_vs_dinov2_generic" and "clinical+lvef+image_dinov2" in ML2:
                    q = ML2["clinical+lvef+image_dinov2"]; r = {"diff": -q["diff_vs_first"], "ci": [-q["diff_ci95"][1], -q["diff_ci95"][0]]}
                if ML2 and ck == "samevendor_vs_trained" and "clinical+lvef+image_samevendor" in ML2:
                    q = ML2["clinical+lvef+image_samevendor"]; r = {"diff": q["diff_vs_first"], "ci": q["diff_ci95"]}
            else: r = PT["determine"].get(tk, {}).get(ck)
            if r is not None: rows.append((tl, cl, col, r))
    p, s = deck(7.5, 0.9 + 0.235 * len(rows) + 0.6); x0, xw, lo, hi = 3.0, 3.9, -0.2, 0.4; y0, rh = 0.45, 0.235
    xp = lambda v: x0 + (v - lo) / (hi - lo) * xw
    for t in (-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4):
        line(s, xp(t), y0 - 0.05, xp(t), y0 + len(rows) * rh, color="ink" if t == 0 else "grid", width=1 if t == 0 else 0.5, dash=False)
        text(s, xp(t) - 0.25, y0 + len(rows) * rh + 0.02, 0.5, 0.18, f"{t:+.1f}".replace("+0.0", "0").replace("-", "−"), size=6.5, color="line")
    text(s, x0, y0 + len(rows) * rh + 0.2, xw, 0.2, "Paired difference (95% CI)", size=7, color="line")
    last = None
    for i, (tl, cl, col, r) in enumerate(rows):
        y = y0 + i * rh + rh / 2
        if tl != last:
            text(s, 0.1, y - 0.1, 1.75, 0.2, tl, size=6.8, bold=True, align=PP_ALIGN.LEFT); last = tl
            if i: line(s, 0.1, y - rh / 2, x0 + xw, y - rh / 2, color="grid", width=0.5)
        text(s, 1.8, y - 0.09, 1.15, 0.18, cl, size=5.8, align=PP_ALIGN.RIGHT, color=col)
        if r is None: text(s, x0 + 0.05, y - 0.09, 1.0, 0.18, "not tested", size=5.6, italic=True, color="line", align=PP_ALIGN.LEFT); continue
        a, b = max(r["ci"][0], lo), min(r["ci"][1], hi); line(s, xp(a), y, xp(b), y, color=col, width=1.5)
        d_ = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(xp(r["diff"]) - 0.045), Inches(y - 0.045), Inches(0.09), Inches(0.09))
        d_.fill.solid(); d_.fill.fore_color.rgb = C[col]; d_.line.fill.background(); d_.shadow.inherit = False
        sig = r["ci"][0] > 0 or r["ci"][1] < 0
        text(s, x0 + xw + 0.02, y - 0.09, 0.55, 0.18, f"{r['diff']:+.2f}".replace("-", "−") + (" *" if sig else ""), size=6, bold=sig, align=PP_ALIGN.LEFT, color=col)
    foot(s, 0.1, y0 + len(rows) * rh + 0.45, 7.3, 0.35, "Head-to-head comparisons on the same examinations, folds and bootstrap samples (2,000). Positive: the first encoder is better. "
         "DETERMINE: sites held out (634 patients, 56 sites). * 95% CI excludes 0.", size=5.8, italic=True, align=PP_ALIGN.LEFT, color="line")
    p.save(os.path.join(OUT, "Figure_14_head_to_head.pptx"))


# ---------------------------------------------------------------- Figure 15: text-to-image search
def figure15():
    """text-to-image search: A, discrimination per typed finding (full width); B, the top-4 test examinations for three
    queries, shown large"""
    SD = J("lge_clip", "search_demo.json")
    if not SD: return
    Q = ["no_lge", "midwall", "sarcoid", "amyloid", "rv_insertion", "infarct"]
    QL = {"no_lge": "No LGE", "midwall": "Mid-wall septal", "sarcoid": "Sarcoidosis-like", "amyloid": "Amyloid-like", "rv_insertion": "RV insertion", "infarct": "Infarct pattern"}
    p, s = deck(7.5, 9.1)
    panel(s, 0.05, 0.02, "A"); text(s, 0.35, 0.04, 7, 0.26, "Type a finding: the model ranks the test examinations by image similarity", size=10, bold=True, align=PP_ALIGN.LEFT)
    text(s, 0.35, 0.3, 7, 0.22, "AUROC for the finding in each examination's own report (95% CI)", size=7.5, italic=True, color="line", align=PP_ALIGN.LEFT)
    x0, xw, y0, rh, lo, hi = 1.75, 4.9, 0.62, 0.33, 0.3, 0.9; xp = lambda v: x0 + (v - lo) / (hi - lo) * xw
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        line(s, xp(t), y0 - 0.05, xp(t), y0 + len(Q) * rh, color="ink" if t == 0.5 else "grid", width=1 if t == 0.5 else 0.5)
        text(s, xp(t) - 0.2, y0 + len(Q) * rh + 0.02, 0.4, 0.18, f"{t:.1f}", size=7, color="line")
    for i, q in enumerate(Q):
        y = y0 + i * rh + rh / 2; text(s, 0.1, y - 0.11, 1.6, 0.22, QL[q], size=8, align=PP_ALIGN.RIGHT)
        for w_, col, off in (("cine", "cine", -0.06), ("trained", "trained", 0.06)):
            r = SD[w_][q]; a_, b_ = r["auroc_ci95"]; line(s, xp(a_), y + off, xp(b_), y + off, color=col, width=1.75)
            d_ = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(xp(r["auroc"]) - 0.05), Inches(y + off - 0.05), Inches(0.1), Inches(0.1))
            d_.fill.solid(); d_.fill.fore_color.rgb = C[col]; d_.line.fill.background(); d_.shadow.inherit = False
        text(s, x0 + xw + 0.08, y - 0.11, 0.6, 0.22, f"{SD['trained'][q]['auroc']:.2f}", size=8, bold=True, color="trained", align=PP_ALIGN.LEFT)
    yl = y0 + len(Q) * rh + 0.22
    text(s, x0 + 0.6, yl, 1.6, 0.2, "● report-pretrained", size=7.5, color="trained")
    text(s, x0 + 2.5, yl, 1.8, 0.2, "● cine checkpoint (not aligned)", size=7.5, color="cine")
    # --- B: the returned examinations, large
    yb = yl + 0.4
    line(s, 0.05, yb - 0.08, 7.45, yb - 0.08, color="grid")
    panel(s, 0.05, yb, "B"); text(s, 0.35, yb + 0.02, 7, 0.26, "Top-4 test examinations returned for three queries", size=10, bold=True, align=PP_ALIGN.LEFT)
    D_ = os.path.join(M.IMG, "search"); w, gx = 1.45, 0.1; xs0 = (7.5 - 4 * w - 3 * gx) / 2
    for i, q in enumerate(["sarcoid", "midwall", "no_lge"]):
        y = yb + 0.36 + i * (w + 0.4)
        text(s, xs0, y, 4 * w + 3 * gx, 0.24, f"“{SD['queries'][q]}”", size=7.8, italic=True, align=PP_ALIGN.LEFT)
        for k in range(4):
            f = os.path.join(D_, f"{q}_{k}.png")
            if os.path.exists(f): s.shapes.add_picture(f, Inches(xs0 + k * (w + gx)), Inches(y + 0.27), Inches(w), Inches(w))
            text(s, xs0 + k * (w + gx) + 0.04, y + 0.3, 0.4, 0.2, f"#{k + 1}", size=8, bold=True, color="white", align=PP_ALIGN.LEFT)
    foot(s, 0.1, 8.2, 7.3, 0.3, "Search figure.", size=5.8, italic=True, align=PP_ALIGN.LEFT, color="line")
    p.save(os.path.join(OUT, "Figure_15_search.pptx"))

if __name__ == "__main__":
    for f in (figure9, figure10, figure11, figure12, figure13, figure14, figure15):
        try: f(); print("built", f.__name__)
        except Exception as ex: print("FAILED", f.__name__, type(ex).__name__, ex)
