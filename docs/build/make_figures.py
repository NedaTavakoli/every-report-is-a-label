#!/usr/bin/env python3
"""
Editable PowerPoint figures for the CLIP/LGE manuscript: every box, arrow and chart is a native PowerPoint object.
One file per figure (JCMR wants separate figure files) in docs/figures/:
  Figure_1_cohort.pptx, Figure_2_design.pptx, Figure_3_retrieval.pptx, Figure_4_examples.pptx,
  Figure_5_downstream.pptx, Figure_6_key_results.pptx, Figure_7_one_exam.pptx, Graphical_abstract.pptx
All numbers are read from outputs/ (cohort_counts.json, cohort_table1.json, lge_clip/*.json, train_log.csv).
Patient images: example slices chosen by objective criteria (extract_example_slices.py);
the layout is checked on the FIG_PLACEHOLDER=1 copy. Report excerpts in Figures 2 and 4 are verbatim (extract_example_reports.py); the author checks them.
Run: .venv/bin/python docs/build/make_figures.py
"""
import os, json, csv
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.chart.data import XyChartData, CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_TICK_LABEL_POSITION
from pptx.oxml.ns import qn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# FIG_PLACEHOLDER=1 builds a layout-check copy with gray placeholders instead of patient images (docs/figures/_layout_check)
PLACEHOLDER = os.environ.get("FIG_PLACEHOLDER") == "1"
OUT = os.path.join(ROOT, "docs", "figures", "_layout_check" if PLACEHOLDER else ""); os.makedirs(OUT, exist_ok=True)
IMG = os.path.join(ROOT, "docs", "figures", "images_placeholder" if PLACEHOLDER else "images")
EXAMPLES = json.load(open(os.path.join(ROOT, "docs", "figures", "images", "examples.json")))
EXC = json.load(open(os.path.join(IMG, "report_excerpts.json")))   # verbatim report excerpts (extract_example_reports.py)
J = lambda f: json.load(open(os.path.join(ROOT, "outputs", f)))
CC, T = J("cohort_counts.json"), J("cohort_table1.json")
RC, PR, MO = J("lge_clip/retrieval_ci.json"), J("lge_clip/clinical_probes.json"), J("lge_clip/mortality.json")
ER = J("lge_clip/eval_runs.json") if os.path.exists(os.path.join(ROOT, "outputs", "lge_clip", "eval_runs.json")) else None
FT = {(w, m): J(f"lge_clip/finetune_deceased_{w}_{m}.json")["auroc"] for w in ["random", "cine", "trained"] for m in ["frozen", "finetune", "transfer"]}
TL = list(csv.DictReader(open(os.path.join(ROOT, "outputs", "lge_clip", "train_log.csv"))))

FONT = "Arial"
TS = float(os.environ.get("FIG_TEXT_SCALE", "1.25"))   # all text and chart fonts scaled for print readability
NOTES = os.environ.get("FIG_NOTES") == "1"             # explanatory notes inside figures: off (they belong in the legends)
C = {"random": RGBColor(0x9E, 0x9E, 0x9E), "cine": RGBColor(0xE6, 0x9F, 0x00), "trained": RGBColor(0x00, 0x72, 0xB2),
     "report": RGBColor(0x00, 0x9E, 0x73), "clinical": RGBColor(0x44, 0x44, 0x44), "ink": RGBColor(0x22, 0x22, 0x22),
     "line": RGBColor(0x66, 0x66, 0x66), "pale": RGBColor(0xF2, 0xF2, 0xF2), "paleblue": RGBColor(0xE3, 0xF0, 0xF8),
     "palegreen": RGBColor(0xE0, 0xF3, 0xEC), "black": RGBColor(0x0B, 0x0B, 0x0B), "scar": RGBColor(0xC0, 0x39, 0x2B), "grid": RGBColor(0xDD, 0xDD, 0xDD), "white": RGBColor(0xFF, 0xFF, 0xFF)}
LBL = {"random": "Random weights", "cine": "Cine checkpoint", "trained": "Report-pretrained"}
n = lambda x: f"{x:,}"


def deck(w, h):
    p = Presentation(); p.slide_width, p.slide_height = Inches(w), Inches(h)
    return p, p.slides.add_slide(p.slide_layouts[6])


def text(s, x, y, w, h, t, size=8, bold=False, color="ink", align=PP_ALIGN.CENTER, italic=False, anchor=MSO_ANCHOR.MIDDLE):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame
    tf.word_wrap = True; tf.vertical_anchor = anchor
    for a in ("margin_left", "margin_right", "margin_top", "margin_bottom"): setattr(tf, a, Inches(0.02))
    for i, line in enumerate(t.split("\n")):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); para.alignment = align
        r = para.add_run(); r.text = line; f = r.font
        f.name, f.size, f.bold, f.italic, f.color.rgb = FONT, Pt(size * TS), bold, italic, C[color]
    return tb


def foot(*a, **k):
    """explanatory note inside a figure: drawn only with FIG_NOTES=1 (JCMR: explanations go in the legend)"""
    return text(*a, **k) if NOTES else None


def box(s, x, y, w, h, t, fill="white", line="line", size=8, bold_first=True, shape=MSO_SHAPE.ROUNDED_RECTANGLE, color="ink", align=PP_ALIGN.CENTER):
    b = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE: b.adjustments[0] = 0.08
    b.fill.solid(); b.fill.fore_color.rgb = C[fill]
    if line: b.line.color.rgb = C[line]; b.line.width = Pt(0.75)
    else: b.line.fill.background()
    b.shadow.inherit = False
    tf = b.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for a in ("margin_left", "margin_right"): setattr(tf, a, Inches(0.05))
    for a in ("margin_top", "margin_bottom"): setattr(tf, a, Inches(0.02))
    for i, line_t in enumerate(t.split("\n")):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); para.alignment = align
        r = para.add_run(); r.text = line_t; f = r.font
        f.name, f.size, f.bold, f.color.rgb = FONT, Pt(size * TS), bold_first and i == 0, C[color]
    return b


def arrow(s, x1, y1, x2, y2, color="line", width=1.0, dash=False):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = C[color]; c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    if dash:
        d = ln.makeelement(qn("a:prstDash"), {"val": "dash"}); ln.append(d)
    ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    return c


def line(s, x1, y1, x2, y2, color="line", width=0.75, dash=False):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = C[color]; c.line.width = Pt(width)
    if dash:
        ln = c.line._get_or_add_ln(); ln.append(ln.makeelement(qn("a:prstDash"), {"val": "dash"}))
    return c


def dot(s, x, y, r, color, filled=True):
    d = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x - r), Inches(y - r), Inches(2 * r), Inches(2 * r))
    d.shadow.inherit = False; d.line.color.rgb = C[color]; d.line.width = Pt(1)
    if filled: d.fill.solid(); d.fill.fore_color.rgb = C[color]
    else: d.fill.solid(); d.fill.fore_color.rgb = C["white"]
    return d


def panel(s, x, y, letter): text(s, x, y, 0.3, 0.25, letter, size=11, bold=True, align=PP_ALIGN.LEFT)


def style_chart(ch, size=7, legend=True, ytitle=None, xtitle=None, ymin=None, ymax=None, ymajor=None, xmin=None, xmax=None, xmajor=None, numfmt=None):
    ch.font.name, ch.font.size = FONT, Pt(size * TS)
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(size * TS)
    va, ca = ch.value_axis, ch.category_axis
    va.has_major_gridlines = True; va.major_gridlines.format.line.color.rgb = C["grid"]; va.major_gridlines.format.line.width = Pt(0.5)
    for ax in (va, ca):
        ax.format.line.color.rgb = C["line"]; ax.tick_labels.font.size = Pt(size * TS)
    if ymin is not None: va.minimum_scale = ymin
    if ymax is not None: va.maximum_scale = ymax
    if ymajor: va.major_unit = ymajor
    if numfmt: va.tick_labels.number_format = numfmt; va.tick_labels.number_format_is_linked = False
    if xmin is not None: ca.minimum_scale = xmin
    if xmax is not None: ca.maximum_scale = xmax
    if xmajor: ca.major_unit = xmajor
    ca.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    for ax, t in ((va, ytitle), (ca, xtitle)):
        if t:
            ax.has_title = True; ax.axis_title.text_frame.text = t
            r = ax.axis_title.text_frame.paragraphs[0].runs[0]; r.font.size = Pt(size * TS); r.font.bold = False; r.font.name = FONT


def color_series(ch, colors, lines=True, width=1.5, dashes=None):
    for i, (sr, col) in enumerate(zip(ch.plots[0].series, colors)):
        if lines:
            sr.format.line.color.rgb = C[col]; sr.format.line.width = Pt(width); sr.smooth = False
            if dashes and dashes[i]:
                ln = sr.format.line._get_or_add_ln(); ln.append(ln.makeelement(qn("a:prstDash"), {"val": "dash"}))
        else:
            sr.format.fill.solid(); sr.format.fill.fore_color.rgb = C[col]; sr.format.line.fill.background()


# ---------------------------------------------------------------- Figure 1: cohort flow
def figure1():
    p, s = deck(7.5, 7.0)
    R, I, L, W = CC["report export"], CC["image delivery (crosswalk)"], CC["LGE magnitude (LGE_MAG_ONLY)"], CC["with report"]
    X, BW, BH = 1.2, 3.6, 0.62; ys = [0.15, 1.15, 2.15, 3.15, 4.15]
    main = [f"CMR report export\n{n(R['rows'])} report rows · {n(R['exams (accession)'])} examinations · {n(R['patients (MRN)'])} patients",
            f"De-identified image delivery\n{n(I['exams'])} examinations · {n(I['patients'])} patients",
            f"Examinations with LGE magnitude series\n{n(L['exam folders'])} examinations · {n(L['patient folders'])} patients",
            f"LGE series and linked report\n{n(W['LGE exams with report'])} examinations · {n(W['LGE patients with report'])} patients",
            f"Study cohort\n{n(T['all']['exams'])} examinations · {n(T['all']['N'])} patients · {n(T['all']['series'])} LGE series"]
    for i, (y, t) in enumerate(zip(ys, main)):
        box(s, X, y, BW, BH, t, fill="paleblue" if i == 4 else "white", size=8)
        if i: arrow(s, X + BW / 2, ys[i - 1] + BH, X + BW / 2, y)
    # side boxes (exclusions)
    SX, SW = 5.25, 2.1
    side = [(0.96, f"Reports linked to images by accession number\n(crosswalk to anonymized IDs)"),
            (2.96, f"Excluded: {n(W['LGE exams without report'])} LGE examinations without a linked report"),
            (3.96, f"Excluded: {n(W['LGE exams with report'] - T['all']['exams'])} examinations with no eligible series\n(long-inversion-time or other views only)")]
    for y, t in side:
        box(s, SX, y - 0.25, SW, 0.5, t, fill="pale", size=7, bold_first=False)
        line(s, X + BW / 2, y, SX, y, dash=True)
    # splits
    sp = [("train", "Training"), ("val", "Validation"), ("test", "Test"), ("expert", "Expert-labeled\n(held out)")]
    yb, bw, gap = 5.45, 1.6, 0.17; x0 = (7.5 - (4 * bw + 3 * gap)) / 2
    line(s, x0 + bw / 2, yb - 0.3, x0 + 3 * (bw + gap) + bw / 2, yb - 0.3)
    line(s, X + BW / 2, ys[-1] + BH, X + BW / 2, yb - 0.3)
    text(s, X + BW / 2 + 0.05, ys[-1] + BH + 0.02, 3.0, 0.3, "Split by patient (80/10/10); expert-labeled patients held out", size=7, align=PP_ALIGN.LEFT, italic=True)
    for k, (key, name) in enumerate(sp):
        x = x0 + k * (bw + gap); t = T[key]
        arrow(s, x + bw / 2, yb - 0.3, x + bw / 2, yb)
        use = {"train": "Pretraining; probe and model fitting", "val": "Monitoring only", "test": "Primary and secondary endpoints", "expert": "Held out of pretraining"}[key]
        box(s, x, yb, bw, 1.1, f"{name}\n{n(t['N'])} patients\n{n(t['exams'])} examinations\n{n(t['series'])} series\n{t['dead']} deceased\n{use}",
            fill="paleblue" if key == "test" else "white", size=7)
    text(s, 0.2, 6.6, 7.1, 0.35, "Deceased: vital status at data extraction. No natural language processing or large language model output was used to select the cohort.",
         size=7, align=PP_ALIGN.LEFT, italic=True)
    p.save(os.path.join(OUT, "Figure_1_cohort.pptx"))


# ---------------------------------------------------------------- Figure 2: design / architecture
VIEWNAME = {"sa": "Short axis", "2ch": "2-chamber", "3ch": "3-chamber", "4ch": "4-chamber"}


CROP = 0.2      # central 60% of the field of view (cardiac scans are centered on the heart); re-crop by hand in PowerPoint if needed
TAG = {"sa": "SAX", "2ch": "2CH", "3ch": "3CH", "4ch": "4CH"}


def img(s, ex, view, x, y, w, label=True, size=6.5, tag=False, tagsize=7):
    pic = s.shapes.add_picture(os.path.join(IMG, f"{ex}_{view}.png"), Inches(x), Inches(y), Inches(w), Inches(w))
    pic.crop_left = pic.crop_right = pic.crop_top = pic.crop_bottom = CROP     # native crop: the full image stays in the file
    pic.line.color.rgb = C["white"]; pic.line.width = Pt(0.5)
    if label: text(s, x, y + w, w, 0.16, VIEWNAME[view], size=size)
    if tag: text(s, x + 0.03, y + 0.02, 0.6, 0.2, TAG[view], size=tagsize, bold=True, color="white", align=PP_ALIGN.LEFT)
    return pic


def bullseye(s, cx, cy, R, scar=(), color="scar"):
    """AHA 17-segment bullseye; scar = segment numbers to fill. Angles: PowerPoint, clockwise from 3 o'clock."""
    six = [(240, 300), (180, 240), (120, 180), (60, 120), (0, 60), (300, 360)]          # anterior, anteroseptal, inferoseptal, inferior, inferolateral, anterolateral
    four = [(225, 315), (135, 225), (45, 135), (315, 405)]                                # anterior, septal, inferior, lateral
    rings = [(R, 0.125, six, 1), (0.75 * R, 1 / 6, six, 7), (0.5 * R, 0.25, four, 13)]
    for r, th, sect, first in rings:
        for k, (a0, a1) in enumerate(sect):
            seg = first + k
            sh = s.shapes.add_shape(MSO_SHAPE.BLOCK_ARC, Inches(cx - r), Inches(cy - r), Inches(2 * r), Inches(2 * r))
            sh.adjustments[0] = 0.6 * (a0 % 360); sh.adjustments[1] = 0.6 * (a1 % 360 if a1 % 360 else 360); sh.adjustments[2] = th
            sh.fill.solid(); sh.fill.fore_color.rgb = C[color] if seg in scar else C["white"]
            sh.line.color.rgb = C["line"]; sh.line.width = Pt(0.5); sh.shadow.inherit = False
    ap = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - R / 4), Inches(cy - R / 4), Inches(R / 2), Inches(R / 2))
    ap.fill.solid(); ap.fill.fore_color.rgb = C["white"]; ap.line.color.rgb = C["line"]; ap.line.width = Pt(0.5); ap.shadow.inherit = False


def vec(s, x, y, n_, col, size=0.09, label=None):
    """embedding drawn as a strip of squares with varying shade"""
    import random; rnd = random.Random(7 + n_ + len(col))
    base = C[col]
    for i in range(n_):
        f = 0.25 + 0.75 * rnd.random()
        rgb = RGBColor(*[int(255 - (255 - c) * f) for c in (base[0], base[1], base[2])])
        q = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x + i * size), Inches(y), Inches(size), Inches(size))
        q.fill.solid(); q.fill.fore_color.rgb = rgb; q.line.color.rgb = C["white"]; q.line.width = Pt(0.25); q.shadow.inherit = False
    if label: text(s, x - 0.1, y + size + 0.01, n_ * size + 0.2, 0.18, label, size=6)


def doc_icon(s, x, y, w, h, col="report", lines=4):
    d = s.shapes.add_shape(MSO_SHAPE.FOLDED_CORNER, Inches(x), Inches(y), Inches(w), Inches(h))
    d.fill.solid(); d.fill.fore_color.rgb = C["white"]; d.line.color.rgb = C[col]; d.line.width = Pt(0.75); d.shadow.inherit = False
    for i in range(lines):
        line(s, x + 0.12 * w, y + h * (0.22 + 0.17 * i), x + w * (0.85 if i % 2 == 0 else 0.6), y + h * (0.22 + 0.17 * i), color=col, width=0.75)


def figure2():
    """study design, drawn large: image and report branches (tinted bands) into one contrastive objective; B: four tests"""
    p, s = deck(7.5, 7.15)
    ex = EXAMPLES[0]
    def band(x, y, w, h, col):
        r = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)); r.adjustments[0] = 0.06
        r.fill.solid(); r.fill.fore_color.rgb = C[col]; r.line.fill.background(); r.shadow.inherit = False
    panel(s, 0.05, 0.02, "A"); text(s, 0.35, 0.03, 7, 0.28, "Report-supervised pretraining: each LGE examination is matched to its own report", size=10, bold=True, align=PP_ALIGN.LEFT)
    band(0.05, 0.42, 5.2, 2.42, "paleblue"); band(0.05, 2.95, 5.2, 1.85, "palegreen")
    text(s, 2.55, 0.48, 2.6, 0.22, "IMAGES", size=7.5, bold=True, color="trained", align=PP_ALIGN.RIGHT)
    text(s, 2.55, 3.0, 2.6, 0.22, "REPORT TEXT", size=7.5, bold=True, color="report", align=PP_ALIGN.RIGHT)
    # --- image branch
    box(s, 0.13, 0.5, 2.15, 2.26, "", fill="black", line=None)
    text(s, 0.13, 0.52, 2.15, 0.22, "One LGE examination", size=8.5, bold=True, color="white")
    for k, v in enumerate(["sa", "2ch", "3ch", "4ch"]):
        img(s, ex, v, 0.2 + (k % 2) * 1.02, 0.77 + (k // 2) * 0.98, 0.96, label=False, tag=True, tagsize=7)
    arrow(s, 2.35, 1.63, 2.62, 1.63, width=2.25)
    box(s, 2.67, 1.08, 1.5, 1.1, "Vision encoder\nvideo transformer,\nstarts from cine CMR", fill="trained", line=None, size=8, color="white")
    arrow(s, 4.22, 1.63, 4.45, 1.63, width=2.25)
    vec(s, 4.5, 1.56, 5, "trained", size=0.13, label="image vector")
    # --- report branch
    box(s, 0.13, 3.05, 2.15, 1.67, "", fill="white", line="report")
    text(s, 0.19, 3.08, 2.05, 0.22, "Its report (excerpt)", size=8.5, bold=True, color="report", align=PP_ALIGN.LEFT)
    text(s, 0.19, 3.32, 2.05, 1.36, "“" + EXC[ex] + "”", size=7.2, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=True)
    arrow(s, 2.35, 3.88, 2.62, 3.88, width=2.25)
    box(s, 2.67, 3.33, 1.5, 1.1, "Text encoder\nclinical language\nmodel", fill="report", line=None, size=8, color="white")
    arrow(s, 4.22, 3.88, 4.45, 3.88, width=2.25)
    vec(s, 4.5, 3.81, 5, "report", size=0.13, label="report vector")
    # --- contrastive matrix
    mx, my, cs = 6.08, 1.62, 0.44
    text(s, 5.35, 0.45, 2.1, 0.25, "Contrastive objective", size=9, bold=True)
    text(s, 5.35, 0.72, 2.1, 0.4, "pull each examination toward\nits own report in the batch", size=7, color="line")
    for i, e in enumerate(EXAMPLES[:3]):
        img(s, e, "sa", mx - cs - 0.04, my + i * cs + 0.03, cs - 0.06, label=False)
        doc_icon(s, mx + i * cs + 0.08, my - cs + 0.04, cs - 0.16, cs - 0.08, lines=3)
        for j in range(3):
            c_ = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(mx + j * cs), Inches(my + i * cs), Inches(cs), Inches(cs))
            c_.fill.solid(); c_.fill.fore_color.rgb = C["trained"] if i == j else C["pale"]; c_.line.color.rgb = C["white"]; c_.line.width = Pt(1.5); c_.shadow.inherit = False
            if i == j: text(s, mx + j * cs, my + i * cs + cs / 2 - 0.1, cs, 0.2, "✓", size=10, bold=True, color="white")
    text(s, mx - cs - 0.05, my + 3 * cs + 0.05, 4 * cs + 0.1, 0.4, "diagonal: matching pairs\n(×32 per batch)", size=7, color="line")
    arrow(s, 5.3, 1.63, mx - cs - 0.08, my + 0.2, width=1.5)
    arrow(s, 5.3, 3.88, mx + 1.5 * cs, my + 3 * cs + 0.47, width=1.5)
    text(s, 0.1, 4.88, 7.3, 0.26, f"{n(T['train']['N'])} training patients  ·  {n(T['train']['series'])} LGE series  ·  no manual labels",
         size=8.5, bold=True, color="ink", align=PP_ALIGN.CENTER)
    # --- panel B: the tests, as cards with an accent bar, icon and headline result
    line(s, 0.05, 5.25, 7.45, 5.25, color="grid")
    panel(s, 0.05, 5.3, "B"); text(s, 0.35, 5.31, 7, 0.28, "What the frozen image features were tested on", size=10, bold=True, align=PP_ALIGN.LEFT)
    tiles = [("Report retrieval", "320 held-out reports"), ("Multicenter trial", "DETERMINE core lab,\n56 sites"),
             ("Mortality beyond LVEF", "added to clinical\nvariables + LVEF"), ("Site shortcut", "fixed by same-\nvendor batches")]
    DFj = J("lge_clip/determine_full_eval.json") if os.path.exists(os.path.join(ROOT, "outputs", "lge_clip", "determine_full_eval.json")) else None
    MLj = J("lge_clip/mortality_lvef.json") if os.path.exists(os.path.join(ROOT, "outputs", "lge_clip", "mortality_lvef.json")) else None
    res = [f"top-5 {100 * RC['trained']['recall@5']['estimate']:.0f}%\nvs {500 / RC['n_exams']:.1f}% chance",
           f"infarct ρ {DFj['trained']['infarct_pct']['site_held_out'][0]:.2f}\nLVEF ρ {DFj['trained']['lvef']['site_held_out'][0]:.2f}" if DFj else "",
           f"AUROC\n{MLj['cv']['clinical+lvef']['estimate']:.2f} → {MLj['cv']['clinical+lvef+image']['estimate']:.2f}" if MLj else "",
           f"same vendor\n{100 * ER['full']['shortcut']['top1_same_vendor']:.0f}% → {100 * ER['samevendor']['shortcut']['top1_same_vendor']:.0f}%" if ER and "samevendor" in ER else ""]
    bw, gap, y = 1.76, 0.11, 5.68
    for k, (t1, t2) in enumerate(tiles):
        x = 0.1 + k * (bw + gap)
        box(s, x, y, bw, 1.42, "", fill="white", line="grid")
        acc = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(bw), Inches(0.07))
        acc.fill.solid(); acc.fill.fore_color.rgb = C["trained"]; acc.line.fill.background(); acc.shadow.inherit = False
        ix, iy = x + 0.1, y + 0.16
        if k == 0:
            doc_icon(s, ix, iy, 0.28, 0.36, col="trained", lines=3)
            o = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(ix + 0.15), Inches(iy + 0.15), Inches(0.2), Inches(0.2)); o.fill.background(); o.line.color.rgb = C["ink"]; o.line.width = Pt(1.5); o.shadow.inherit = False
            line(s, ix + 0.32, iy + 0.32, ix + 0.42, iy + 0.42, color="ink", width=2)
        elif k == 1:
            bullseye(s, ix + 0.21, iy + 0.21, 0.2, scar=(4, 10, 15), color="scar")
        elif k == 2:
            h_ = s.shapes.add_shape(MSO_SHAPE.HEART, Inches(ix), Inches(iy), Inches(0.42), Inches(0.39))
            h_.fill.solid(); h_.fill.fore_color.rgb = C["scar"]; h_.line.fill.background(); h_.shadow.inherit = False
            pts = [(0, .2), (.12, .2), (.16, .08), (.21, .32), (.26, .2), (.42, .2)]
            for (x1, y1), (x2, y2) in zip(pts, pts[1:]): line(s, ix + x1, iy + y1, ix + x2, iy + y2, color="white", width=1.25)
        else:
            for j, (hgt, col) in enumerate([(0.36, "trained"), (0.26, "samevendor_c")]):
                q = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(ix + 0.05 + j * 0.17), Inches(iy + 0.4 - hgt), Inches(0.13), Inches(hgt))
                q.fill.solid(); q.fill.fore_color.rgb = C[col] if col in C else RGBColor(0x00, 0x80, 0x80); q.line.fill.background(); q.shadow.inherit = False
            line(s, ix, iy + 0.4, ix + 0.42, iy + 0.4, color="ink")
        text(s, x + 0.56, y + 0.13, bw - 0.6, 0.25, t1, size=8.5, bold=True, align=PP_ALIGN.LEFT)
        text(s, x + 0.56, y + 0.38, bw - 0.6, 0.36, t2, size=6.8, color="line", align=PP_ALIGN.LEFT)
        text(s, x + 0.06, y + 0.8, bw - 0.12, 0.55, res[k], size=9.5, bold=True, color="trained")
    p.save(os.path.join(OUT, "Figure_2_design.pptx"))


# ---------------------------------------------------------------- Figure 4: example test examinations
def figure_examples():
    p, s = deck(7.5, 7.45)
    w, gx, RH = 1.76, 0.06, 2.45
    for i, e in enumerate(EXAMPLES):
        y = 0.05 + i * RH
        text(s, 0.08, y, 7.3, 0.22, f"Example {i + 1}  ·  its own report ranked 1st of {RC['n_exams']} by the pretrained model", size=8, bold=True, align=PP_ALIGN.LEFT)
        bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.08), Inches(y + 0.25), Inches(7.34), Inches(w + 0.04))
        bg.fill.solid(); bg.fill.fore_color.rgb = C["black"]; bg.line.fill.background(); bg.shadow.inherit = False
        for k, v in enumerate(["sa", "2ch", "3ch", "4ch"]):
            img(s, e, v, 0.12 + k * (w + gx), y + 0.27, w, label=False, tag=True, tagsize=9)
        text(s, 0.08, y + 0.27 + w + 0.04, 7.34, 0.36, "Report: “" + EXC[e] + "”", size=7, italic=True, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, color="ink")
    p.save(os.path.join(OUT, "Figure_4_examples.pptx"))


# ---------------------------------------------------------------- Figure 3: training and retrieval
def figure3():
    p, s = deck(7.5, 6.3)
    ep = [int(r["epoch"]) + 1 for r in TL]; loss = [float(r["loss"]) for r in TL]
    cd = XyChartData(); sr = cd.add_series("Training loss")
    for e, l in zip(ep, loss): sr.add_data_point(e, l)
    ch = s.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, Inches(0.2), Inches(0.3), Inches(3.6), Inches(2.5), cd).chart
    style_chart(ch, size=8, legend=False, ytitle="Contrastive loss", xtitle="Epoch", ymin=0, ymax=3.5, ymajor=0.5, xmin=0, xmax=200, xmajor=50, numfmt="0.0")
    color_series(ch, ["clinical"], width=2.25); panel(s, 0.05, 0.02, "A")
    v = [(int(r["epoch"]) + 1, float(r["val_recall@1"]), float(r["val_recall@5"])) for r in TL if r["val_recall@1"]]
    cd = XyChartData()
    for name, idx in (("Top-1 recall", 1), ("Top-5 recall", 2)):
        sr = cd.add_series(name)
        for row in v: sr.add_data_point(row[0], row[idx])
    ch = s.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES, Inches(3.85), Inches(0.3), Inches(3.6), Inches(2.5), cd).chart
    style_chart(ch, size=8, ytitle="Validation recall", xtitle="Epoch", ymin=0, ymax=0.4, ymajor=0.1, xmin=0, xmax=200, xmajor=50, numfmt="0%")
    color_series(ch, ["trained", "trained"], dashes=[False, True], width=2.25)
    for sr in ch.plots[0].series: sr.marker.size = 3; sr.marker.format.fill.solid(); sr.marker.format.fill.fore_color.rgb = C["trained"]; sr.marker.format.line.color.rgb = C["trained"]
    panel(s, 3.7, 0.02, "B")
    cd = XyChartData(); K = list(range(1, 51))
    for w in ["trained", "cine", "random"]:
        sr = cd.add_series(LBL[w])
        for k, r in zip(K, RC["recall_curve"][w]): sr.add_data_point(k, r)
    sr = cd.add_series("Chance")
    for k in K: sr.add_data_point(k, k / RC["n_exams"])
    ch = s.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, Inches(0.2), Inches(3.05), Inches(4.3), Inches(2.85), cd).chart
    style_chart(ch, size=8, ytitle=f"Test recall@k ({RC['n_exams']} candidate reports)", xtitle="k", ymin=0, ymax=0.8, ymajor=0.2, xmin=0, xmax=50, xmajor=10, numfmt="0%")
    color_series(ch, ["trained", "cine", "random", "clinical"], dashes=[False, False, False, True], width=2.25)
    panel(s, 0.05, 2.85, "C")
    # results table drawn as text rows (large, no grid): estimate on top, 95% CI below
    x0, y0 = 4.65, 3.15; cw = [0.95, 0.62, 0.62, 0.62]; xs_ = [x0 + sum(cw[:j]) for j in range(4)]
    panel(s, x0 - 0.15, 2.85, "D")
    for j, h in enumerate(["", "Top-1", "Top-5", "Median\nrank"]): text(s, xs_[j], y0, cw[j], 0.36, h, size=8, bold=True, align=PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER)
    line(s, x0, y0 + 0.4, x0 + sum(cw), y0 + 0.4, color="ink", width=0.75)
    for i, w in enumerate(["random", "cine", "trained"]):
        y = y0 + 0.5 + i * 0.72; col = {"random": "line", "cine": "cine", "trained": "trained"}[w]
        text(s, xs_[0], y, cw[0], 0.5, {"random": "Random\nweights", "cine": "Cine\ncheckpoint", "trained": "Report-\npretrained"}[w], size=8, bold=w == "trained", color=col, align=PP_ALIGN.LEFT)
        vals = [(f"{100 * RC[w]['recall@1']['estimate']:.1f}%", f"{100 * RC[w]['recall@1']['ci95'][0]:.1f}–{100 * RC[w]['recall@1']['ci95'][1]:.1f}"),
                (f"{100 * RC[w]['recall@5']['estimate']:.1f}%", f"{100 * RC[w]['recall@5']['ci95'][0]:.1f}–{100 * RC[w]['recall@5']['ci95'][1]:.1f}"),
                (f"{RC[w]['median_rank']['estimate']:g}", f"{RC[w]['median_rank']['ci95'][0]:g}–{RC[w]['median_rank']['ci95'][1]:g}")]
        for j, (v1, v2) in enumerate(vals):
            text(s, xs_[j + 1], y, cw[j + 1], 0.28, v1, size=9, bold=w == "trained", color=col)
            text(s, xs_[j + 1], y + 0.27, cw[j + 1], 0.22, v2, size=6.5, color="line")
    line(s, x0, y0 + 2.7, x0 + sum(cw), y0 + 2.7, color="ink", width=0.75)
    text(s, x0, y0 + 2.72, sum(cw), 0.22, f"chance: top-1 {100 / RC['n_exams']:.1f}%, top-5 {500 / RC['n_exams']:.1f}%", size=6.5, italic=True, color="line", align=PP_ALIGN.LEFT)
    p.save(os.path.join(OUT, "Figure_3_retrieval.pptx"))


# ---------------------------------------------------------------- Figure 5: clinical variables and vital status
def figure4():
    p, s = deck(7.5, 7.4)
    feats = [("image_random", "Image, random", "random"), ("image_cine", "Image, cine", "cine"), ("image_trained", "Image, report-pretrained", "trained"), ("report_trained", "Report text", "report")]
    bins = [("male", "Male sex"), ("diabetes", "Diabetes"), ("ckd", "CKD"), ("egfr_lt60", "eGFR < 60"), ("ever_smoker", "Ever smoker")]
    cd = CategoryChartData(); cd.categories = [b[1] for b in bins]
    for f, name, _ in feats: cd.add_series(name, [PR[k][f]["estimate"] for k, _ in bins])
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.2), Inches(0.25), Inches(4.3), Inches(2.4), cd).chart
    style_chart(ch, ytitle="Test AUROC", ymin=0.4, ymax=1.0, ymajor=0.1, numfmt="0.00"); color_series(ch, [x[2] for x in feats], lines=False)
    ch.plots[0].gap_width = 60; panel(s, 0.05, 0.02, "A")
    cont = [("age", "Age"), ("bmi", "BMI"), ("hdl", "HDL"), ("ldl", "LDL")]
    cd = CategoryChartData(); cd.categories = [c[1] for c in cont]
    for f, name, _ in feats: cd.add_series(name, [max(PR[k][f]["estimate"], 0) for k, _ in cont])
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(4.6), Inches(0.25), Inches(2.8), Inches(2.4), cd).chart
    style_chart(ch, legend=False, ytitle="Test R² (negative shown as 0)", ymin=0, ymax=0.7, ymajor=0.1, numfmt="0.00"); color_series(ch, [x[2] for x in feats], lines=False)
    ch.plots[0].gap_width = 60; panel(s, 4.45, 0.02, "B")

    # C: forest plot of AUROC difference vs the clinical model (drawn with shapes so every element is editable)
    panel(s, 0.05, 2.8, "C"); text(s, 0.35, 2.8, 4.2, 0.25, "Vital status: AUROC difference from the clinical model", size=8, bold=True, align=PP_ALIGN.LEFT)
    models = [("image", "Image only"), ("report", "Report only"), ("clinical+image", "Clinical + image"), ("clinical+report", "Clinical + report"),
              ("clinical+image+report", "Clinical + image + report"), ("clinical+image_cine", "Clinical + cine image"), ("clinical+image_random", "Clinical + random image")]
    lo, hi, X0, X1, Y0, RH = -0.15, 0.20, 1.6, 3.55, 3.3, 0.4
    xp = lambda v: X0 + (v - lo) / (hi - lo) * (X1 - X0)
    for t in [-0.1, 0, 0.1, 0.2]:
        line(s, xp(t), Y0 - 0.05, xp(t), Y0 + RH * len(models), color="grid" if t else "line", dash=t == 0)
        text(s, xp(t) - 0.25, Y0 + RH * len(models) + 0.02, 0.5, 0.2, f"{t:+.1f}" if t else "0", size=6.5)
    text(s, X0, Y0 + RH * len(models) + 0.2, X1 - X0, 0.22, "ΔAUROC (95% CI)", size=7)
    for i, (k, name) in enumerate(models):
        yc = Y0 + RH * i + RH / 2
        text(s, 0.05, yc - 0.12, 1.5, 0.24, name, size=7, align=PP_ALIGN.RIGHT)
        for part, dy, filled, col in (("cv", -0.07, True, "trained"), ("test", 0.07, False, "clinical")):
            r = MO[part][k]; a, b = r["diff_ci95"]
            line(s, xp(max(a, lo)), yc + dy, xp(min(b, hi)), yc + dy, color=col, width=1.25)
            dot(s, xp(r["diff_vs_first"]), yc + dy, 0.045, col, filled)
        cvr = MO["cv"][k]
        text(s, X1 + 0.05, yc - 0.12, 1.35, 0.24, f"{cvr['diff_vs_first']:+.3f} ({cvr['diff_ci95'][0]:+.3f}, {cvr['diff_ci95'][1]:+.3f})", size=6.5, align=PP_ALIGN.LEFT, color="trained")
    dot(s, 0.3, 6.62, 0.045, "trained", True); text(s, 0.4, 6.52, 2.6, 0.22, f"Cross-validation ({MO['events_cv']} deaths / {n(MO['n_cv'])} examinations)", size=6.5, align=PP_ALIGN.LEFT)
    dot(s, 0.3, 6.84, 0.045, "clinical", False); text(s, 0.4, 6.74, 2.6, 0.22, f"Test set ({MO['events_test']} deaths / {MO['n_test']} examinations)", size=6.5, align=PP_ALIGN.LEFT)
    foot(s, 0.15, 7.0, 7.2, 0.35, f"D: encoder frozen, projection fine-tuned (partial) or fully fine-tuned. Clinical model AUROC: {MO['cv']['clinical']['estimate']:.2f} (cross-validation), {MO['test']['clinical']['estimate']:.2f} (test). "
         "Text at right: cross-validation difference (95% CI).", size=6.5, italic=True, align=PP_ALIGN.LEFT)

    # D: attention head, image only
    modes = [("frozen", "Frozen"), ("finetune", "Partial"), ("transfer", "Full")]
    cd = CategoryChartData(); cd.categories = [m[1] for m in modes]
    for w in ["random", "cine", "trained"]: cd.add_series(LBL[w], [FT[(w, m)] for m, _ in modes])
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(5.1), Inches(3.05), Inches(2.35), Inches(3.4), cd).chart
    style_chart(ch, legend=True, ytitle="Test AUROC, image only", ymin=0.5, ymax=0.8, ymajor=0.05, numfmt="0.00")
    color_series(ch, ["random", "cine", "trained"], lines=False); ch.plots[0].gap_width = 40
    panel(s, 5.0, 2.8, "D"); text(s, 5.25, 2.8, 2.2, 0.25, "Image only, attention head", size=8, bold=True, align=PP_ALIGN.LEFT)
    p.save(os.path.join(OUT, "Figure_5_downstream.pptx"))


# ---------------------------------------------------------------- graphical abstract (JCMR: >= 1328 x 531 px, readable at 13 x 5 cm)
def graphical_abstract():
    p, s = deck(13.28, 5.31)
    text(s, 0.3, 0.15, 12.7, 0.6, "Every radiology report is a label: a multimodal LGE foundation model", size=20, bold=True)
    box(s, 0.3, 1.0, 3.4, 3.9, f"Data\n{n(T['all']['N'])} patients\n{n(T['all']['exams'])} LGE examinations\neach with its written report\n\nSingle center, 2006–2025", fill="pale", size=18)
    arrow(s, 3.75, 2.95, 4.35, 2.95, width=3)
    box(s, 4.4, 1.0, 3.6, 1.7, "", fill="black", line=None)
    img(s, EXAMPLES[0], "sa", 4.5, 1.08, 1.54, label=False, tag=True, tagsize=12)
    text(s, 6.1, 1.0, 1.85, 1.7, "LGE images\n→ vision\nencoder", size=18, bold=True, color="white")
    box(s, 4.4, 3.2, 3.6, 1.7, "Report text\n→ language model", fill="palegreen", size=18)
    text(s, 4.4, 2.68, 3.6, 0.55, "↕  contrastive alignment", size=16, bold=True, color="trained")
    arrow(s, 8.05, 2.95, 8.65, 2.95, width=3)
    DFj = J("lge_clip/determine_full_eval.json") if os.path.exists(os.path.join(ROOT, "outputs", "lge_clip", "determine_full_eval.json")) else None
    MLj = J("lge_clip/mortality_lvef.json") if os.path.exists(os.path.join(ROOT, "outputs", "lge_clip", "mortality_lvef.json")) else None
    res = [(f"All-cause mortality beyond clinical + LVEF: AUROC {MLj['cv']['clinical+lvef']['estimate']:.2f} → {MLj['cv']['clinical+lvef+image']['estimate']:.2f}" if MLj else "Death beyond LVEF", "trained"),
           (f"56-site trial, unseen sites: infarct size ρ {DFj['trained']['infarct_pct']['site_held_out'][0]:.2f}, LVEF ρ {DFj['trained']['lvef']['site_held_out'][0]:.2f}" if DFj else "Multicenter trial", "trained"),
           (f"Matches its own report: top-5 {100 * RC['trained']['recall@5']['estimate']:.0f}% vs {500 / RC['n_exams']:.1f}% chance", "trained"),
           ("Same-vendor batches remove a site shortcut and improve transfer", "samevendor" if "samevendor" in C else "report")]
    for k, (t, col) in enumerate(res):
        box(s, 8.7, 1.0 + k * 1.0, 4.3, 0.85, t, fill="white", size=15, bold_first=False, color=col)
    p.save(os.path.join(OUT, "Graphical_abstract.pptx"))



# ---------------------------------------------------------------- Figure 6: key results (one row, same three encoders everywhere)
def bars(s, x0, y0, w, h, vals, lo, hi, fmt, ci=None, ref=None, reflabel=None, ticks=(), tfmt=None,
         keys=("random", "cine", "trained"), labels=None, main="trained", labsize=6.5, valsize=None):
    """three bars drawn with shapes: random, cine, report-pretrained; value labels above; optional CIs and reference line"""
    yp = lambda v: y0 + h - (v - lo) / (hi - lo) * h
    line(s, x0, y0 + h, x0 + w, y0 + h, color="ink", width=1)
    for t in ticks:
        line(s, x0, yp(t), x0 + w, yp(t), color="grid", width=0.5)
        text(s, x0 - 0.42, yp(t) - 0.08, 0.38, 0.16, (tfmt or fmt)(t), size=6.5, align=PP_ALIGN.RIGHT, color="line")
    if ref is not None:
        line(s, x0, yp(ref), x0 + w, yp(ref), color="clinical", width=1, dash=True)
        if reflabel: text(s, x0 + w + 0.02, yp(ref) - 0.08, 0.5, 0.16, reflabel, size=6, align=PP_ALIGN.LEFT, color="clinical", italic=True)
    bw = w / (1.3 * len(keys) + 0.3)       # 3 bars: w / 4.2, as before
    labels = labels or {"random": "Random", "cine": "Cine", "trained": "Report-\npretrained"}
    for k, (w_, v) in enumerate(zip(keys, vals)):
        x = x0 + bw * 0.45 + k * bw * 1.3
        if v is None:
            text(s, x - 0.12, y0 + h - 0.22, bw + 0.24, 0.2, "n/a", size=6, color="line", italic=True)
            text(s, x - 0.12, y0 + h + 0.03, bw + 0.24, 0.32, labels[w_], size=labsize); continue
        top, base = yp(max(v, lo)), yp(max(lo, min(0, hi)) if lo < 0 else lo)
        y_hi, y_lo = min(top, base), max(top, base)
        b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y_hi), Inches(bw), Inches(max(y_lo - y_hi, 0.01)))
        b.fill.solid(); b.fill.fore_color.rgb = C[w_]; b.line.fill.background(); b.shadow.inherit = False
        lab_y = y_hi - 0.22
        if ci:
            a_, c_ = ci[k]; xc = x + bw / 2
            line(s, xc, yp(a_), xc, yp(c_), color="ink", width=1)
            for yy in (yp(a_), yp(c_)): line(s, xc - 0.05, yy, xc + 0.05, yy, color="ink", width=1)
            lab_y = min(lab_y, yp(c_) - 0.22)
        if v is not None: text(s, x - 0.15, lab_y, bw + 0.3, 0.2, fmt(v), size=valsize or (9 if w_ == main else 7.5), bold=w_ == main,
                               color=w_ if w_ not in ("random", "t25", "dinov2clip", "imgssl_l", "low") else "line")
        text(s, x - 0.12, y0 + h + 0.03, bw + 0.24, 0.32, labels[w_], size=labsize)


def figure6():
    p, s = deck(7.5, 3.0)
    W, H, Y, gap = 1.3, 1.75, 0.72, 0.58
    xs = [0.45 + k * (W + gap) for k in range(4)]
    heads = [("A", "Matches its own report", "test top-5 recall, 320 reports"),
             ("B", "Reads the clinical profile", "mean test AUROC, 5 variables"),
             ("C", "Adds to clinical risk", "mortality ΔAUROC over clinical model"),
             ("D", "Predicts mortality alone", "test AUROC, image only")]
    for x, (L, t1, t2) in zip(xs, heads):
        panel(s, x - 0.45, 0.02, L)
        text(s, x - 0.15, 0.05, W + 0.3, 0.22, t1, size=8, bold=True)
        text(s, x - 0.2, 0.27, W + 0.4, 0.2, t2, size=6, italic=True, color="line")
    W3 = ["random", "cine", "trained"]
    pc_ = lambda v: f"{100 * v:.0f}%" if v >= 0.1 else f"{100 * v:.1f}%"
    bars(s, xs[0], Y, W, H, [RC[w]["recall@5"]["estimate"] for w in W3], 0, 0.4, pc_,
         ci=[RC[w]["recall@5"]["ci95"] for w in W3], ref=RC["chance@5"], reflabel="chance", ticks=(0, 0.1, 0.2, 0.3, 0.4), tfmt=lambda t: f"{100 * t:.0f}%")
    B5 = ["male", "diabetes", "ckd", "egfr_lt60", "ever_smoker"]
    mean = [sum(PR[k][f"image_{w}"]["estimate"] for k in B5) / len(B5) for w in W3]
    bars(s, xs[1], Y, W, H, mean, 0.5, 0.75, lambda v: f"{v:.2f}", ref=0.5, reflabel="chance", ticks=(0.5, 0.6, 0.7))
    key = {"random": "clinical+image_random", "cine": "clinical+image_cine", "trained": "clinical+image"}
    bars(s, xs[2], Y, W, H, [MO["cv"][key[w]]["diff_vs_first"] for w in W3], -0.02, 0.08, lambda v: f"{v:+.3f}".replace("-", "−"),
         ci=[MO["cv"][key[w]]["diff_ci95"] for w in W3], ref=0, reflabel="clinical\nmodel", ticks=(0, 0.04, 0.08), tfmt=lambda t: f"{t:+.2f}" if t else "0")
    bars(s, xs[3], Y, W, H, [FT[(w, "frozen")] for w in W3], 0.5, 0.8, lambda v: f"{v:.2f}", ref=0.5, ticks=(0.5, 0.6, 0.7, 0.8))
    foot(s, 0.1, 2.78, 7.3, 0.2, f"Same encoder in every panel: random weights, cine CMR checkpoint, or after report pretraining. Bars with whiskers: 95% CI (patient bootstrap). "
         f"C: {MO['events_cv']} deaths / {n(MO['n_cv'])} examinations; D: frozen encoder, {MO['events_test']} test deaths.", size=5.8, italic=True, align=PP_ALIGN.LEFT, color="line")
    p.save(os.path.join(OUT, "Figure_6_key_results.pptx"))


# ---------------------------------------------------------------- Figure 7: one patient, the whole examination
WE = [("Function, anatomy and flow", [("cine_4ch", "4CH", "Cine (bSSFP), end-diastole"), ("cine_sa", "SAX", "Cine (bSSFP), mid ventricle"),
                                      ("blackblood", "4CH", "Black-blood (double IR)"), ("flow_phase", "Ao", "Phase contrast: velocity"),
                                      ("flow_mag", "Ao", "Phase contrast: magnitude")]),
      ("Tissue characterization", [("t1_native", "SAX", "Native T1 map (200-2000 ms)"), ("t1_post", "SAX", "Post-contrast T1 map (100-900 ms)"),
                                   ("t2_map", "SAX", "T2 map (20-100 ms)"), ("lge_psir_sa", "SAX", "LGE, phase-sensitive (PSIR)"),
                                   ("lge_psir_4ch", "4CH", "LGE, phase-sensitive (PSIR)")])]


def wimg(s, key, x, y, w, tag, label, frame="white", fw=0.5):
    pic = s.shapes.add_picture(os.path.join(IMG, "whole_exam", key + ".png"), Inches(x), Inches(y), Inches(w), Inches(w))
    pic.crop_left = pic.crop_right = pic.crop_top = pic.crop_bottom = CROP
    pic.line.color.rgb = C[frame]; pic.line.width = Pt(fw)
    text(s, x + 0.03, y + 0.02, 0.6, 0.2, tag, size=8, bold=True, color="white", align=PP_ALIGN.LEFT)
    text(s, x, y + w + 0.01, w, 0.17, label, size=6, color="white")


def figure7():
    ex = EXAMPLES[0]; w, gx = 1.38, 0.07; x0 = 0.12 + (7.34 - 5 * w - 4 * gx) / 2
    p, s = deck(7.5, 7.75)
    text(s, 0.08, 0.02, 7.34, 0.24, f"One patient, one CMR examination (report ID {ex}): what a cardiologist reads and what the model reads",
         size=8.5, bold=True, align=PP_ALIGN.LEFT)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.08), Inches(0.3), Inches(7.34), Inches(5.36))
    bg.fill.solid(); bg.fill.fore_color.rgb = C["black"]; bg.line.fill.background(); bg.shadow.inherit = False
    y = 0.34
    for title, row in WE:
        text(s, 0.14, y, 7.2, 0.2, title + "  ·  in the examination, not used by this model", size=7, bold=True, color="white", align=PP_ALIGN.LEFT)
        for k, (key, tag, lab) in enumerate(row):
            wimg(s, key, x0 + k * (w + gx), y + 0.21, w, tag, lab)
        y += 0.21 + w + 0.24
    hdr = text(s, 0.14, y, 7.2, 0.2, "Late gadolinium enhancement, magnitude  ·  the four views the model reads", size=7, bold=True, color="trained", align=PP_ALIGN.LEFT)
    for k, v in enumerate(["sa", "2ch", "3ch", "4ch"]):
        pic = img(s, ex, v, x0 + k * (w + gx), y + 0.21, w, label=False, tag=True, tagsize=8)
        pic.line.color.rgb = C["trained"]; pic.line.width = Pt(2.0)
    bx = x0 + 4 * (w + gx); by = y + 0.21
    b = box(s, bx, by, w, w, "", fill="trained", line="trained", size=7)
    text(s, bx + 0.05, by + 0.06, w - 0.1, 0.5, "Report-pretrained LGE encoder", size=7.5, bold=True, color="white")
    arrow(s, bx + w / 2, by + 0.58, bx + w / 2, by + 0.78, color="white", width=1.5)
    text(s, bx + 0.05, by + 0.8, w - 0.1, 0.5, f"its own report ranked 1st of {RC['n_exams']} test reports", size=7, color="white")
    rep = json.load(open(os.path.join(IMG, "whole_exam", "report_full.json")))
    rb = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.08), Inches(5.72), Inches(7.34), Inches(1.95))
    rb.fill.solid(); rb.fill.fore_color.rgb = C["white"]; rb.line.color.rgb = C["report"]; rb.line.width = Pt(1.25); rb.shadow.inherit = False
    text(s, 0.14, 5.75, 7.2, 0.2, "The examination's clinical report (verbatim): the text paired with these images in pretraining",
         size=7, bold=True, color="report", align=PP_ALIGN.LEFT)
    tb = s.shapes.add_textbox(Inches(0.14), Inches(5.98), Inches(7.22), Inches(1.66)); tf = tb.text_frame
    tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.TOP
    for a in ("margin_left", "margin_right", "margin_top", "margin_bottom"): setattr(tf, a, Inches(0.02))
    for i, (lab, body) in enumerate([("Findings: ", rep["findings"]), ("Impression: ", rep["impression"])]):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); para.alignment = PP_ALIGN.LEFT; para.space_before = Pt(2 if i else 0)
        for t_, b_ in ((lab, True), (body, False)):
            r = para.add_run(); r.text = t_; f = r.font; f.name, f.size, f.bold, f.color.rgb = FONT, Pt(6.3 * TS), b_, C["ink"]
    foot(s, 0.08, 6.97, 7.34, 0.3, "All images from the same examination, one image per sequence chosen by fixed rules (first cardiac phase, middle slice, peak-systolic "
         "flow frame). T2 map fitted from the four T2-prepared images. Pretraining used only the LGE magnitude views and the report text.", size=5.8, italic=True,
         align=PP_ALIGN.LEFT, color="line")
    p.save(os.path.join(OUT, "Figure_7_one_exam.pptx"))


# ---------------------------------------------------------------- Figure 8: data scaling
def figure8():
    if not ER or not all(k in ER for k in ("frac25", "frac50", "full")): return
    C["t25"], C["t50"], C["t100"] = RGBColor(0x9F, 0xDA, 0x8E), RGBColor(0x2F, 0xB4, 0x8E), RGBColor(0x1F, 0x5F, 0x8B)
    K = ["frac25", "frac50", "full"]; KC = ["t25", "t50", "t100"]
    lab = {"t25": "568\n(25%)", "t50": "1,136\n(50%)", "t100": "2,273\n(100%)"}
    p, s = deck(7.5, 3.15)
    W, H, Y, gap = 1.3, 1.7, 0.75, 0.58
    xs = [0.45 + k * (W + gap) for k in range(4)]
    heads = [("A", "Report retrieval", "test top-1 recall"), ("B", "Report retrieval", "test top-5 recall"),
             ("C", "EMIDEC: infarction", "vs normal, AUROC"), ("D", "EMIDEC: infarct size", "Spearman ρ")]
    for x, (L, t1, t2) in zip(xs, heads):
        panel(s, x - 0.45, 0.02, L)
        text(s, x - 0.15, 0.05, W + 0.3, 0.22, t1, size=8, bold=True)
        text(s, x - 0.2, 0.27, W + 0.4, 0.2, t2, size=6, italic=True, color="line")
    pc_ = lambda v: f"{100 * v:.1f}%" if v < 0.1 else f"{100 * v:.0f}%"
    r = lambda k, m: ER[k]["retrieval"][m]
    bars(s, xs[0], Y, W, H, [r(k, "recall@1")[0] for k in K], 0, 0.16, pc_, ci=[r(k, "recall@1")[1:] for k in K], ref=1 / 320, reflabel=None,
         ticks=(0, 0.05, 0.1, 0.15), tfmt=lambda t: f"{100 * t:.0f}%", keys=KC, labels=lab, main="t100")
    bars(s, xs[1], Y, W, H, [r(k, "recall@5")[0] for k in K], 0, 0.4, pc_, ci=[r(k, "recall@5")[1:] for k in K], ref=5 / 320, reflabel=None,
         ticks=(0, 0.1, 0.2, 0.3, 0.4), tfmt=lambda t: f"{100 * t:.0f}%", keys=KC, labels=lab, main="t100")  # chance label on panel A only
    bars(s, xs[2], Y, W, H, [ER[k]["emidec"]["auroc"] for k in K], 0.4, 0.9, lambda v: f"{v:.2f}", ci=[ER[k]["emidec"]["ci"] for k in K], ref=0.5,
         reflabel=None, ticks=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9), keys=KC, labels=lab, main="t100")
    bars(s, xs[3], Y, W, H, [ER[k]["emidec"]["infarct_spearman"] for k in K], 0, 0.6, lambda v: f"{v:.2f}", ref=0, ticks=(0, 0.2, 0.4, 0.6),
         keys=KC, labels=lab, main="t100")
    foot(s, 0.1, 2.88, 7.3, 0.24, "Report-image pretraining on a random 25%, 50% or all of the training patients (same epochs and settings); x-axis: training patients. "
         "Whiskers: 95% CI (A-B: patient bootstrap; C: bootstrap of the 100 EMIDEC examinations).", size=5.8, italic=True, align=PP_ALIGN.LEFT, color="line")
    p.save(os.path.join(OUT, "Figure_8_scaling.pptx"))


if __name__ == "__main__":
    figure1(); figure2(); figure3(); figure_examples(); figure4(); figure6(); figure7(); figure8(); graphical_abstract()
    print("wrote", sorted(os.listdir(OUT)))
