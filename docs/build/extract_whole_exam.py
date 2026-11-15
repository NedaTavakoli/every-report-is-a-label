#!/usr/bin/env python3
"""
Whole-examination panels for one example patient (the first rank-1 test example of the examples figure; set FIG_EXAMPLE_ACC to override), for the
"one patient, one examination" figure. Series are chosen by their DICOM SeriesNumber / description (GE), images by fixed
rules (no picking by eye):
  cine_4ch     4CH FIESTA, first cardiac phase (end-diastole)
  cine_sa      SA FIESTA STACK, middle slice location, first phase
  blackblood   4ch T1 DIR (double inversion recovery, black blood)
  t1_native    T1 map pre-contrast (MOLLI, mid SA; stored in ms; window 200-2000 ms)
  t1_post      T1 map post-contrast (window 100-900 ms)
  t2_map       T2 map fitted per pixel (log-linear) from the 4 T2-prepared echoes (9.8-97.8 ms) at the same mid SA slice;
               window 20-100 ms; background (low first-echo signal) set to 0
  flow_phase   Aorta PC (corrected), phase (velocity) frame with the highest central velocity (peak systole), mid-gray = 0
  flow_mag     the matching magnitude frame
  lge_psir_sa / lge_mag_sa      PS:SA and MAG:SA PSMDE BH, middle slice (index 6 of 12, as in Figure 4)
  lge_psir_4ch / lge_mag_4ch    PS:4ch and MAG:4ch PSMDE BH
The model uses only the four magnitude LGE views (MAG:SA/2ch/3ch/4ch PSMDE BH). Outputs docs/figures/images/whole_exam/*.png
(512 x 512, 8-bit; local only) and gray placeholders in docs/figures/images_placeholder/whole_exam/.
"""
import os, sys, json
import numpy as np, pandas as pd, pydicom
from PIL import Image, ImageDraw

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ACC = os.environ.get("FIG_EXAMPLE_ACC") or json.load(open(os.path.join(ROOT, "docs", "figures", "images", "examples.json")))[0]   # first rank-1 test example
SHARE = os.environ.get("FULL_EXAM_DICOM_ROOT", "/path/to/full_exam_DICOM_root")   # all sequences of each exam


def load(d):
    by = {}
    for f in os.listdir(d):
        try: h = pydicom.dcmread(os.path.join(d, f))
        except Exception: continue
        by.setdefault(int(h.SeriesNumber), []).append(h)
    for k in by: by[k].sort(key=lambda h: int(h.InstanceNumber))
    return by


def px(h): return h.pixel_array.astype(np.float32) * float(getattr(h, "RescaleSlope", 1) or 1) + float(getattr(h, "RescaleIntercept", 0) or 0)


def win(a, lo, hi): return np.clip((a - lo) / (hi - lo), 0, 1)


def auto(a):
    c = a[a.shape[0] // 4: 3 * a.shape[0] // 4, a.shape[1] // 4: 3 * a.shape[1] // 4]
    return win(a, np.percentile(c, 1), np.percentile(c, 99.5))


def mid_loc(hs):
    locs = sorted({round(float(h.SliceLocation), 1) for h in hs}); m = locs[len(locs) // 2]
    return [h for h in hs if round(float(h.SliceLocation), 1) == m]


def main():
    s = pd.read_csv(os.path.join(ROOT, "data", "manifest", "series.csv"), dtype=str)
    top = s[s.anon_acc == ACC].rel_path.iloc[0].split("/")[0]
    by = load(os.path.join(SHARE, top, ACC))
    out = {}
    out["cine_4ch"] = auto(px(min(by[5], key=lambda h: float(h.TriggerTime))))
    out["cine_sa"] = auto(px(min(mid_loc(by[21]), key=lambda h: float(h.TriggerTime))))
    out["blackblood"] = auto(px(by[9][0]))
    def tmap(a, lo, hi):     # background noise of the fitted map lies far above the window: shown black, not white
        a = a.copy(); a[(a > hi * 1.25) | (a <= 0)] = 0; return win(a, lo, hi)
    out["t1_native"] = tmap(px(by[11][0]), 200, 2000)
    out["t1_post"] = tmap(px(by[27][0]), 100, 900)
    e = sorted(by[15], key=lambda h: float(h.EchoTime)); te = np.array([float(h.EchoTime) for h in e])
    S = np.stack([np.maximum(px(h), 1) for h in e]); y = np.log(S)
    tm = te.mean(); slope = ((te - tm)[:, None, None] * (y - y.mean(0))).sum(0) / ((te - tm) ** 2).sum()
    t2 = np.where(slope < 0, -1 / np.minimum(slope, -1e-6), 0)
    fg = (S[0] > np.percentile(S[0], 75) * 0.5) & (t2 < 250)     # tissue: enough first-echo signal and a plausible T2
    out["t2_map"] = win(np.where(fg, t2, 0), 20, 100)
    pc = by[1200]; ph = [h for h in pc if float(getattr(h, "WindowCenter", 1) or 1) == 0] or pc[:len(pc) // 2]
    mg = [h for h in pc if h not in ph]
    def central_v(h):
        a = np.abs(px(h)); c = a[a.shape[0] // 4: 3 * a.shape[0] // 4, a.shape[1] // 4: 3 * a.shape[1] // 4]; return np.percentile(c, 99.5)
    pk = max(ph, key=central_v); a = px(pk); vmax = np.percentile(np.abs(a), 99.5)
    out["flow_phase"] = win(a, -vmax, vmax)
    out["flow_mag"] = auto(px(min(mg, key=lambda h: abs(float(h.TriggerTime) - float(pk.TriggerTime))))) if mg else out["flow_phase"]
    for key, sn in [("lge_psir_sa", 23), ("lge_mag_sa", 2300)]:
        hs = sorted(by[sn], key=lambda h: int(h.InstanceNumber)); out[key] = auto(px(hs[len(hs) // 2]))
    out["lge_psir_4ch"] = auto(px(by[24][0])); out["lge_mag_4ch"] = auto(px(by[2400][0]))
    d = os.path.join(ROOT, "docs", "figures", "images", "whole_exam"); p = os.path.join(ROOT, "docs", "figures", "images_placeholder", "whole_exam")
    os.makedirs(d, exist_ok=True); os.makedirs(p, exist_ok=True)
    for k, a in out.items():
        Image.fromarray((255 * a).astype(np.uint8)).resize((512, 512), Image.BICUBIC).save(os.path.join(d, k + ".png"))
        g = Image.new("L", (512, 512), 90); ImageDraw.Draw(g).text((20, 240), k, fill=230); g.save(os.path.join(p, k + ".png"))
    print("panels:", list(out), "| T2 fit TEs (ms):", te.round(1).tolist(), "| median T2 in fg (ms):", round(float(np.median(t2[fg & (t2 < 200)])), 1),
          "| flow phase frames:", len(ph), "magnitude frames:", len(mg))


if __name__ == "__main__":
    main()
