"""OCR + consistency + forensic feature extraction for the synthetic specimen cards.

Pipeline per card:
  1. Crop the 6 visible fields and 3 MRZ lines (fixed template boxes; a real
     system would use a field detector such as YOLO here).
  2. OCR each crop with Tesseract (single-line mode, character whitelists).
  3. Build three feature groups:
       consistency : visible fields vs MRZ, MRZ check digits, date logic
       ocr_quality : OCR confidence statistics
       forensic    : cross-region sharpness / ink / character-height deviations
"""
import difflib
import re
import sys
from datetime import date
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import pytesseract

from gen_dataset import FIELD_BOX, MRZ_BOX, chk

# --- locate the Tesseract program (the pip package pytesseract is only a wrapper) ---
import os
import shutil

_TESS_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
]
if not shutil.which("tesseract"):
    for _p in _TESS_CANDIDATES:
        if os.path.isfile(_p):
            pytesseract.pytesseract.tesseract_cmd = _p
            break


def check_tesseract():
    try:
        return pytesseract.get_tesseract_version()
    except Exception:
        sys.exit(
            "\nERROR: the Tesseract OCR program was not found.\n"
            "  1. Install it (Windows): https://github.com/UB-Mannheim/tesseract/wiki  (tesseract-ocr-w64-setup ...exe)\n"
            "  2. Re-run this script. If it still fails, add this line under 'import pytesseract' in extract_features.py:\n"
            '       pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"\n'
            "  (Check with:  tesseract --version  in a NEW terminal.)\n"
        )


PAD_X = 6
NAMES_WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DATE_WL = "0123456789."
DOC_WL = "EX0123456789"  # this issuer: EX + 7 digits
MRZ_WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
WL = {"surname": NAMES_WL, "given": NAMES_WL, "dob": DATE_WL, "docno": DOC_WL, "issue": DATE_WL, "expiry": DATE_WL}
FIELDS = list(FIELD_BOX)


def crop(gray, box, trim=0):
    x0, y0, x1, y1 = box
    return gray[y0 + 3:y1 + 2 - trim, max(0, x0 - PAD_X):x1 + PAD_X]  # skip the label line above


def ocr(img, whitelist):
    big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cfg = f"--psm 7 -c tessedit_char_whitelist={whitelist}"
    d = pytesseract.image_to_data(th, config=cfg, output_type=pytesseract.Output.DICT)
    words = [(t, float(c)) for t, c in zip(d["text"], d["conf"]) if t.strip() and float(c) >= 0]
    text = "".join(t for t, _ in words)
    conf = float(np.mean([c for _, c in words])) if words else 0.0
    return text, conf


def parse_date(s):
    digits = re.sub(r"\D", "", s)
    if len(digits) != 8:
        return None
    try:
        return date(int(digits[4:]), int(digits[2:4]), int(digits[:2]))
    except ValueError:
        return None


def sim(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio() if a or b else 0.0


def region_stats(gray_crop):
    lap = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
    _, th = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = float((th > 0).mean())
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hs = [cv2.boundingRect(c)[3] for c in cnts if cv2.contourArea(c) > 20]
    ch = float(np.median(hs)) if hs else 0.0
    return np.log(lap + 1e-6), ink, ch


def extract(path):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    f = {}
    text, conf = {}, {}
    for k, box in FIELD_BOX.items():
        text[k], conf[k] = ocr(crop(gray, box, trim=8), WL[k])
    mrz, mconf = [], []
    for box in MRZ_BOX:
        t, c = ocr(crop(gray, box), MRZ_WL)
        mrz.append(t)
        mconf.append(c)

    # ---- MRZ parse (pad/truncate to 30 and flag length)
    # Tesseract often drops runs of '<', so only the first 15 chars of lines 1-2 are position-reliable
    f["mrz_core_len_ok"] = float(len(mrz[0]) >= 15 and len(mrz[1]) >= 15)
    l1, l2, l3 = [(l + "<" * 30)[:30] for l in mrz]
    overall_c = mrz[1][-1] if mrz[1] and mrz[1][-1].isdigit() else "?"
    m_doc, m_doc_c = l1[5:14], l1[14]
    m_dob, m_dob_c = l2[0:6], l2[6]
    m_exp, m_exp_c = l2[8:14], l2[14]
    m_all_c = overall_c
    parts = l3.split("<<")
    m_sur = parts[0].replace("<", "")
    m_giv = parts[1].replace("<", "") if len(parts) > 1 else ""

    # ---- consistency features
    v_dob, v_exp, v_iss = parse_date(text["dob"]), parse_date(text["expiry"]), parse_date(text["issue"])
    f["parse_ok_dob"] = float(v_dob is not None)
    f["parse_ok_issue"] = float(v_iss is not None)
    f["parse_ok_expiry"] = float(v_exp is not None)
    f["name_sim_surname"] = sim(text["surname"], m_sur)
    f["name_sim_given"] = sim(text["given"], m_giv)
    f["doc_sim"] = sim(text["docno"], m_doc)
    f["dob_sim"] = sim(re.sub(r"\D", "", text["dob"])[6:8] + re.sub(r"\D", "", text["dob"])[2:4] + re.sub(r"\D", "", text["dob"])[:2], m_dob)
    f["exp_sim"] = sim(re.sub(r"\D", "", text["expiry"])[6:8] + re.sub(r"\D", "", text["expiry"])[2:4] + re.sub(r"\D", "", text["expiry"])[:2], m_exp)
    f["mrz_chk_doc"] = float(chk(m_doc) == m_doc_c)
    f["mrz_chk_dob"] = float(chk(m_dob) == m_dob_c)
    f["mrz_chk_exp"] = float(chk(m_exp) == m_exp_c)
    f["mrz_chk_all"] = float(chk(m_doc + m_doc_c + m_dob + m_dob_c + m_exp + m_exp_c) == m_all_c)
    f["vis_docno_format_ok"] = float(bool(re.fullmatch(r"EX\d{7}", text["docno"])))
    if v_dob and v_iss and v_exp:
        f["exp_after_issue"] = float(v_exp > v_iss)
        f["issue_age_ok"] = float(16 <= (v_iss - v_dob).days / 365.25 <= 100)
        span = (v_exp - v_iss).days / 365.25
        f["validity_span_ok"] = float(9.9 <= span <= 10.1)  # this issuer always grants 10 years
    else:
        f["exp_after_issue"] = f["issue_age_ok"] = f["validity_span_ok"] = 0.0

    # ---- OCR quality
    allc = list(conf.values()) + mconf
    f["ocr_conf_mean_fields"] = float(np.mean(list(conf.values())))
    f["ocr_conf_min_fields"] = float(np.min(list(conf.values())))
    f["ocr_conf_mean_mrz"] = float(np.mean(mconf))
    f["ocr_conf_std_all"] = float(np.std(allc))
    f["ocr_conf_min_all"] = float(np.min(allc))

    # ---- forensic: per-region stats, then cross-region deviation
    regs = [crop(gray, b, trim=8) for b in FIELD_BOX.values()] + [crop(gray, b) for b in MRZ_BOX]
    st = np.array([region_stats(r) for r in regs])  # (9, 3): loglap, ink, charheight
    names = ["loglap", "ink", "charh"]
    for j, n in enumerate(names):
        col = st[:, j]
        med = np.median(col)
        dev = col - med
        f[f"for_{n}_std"] = float(col.std())
        f[f"for_{n}_maxdev"] = float(np.max(np.abs(dev)))
        for i, region in enumerate(FIELDS + ["mrz1", "mrz2", "mrz3"]):
            f[f"for_{n}_dev_{region}"] = float(dev[i])
    f["_ocr_text"] = "|".join(list(text.values()) + mrz)
    return f


def group_of(col):
    if col.startswith("for_"):
        return "forensic"
    if col.startswith("ocr_"):
        return "ocr_quality"
    return "consistency"


def _work(args):
    doc_id, path = args
    try:
        r = extract(path)
    except Exception as e:  # keep going; record the failure
        r = {"_error": str(e)}
    r["doc_id"] = doc_id
    return r


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "data"
    print("Tesseract version:", check_tesseract())
    lab = pd.read_csv(f"{root}/labels.csv")
    jobs = [(r.doc_id, f"{root}/{r.path}") for r in lab.itertuples()]
    with Pool(2) as p:
        rows = p.map(_work, jobs, chunksize=8)
    feats = pd.DataFrame(rows)
    if "_error" in feats.columns and feats["_error"].notna().any():
        n_err = int(feats["_error"].notna().sum())
        print(f"errors on {n_err}/{len(feats)} cards:", feats["_error"].dropna().unique()[:5])
        if n_err > 0.05 * len(feats):
            sys.exit("Too many failures; not writing features.csv. Fix the error above and re-run.")
    out = lab.merge(feats, on="doc_id")
    out.to_csv(f"{root}/features.csv", index=False)
    print(out.shape, "->", f"{root}/features.csv")
