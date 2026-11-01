"""Generate a small SYNTHETIC dataset of fictional specimen ID cards.

Everything here is fictional: "Republic of Examplia", invented names, no real
faces, and every card carries a SPECIMEN marking. The purpose is only to test
a document + OCR-consistency pipeline, not to imitate any real document.

Each identity yields one genuine card and one tampered card. Tamper types:
  A  naive      : visible fields edited, MRZ left untouched (mismatch)
  B  skilled    : visible fields AND MRZ edited, check digits recomputed
  C  sloppy     : visible fields AND MRZ edited, check digits NOT recomputed
All cards get the same random capture degradation (blur, brightness, noise,
JPEG, small shift) so global capture quality is not a shortcut.
"""
import argparse
import csv
import os
import random
from datetime import date

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 856, 540
BG = (240, 243, 248)
# --- portable font lookup (Windows / Linux / Mac) -------------------------
# Preferred: DejaVu fonts bundled with matplotlib (pip install matplotlib) or
# installed on Linux. Fallback: standard Windows / macOS system fonts.
try:
    import matplotlib
    _MPL = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
except ImportError:
    _MPL = ""
_DIRS = [_MPL, "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation"]
_SYS = [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
        "/System/Library/Fonts/Supplemental", "/Library/Fonts"]


def _find(*names):
    """Return the first existing font file among `names` (searched in all font dirs)."""
    for n in names:
        for d in _DIRS + _SYS:
            p = os.path.join(d, n) if d else ""
            if p and os.path.isfile(p):
                return p
    raise FileNotFoundError("No usable font found. Tried: " + ", ".join(names) +
                            ". Fix: pip install matplotlib (it bundles DejaVu fonts).")


VALUE_FONT = BOLD = _find("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf")
LABEL_FONT = _find("DejaVuSans.ttf", "arial.ttf", "Arial.ttf")
MRZ_FONT = _find("DejaVuSansMono-Bold.ttf", "courbd.ttf", "Courier New Bold.ttf")
ALT_FONTS = [
    _find("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"),
    _find("LiberationSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "DejaVuSerif-Bold.ttf"),
    _find("LiberationSans-Regular.ttf", "arial.ttf", "Arial.ttf", "DejaVuSerif.ttf"),
]
ALT_MRZ = [
    _find("DejaVuSansMono.ttf", "cour.ttf", "Courier New.ttf"),
    _find("LiberationMono-Bold.ttf", "courbd.ttf", "Courier New Bold.ttf", "DejaVuSansMono-Oblique.ttf"),
]
VSIZE, MSIZE = 30, 32

# field name -> (label, x, y_label)   value drawn at y_label + 22
FIELDS = {
    "surname": ("SURNAME", 280, 100),
    "given": ("GIVEN NAME", 280, 160),
    "dob": ("DATE OF BIRTH", 280, 220),
    "docno": ("DOCUMENT NO", 560, 220),
    "issue": ("DATE OF ISSUE", 280, 280),
    "expiry": ("DATE OF EXPIRY", 560, 280),
}
FIELD_BOX = {  # crop boxes for values (x0, y0, x1, y1) on the clean card
    "surname": (280, 118, 830, 166),
    "given": (280, 178, 830, 226),
    "dob": (280, 238, 540, 286),
    "docno": (560, 238, 830, 286),
    "issue": (280, 298, 540, 346),
    "expiry": (560, 298, 830, 346),
}
MRZ_X, MRZ_Y, MRZ_DY = 40, 398, 42
MRZ_BOX = [(30, MRZ_Y + i * MRZ_DY - 4, 640, MRZ_Y + i * MRZ_DY + 40) for i in range(3)]

SURNAMES = ["MARLOWE", "KESTREL", "VANTERPOOL", "ASHDOWN", "BRENNAN", "CALLOWAY", "DRAYCOTT", "ELLINGTON",
            "FAIRBANK", "GRANTHAM", "HOLLOWAY", "IRONSIDE", "JAMESON", "KINGSLEY", "LOCKHART", "MORELAND",
            "NORTHCOTT", "OAKLEY", "PEMBROKE", "QUILLEN", "RAVENSCROFT", "SINCLAIR", "THORNBURY", "UNDERHILL",
            "VALDEZ", "WHITLOCK", "YARBOROUGH", "ZELLER", "ABERNATHY", "BLACKWELL", "CARMICHAEL", "DUNMORE"]
GIVEN = ["ALDEN", "BRIONY", "CORWIN", "DELPHINE", "EZRA", "FIONA", "GIDEON", "HESTER", "IVO", "JOSEPHINE",
         "KIERAN", "LEONORA", "MAGNUS", "NERISSA", "OSWIN", "PHILIPPA", "QUINN", "ROWAN", "SABINE", "TOBIAS",
         "ULRIKA", "VICTOR", "WILHELMINA", "XAVIER", "YVETTE", "ZACHARY", "AMARA", "BASTIAN", "CELESTE", "DOMINIC"]


# ---------------------------------------------------------------- MRZ helpers
def chk(s):
    vals = {c: i for i, c in enumerate("0123456789")}
    vals.update({c: 10 + i for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")})
    vals["<"] = 0
    w = [7, 3, 1]
    return str(sum(vals[c] * w[i % 3] for i, c in enumerate(s)) % 10)


def fmt_date(d):
    return d.strftime("%d.%m.%Y")


def yymmdd(d):
    return d.strftime("%y%m%d")


def build_mrz(rec, recompute=True, orig=None):
    """Return 3 MRZ lines (30 chars). If recompute is False, reuse the check
    digits of `orig` (what a sloppy forger leaves in place)."""
    doc = rec["docno"]
    dob, exp = yymmdd(rec["dob"]), yymmdd(rec["expiry"])
    if recompute or orig is None:
        c_doc, c_dob, c_exp = chk(doc), chk(dob), chk(exp)
        c_all = chk(doc + c_doc + dob + c_dob + exp + c_exp)
    else:
        od = orig
        c_doc, c_dob, c_exp = chk(od["docno"]), chk(yymmdd(od["dob"])), chk(yymmdd(od["expiry"]))
        c_all = chk(od["docno"] + c_doc + yymmdd(od["dob"]) + c_dob + yymmdd(od["expiry"]) + c_exp)
    l1 = ("I<EXA" + doc + c_doc).ljust(29, "<") + "<"
    l1 = l1[:30]
    l2 = (dob + c_dob + "<" + exp + c_exp + "EXA").ljust(29, "<") + c_all
    l3 = (rec["surname"] + "<<" + rec["given"]).ljust(30, "<")[:30]
    return [l1, l2, l3]


# ------------------------------------------------------------ record sampling
def rand_date(rng, y0, y1):
    return date(rng.randint(y0, y1), rng.randint(1, 12), rng.randint(1, 28))


def new_record(rng):
    issue = rand_date(rng, 2018, 2025)
    return {
        "surname": rng.choice(SURNAMES),
        "given": rng.choice(GIVEN),
        "dob": rand_date(rng, 1950, 2005),
        "docno": "EX" + "".join(str(rng.randint(0, 9)) for _ in range(7)),
        "issue": issue,
        "expiry": date(issue.year + 10, issue.month, issue.day),
    }


def edit_record(rec, rng):
    r = dict(rec)
    fields = rng.sample(["surname", "given", "dob", "docno", "issue", "expiry"], rng.choice([1, 1, 2]))
    for f in fields:
        if f == "surname":
            r[f] = rng.choice([s for s in SURNAMES if s != rec[f]])
        elif f == "given":
            r[f] = rng.choice([s for s in GIVEN if s != rec[f]])
        elif f == "docno":
            d = list(rec[f])
            for i in rng.sample(range(2, 9), 3):
                d[i] = str((int(d[i]) + rng.randint(1, 9)) % 10)
            r[f] = "".join(d)
        else:
            old = rec[f]
            shift = rng.choice([-3, -2, -1, 1, 2, 3])
            r[f] = date(min(max(old.year + shift, 1930), 2035), rng.randint(1, 12), rng.randint(1, 28))
    return r, fields


# ------------------------------------------------------------------ rendering
def font(path, size):
    return ImageFont.truetype(path, size)


def draw_text_box(img, box, text, fpath, size, rng=None, patch=False):
    """Draw text in a crop box. patch=True imitates a pasted edit: different
    font/size, small offset, slight blur confined to the patch."""
    x0, y0, x1, y1 = box
    if not patch:
        ImageDraw.Draw(img).text((x0, y0 + 2), text, font=font(fpath, size), fill=(20, 20, 30))
        return
    r = rng
    reg = Image.new("RGB", (x1 - x0, y1 - y0), BG)
    d = ImageDraw.Draw(reg)
    f = r.choice(ALT_FONTS) if fpath in (VALUE_FONT,) or fpath in ALT_FONTS else r.choice(ALT_MRZ)
    if fpath in (MRZ_FONT, *ALT_MRZ):
        f = r.choice(ALT_MRZ)
    sz = size + r.randint(-2, 2)
    d.text((r.randint(0, 3), 2 + r.randint(-1, 2)), text, font=font(f, sz), fill=(20, 20, 30))
    reg = reg.filter(ImageFilter.GaussianBlur(r.uniform(0.3, 0.8)))
    img.paste(reg, (x0, y0))


def render_card(vis, mrz_lines, patch_fields, patch_mrz, rng):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 70], fill=(30, 60, 120))
    d.text((30, 18), "REPUBLIC OF EXAMPLIA", font=font(BOLD, 26), fill="white")
    d.text((560, 24), "SPECIMEN - SYNTHETIC", font=font(BOLD, 18), fill=(255, 220, 120))
    # placeholder portrait (no real face)
    d.rectangle([40, 110, 240, 370], fill=(200, 205, 215), outline=(90, 90, 110), width=2)
    d.ellipse([105, 150, 175, 225], fill=(150, 155, 170))
    d.ellipse([70, 235, 210, 360], fill=(150, 155, 170))
    d.text((70, 372), "SPECIMEN", font=font(BOLD, 16), fill=(160, 40, 40))
    for k, (label, x, y) in FIELDS.items():
        d.text((x, y), label, font=font(LABEL_FONT, 15), fill=(90, 90, 110))
    for k, box in FIELD_BOX.items():
        draw_text_box(img, box, vis[k], VALUE_FONT, VSIZE, rng, patch=(k in patch_fields))
    d.rectangle([0, 392, W, H], fill=(250, 250, 252))
    for i, line in enumerate(mrz_lines):
        draw_text_box(img, MRZ_BOX[i], line, MRZ_FONT, MSIZE, rng, patch=patch_mrz)
    return img


def degrade(img, rng):
    # same degradation distribution for genuine and tampered cards
    dx, dy = rng.randint(-4, 4), rng.randint(-4, 4)
    canvas = Image.new("RGB", (W, H), BG)
    canvas.paste(img, (dx, dy))
    canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.0, 0.9)))
    a = np.asarray(canvas).astype(np.float32) * rng.uniform(0.82, 1.12)
    a += np.random.RandomState(rng.randint(0, 10**6)).normal(0, rng.uniform(1.5, 7), a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def vis_strings(rec):
    return {"surname": rec["surname"], "given": rec["given"], "dob": fmt_date(rec["dob"]),
            "docno": rec["docno"], "issue": fmt_date(rec["issue"]), "expiry": fmt_date(rec["expiry"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=180, help="identities (2 cards each)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    rows = []
    for i in range(a.n):
        rec = new_record(rng)
        # genuine
        g = degrade(render_card(vis_strings(rec), build_mrz(rec), set(), False, rng), rng)
        gp = f"images/id{i:04d}_genuine.jpg"
        g.save(os.path.join(a.out, gp), quality=rng.randint(55, 95))
        rows.append([f"id{i:04d}_genuine", i, 0, "genuine", "", gp])
        # tampered
        ftype = rng.choices(["A", "B", "C"], weights=[4, 4, 2])[0]
        rec2, edited = edit_record(rec, rng)
        mrz_edit_rec = rec2
        if ftype == "A":
            mrz = build_mrz(rec)  # MRZ untouched
            patch_mrz = False
        elif ftype == "B":
            mrz = build_mrz(rec2, recompute=True)
            patch_mrz = True
        else:
            mrz = build_mrz(rec2, recompute=False, orig=rec)
            patch_mrz = True
        f = degrade(render_card(vis_strings(rec2), mrz, set(edited), patch_mrz, rng), rng)
        fp = f"images/id{i:04d}_forged{ftype}.jpg"
        f.save(os.path.join(a.out, fp), quality=rng.randint(55, 95))
        rows.append([f"id{i:04d}_forged{ftype}", i, 1, f"forged_{ftype}", "+".join(edited), fp])
    with open(os.path.join(a.out, "labels.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["doc_id", "identity", "label", "kind", "edited_fields", "path"])
        w.writerows(rows)
    print(f"wrote {len(rows)} cards to {a.out}/")


if __name__ == "__main__":
    main()
