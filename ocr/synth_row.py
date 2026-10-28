"""Synthetic Bangladeshi plates for training the digit-row reader (training-only; needs PIL + fonts).

Renders a full plate - frame, bolts, city row, `NN-NNNN` digit row with emboss shadow - in a random
body/ink colour, tilts and re-crops it like the detector's rectified crop would be, then degrades it
like low-resolution CCTV (downscale, blur, motion, noise, JPEG, exposure). The model input is the
lower region of that crop, so the top row is partly visible and must be ignored, as on real plates.
"""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from ocr.digits import DIGITS, FONT_PATH
from ocr.degrade import jpeg, motion_blur

from ocr.rowprep import IN_H, IN_W, digit_row_image  # model input size + the shared normaliser
REGION_TOP = 0.40  # the model sees the plate from 40 % of its height down

# city/district words seen on plates (the first six are in the labelled data) plus common others
CITY_WORDS = ["চট্ট মেট্রো", "ঢাকা মেট্রো", "ঝালকাঠি", "নোয়াখালী", "কুমিল্লা", "রাঙ্গামাটি", "চট্টগ্রাম", "খুলনা মেট্রো",
              "রাজশাহী মেট্রো", "সিলেট মেট্রো", "বরিশাল মেট্রো", "রংপুর মেট্রো", "গাজীপুর", "ময়মনসিংহ", "নারায়ণগঞ্জ",
              "বগুড়া", "ফেনী", "কক্সবাজার"]
LETTERS = "কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ"
TOP_H, TOP_W = 48, 160  # top-row model input
TOP_REGION = 0.52  # the top-row model sees the plate down to 52 % of its height

BODIES = [(236, 236, 232), (225, 225, 222), (245, 240, 215), (95, 200, 215), (150, 215, 210), (70, 165, 150),
          (60, 140, 120), (215, 205, 170), (240, 215, 80), (210, 225, 240), (200, 205, 210), (250, 250, 250)]
INKS = [(15, 15, 15), (40, 25, 20), (60, 30, 25), (30, 40, 20), (25, 25, 60), (20, 20, 20), (50, 50, 50)]


_FONT_DIR = Path(__file__).with_name("fonts")
# (file, collection index, sampling weight): bold/medium plate-like faces most often
_FONT_SPECS = [
    ("HindSiliguri-Bold.ttf", 0, 3), ("HindSiliguri-SemiBold.ttf", 0, 3), ("HindSiliguri-Medium.ttf", 0, 2),
    ("Mina-Bold.ttf", 0, 3), ("Mina-Regular.ttf", 0, 2), ("Atma-Bold.ttf", 0, 2), ("Atma-SemiBold.ttf", 0, 2),
    ("BalooDa2_wght.ttf", 0, 2), ("TiroBangla-Regular.ttf", 0, 1), ("Galada-Regular.ttf", 0, 1),
]


@lru_cache(maxsize=1)
def _font_table() -> tuple[list[tuple[str, int]], np.ndarray]:
    """Available (path, index) faces with sampling probabilities: package fonts plus Nirmala UI."""
    faces, weights = [], []
    for name, idx, w in _FONT_SPECS:
        path = _FONT_DIR / name
        if path.exists():
            faces.append((str(path), idx))
            weights.append(w)
    for idx, w in ((0, 2), (1, 3), (2, 1), (3, 1), (4, 2)):  # Nirmala UI / Text, regular+bold+semilight
        if Path(FONT_PATH).exists():
            faces.append((FONT_PATH, idx))
            weights.append(w)
    w = np.array(weights, float)
    return faces, w / w.sum()


@lru_cache(maxsize=1024)
def _font_cached(size: int, path: str, index: int):
    from PIL import ImageFont

    return ImageFont.truetype(path, max(8, size), index=index)


def _font(rng: np.random.Generator, size: int):
    faces, p = _font_table()
    path, index = faces[int(rng.choice(len(faces), p=p))]
    return _font_cached(int(size), path, index)


def render_plate(rng: np.random.Generator) -> tuple[np.ndarray, str, tuple[int, int]]:
    h = 300
    w = int(h * rng.uniform(1.25, 1.95))
    body = np.array(BODIES[int(rng.integers(len(BODIES)))], np.float32) + rng.normal(0, 8, 3)
    ink = np.array(INKS[int(rng.integers(len(INKS)))], np.float32)
    img = np.empty((h, w, 3), np.uint8)
    img[:] = np.clip(body, 0, 255).astype(np.uint8)
    if rng.random() < 0.7:  # uneven lighting across the plate
        grad = np.linspace(rng.uniform(0.75, 1.0), rng.uniform(0.85, 1.1), w, dtype=np.float32)[None, :, None]
        img = np.clip(img.astype(np.float32) * grad, 0, 255).astype(np.uint8)
    m = int(h * rng.uniform(0.025, 0.06))
    t = int(rng.integers(3, 11))
    if rng.random() < 0.85:
        cv2.rectangle(img, (m, m), (w - m, h - m), tuple(int(c) for c in ink), t, cv2.LINE_AA)
    if rng.random() < 0.55:
        for cx in (int(w * rng.uniform(0.10, 0.2)), int(w * rng.uniform(0.8, 0.9))):
            cv2.circle(img, (cx, int(h * rng.uniform(0.04, 0.09))), int(h * rng.uniform(0.03, 0.05)), tuple(int(c) for c in ink * 0.5), -1, cv2.LINE_AA)

    from PIL import Image, ImageDraw

    pil = Image.fromarray(img[..., ::-1])
    d = ImageDraw.Draw(pil)
    inkc = tuple(int(c) for c in ink[::-1])

    # bottom row: NN-NNNN (kept inside the plate: width <= 85 % of it)
    values = [int(rng.integers(10)) for _ in range(6)]
    digits = "".join(DIGITS[v] for v in values)  # the label: a digit is the same class in either script
    shown = [DIGITS[v] for v in values]  # Bangladeshi plates carry Bengali numerals only
    layout_b = rng.random() < 0.3  # city word alone on top; class letter in front of the number on the bottom row
    size = int(h * rng.uniform(0.34, 0.46))
    pitch = min(size * rng.uniform(0.62, 0.78), 0.84 * w / 6.7)
    cy = h * rng.uniform(0.66, 0.80)
    cx0 = w * rng.uniform(0.48, 0.52) - pitch * 6.7 / 2 + (pitch * 0.55 if layout_b else 0.0)
    font = _font(rng, size)
    shadow = tuple(int(c) for c in np.clip(body * 0.75, 0, 255)[::-1])
    emboss = rng.random() < 0.5
    for i, dg in enumerate(shown):
        k = i if i < 2 else i + 0.7  # the separator takes ~0.7 of a pitch
        x = cx0 + pitch * (k + 0.5) + rng.normal(0, pitch * 0.02)
        if emboss:
            d.text((x + 3, cy + 3), dg, font=font, fill=shadow, anchor="mm")
        d.text((x, cy + rng.normal(0, 2)), dg, font=font, fill=inkc, anchor="mm")
    letter_i = int(rng.integers(len(LETTERS)))
    if layout_b:
        d.text((cx0 - pitch * 0.55, cy + rng.normal(0, 2)), LETTERS[letter_i], font=font, fill=inkc, anchor="mm")
    img = np.ascontiguousarray(np.asarray(pil)[..., ::-1])
    sx = cx0 + pitch * 2.35
    if rng.random() < 0.6:
        cv2.line(img, (int(sx - pitch * 0.2), int(cy)), (int(sx + pitch * 0.2), int(cy)), tuple(int(c) for c in ink), max(3, size // 12), cv2.LINE_AA)
    else:
        cv2.circle(img, (int(sx), int(cy + size * 0.18)), max(3, size // 14), tuple(int(c) for c in ink), -1, cv2.LINE_AA)
    # top row: city word + class letter, properly shaped (needs textrender; plain PIL scrambles Bengali)
    city_i = int(rng.integers(len(CITY_WORDS)))
    try:
        from ocr import textrender

        top_text = CITY_WORDS[city_i] + ("" if layout_b else "-" + LETTERS[letter_i])
        mask = textrender.render(top_text, int(h * rng.uniform(0.17, 0.25)))
        mh, mw = mask.shape
        if mw > 0.9 * w:
            mask = cv2.resize(mask, (int(0.9 * w), int(mh * 0.9 * w / mw)), interpolation=cv2.INTER_AREA)
            mh, mw = mask.shape
        x0 = int(w * rng.uniform(0.47, 0.53) - mw / 2)
        y0 = int(h * rng.uniform(0.10, 0.20))
        x0, y0 = max(0, min(x0, w - mw)), max(0, min(y0, h - mh))
        alpha = (255 - mask).astype(np.float32)[..., None] / 255.0
        roi = img[y0: y0 + mh, x0: x0 + mw].astype(np.float32)
        img[y0: y0 + mh, x0: x0 + mw] = np.clip(roi * (1 - alpha) + ink * alpha, 0, 255).astype(np.uint8)
    except Exception:  # no shaping stack / font: leave the top row blank
        pass
    return img, digits, (city_i, -1 if layout_b else letter_i)


def to_region(crop: np.ndarray) -> np.ndarray:
    """Model input from a rectified plate crop: lower region, grey, resized to IN_H x IN_W (uint8)."""
    h = crop.shape[0]
    region = crop[int(REGION_TOP * h):]
    g = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
    return cv2.resize(g, (IN_W, IN_H), interpolation=cv2.INTER_AREA if g.shape[1] > IN_W else cv2.INTER_CUBIC)


def degraded_crop(plate: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A rendered plate seen the way the rectified crop of a real (mostly clear, sometimes soft) frame looks:
    the plate fills the crop, with a little tilt, perspective and an occasional strip of surroundings."""
    h, w = plate.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jit = lambda: rng.normal(0, 0.012, 2) * np.array([w, h])  # noqa: E731
    dst = src + np.float32([jit(), jit(), jit(), jit()])
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-7, 7), 1.0)
    dst = (np.c_[dst, np.ones(4)] @ rot.T).astype(np.float32)
    pad = int(0.25 * max(w, h))
    big = cv2.copyMakeBorder(plate, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    mat = cv2.getPerspectiveTransform(src + pad, dst + pad)
    warped = cv2.warpPerspective(big, mat, (w + 2 * pad, h + 2 * pad), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    cx0, cy0 = pad + int(rng.uniform(0, 0.03) * w), pad + int(rng.uniform(0, 0.03) * h)
    crop = warped[cy0: pad + h - int(rng.uniform(0, 0.03) * h), cx0: pad + w - int(rng.uniform(0, 0.03) * w)]
    if rng.random() < 0.3:  # strip of dark surroundings on one or two sides, as on real crops
        m = int(rng.uniform(0.02, 0.07) * crop.shape[1])
        colour = [int(c) for c in rng.integers(10, 90, 3)]
        side = int(rng.integers(4))
        crop = cv2.copyMakeBorder(crop, m if side == 0 else 0, m if side == 1 else 0, m if side == 2 else 0, m if side == 3 else 0,
                                  cv2.BORDER_CONSTANT, value=colour)

    # camera: mostly clear, sometimes soft or compressed
    s = rng.uniform(85, 170) / crop.shape[1]
    low = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if rng.random() < 0.65:
        low = cv2.GaussianBlur(low, (0, 0), rng.uniform(0.3, 1.2))
    if rng.random() < 0.12:
        low = motion_blur(low, rng.uniform(2, 4), rng.uniform(-12, 12))
    low = low.astype(np.float32) * rng.uniform(0.75, 1.1) + rng.uniform(-10, 10)
    low = np.clip(low + rng.normal(0, rng.uniform(0, 5), low.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.7:
        low = jpeg(low, int(rng.integers(35, 95)))
    return cv2.resize(low, (160, 120) if rng.random() < 0.5 else (max(80, low.shape[1] * 2), max(60, low.shape[0] * 2)),
                      interpolation=cv2.INTER_CUBIC)


def sample(rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """(digit-row model input, 6 Bengali digits)."""
    plate, label, _ = render_plate(rng)
    return to_region(degraded_crop(plate, rng)), label


def to_top_region(crop: np.ndarray) -> np.ndarray:
    """Top-row model input from a rectified plate crop: upper region, grey, TOP_H x TOP_W (uint8)."""
    h = crop.shape[0]
    region = crop[: int(TOP_REGION * h)]
    g = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
    return cv2.resize(g, (TOP_W, TOP_H), interpolation=cv2.INTER_AREA if g.shape[1] > TOP_W else cv2.INTER_CUBIC)


def sample_top(rng: np.random.Generator) -> tuple[np.ndarray, int, int]:
    """(top-row model input, city index, letter index)."""
    plate, _, (city, letter) = render_plate(rng)
    return to_top_region(degraded_crop(plate, rng)), city, letter
