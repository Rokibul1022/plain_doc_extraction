"""OCR-guided deblurring.

Blind kernel estimation from image statistics is unreliable on tiny, noisy plates, but the plate
reader itself is an excellent judge: the right motion/defocus kernel makes the characters
suddenly readable. Candidates are deconvolved on the plate's native pixels and the one the OCR
is most confident about is kept - but only if it clearly beats the unblurred read.
"""

import cv2
import numpy as np

from ocr.ocr import PlateOCR, PlateReading
from opencv.restore import _disk_psf, _gauss_psf, _motion_psf, _wiener
from opencv.quality import estimate_noise, normalise_height, to_gray

MARGIN = 0.06  # required confidence gain over the un-deblurred read
PLAUSIBLE = 0.55  # a deblurred read must itself look like a real plate read, not noise


def _score(r: PlateReading) -> float:
    if not r.text:
        return 0.0
    return r.confidence * (1.08 if r.format_ok else 0.9) * min(1.0, 0.55 + 0.1 * len(r.text))


def _apply(ycc: np.ndarray, psf: np.ndarray, k_reg: float) -> np.ndarray:
    out = ycc.copy()
    out[..., 0] = np.clip(_wiener(ycc[..., 0], psf, k_reg), 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)


def deblur_guided(crop: np.ndarray, ocr: PlateOCR) -> tuple[np.ndarray, str, PlateReading] | None:
    """Return (deblurred crop, description, reading) if a kernel clearly improves legibility."""
    if not ocr.available or crop.ndim != 3 or min(crop.shape[:2]) < 10:
        return None
    base = ocr.read(crop)
    base_score = _score(base)
    if base_score > 0.93:  # already confidently read
        return None

    noise = estimate_noise(normalise_height(to_gray(crop)))
    k_reg = 0.02 if noise < 5 else 0.04
    ycc = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)

    # stage 1: coarse grid of motion kernels (CCTV motion is mostly near-horizontal) + defocus
    cands: list[tuple[str, np.ndarray]] = []
    for ang in (0, -8, 8, -16, 16, 28, -28):
        for length in (3, 5, 7, 9):
            cands.append((f"motion L={length} a={ang}", _motion_psf(length, ang)))
    for r in (1.0, 1.5, 2.0):
        cands.append((f"defocus r={r}", _disk_psf(r)))
    for sg in (0.8, 1.1, 1.4, 1.8, 2.2):
        cands.append((f"gauss s={sg}", _gauss_psf(sg)))
    imgs = [_apply(ycc, psf, k_reg) for _, psf in cands]
    reads = ocr.read_many(imgs)
    scores = [_score(r) for r in reads]
    i = int(np.argmax(scores))
    best_img, best_desc, best_read, best_score = imgs[i], cands[i][0], reads[i], scores[i]

    # stage 2: refine length/angle around a winning motion kernel
    if cands[i][0].startswith("motion"):
        length0 = int(cands[i][0].split("L=")[1].split()[0])
        ang0 = int(cands[i][0].split("a=")[1])
        fine = []
        for dl in (-1, 0, 1):
            for da in (-4, 0, 4):
                length, ang = max(2, length0 + dl), ang0 + da
                if (length, ang) != (length0, ang0):
                    fine.append((f"motion L={length} a={ang}", _motion_psf(length, ang)))
        if fine:
            fimgs = [_apply(ycc, psf, k_reg) for _, psf in fine]
            freads = ocr.read_many(fimgs)
            for (desc, _), im, rd in zip(fine, fimgs, freads):
                sc = _score(rd)
                if sc > best_score:
                    best_img, best_desc, best_read, best_score = im, desc, rd, sc

    # Improving the confidence of a garbage read (e.g. Latin letters hallucinated from another
    # script) is meaningless, so the winner must also be a plausible read in its own right.
    if best_score >= base_score + MARGIN and best_score >= PLAUSIBLE:
        return best_img, best_desc, best_read
    return None


# ---------------------------------------------------------------- Bengali plates
def _bn_score(reader, imgs: list[np.ndarray], width: float):
    """(reliability, reading) for each candidate image, judged by the Bengali digit reader alone."""
    from ocr import verdict_bn

    out = []
    for im in imgs:
        row = reader.read([im])
        out.append((verdict_bn.reliability(row, width) if row.ok else 0.0, row))
    return out


def deblur_guided_bn(crop: np.ndarray, reader, width: float | None = None, margin: float = 0.10):
    """Try motion/defocus/Gaussian kernels on a Bengali plate crop; keep the one the digit reader trusts most.

    Needs no labels: a kernel is accepted only if the reader's own calibrated reliability rises by `margin`
    over the un-deblurred crop. Returns (image, description, reliability) or None."""
    if not reader.available or crop.ndim != 3 or min(crop.shape[:2]) < 10:
        return None
    width = float(width if width is not None else crop.shape[1])
    base_rel, _ = _bn_score(reader, [crop], width)[0]
    if base_rel >= 0.93:  # already read with high confidence
        return None
    noise = estimate_noise(normalise_height(to_gray(crop)))
    k_reg = 0.02 if noise < 5 else 0.04
    ycc = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)

    cands: list[tuple[str, np.ndarray]] = []
    for ang in (0, -8, 8, -16, 16):
        for length in (3, 5, 7, 9):
            cands.append((f"motion L={length} a={ang}", _motion_psf(length, ang)))
    for r in (1.0, 1.5, 2.0):
        cands.append((f"defocus r={r}", _disk_psf(r)))
    for sg in (0.8, 1.1, 1.4, 1.8, 2.2):
        cands.append((f"gauss s={sg}", _gauss_psf(sg)))
    imgs = [_apply(ycc, psf, k_reg) for _, psf in cands]
    scored = _bn_score(reader, imgs, width)
    i = int(np.argmax([s for s, _ in scored]))
    best_rel, best_img, best_desc = scored[i][0], imgs[i], cands[i][0]

    if best_desc.startswith("motion"):  # refine length/angle around a winning motion kernel
        length0, ang0 = int(best_desc.split("L=")[1].split()[0]), int(best_desc.split("a=")[1])
        fine = [(f"motion L={max(2, length0 + dl)} a={ang0 + da}", _motion_psf(max(2, length0 + dl), ang0 + da))
                for dl in (-1, 0, 1) for da in (-4, 0, 4) if (dl, da) != (0, 0)]
        fimgs = [_apply(ycc, psf, k_reg) for _, psf in fine]
        for (desc, _), im, (rel, _) in zip(fine, fimgs, _bn_score(reader, fimgs, width)):
            if rel > best_rel:
                best_rel, best_img, best_desc = rel, im, desc

    if best_rel >= base_rel + margin:
        return best_img, best_desc, best_rel
    return None
