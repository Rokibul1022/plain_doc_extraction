"""Evaluate the detector + pipeline on the real-world images in `new samples/`.

Writes `eval/new_samples_contact_sheet.jpg` (detected quad on the raw input beside the
enhanced output) and `eval/new_samples_metrics.json`. There is no ground truth for these
images, so crops are judged visually from the sheet; the metrics record what the detector
reported (area covered, confidence, timings).
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from docscan.core.detect import detect_document
from docscan.core.geometry import order_points
from docscan.core.pipeline import process
from docscan.schemas import ProcessParams
from scripts.evaluate import evaluate_ocr, fit_image  # noqa: E402

CELL_W, CELL_H = 440, 280
COLS = 2
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def natural_key(path: Path) -> list:
    import re

    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def draw_quad(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    vis = img.copy()
    thick = max(2, max(vis.shape[:2]) // 180)
    cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 230, 0), thick, cv2.LINE_AA)
    for x, y in quad:
        cv2.circle(vis, (int(x), int(y)), thick * 2, (0, 140, 255), -1, cv2.LINE_AA)
    return vis


def build_sheet(rows: list[dict]) -> np.ndarray:
    """Grid of (input + detected quad | enhanced output) pairs, COLS pairs per row."""
    pair_w = CELL_W * 2 + 30
    block_h = CELL_H + 52
    n_rows = (len(rows) + COLS - 1) // COLS
    total_w = pair_w * COLS + 30
    total_h = 80 + n_rows * block_h
    canvas = np.full((total_h, total_w, 3), 20, dtype=np.uint8)

    cv2.putText(canvas, "DOCSCAN NEW SAMPLES: DETECTED CROP (LEFT) VS ENHANCED OUTPUT (RIGHT)",
                (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA)

    for i, r in enumerate(rows):
        col, row = i % COLS, i // COLS
        x = 24 + col * pair_w
        y = 70 + row * block_h
        flag = "" if r["detected"] else "  [FULL FRAME]"
        label = (f"{r['sample']}  area {r['area_pct']:.0f}%  conf {int(r['confidence'] * 100)}%"
                 f"{flag}")
        cv2.putText(canvas, label, (x, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (90, 230, 130) if r["detected"] else (80, 160, 255), 1, cv2.LINE_AA)
        canvas[y + 26:y + 26 + CELL_H, x:x + CELL_W] = fit_image(r["_vis"], CELL_W, CELL_H)
        canvas[y + 26:y + 26 + CELL_H, x + CELL_W + 10:x + 2 * CELL_W + 10] = fit_image(
            r["_out"], CELL_W, CELL_H
        )
    return canvas


def main() -> None:
    root = Path(__file__).parent.parent
    samples_dir = root / "new samples"
    eval_dir = root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        (p for p in samples_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS), key=natural_key
    )

    print("\n" + "=" * 100)
    print(" DOCSCAN EVALUATION: NEW SAMPLES")
    print("=" * 100)
    print(f"{'Sample':<18} | {'Resolution':<10} | {'Detected':<8} | {'Area %':<7} | "
          f"{'Conf %':<6} | {'Total ms':<8} | {'Out size':<10} | Mode")
    print("-" * 100)

    rows = []
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"{path.name:<18} | unreadable, skipped")
            continue
        h, w = img.shape[:2]

        corners, confidence, detected = detect_document(img, det_h=600, target="auto")
        quad = order_points(corners)
        area_pct = 100.0 * float(cv2.contourArea(quad.astype(np.float32))) / (w * h)

        result = process(img, quad, ProcessParams(mode="plate_text", auto_invert=True), preview=False)
        ocr_before, ocr_after = evaluate_ocr(img, result.image)
        oh, ow = result.image.shape[:2]
        total_ms = float(result.timings_ms.get("total_ms", 0.0))

        rows.append({
            "sample": path.name,
            "resolution": f"{w}x{h}",
            "detected": bool(detected),
            "confidence": round(float(confidence), 2),
            "area_pct": round(area_pct, 1),
            "corners": [[round(float(a), 1), round(float(b), 1)] for a, b in quad],
            "mode_used": str(result.mode_used),
            "output_size": f"{ow}x{oh}",
            "ocr_chars_before": ocr_before,
            "ocr_chars_after": ocr_after,
            "timings_ms": {k: float(v) for k, v in result.timings_ms.items()},
            "_vis": draw_quad(img, quad),
            "_out": result.image,
        })
        print(f"{path.name:<18} | {w}x{h:<7} | {str(detected):<8} | {area_pct:>6.1f} | "
              f"{int(confidence * 100):>5}% | {total_ms:>6.1f}ms | {ow}x{oh:<6} | {result.mode_used}")

    print("-" * 100)
    detected_n = sum(r["detected"] for r in rows)
    print(f"Detected: {detected_n}/{len(rows)}   Full-frame fallbacks: {len(rows) - detected_n}")

    sheet_path = eval_dir / "new_samples_contact_sheet.jpg"
    cv2.imwrite(str(sheet_path), build_sheet(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])

    metrics_path = eval_dir / "new_samples_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_samples": len(rows),
                "detected": detected_n,
                "samples": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
            },
            f,
            indent=2,
        )
    print(f"\nContact sheet: {sheet_path}\nMetrics:       {metrics_path}\n" + "=" * 100)


if __name__ == "__main__":
    main()
