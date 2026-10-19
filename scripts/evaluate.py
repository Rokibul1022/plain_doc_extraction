"""Run evaluation benchmarks across synthetic samples and output contact sheets."""

import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np

# Ensure docscan package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from docscan.core.detect import detect_document
from docscan.core.geometry import order_points
from docscan.core.pipeline import process
from docscan.export.ocr import is_ocr_available
from docscan.schemas import ProcessParams


def evaluate_ocr(orig_img: np.ndarray, proc_img: np.ndarray) -> tuple[int, int]:
    """If Tesseract is installed, compare character count extracted before and after."""
    if not is_ocr_available():
        return 0, 0
    try:
        import pytesseract
        text_before = pytesseract.image_to_string(orig_img)
        text_after = pytesseract.image_to_string(proc_img)
        return len(text_before.strip()), len(text_after.strip())
    except Exception:
        return 0, 0


def fit_image(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Letterbox-fit image into (target_w, target_h) preserving aspect ratio without distortion."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    scale = min(target_w / max(1, w), target_h / max(1, h))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(
        img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
    )
    out = np.full((target_h, target_w, 3), 28, dtype=np.uint8)
    ox = (target_w - nw) // 2
    oy = (target_h - nh) // 2
    out[oy : oy + nh, ox : ox + nw] = resized
    return out


def create_contact_sheet(pairs: list[tuple[str, np.ndarray, np.ndarray]]) -> np.ndarray:
    """Combine before and after pairs into a unified visual contact sheet image."""
    cell_w, cell_h = 440, 280
    num_rows = len(pairs)
    total_w = cell_w * 2 + 80
    row_stride = cell_h + 46
    total_h = 90 + num_rows * row_stride

    canvas = np.full((total_h, total_w, 3), 20, dtype=np.uint8)

    # Header title
    cv2.putText(
        canvas,
        "DOCSCAN BENCHMARK CONTACT SHEET: BEFORE VS AFTER (DOCS & PLATES)",
        (30, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    for idx, (title, before_img, after_img) in enumerate(pairs):
        y_offset = 75 + idx * row_stride

        # Letterboxed thumbnails
        b_thumb = fit_image(before_img, cell_w, cell_h)
        a_thumb = fit_image(after_img, cell_w, cell_h)

        # Draw labels
        cv2.putText(
            canvas,
            f"{title} - Raw Input",
            (30, y_offset + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{title} - DocScan Enhanced & Cleaned",
            (cell_w + 50, y_offset + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (90, 230, 130),
            1,
            cv2.LINE_AA,
        )

        canvas[y_offset + 26 : y_offset + 26 + cell_h, 30 : 30 + cell_w] = b_thumb
        canvas[
            y_offset + 26 : y_offset + 26 + cell_h,
            cell_w + 50 : cell_w + 50 + cell_w,
        ] = a_thumb

    return canvas


def main() -> None:
    root_dir = Path(__file__).parent.parent
    samples_dir = root_dir / "samples"
    eval_dir = root_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    gt_file = samples_dir / "ground_truth.json"
    ground_truth = {}
    if gt_file.is_file():
        with open(gt_file, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)

    print("\n" + "=" * 95)
    print(" DOCSCAN SYSTEM EVALUATION & ACCURACY BENCHMARK (DOCUMENTS & PLATES)")
    print("=" * 95)

    results = []
    pairs_for_sheet = []

    # 1. Evaluate synthetic documents against ground truth
    print("\n--- [PART 1: DOCUMENT BENCHMARKS] ---")
    print(
        f"{'Sample':<22} | {'Corners (px)':<12} | {'Err % Diag':<10} | {'Detected':<8} | {'Total ms':<8} | {'Mode'}"
    )
    print("-" * 95)

    success_count = 0
    total_docs = len(ground_truth)

    for filename, meta in ground_truth.items():
        img_path = samples_dir / filename
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        diag = np.sqrt(h**2 + w**2)
        gt_corners = order_points(np.array(meta["ground_truth_corners"], dtype=np.float32))

        # Detect
        det_corners, confidence, detected = detect_document(img, det_h=600, target="document")

        # Compute corner distances
        ordered_det = order_points(det_corners)
        dists = np.linalg.norm(gt_corners - ordered_det, axis=1)
        mean_err_px = float(np.mean(dists))
        err_pct_diag = (mean_err_px / diag) * 100.0

        is_success = detected and (err_pct_diag < 2.5)
        if is_success:
            success_count += 1

        params = ProcessParams(mode="auto")
        proc_result = process(img, ordered_det, params, preview=False)

        results.append(
            {
                "sample": filename,
                "category": "document",
                "mean_err_px": round(float(mean_err_px), 1),
                "err_pct_diag": round(float(err_pct_diag), 2),
                "detected": bool(detected),
                "is_success": bool(is_success),
                "confidence": round(float(confidence), 2),
                "mode_used": str(proc_result.mode_used),
                "timings_ms": {k: float(v) for k, v in proc_result.timings_ms.items()},
            }
        )

        pairs_for_sheet.append((filename, img, proc_result.image))

        print(
            f"{filename:<22} | {mean_err_px:>10.1f} px | {err_pct_diag:>9.2f}% | "
            f"{str(detected):<8} | {proc_result.timings_ms.get('total_ms', 0):>6.1f}ms | {proc_result.mode_used}"
        )

    # 2. Evaluate real-world license plate & factory label samples
    print("\n--- [PART 2: LICENSE PLATES & FACTORY LABELS (REAL-WORLD)] ---")
    print(
        f"{'Sample':<22} | {'Resolution':<12} | {'Detected':<8} | {'Conf %':<8} | {'Total ms':<8} | {'Mode'}"
    )
    print("-" * 95)

    real_samples = sorted(
        [
            p
            for p in samples_dir.glob("*.jpeg")
            if p.name not in ground_truth
        ]
    )
    # Also check if images.jpeg or others exist
    for img_path in real_samples:
        filename = img_path.name
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        res_str = f"{w}x{h}"

        # Detect with plate target
        det_corners, confidence, detected = detect_document(img, det_h=600, target="plate")
        ordered_det = order_points(det_corners)

        params = ProcessParams(mode="plate_text", auto_invert=True)
        proc_result = process(img, ordered_det, params, preview=False)

        results.append(
            {
                "sample": filename,
                "category": "plate_or_label",
                "resolution": res_str,
                "detected": bool(detected),
                "confidence": round(float(confidence), 2),
                "mode_used": str(proc_result.mode_used),
                "timings_ms": {k: float(v) for k, v in proc_result.timings_ms.items()},
            }
        )

        pairs_for_sheet.append((filename, img, proc_result.image))

        print(
            f"{filename:<22} | {res_str:<12} | {str(detected):<8} | "
            f"{int(confidence * 100):>6}% | {proc_result.timings_ms.get('total_ms', 0):>6.1f}ms | {proc_result.mode_used}"
        )

    print("-" * 95)
    if total_docs > 0:
        doc_success_rate = (success_count / total_docs) * 100.0
        print(f"Document Accuracy: {doc_success_rate:.1f}% ({success_count}/{total_docs})")

    plate_detect_count = sum(
        1 for r in results if r.get("category") == "plate_or_label" and r["detected"]
    )
    total_plates = len(real_samples)
    if total_plates > 0:
        print(
            f"Plate/Tag Detection Rate: {(plate_detect_count / total_plates) * 100:.1f}% "
            f"({plate_detect_count}/{total_plates})"
        )

    # Save contact sheet
    contact_sheet = create_contact_sheet(pairs_for_sheet)
    sheet_path = eval_dir / "contact_sheet.jpg"
    cv2.imwrite(str(sheet_path), contact_sheet)
    print(f"\nContact sheet saved to: {sheet_path}")

    # Write evaluation metrics json
    metrics_path = eval_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_samples": len(results),
                "samples": results,
            },
            f,
            indent=2,
        )
    print(f"Detailed metrics saved to: {metrics_path}\n" + "=" * 95)


if __name__ == "__main__":
    main()
