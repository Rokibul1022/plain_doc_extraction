"""
plate_scan.py - CamScanner-style license plate cleaner (OpenCV only)

Usage:
    python plate_scan.py plate.jpg
    python plate_scan.py plate.jpg --ocr        # also try to read the text (needs pytesseract)

Outputs (saved next to the input, inside ./output/):
    <name>_1_corners.jpg   -> detected 4 corners drawn on original
    <name>_2_warped.jpg    -> straightened plate (top-down view)
    <name>_3_clean.png     -> enhanced grayscale (denoised, sharpened)
    <name>_4_scan.png      -> final black text on white background (scanner look)
"""
import argparse
import os
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------- helpers
def order_points(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(pts, dtype="float32").reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()],
                     pts[s.argmax()], pts[d.argmax()]], dtype="float32")


def quad_ok(quad, img_area):
    """Check that a 4-point shape looks like a license plate (wide rectangle)."""
    q = order_points(quad)
    w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    if h < 1:
        return False
    ratio = w / h
    area = cv2.contourArea(q)
    # plates: wide (about 2:1 to 6:1), and not tiny
    return 1.3 <= ratio <= 7.0 and area > 0.01 * img_area


# ---------------------------------------------------------------- step 1: find 4 corners
def find_plate_corners(img):
    """Return (4 ordered corners, found_flag). Works on a resized copy for speed."""
    h0, w0 = img.shape[:2]
    scale = 700.0 / max(h0, w0)
    small = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img.copy()
    s = scale if scale < 1 else 1.0
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    area_small = small.shape[0] * small.shape[1]

    candidates = []

    # Method A: edges -> contours -> 4-point polygons
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    for lo, hi in [(30, 200), (50, 150), (20, 100)]:
        edges = cv2.Canny(blur, lo, hi)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:25]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx) and quad_ok(approx, area_small):
                candidates.append(approx.reshape(4, 2))
            else:
                # Method B: rotated rectangle of a big contour (handles rounded corners)
                if cv2.contourArea(c) > 0.01 * area_small:
                    box = cv2.boxPoints(cv2.minAreaRect(c))
                    if quad_ok(box, area_small):
                        candidates.append(box)
        if candidates:
            break

    # Method C: blackhat + threshold (finds dark text areas, then plate box around them)
    if not candidates:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, th = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 9)))
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:10]:
            box = cv2.boxPoints(cv2.minAreaRect(c))
            if quad_ok(box, area_small):
                candidates.append(box)

    if candidates:
        # choose the biggest valid candidate
        best = max(candidates, key=lambda q: cv2.contourArea(q.astype("float32")))
        return order_points(best / s), True

    # Fallback: whole image
    return order_points([[0, 0], [w0 - 1, 0], [w0 - 1, h0 - 1], [0, h0 - 1]]), False


# ---------------------------------------------------------------- step 2: straighten
def warp_plate(img, corners, pad=0.03):
    """Perspective-warp the plate to a flat rectangle."""
    tl, tr, br, bl = corners
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    w, h = max(w, 10), max(h, 10)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC)
    # add a small safety margin trim so the plate border does not hurt thresholding
    dy, dx = int(h * pad), int(w * pad)
    if h > 2 * dy + 10 and w > 2 * dx + 10:
        warped = warped[dy:h - dy, dx:w - dx]
    return warped


# ---------------------------------------------------------------- step 3: make it clear
def enhance_plate(warped, target_width=900):
    """Upscale -> denoise -> fix lighting -> contrast -> sharpen. Returns clean gray image."""
    h, w = warped.shape[:2]
    scale = target_width / w
    big = cv2.resize(warped, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

    # 1) denoise (keeps edges of letters)
    gray = cv2.fastNlMeansDenoising(gray, None, h=12, templateWindowSize=7, searchWindowSize=21)

    # 2) remove shadow / uneven light: divide by blurred background
    bg = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), 31)
    gray = cv2.divide(gray, bg, scale=255)

    # 3) local contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    # 4) sharpen (unsharp mask)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    gray = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
    return gray


# ---------------------------------------------------------------- step 4: scanner look
def to_scan(gray):
    """Black text on white background (smooth edges)."""
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.0)
    # Otsu works well here because lighting was already flattened in enhance_plate()
    t, _ = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bin_img = cv2.threshold(smooth, t * 0.95, 255, cv2.THRESH_BINARY)  # slight bias keeps strokes thin
    # local fallback: if Otsu gives a nearly empty/full page, use adaptive threshold
    black_ratio = (bin_img == 0).mean()
    if black_ratio < 0.02 or black_ratio > 0.6:
        bin_img = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 51, 10)
    # ensure background is white (plates can be dark with light text)
    border = np.concatenate([bin_img[:5].ravel(), bin_img[-5:].ravel(),
                             bin_img[:, :5].ravel(), bin_img[:, -5:].ravel()])
    if border.mean() < 127:
        bin_img = 255 - bin_img
    # tidy: remove tiny specks (keep letters), smooth jagged edges
    inv = 255 - bin_img
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    min_area = 0.00012 * bin_img.shape[0] * bin_img.shape[1]
    clean = np.zeros_like(inv)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    clean = cv2.GaussianBlur(clean, (0, 0), 0.9)
    _, clean = cv2.threshold(clean, 127, 255, cv2.THRESH_BINARY)
    result = 255 - clean
    return cv2.copyMakeBorder(result, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)


# ---------------------------------------------------------------- optional OCR
def try_ocr(img):
    try:
        import pytesseract
    except ImportError:
        print("[OCR] pytesseract not installed. Run: pip install pytesseract (and install Tesseract app).")
        return None
    cfg = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- "
    text = pytesseract.image_to_string(img, config=cfg).strip()
    return text


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Clean and straighten a license plate image.")
    ap.add_argument("image", help="path to input photo")
    ap.add_argument("--out", default="output", help="output folder (default: ./output)")
    ap.add_argument("--ocr", action="store_true", help="try to read text with pytesseract")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Cannot read image: {args.image}")
    os.makedirs(args.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.image))[0]

    # 1. corners
    corners, found = find_plate_corners(img)
    vis = img.copy()
    cv2.polylines(vis, [corners.astype(int)], True, (0, 255, 0), max(2, img.shape[1] // 300))
    for p in corners.astype(int):
        cv2.circle(vis, tuple(p), max(5, img.shape[1] // 150), (0, 0, 255), -1)
    print("4 corners detected." if found else "Corners NOT found -> using whole image.")
    cv2.imwrite(os.path.join(args.out, f"{name}_1_corners.jpg"), vis)

    # 2. straighten
    warped = warp_plate(img, corners)
    cv2.imwrite(os.path.join(args.out, f"{name}_2_warped.jpg"), warped)

    # 3. enhance
    clean = enhance_plate(warped)
    cv2.imwrite(os.path.join(args.out, f"{name}_3_clean.png"), clean)

    # 4. final scan
    scan = to_scan(clean)
    cv2.imwrite(os.path.join(args.out, f"{name}_4_scan.png"), scan)
    print(f"Saved results in: {args.out}/")

    if args.ocr:
        text = try_ocr(scan)
        if text is not None:
            print("OCR text:", text if text else "(nothing read)")


if __name__ == "__main__":
    main()
