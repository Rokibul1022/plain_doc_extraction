"""Generate synthetic photographic test documents with ground-truth corner annotations."""

import json
from pathlib import Path
import sys
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from docscan.core.geometry import order_points


def create_table_background(h: int, w: int) -> np.ndarray:
    """Create a textured wooden/desktop background canvas."""
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:, :] = (38, 48, 62)  # Dark desktop slate/wood
    noise = np.random.randint(-8, 8, (h, w, 3), dtype=np.int16)
    return np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def apply_shadow_and_noise(img: np.ndarray) -> np.ndarray:
    """Apply non-uniform illumination gradient and sensor noise."""
    h, w = img.shape[:2]
    # Linear gradient shadow from top-left to bottom-right
    x_grad = np.linspace(0.5, 1.1, w, dtype=np.float32)
    y_grad = np.linspace(0.6, 1.05, h, dtype=np.float32)[:, None]
    lighting = np.clip(x_grad * y_grad, 0.35, 1.15)[:, :, None]
    shadowed = np.clip(img.astype(np.float32) * lighting, 0, 255).astype(np.uint8)

    noise = np.random.normal(0, 2.5, img.shape).astype(np.float32)
    return np.clip(shadowed.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def warp_onto_background(
    doc_img: np.ndarray, bg_w: int, bg_h: int, corner_margins: list[tuple[float, float]]
) -> tuple[np.ndarray, list[list[float]]]:
    """Perspective warp document into canvas margins."""
    doc_h, doc_w = doc_img.shape[:2]
    src_pts = np.array(
        [[0, 0], [doc_w - 1, 0], [doc_w - 1, doc_h - 1], [0, doc_h - 1]],
        dtype=np.float32,
    )

    dst_pts = np.array(
        [[bg_w * mx, bg_h * my] for mx, my in corner_margins], dtype=np.float32
    )

    bg = create_table_background(bg_h, bg_w)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_doc = cv2.warpPerspective(doc_img, M, (bg_w, bg_h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(
        np.full((doc_h, doc_w), 255, dtype=np.uint8),
        M,
        (bg_w, bg_h),
        flags=cv2.INTER_NEAREST,
    )

    mask_3d = mask[:, :, None] > 0
    composite = np.where(mask_3d, warped_doc, bg)
    photo = apply_shadow_and_noise(composite)
    ordered_gt = order_points(dst_pts).tolist()
    return photo, ordered_gt


def generate_invoice() -> tuple[np.ndarray, list[list[float]]]:
    """Generate business invoice sample."""
    doc = np.full((1000, 750, 3), 245, dtype=np.uint8)
    cv2.putText(doc, "TAX INVOICE", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3)
    cv2.putText(doc, "Acme Solutions Corp #INV-2026-88", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
    cv2.line(doc, (50, 130), (700, 130), (40, 40, 40), 2)

    # Table header
    cv2.rectangle(doc, (50, 180), (700, 220), (220, 220, 220), -1)
    cv2.putText(doc, "Item Description", (60, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)
    cv2.putText(doc, "Qty", (450, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)
    cv2.putText(doc, "Amount", (580, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)

    # Line items
    for idx, y in enumerate(range(260, 600, 60)):
        cv2.putText(doc, f"Consulting & Development Service Stage {idx+1}", (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
        cv2.putText(doc, "1.0", (460, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
        cv2.putText(doc, f"${(idx+2)*1450:.2f}", (580, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
        cv2.line(doc, (50, y + 20), (700, y + 20), (200, 200, 200), 1)

    # Stamp
    cv2.rectangle(doc, (480, 780), (680, 860), (30, 30, 190), 3)
    cv2.putText(doc, "PAID IN FULL", (500, 830), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 190), 2)

    return warp_onto_background(
        doc, 1400, 1100, [(0.15, 0.12), (0.84, 0.08), (0.88, 0.88), (0.12, 0.82)]
    )


def generate_certificate() -> tuple[np.ndarray, list[list[float]]]:
    """Generate colorful achievement certificate."""
    doc = np.full((800, 1100, 3), 250, dtype=np.uint8)
    # Double ornamental border
    cv2.rectangle(doc, (30, 30), (1070, 770), (40, 120, 200), 6)
    cv2.rectangle(doc, (45, 45), (1055, 755), (180, 120, 20), 2)

    cv2.putText(doc, "CERTIFICATE OF EXCELLENCE", (220, 150), cv2.FONT_HERSHEY_TRIPLEX, 1.3, (30, 30, 30), 2)
    cv2.putText(doc, "This certificate is proudly presented to", (340, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 1)
    cv2.putText(doc, "ALEXANDER HAMILTON", (320, 330), cv2.FONT_HERSHEY_TRIPLEX, 1.4, (20, 80, 180), 3)
    cv2.putText(doc, "For outstanding achievements in classical computer vision engineering.", (160, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 1)

    # Golden seal badge
    cv2.circle(doc, (900, 620), 65, (30, 180, 220), -1)
    cv2.circle(doc, (900, 620), 60, (20, 140, 180), 3)
    cv2.putText(doc, "VERIFIED", (845, 628), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)

    return warp_onto_background(
        doc, 1500, 1100, [(0.10, 0.16), (0.88, 0.10), (0.92, 0.84), (0.08, 0.90)]
    )


def generate_receipt() -> tuple[np.ndarray, list[list[float]]]:
    """Generate long receipt document."""
    doc = np.full((1200, 500, 3), 242, dtype=np.uint8)
    cv2.putText(doc, "*** FRESH MARKET ***", (90, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.putText(doc, "Store #402 - Terminal 12", (130, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
    cv2.line(doc, (30, 150), (470, 150), (50, 50, 50), 1)

    items = [
        ("Organic Whole Milk 1 Gal", "$4.89"),
        ("Sourdough Artisan Bread", "$5.49"),
        ("Honey Crisp Apples 2.3 lb", "$6.12"),
        ("Dark Roast Coffee Beans", "$14.99"),
        ("Greek Yogurt 32oz", "$5.29"),
        ("Olive Oil Extra Virgin", "$12.49"),
        ("Organic Free Range Eggs", "$4.99"),
    ]

    y = 200
    for name, price in items:
        cv2.putText(doc, name, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
        cv2.putText(doc, price, (380, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
        y += 50

    cv2.line(doc, (30, y + 20), (470, y + 20), (50, 50, 50), 2)
    cv2.putText(doc, "SUBTOTAL", (40, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    cv2.putText(doc, "$54.26", (370, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    cv2.putText(doc, "TAX 8.25%", (40, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
    cv2.putText(doc, "$4.48", (390, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
    cv2.putText(doc, "TOTAL PAID", (40, y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 2)
    cv2.putText(doc, "$58.74", (350, y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 2)

    return warp_onto_background(
        doc, 1300, 1200, [(0.24, 0.08), (0.76, 0.12), (0.72, 0.92), (0.28, 0.88)]
    )


def generate_handwritten() -> tuple[np.ndarray, list[list[float]]]:
    """Generate notebook paper with simulated handwriting lines."""
    doc = np.full((1000, 750, 3), 246, dtype=np.uint8)
    # Notebook margin line (red)
    cv2.line(doc, (110, 0), (110, 1000), (90, 90, 230), 2)
    # Feint blue ruled lines
    for y in range(80, 960, 40):
        cv2.line(doc, (0, y), (750, y), (220, 200, 180), 1)
        # Simulated handwriting text
        length = np.random.randint(300, 680)
        cv2.line(doc, (130, y - 5), (length, y - 5), (120, 40, 20), 2)

    return warp_onto_background(
        doc, 1400, 1100, [(0.18, 0.10), (0.82, 0.14), (0.86, 0.88), (0.14, 0.84)]
    )


def main() -> None:
    samples_dir = Path(__file__).parent.parent / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    generators = {
        "invoice.jpg": generate_invoice,
        "certificate.jpg": generate_certificate,
        "receipt.jpg": generate_receipt,
        "handwritten_notes.jpg": generate_handwritten,
    }

    metadata: dict[str, dict] = {}
    print(f"Synthesizing sample documents into {samples_dir} ...")

    for filename, gen_fn in generators.items():
        img, gt_corners = gen_fn()
        out_path = samples_dir / filename
        cv2.imwrite(str(out_path), img)

        metadata[filename] = {
            "ground_truth_corners": gt_corners,
            "width": img.shape[1],
            "height": img.shape[0],
        }
        print(f" -> Generated {filename} ({img.shape[1]}x{img.shape[0]} px)")

    gt_json_path = samples_dir / "ground_truth.json"
    with open(gt_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"All samples saved. Ground truth metadata written to {gt_json_path}")


if __name__ == "__main__":
    main()
