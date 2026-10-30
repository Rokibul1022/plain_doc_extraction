"""Write ocr_merge/priors.json from the labelled plates (city / letter counts only; no numbers, no photos).

    python ocr_merge/scripts/make_priors.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ocr_merge.prior import PATH, Priors  # noqa: E402
from train_merged import labelled_plates  # noqa: E402

plates = labelled_plates(ROOT / "ocr_merge" / "weights" / "real_plates.pkl")
pri = Priors([p["city"] for p in plates.values()], [p["letter"] for p in plates.values()])
pri.to_json()
print(f"{len(plates)} labelled plates -> {PATH} ({int(pri.city_n.sum())} cities, {int(pri.letter_n.sum())} letters)")
