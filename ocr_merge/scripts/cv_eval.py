"""Score cross-validated merged models: every labelled plate is read by a model that never saw it (or its number).

    python ocr_merge/scripts/cv_eval.py NAME [--decode6] [--out eval/ocr_merge/cv171/NAME.json]

Reads ocr_merge/weights/cv/NAME/fold{0..4}.pt (made by cv_train.sh), runs each fold's model on the GPU over that fold's plates
through the real decision code (`ocr.pipeline.build_report`), and pools the results over all labelled plates. Also writes the
per-plate readings so mistakes can be inspected. Letter scoring only counts labels that are single letters the model knows.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import ocr.pipeline as sep  # noqa: E402
from ocr.synth_row import CITY_WORDS, LETTERS  # noqa: E402
from ocr_merge.cv import folds  # noqa: E402
from ocr_merge.decode import decode6  # noqa: E402
from ocr_merge.prior import Priors  # noqa: E402
from ocr_merge.gpu import TorchSession  # noqa: E402
from ocr_merge.reader import MergedReader  # noqa: E402
from train_merged import labelled_plates  # noqa: E402

from ocr_merge.views import VIEW_SETS  # noqa: E402

sep.make_document = sep.relief_page = lambda x: x  # the display page is not part of the reading


class Scan:  # what build_report needs from a scan
    def __init__(self, crops):
        self.rectified, self.restored = crops
        self.corners, self.method = np.zeros((4, 2)), "cache"


def pick_top(c: dict):
    """The reader whose top row is used (same rule as ocr.pipeline: ink first, relief if the ink reader is unsure)."""
    ink, rel = c.get("ink"), c.get("relief")
    pick = ink if ink and ink.city else rel
    if ink and rel and ink.city and rel.city and ink.city_conf < 0.5 and rel.city_conf > ink.city_conf:
        pick = rel
    return pick


def evaluate(session_for_fold, plates: dict, k: int = 5, decode=None, seed: int = 0, views=None, prior=None, retry=None) -> list[dict]:
    """Per-plate results; `session_for_fold(i)` returns the model (an object with .run) for fold i's held-out plates.
    `prior=(a_city, a_letter)` decides the top row with class priors estimated from the other folds' labels only."""
    out = []
    all_folds = folds(plates, k, seed)
    for i, held in enumerate(all_folds):
        reader = MergedReader()
        if prior:
            rest = [plates[f] for j, fl in enumerate(all_folds) if j != i for f in fl]
            reader.priors, (reader.a_city, reader.a_letter) = Priors([p["city"] for p in rest], [p["letter"] for p in rest]), prior
        reader._sess, reader._tried, reader.decode = session_for_fold(i), True, decode
        if views:
            reader.views = views
        for f in held:
            p = plates[f]
            cands = reader.candidates(p["crops"])
            rep = sep.build_report(Scan(p["crops"]), cands)
            if retry and rep.verdict != "readable":  # same as ocr_merge.pipeline.read_plate: doubtful plates get a denser vote
                cands = reader.candidates(p["crops"], VIEW_SETS[retry])
                rep = sep.build_report(Scan(p["crops"]), cands)
            pk = pick_top(cands)
            out.append({"file": f, "fold": i, "truth": p["digits"], "digits": rep.digits, "digits_right": sum(a == b for a, b in zip(rep.digits, p["digits"])),
                        "exact": int(rep.digits == p["digits"]), "verdict": rep.verdict, "mode": rep.mode,
                        "city": rep.top, "city_label": CITY_WORDS[p["city"]] if p["city"] >= 0 else "",
                        "letter_raw": pk.letter if pk else "", "letter_conf": round(pk.letter_conf, 3) if pk else 0.0,
                        "letter_shown": rep.letter, "letter_label": LETTERS[p["letter"]] if p["letter"] >= 0 else "",
                        "city_probs": [round(float(x), 4) for x in pk.city_probs] if pk else [], "letter_probs": [round(float(x), 4) for x in pk.letter_probs] if pk else []})
    return out


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    city = [r for r in rows if r["city_label"]]
    let = [r for r in rows if r["letter_label"]]
    shown = [r for r in let if r["letter_shown"]]
    s = {"plates": n, "digits": f"{sum(r['digits_right'] for r in rows)}/{6 * n}", "digits_pct": round(100 * sum(r['digits_right'] for r in rows) / (6 * n), 1),
         "whole": f"{sum(r['exact'] for r in rows)}/{n}", "whole_pct": round(100 * sum(r['exact'] for r in rows) / n, 1),
         "city": f"{sum(r['city'] == r['city_label'] for r in city)}/{len(city)}", "city_pct": round(100 * sum(r['city'] == r['city_label'] for r in city) / len(city), 1),
         "letter_best_guess": f"{sum(r['letter_raw'] == r['letter_label'] for r in let)}/{len(let)}",
         "letter_best_guess_pct": round(100 * sum(r['letter_raw'] == r['letter_label'] for r in let) / len(let), 1),
         "letter_shown": f"{sum(r['letter_shown'] == r['letter_label'] for r in shown)}/{len(shown)} shown",
         "by_verdict": {v: f"{sum(r['exact'] for r in rows if r['verdict'] == v)}/{sum(r['verdict'] == v for r in rows)}" for v in ("readable", "uncertain", "unreadable")}}
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--decode6", action="store_true", help="force exactly six digits per view")
    ap.add_argument("--views", choices=tuple(VIEW_SETS), default="6", help="digit-row views per crop")
    ap.add_argument("--prior", nargs=2, type=float, metavar=("A_CITY", "A_LETTER"), help="decide city/letter with class priors of this strength")
    ap.add_argument("--retry", choices=tuple(VIEW_SETS), help="re-read plates that are not readable with this view set")
    ap.add_argument("--out")
    ap.add_argument("--weights-dir", default=str(ROOT / "ocr_merge" / "weights" / "cv"))
    a = ap.parse_args()
    plates = labelled_plates(ROOT / "ocr_merge" / "weights" / "real_plates.pkl")
    rows = evaluate(lambda i: TorchSession(Path(a.weights_dir) / a.name / f"fold{i}.pt"), plates, decode=decode6 if a.decode6 else None, views=VIEW_SETS[a.views] if a.views != "6" else None, prior=tuple(a.prior) if a.prior else None, retry=a.retry)
    s = summarize(rows)
    print(a.name + (" +decode6" if a.decode6 else ""), json.dumps(s, ensure_ascii=False))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"summary": s, "plates": rows}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
