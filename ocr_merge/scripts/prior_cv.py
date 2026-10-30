"""Does the class-prior decision help? Cross-validated: priors for a fold come from the other folds' labels only.

    python ocr_merge/scripts/prior_cv.py eval/ocr_merge/cv171/base.json
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ocr.synth_row import CITY_WORDS, LETTERS  # noqa: E402
from ocr_merge.prior import Priors  # noqa: E402


def scan(rows: list[dict], a_city: float, a_letter: float, show_min: float = 0.6) -> dict:
    ci_ok = ci_n = li_ok = li_n = sh_ok = sh_n = 0
    for fold in sorted({r["fold"] for r in rows}):
        tr = [r for r in rows if r["fold"] != fold]
        pri = Priors([CITY_WORDS.index(r["city_label"]) if r["city_label"] and r["city_label"] in CITY_WORDS else -1 for r in tr],
                     [LETTERS.index(r["letter_label"]) if r["letter_label"] and r["letter_label"] in LETTERS else -1 for r in tr])
        for r in (x for x in rows if x["fold"] == fold):
            if not r["city_probs"]:
                continue
            c, _, l, conf = pri.decide(r["city_probs"], r["letter_probs"], a_city, a_letter)
            if r["city_label"] and r["city_label"] in CITY_WORDS:
                ci_n += 1
                ci_ok += CITY_WORDS[c] == r["city_label"]
            if r["letter_label"] and r["letter_label"] in LETTERS:
                li_n += 1
                li_ok += LETTERS[l] == r["letter_label"]
                if conf >= show_min:
                    sh_n += 1
                    sh_ok += LETTERS[l] == r["letter_label"]
    return {"city": f"{ci_ok}/{ci_n} = {100 * ci_ok / ci_n:.1f}%", "letter": f"{li_ok}/{li_n} = {100 * li_ok / li_n:.1f}%",
            "letter_shown_right": f"{sh_ok}/{sh_n}"}


def main() -> None:
    rows = json.load(open(sys.argv[1], encoding="utf-8"))["plates"]
    print(f"{'city prior':>10} {'letter prior':>12} | city | letter (best guess) | letters shown (conf>=0.6) and right")
    for ac, al in [(0, 0), (0, 0.5), (0, 1.0), (0, 1.5), (0.5, 1.0), (0.5, 1.5), (1.0, 1.0)]:
        s = scan(rows, ac, al)
        print(f"{ac:>10} {al:>12} | {s['city']:>14} | {s['letter']:>16} | {s['letter_shown_right']}")


if __name__ == "__main__":
    main()
