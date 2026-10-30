"""Class priors for the top-row words (a decision step on top of the model's probabilities, no retraining).

The synthetic plates the model learns from draw every city and letter about equally often, but real plates do not (in the labelled
set one letter is 39 % of all letters, one city 59 % of the cities, and each city has its own few letters). When the picture is
blurry the model's probabilities are close and the prior decides: posterior ~ p_model(class) * prior(class) ** alpha.
Priors come from labelled plates only (smoothed, the letter prior conditioned on the city). They describe the plates the system
was labelled on; on a different camera or region they should be re-estimated (`ocr_merge/scripts/make_priors.py`) or switched off
(alpha = 0, or MERGE_PRIOR=0).
"""

import json
from pathlib import Path

import numpy as np

from ocr.synth_row import CITY_WORDS, LETTERS

K_SMOOTH = 5.0  # a city with few labelled plates falls back towards the overall letter frequencies
PATH = Path(__file__).with_name("priors.json")


class Priors:
    def __init__(self, city_idx: list[int], letter_idx: list[int]) -> None:
        """city_idx / letter_idx: labels of the labelled plates as indices (-1 = unknown)."""
        nc, nl = len(CITY_WORDS), len(LETTERS)
        self.city_n = np.zeros(nc)
        self.letter_n = np.zeros(nl)
        self.pair = np.zeros((nc, nl))
        for c, l in zip(city_idx, letter_idx):
            if c >= 0:
                self.city_n[c] += 1
            if l >= 0:
                self.letter_n[l] += 1
                if c >= 0:
                    self.pair[c, l] += 1
        self._norm()

    def _norm(self) -> None:
        self.city = (self.city_n + 1) / (self.city_n + 1).sum()
        self.letter = (self.letter_n + 1) / (self.letter_n + 1).sum()

    def letter_given(self, c: int) -> np.ndarray:
        n = self.pair[c]
        return (n + K_SMOOTH * self.letter) / (n.sum() + K_SMOOTH)

    def decide(self, city_probs, letter_probs, a_city: float = 0.5, a_letter: float = 1.0):
        """(city index, its posterior, letter index, its posterior) after applying the priors."""
        city_probs, letter_probs = np.asarray(city_probs, float), np.asarray(letter_probs, float)
        lc = np.log(np.maximum(city_probs, 1e-9)) + a_city * np.log(self.city)
        pc = np.exp(lc - lc.max())
        pc /= pc.sum()
        ci = int(pc.argmax())
        ll = np.log(np.maximum(letter_probs, 1e-9)) + a_letter * np.log(self.letter_given(ci))
        pl = np.exp(ll - ll.max())
        pl /= pl.sum()
        li = int(pl.argmax())
        return ci, float(pc[ci]), li, float(pl[li])

    # --- storage: counts only (no plate numbers, no photos)
    def to_json(self, path: Path = PATH) -> None:
        d = {"cities": {CITY_WORDS[i]: int(n) for i, n in enumerate(self.city_n) if n},
             "letters": {LETTERS[i]: int(n) for i, n in enumerate(self.letter_n) if n},
             "pairs": {f"{CITY_WORDS[c]}|{LETTERS[l]}": int(self.pair[c, l]) for c, l in zip(*np.nonzero(self.pair))}}
        Path(path).write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path = PATH) -> "Priors":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        p = cls([], [])
        for w, n in d["cities"].items():
            p.city_n[CITY_WORDS.index(w)] = n
        for l, n in d["letters"].items():
            p.letter_n[LETTERS.index(l)] = n
        for k, n in d["pairs"].items():
            w, l = k.split("|")
            p.pair[CITY_WORDS.index(w), LETTERS.index(l)] = n
        p._norm()
        return p
