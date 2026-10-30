"""Digit-row views: (top fraction of the plate where the row starts, left/right trim). The reader votes over all of them.

More views give a steadier vote (cross-validated whole-number accuracy: 6 views 65.5 %, 20 views 70.2 %, 36 views 71.3 %, 55 views 72.5 %)
but every view is one more image through the network, so latency grows with the count.
"""

from ocr.bangla_reader import VIEWS


def _grid(tops, trims):
    return [(round(t, 3), tr) for tr in trims for t in tops]


VIEW_SETS = {
    "6": VIEWS,
    "12": VIEWS + [(0.31, 0.0), (0.49, 0.0), (0.34, 0.04), (0.46, 0.04), (0.40, 0.06), (0.37, 0.05)],
    "20": VIEWS + [(0.31, 0.0), (0.49, 0.0), (0.34, 0.04), (0.46, 0.04), (0.40, 0.06), (0.37, 0.05), (0.43, 0.05), (0.28, 0.0), (0.52, 0.0),
                   (0.31, 0.04), (0.49, 0.04), (0.34, 0.07), (0.46, 0.07), (0.40, 0.09)],
    "36": _grid([0.28 + 0.03 * i for i in range(9)], [0.0, 0.03, 0.06, 0.09]),
    "55": _grid([0.28 + 0.025 * i for i in range(11)], [0.0, 0.025, 0.05, 0.075, 0.1]),
}
