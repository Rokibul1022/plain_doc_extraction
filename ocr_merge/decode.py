"""CTC decoding constrained to exactly six digits.

Every plate number has six digits, so a view that the plain greedy decode reads as 5 or 7 symbols is still evidence about the six
positions. `decode6` finds, per view, the most probable frame-by-frame path whose collapsed output has exactly six digits
(Viterbi over states (digits emitted so far, last symbol)), so no view is thrown away and the vote never ends in "?".
"""

import numpy as np

BLANK = 10
DIGITS = "০১২৩৪৫৬৭৮৯"


def decode6(p: np.ndarray, n: int = 6) -> list[tuple[str, list[float]]]:
    """p: (V, T, 11) probabilities -> per view (digits, per-digit confidence), always n digits."""
    V, T, C = p.shape
    lp = np.log(np.maximum(p, 1e-9))
    NEG = -1e18
    score = np.full((V, n + 1, C), NEG)
    score[:, 0, BLANK] = lp[:, 0, BLANK]
    score[:, 1, :BLANK] = lp[:, 0, :BLANK]
    back_j = np.zeros((T, V, n + 1, C), np.int8)
    back_s = np.zeros((T, V, n + 1, C), np.int8)
    jj = np.broadcast_to(np.arange(n + 1), (V, n + 1))
    vi = np.arange(V)[:, None]
    for t in range(1, T):
        new = np.full_like(score, NEG)
        s_best = score.argmax(2)
        best = np.take_along_axis(score, s_best[..., None], 2)[..., 0]
        new[:, :, BLANK] = best + lp[:, t, BLANK][:, None]
        back_j[t, :, :, BLANK], back_s[t, :, :, BLANK] = jj, s_best
        for c in range(BLANK):
            stay = score[:, :, c]
            src = score[:, :n, :].copy()
            src[:, :, c] = NEG  # a repeat of the same symbol without a blank is not a new digit
            e_s = src.argmax(2)
            ex = np.take_along_axis(src, e_s[..., None], 2)[..., 0]  # (V, n): emit c as digit j+1 after j digits
            cand = np.full((V, n + 1), NEG)
            cand[:, 1:] = ex
            use_new = cand > stay
            val = np.where(use_new, cand, stay) + lp[:, t, c][:, None]
            new[:, :, c] = np.where(val > NEG / 2, val, NEG)
            back_j[t, :, :, c] = np.where(use_new, jj - 1, jj)
            back_s[t, :, :, c] = np.where(use_new, np.concatenate([np.zeros((V, 1), e_s.dtype), e_s], 1), c)
        score = new
    out = []
    for v in range(V):
        s = int(score[v, n].argmax())
        j = n
        path = [s]
        for t in range(T - 1, 0, -1):
            j, s = int(back_j[t, v, j, s]), int(back_s[t, v, j, s])
            path.append(s)
        path.reverse()
        chars, conf, prev = [], [], -1
        for t, k in enumerate(path):
            if k != prev and k != BLANK:
                chars.append(DIGITS[k])
                conf.append(float(p[v, t, k]))
            elif k == prev and k != BLANK:
                conf[-1] = max(conf[-1], float(p[v, t, k]))
            prev = k
        out.append(("".join(chars), conf))
    return out
