"""Shared constants for Bengali plate digits."""

DIGITS = "০১২৩৪৫৬৭৮৯"
FONT_PATH = "C:/Windows/Fonts/Nirmala.ttc"  # Nirmala UI: Bengali digits/letters for synthetic training data


def normalize_digits(text: str) -> str:
    """Keep only digits, written as Bengali numerals (Latin 0-9 are mapped to ০-৯)."""
    out = []
    for ch in text:
        if ch in DIGITS:
            out.append(ch)
        elif "0" <= ch <= "9":
            out.append(DIGITS[int(ch)])
    return "".join(out)
