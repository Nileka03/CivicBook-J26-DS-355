"""
UC Maharagama + MC Kaduwela complaints - remove empty rows and duplicates

Reads   uc_development_null_removed.xlsx      (one column: the complaint)
Writes  uc_development_clean_kept.xlsx        complaints that stay
        uc_development_clean_removed.xlsx     empty rows + duplicate copies
        uc_development_near_duplicates.xlsx   almost-identical pairs, NOT removed,
                                              for you to check (typing differences)

Empty row     = blank, or only spaces / line breaks / punctuation / hidden characters.
Duplicate     = same complaint, compared Sinhala-aware (the first one is kept):
                - Unicode normalised (NFC)
                - zero-width joiner / non-joiner ignored (ශ්‍රී = ශ්රී)
                - spaces, line breaks and punctuation ignored
The complaint text in the output files is NOT changed.

Run (in the folder that has the Excel file):
    pip install pandas openpyxl
    python uc_clean.py
"""

import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

INPUT_FILE = Path("health_original.xlsx")
KEPT_FILE = Path("health_kept.xlsx")
REMOVED_FILE = Path("health_removed.xlsx")
NEAR_FILE = Path("health_duplicates.xlsx")

# Pairs at least this similar (0-1) are listed for review. Nothing is removed for this.
NEAR_DUPLICATE_THRESHOLD = 0.90

ZERO_WIDTH = {"‌", "‍", "​", "﻿", "⁠"}


def compare_key(value) -> str:
    """Text used only for comparing. Sinhala letters, vowel signs and virama kept;
    spaces, punctuation and hidden characters dropped."""
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if ch not in ZERO_WIDTH).lower()
    return "".join(ch for ch in text
                   if ch.isalnum() or unicodedata.category(ch).startswith("M"))


def main() -> None:
    df = pd.read_excel(INPUT_FILE)
    col = df.columns[0]                         # the complaint column
    key = df[col].apply(compare_key)

    empty = key == ""
    duplicate = key.duplicated(keep="first") & ~empty
    remove = empty | duplicate

    kept = df[~remove]
    df[remove].to_excel(REMOVED_FILE, index=False)
    kept.to_excel(KEPT_FILE, index=False)

    # near-duplicates among the kept complaints (review only)
    rows = list(zip(kept.index + 2, key[~remove], kept[col]))
    pairs = []
    for i in range(len(rows)):
        r1, k1, t1 = rows[i]
        for j in range(i + 1, len(rows)):
            r2, k2, t2 = rows[j]
            if min(len(k1), len(k2)) / max(len(k1), len(k2)) < NEAR_DUPLICATE_THRESHOLD:
                continue
            sim = SequenceMatcher(None, k1, k2).ratio()
            if sim >= NEAR_DUPLICATE_THRESHOLD:
                pairs.append({"row_a": r1, "complaint_a": t1,
                              "row_b": r2, "complaint_b": t2,
                              "similarity": round(sim, 3)})
    near = pd.DataFrame(pairs, columns=["row_a", "complaint_a",
                                        "row_b", "complaint_b", "similarity"])
    near.sort_values("similarity", ascending=False).to_excel(NEAR_FILE, index=False)

    print(f"Complaint column        : {col}")
    print(f"Input rows              : {len(df)}")
    print(f"  empty rows removed    : {empty.sum()}")
    print(f"  duplicates removed    : {duplicate.sum()}")
    print(f"Rows kept               : {len(kept)}")
    print(f"Near-duplicate pairs    : {len(near)}   (in {NEAR_FILE.name}, not removed)")


if __name__ == "__main__":
    main()