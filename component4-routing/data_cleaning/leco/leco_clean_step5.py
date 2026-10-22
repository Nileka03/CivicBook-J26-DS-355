"""
LECO complaint cleaning - step 5: exact copies and label conflicts

Reads   after_step4_leco_clean.xlsx              (columns: Category, Description)
Writes  after_step5_leco_clean_kept.xlsx         one copy of every complaint
        after_step5_leco_clean_removed.xlsx      the extra copies that were removed
        after_step5_label_conflicts.xlsx         same text under different categories

Exact copy     = same Description AND same Category. The first one is kept.
Label conflict = same Description under two or more different Categories.
                 These rows are NOT removed - they are only listed for you to check.

Text is compared ignoring capital letters, extra spaces and hidden characters,
so "No power" and "no  power" are the same. "wrong calculation" and
"wrongly calculation" are different and both stay.

Run:
    python leco_clean_step5.py
"""

import unicodedata
from pathlib import Path

import pandas as pd

FOLDER = Path(r"D:\SLIIT\Y4S1\Research\data_cleaning\leco")
INPUT_FILE = FOLDER / "after_step4_leco_clean.xlsx"
KEPT_FILE = FOLDER / "after_step5_leco_clean_kept.xlsx"
REMOVED_FILE = FOLDER / "after_step5_leco_clean_removed.xlsx"
CONFLICT_FILE = FOLDER / "after_step5_label_conflicts.xlsx"
TEXT_COL = "Description"
LABEL_COL = "Category"


def norm(value) -> str:
    """Lower-case, hidden characters removed, spaces collapsed."""
    text = "" if pd.isna(value) else str(value)
    text = "".join(" " if unicodedata.category(ch)[0] in "CZ" else ch for ch in text)
    return " ".join(text.lower().split())


def main() -> None:
    df = pd.read_excel(INPUT_FILE)
    text_key = df[TEXT_COL].apply(norm)
    label_key = df[LABEL_COL].apply(norm)

    # ---- exact copies: same text + same category, keep the first
    is_copy = pd.DataFrame({"t": text_key, "l": label_key}).duplicated(keep="first")
    kept = df[~is_copy]
    removed = df[is_copy]
    kept.to_excel(KEPT_FILE, index=False)
    removed.to_excel(REMOVED_FILE, index=False)

    # ---- label conflicts: same text under more than one category
    counts = (pd.DataFrame({"text_key": text_key, LABEL_COL: df[LABEL_COL],
                            TEXT_COL: df[TEXT_COL], "label_key": label_key})
              .groupby(["text_key", "label_key"], sort=False)
              .agg(**{TEXT_COL: (TEXT_COL, "first"),
                      LABEL_COL: (LABEL_COL, "first"),
                      "rows_in_input": (TEXT_COL, "size")})
              .reset_index())
    n_labels = counts.groupby("text_key")["label_key"].transform("nunique")
    conflicts = counts[n_labels > 1].copy()
    conflicts["categories_for_this_text"] = n_labels[n_labels > 1]
    conflicts = conflicts.sort_values(["text_key", "rows_in_input"],
                                      ascending=[True, False])
    conflicts = conflicts[[TEXT_COL, LABEL_COL, "rows_in_input",
                           "categories_for_this_text"]]
    conflicts.to_excel(CONFLICT_FILE, index=False)

    print(f"Input rows                 : {len(df)}")
    print(f"Exact copies removed       : {len(removed)}")
    print(f"Rows kept                  : {len(kept)}")
    print(f"Texts with a label conflict: {conflicts[TEXT_COL].map(norm).nunique()}"
          f"   ({len(conflicts)} lines in {CONFLICT_FILE.name})")


if __name__ == "__main__":
    main()