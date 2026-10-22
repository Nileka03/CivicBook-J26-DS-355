"""
LECO complaint cleaning - step 7: remove empty rows

Removes rows where
  - Description is empty (or only spaces, line breaks or hidden characters), or
  - Category is empty (a complaint without a label cannot be used for training)

Reads   after_step6_leco_clean_kept.xlsx
Writes  after_step7_leco_clean_final.xlsx
        after_step7_leco_clean_removed.xlsx

Run:
    python leco_clean_step7.py
"""

import unicodedata
from pathlib import Path

import pandas as pd

FOLDER = Path(r"D:\SLIIT\Y4S1\Research\data_cleaning\leco")
INPUT_FILE = FOLDER / "after_step6_leco_clean_kept.xlsx"
FINAL_FILE = FOLDER / "after_step7_leco_clean_final.xlsx"
REMOVED_FILE = FOLDER / "after_step7_leco_clean_removed.xlsx"
TEXT_COL = "Description"
LABEL_COL = "Category"


def is_blank(value) -> bool:
    """True for empty cells and cells with only spaces / hidden characters."""
    if pd.isna(value):
        return True
    text = str(value)
    return all(unicodedata.category(ch)[0] in "CZ" for ch in text)


def main() -> None:
    df = pd.read_excel(INPUT_FILE)
    no_text = df[TEXT_COL].apply(is_blank)
    no_label = df[LABEL_COL].apply(is_blank)
    remove = no_text | no_label

    df[~remove].to_excel(FINAL_FILE, index=False)
    df[remove].to_excel(REMOVED_FILE, index=False)

    print(f"Input rows                       : {len(df)}")
    print(f"  empty Category and Description : {(no_text & no_label).sum()}")
    print(f"  empty Description only         : {(no_text & ~no_label).sum()}")
    print(f"  empty Category only            : {(no_label & ~no_text).sum()}")
    print(f"Rows removed                     : {remove.sum()}")
    print(f"Rows in final file               : {(~remove).sum()}")


if __name__ == "__main__":
    main()