"""
LECO complaint cleaning - step 2: remove mobile-app rows whose comment is not a
complaint (a person's name, a place name, a meaningless word).

Instead of checking every row, you check each DIFFERENT short comment once.

    python leco_clean_step2.py          # 1. builds the review list
    (mark the review file in Excel)
    python leco_clean_step2.py apply    # 2. removes the rows you marked

Step 1 of this script reads  after_step1_leco_clean_kept.xlsx  and writes
after_step2_review.xlsx : one line per different short comment, with
    comment      - the text after "Comment:"
    words        - number of words
    rows         - how many rows have exactly this comment
    has_keyword  - "yes" if it contains a complaint word (power, bill, meter ...)
    remove       - EMPTY. Type x here if the comment is not a complaint.

Lines without a keyword come first: names and places are almost all there.
The "yes" lines at the bottom usually need only a quick look.

Step 2 ("apply") writes
    after_step2_leco_clean_kept.xlsx
    after_step2_leco_clean_removed.xlsx
"""

import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

FOLDER = Path(r"D:\SLIIT\Y4S1\Research\data_cleaning\leco")
INPUT_FILE = FOLDER / "after_step1_leco_clean_kept.xlsx"
REVIEW_FILE = FOLDER / "after_step2_review.xlsx"
KEPT_FILE = FOLDER / "after_step2_leco_clean_kept.xlsx"
REMOVED_FILE = FOLDER / "after_step2_leco_clean_removed.xlsx"
TEXT_COL = "Description"

# Comments with this many words or fewer are put in the review list.
# Longer comments are kept without review.
MAX_WORDS = 4

# Only used to sort the review list - nothing is removed or kept because of these.
KEYWORDS = [
    "power", "light", "current", "electric", "supply", "bill", "meter", "reading",
    "payment", "paid", "breakdown", "break down", "failure", "fault", "outage",
    "line", "wire", "cable", "pole", "post", "transformer", "fuse", "trip",
    "voltage", "lamp", "bulb", "spark", "fire", "tree", "branch", "connection",
    "disconnect", "reconnect", "account", "unit", "solar", "cut",
    "විදුලි", "බිල", "මීටර", "ලයිට්", "කණු", "වයර්", "කැඩ", "විසන්ධි",
]

COMMENT_RE = re.compile(r"Comment\s*:", re.IGNORECASE)


def is_template(text: str) -> bool:
    t = text.lower()
    return "mobile number" in t and "complaint category" in t


def clean(text: str) -> str:
    """Lower-case, drop invisible characters, collapse spaces."""
    text = re.sub(r"_x[0-9A-Fa-f]{4}_", " ", text)          # Excel escape codes
    text = "".join(" " if unicodedata.category(ch)[0] in "CZ" else ch
                   for ch in text)                           # invisible / control
    return " ".join(text.lower().split())


def match_key(text: str) -> str:
    """Letters, digits, * and / only - used to match marked comments to rows,
    so hidden characters or punctuation changed by Excel cannot break it."""
    return " ".join(re.sub(r"[^\w*/]+", " ", clean(text)).replace("_", " ").split())


def comment_key(text: str) -> str:
    """The comment of a mobile-app row, cleaned.
    Returns '' for rows that are not mobile-app rows."""
    if not is_template(text):
        return ""
    m = COMMENT_RE.search(text)
    if not m:
        return ""
    return clean(text[m.end():]).strip(" *")


def load():
    df = pd.read_excel(INPUT_FILE)
    key = df[TEXT_COL].fillna("").astype(str).apply(comment_key)
    return df, key


def build_review() -> None:
    if REVIEW_FILE.exists() and "rebuild" not in sys.argv:
        sys.exit(f"{REVIEW_FILE.name} already exists - not overwriting your marks.\n"
                 "Run with 'apply' to use it, or 'rebuild' to start a new list.")
    df, key = load()
    words = key.str.split().str.len()
    short = key[(key != "") & (words <= MAX_WORDS)]

    review = short.value_counts().rename_axis("comment").reset_index(name="rows")
    review["words"] = review["comment"].str.split().str.len()
    review["has_keyword"] = review["comment"].apply(
        lambda c: "yes" if any(k in c for k in KEYWORDS) else "")
    review["remove"] = ""
    review = review.sort_values(["has_keyword", "words", "comment"])
    review = review[["comment", "words", "rows", "has_keyword", "remove"]]
    review.to_excel(REVIEW_FILE, index=False)

    print(f"Rows in input              : {len(df)}")
    print(f"Mobile-app rows            : {(key != '').sum()}")
    print(f"  with a short comment     : {len(short)}")
    print(f"Different short comments   : {len(review)}   <- lines to check")
    print(f"  without a keyword        : {(review['has_keyword'] == '').sum()}")
    print(f"\nOpen {REVIEW_FILE.name}, type x in 'remove', save, then run:")
    print("    python leco_clean_step2.py apply")


def apply_review() -> None:
    df, key = load()
    review = pd.read_excel(REVIEW_FILE, keep_default_na=False, dtype=str)
    marked = review["remove"].str.strip() != ""
    marked_keys = {match_key(c) for c in review.loc[marked, "comment"]} - {""}

    row_keys = key.apply(match_key)
    remove = (key != "") & row_keys.isin(marked_keys)
    df[~remove].to_excel(KEPT_FILE, index=False)
    df[remove].to_excel(REMOVED_FILE, index=False)

    print(f"Comments marked : {len(marked_keys)}")
    print(f"Input rows      : {len(df)}")
    print(f"Kept            : {(~remove).sum()}")
    print(f"Removed         : {remove.sum()}")

    unmatched = sorted(marked_keys - set(row_keys[remove]))
    if unmatched:
        print(f"\nWARNING: {len(unmatched)} marked comments matched no row. First few:")
        for c in unmatched[:10]:
            print("   ", ascii(c))


if __name__ == "__main__":
    if "apply" in sys.argv:
        apply_review()
    else:
        build_review()