"""
LECO complaint cleaning - step 6: resolve label conflicts by majority

A label conflict is the same Description under two or more Categories.
For each such text:
  - if one category clearly has the most rows, that category is KEPT and the
    rows under the other categories are removed (most likely mislabelled)
  - if there is no clear majority (a tie, or the top share is too low),
    ALL rows for that text are removed (the text is ambiguous)

Reads   after_step5_leco_clean_kept.xlsx        the rows to filter
        after_step4_leco_clean.xlsx             used only to count how often each
                                                text/category pair occurred
                                                (step 5 removed the copies)
Writes  after_step6_leco_clean_kept.xlsx
        after_step6_leco_clean_removed.xlsx
        after_step6_conflict_decisions.xlsx     what was decided for each conflict

Run:
    python leco_clean_step6.py
"""

import unicodedata
from pathlib import Path

import pandas as pd

FOLDER = Path(r"D:\SLIIT\Y4S1\Research\data_cleaning\leco")
INPUT_FILE = FOLDER / "after_step5_leco_clean_kept.xlsx"
COUNT_FILE = FOLDER / "after_step4_leco_clean.xlsx"
KEPT_FILE = FOLDER / "after_step6_leco_clean_kept.xlsx"
REMOVED_FILE = FOLDER / "after_step6_leco_clean_removed.xlsx"
DECISION_FILE = FOLDER / "after_step6_conflict_decisions.xlsx"
TEXT_COL = "Description"
LABEL_COL = "Category"

# The top category is kept only if it has at least this share of the rows for
# that text AND is not tied with another category.
#   0.6 -> 2 vs 1 keeps the majority, 1 vs 1 removes both, 3 vs 2 keeps (0.6)
MIN_MAJORITY_SHARE = 0.6


def norm(value) -> str:
    """Lower-case, hidden characters removed, spaces collapsed."""
    text = "" if pd.isna(value) else str(value)
    text = "".join(" " if unicodedata.category(ch)[0] in "CZ" else ch for ch in text)
    return " ".join(text.lower().split())


def main() -> None:
    df = pd.read_excel(INPUT_FILE)
    df_text = df[TEXT_COL].apply(norm)
    df_label = df[LABEL_COL].apply(norm)

    # how often each (text, category) pair occurred before copies were removed
    src = pd.read_excel(COUNT_FILE)
    counts = (pd.DataFrame({"text": src[TEXT_COL].apply(norm),
                            "label": src[LABEL_COL].apply(norm)})
              .value_counts().rename("rows").reset_index())

    # only texts that have more than one category in the file being filtered
    labels_per_text = (pd.DataFrame({"text": df_text, "label": df_label})
                       .drop_duplicates().groupby("text")["label"].nunique())
    conflict_texts = set(labels_per_text[labels_per_text > 1].index)
    counts = counts[counts["text"].isin(conflict_texts)].copy()

    counts["total"] = counts.groupby("text")["rows"].transform("sum")
    counts["share"] = (counts["rows"] / counts["total"]).round(3)
    counts["top"] = counts.groupby("text")["rows"].transform("max")
    counts["n_top"] = (counts["rows"] == counts["top"]).groupby(counts["text"]).transform("sum")

    def decide(r) -> str:
        clear = r["n_top"] == 1 and r["top"] / r["total"] >= MIN_MAJORITY_SHARE
        if not clear:
            return "removed - no clear majority"
        return "kept - majority" if r["rows"] == r["top"] else "removed - minority"

    counts["decision"] = counts.apply(decide, axis=1)
    decision = {(t, l): d for t, l, d in
                zip(counts["text"], counts["label"], counts["decision"])}

    remove = pd.Series(
        [decision.get((t, l), "kept").startswith("removed")
         for t, l in zip(df_text, df_label)], index=df.index)

    df[~remove].to_excel(KEPT_FILE, index=False)
    df[remove].to_excel(REMOVED_FILE, index=False)

    # decisions file, with the original text and category spelling
    first = (df.assign(text=df_text, label=df_label)
             .drop_duplicates(["text", "label"])[["text", "label", TEXT_COL, LABEL_COL]])
    report = (counts.merge(first, on=["text", "label"], how="left")
              .sort_values(["text", "rows"], ascending=[True, False])
              [[TEXT_COL, LABEL_COL, "rows", "share", "decision"]])
    report.to_excel(DECISION_FILE, index=False)

    by_text = counts.groupby("text")["decision"].first()
    print(f"Input rows                      : {len(df)}")
    print(f"Texts with a label conflict     : {len(conflict_texts)}")
    print(f"  majority category kept        : {(~by_text.str.contains('no clear')).sum()}")
    print(f"  removed completely (no clear) : {by_text.str.contains('no clear').sum()}")
    print(f"Rows removed                    : {remove.sum()}")
    print(f"Rows kept                       : {(~remove).sum()}")


if __name__ == "__main__":
    main()