"""
LECO cleaning - counts for the methodology section.

Read-only: opens the Excel files the cleaning steps already created and prints
row counts. It does not run any step again and does not change or create any
file. Only numbers and category names are printed - no complaint text - so the
output is safe to paste into a chat.

Run (in the same folder as the Excel files):
    python leco_counts.py
"""

import re
from pathlib import Path

import pandas as pd

FOLDER = Path(r"D:\SLIIT\Y4S1\Research\data_cleaning\leco")
TEXT_COL = "Description"
LABEL_COL = "Category"

FILES = [
    ("Original export",                    "leco_complaints_original.xlsx"),
    ("Category + Description only",        "leco_complaints.xlsx"),
    ("Empty descriptions removed",         "leco_null_removed.xlsx"),
    ("Step 1 kept",                        "after_step1_leco_clean_kept.xlsx"),
    ("Step 1 removed",                     "after_step1_leco_clean_removed.xlsx"),
    ("Step 2 review list (unique comments)", "after_step2_review.xlsx"),
    ("Step 2 kept",                        "after_step2_leco_clean_kept.xlsx"),
    ("Step 2 removed",                     "after_step2_leco_clean_removed.xlsx"),
    ("Step 3 output",                      "after_step3_leco_clean.xlsx"),
    ("Step 4 output",                      "after_step4_leco_clean.xlsx"),
    ("Step 4 emptied (removed)",           "after_step4_emptied.xlsx"),
    ("Step 4 review (email rows)",         "after_step4_review.xlsx"),
    ("Step 5 kept",                        "after_step5_leco_clean_kept.xlsx"),
    ("Step 5 removed (exact copies)",      "after_step5_leco_clean_removed.xlsx"),
    ("Step 5 label-conflict lines",        "after_step5_label_conflicts.xlsx"),
    ("Step 6 kept",                        "after_step6_leco_clean_kept.xlsx"),
    ("Step 6 removed",                     "after_step6_leco_clean_removed.xlsx"),
    ("Step 6 decision lines",              "after_step6_conflict_decisions.xlsx"),
    ("Step 7 final (FINAL DATASET)",       "after_step7_leco_clean_final.xlsx"),
    ("Step 7 removed (empty rows)",        "after_step7_leco_clean_removed.xlsx"),
]
FINAL = "after_step7_leco_clean_final.xlsx"

COMMENT_RE = re.compile(r"Comment\s*:", re.IGNORECASE)
CODE_RE = re.compile(r"^[A-Za-z]{2,6}(?:[/-][A-Za-z]{1,6})*[/-]?\d{2,}(?:[/-]\d+)*$")


def load(name):
    path = FOLDER / name
    return pd.read_excel(path) if path.exists() else None


def is_template(t: str) -> bool:
    t = t.lower()
    return "mobile number" in t and "complaint category" in t


def step1_reason(text: str) -> str:
    body = text
    if is_template(text):
        m = COMMENT_RE.search(text)
        if not m:
            return "app template, no Comment part"
        body = text[m.end():].strip(" *\n\r\t")
        if not body:
            return "app template, empty Comment"
    chars = re.sub(r"\s", "", body)
    if chars and chars.count("?") / len(chars) >= 0.5:
        return "mostly question marks"
    tokens = [t for t in re.split(r"[\s,;]+", body.strip()) if t]
    if tokens and all(CODE_RE.match(t) for t in tokens):
        return "reference code only"
    return "other"


def text_of(df):
    return df[TEXT_COL].fillna("").astype(str)


def main() -> None:
    data = {}
    print("== Row counts per file")
    for label, name in FILES:
        df = load(name)
        data[name] = df
        shown = "not found" if df is None else f"{len(df):>6}"
        print(f"  {label:<40}: {shown}")

    df = data["after_step1_leco_clean_removed.xlsx"]
    if df is not None:
        print("\n== Step 1 removed, by reason")
        for reason, n in text_of(df).apply(step1_reason).value_counts().items():
            print(f"  {reason:<40}: {n:>6}")

    df = data["after_step2_review.xlsx"]
    if df is not None and "remove" in df.columns:
        marked = df["remove"].fillna("").astype(str).str.strip() != ""
        print("\n== Step 2 review")
        print(f"  {'unique short comments listed':<40}: {len(df):>6}")
        print(f"  {'marked x (name/place/meaningless)':<40}: {marked.sum():>6}")

    df = data["after_step2_leco_clean_kept.xlsx"]
    if df is not None:
        print("\n== Step 3 input")
        print(f"  {'app-template rows reduced to Comment':<40}: "
              f"{text_of(df).apply(is_template).sum():>6}")

    df = data["after_step4_leco_clean.xlsx"]
    if df is not None:
        t = text_of(df)
        print("\n== Step 4 output")
        print(f"  {'rows containing <NUMBER>':<40}: {t.str.contains('<NUMBER>').sum():>6}")
        print(f"  {'<NUMBER> tags in total':<40}: {t.str.count('<NUMBER>').sum():>6}")
        print(f"  {'rows containing <EMAIL>':<40}: {t.str.contains('<EMAIL>').sum():>6}")

    df = data["after_step6_conflict_decisions.xlsx"]
    if df is not None and "decision" in df.columns:
        print("\n== Step 6 decisions (lines = text/category pairs)")
        for d, n in df["decision"].value_counts().items():
            print(f"  {d:<40}: {n:>6}")

    df = data[FINAL]
    if df is not None:
        t = text_of(df)
        words = t.str.split().str.len()
        sinhala = t.str.contains(r"[\u0D80-\u0DFF]")
        latin = t.str.contains(r"[A-Za-z]")
        print("\n== Final dataset")
        print(f"  {'rows':<40}: {len(df):>6}")
        print(f"  {'words per complaint (median)':<40}: {words.median():>6.0f}")
        print(f"  {'words per complaint (mean)':<40}: {words.mean():>6.1f}")
        print(f"  {'Sinhala script only':<40}: {(sinhala & ~latin).sum():>6}")
        print(f"  {'Latin script only (English/Singlish)':<40}: {(latin & ~sinhala).sum():>6}")
        print(f"  {'both scripts (code-mixed)':<40}: {(sinhala & latin).sum():>6}")

        counts = df[LABEL_COL].fillna("(empty)").astype(str).str.strip().value_counts()
        print(f"\n== Final complaints per {LABEL_COL}   ({len(counts)} categories)")
        for cat, n in counts.items():
            print(f"  {cat[:40]:<40}: {n:>6}   {100 * n / len(df):5.1f}%")
        print(f"  {'TOTAL':<40}: {counts.sum():>6}")

if __name__ == "__main__":
    main()