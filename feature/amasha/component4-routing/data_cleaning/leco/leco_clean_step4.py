"""
LECO complaint cleaning - step 4

Cleans the text inside each Description:
  1. removes pasted-image markers            ![](clipboard-....png)
  2. emails: removes From/Date/Subject/To lines, older forwarded messages,
     greetings, sign-offs and signatures - keeps the newest message body
  3. removes filler such as "please find the attached"
  4. replaces every identifying number with <NUMBER>: phone numbers, account
     numbers, reference codes, masked numbers (07******57)
     NOT changed: dates, prices / amounts, and short numbers (100 units, 45/2)
  5. replaces email addresses with <EMAIL>

Reads   after_step3_leco_clean.xlsx          (columns: Category, Description)
Writes  after_step4_leco_clean.xlsx          cleaned rows
        after_step4_emptied.xlsx             rows with nothing left (original text)
        after_step4_review.xlsx              emails / signature rows, before + after

    python leco_clean_step4.py           # clean + build the review file
    python leco_clean_step4.py apply     # optional: apply your edits

The review file has one line per email-like row:
    row      - Excel row number in after_step3_leco_clean.xlsx
    before   - the text before cleaning
    cleaned  - the text after cleaning   <- edit this cell if something is wrong
                                            (e.g. delete a name that was left)
Clear the 'cleaned' cell completely to drop that row.
Then run with 'apply'. If you change nothing, you do not need to run 'apply'.
"""

import re
import sys
from pathlib import Path

import pandas as pd

FOLDER = Path(r"D:\SLIIT\Y4S1\Research\data_cleaning\leco")
INPUT_FILE = FOLDER / "after_step3_leco_clean.xlsx"
OUTPUT_FILE = FOLDER / "after_step4_leco_clean.xlsx"
EMPTIED_FILE = FOLDER / "after_step4_emptied.xlsx"
REVIEW_FILE = FOLDER / "after_step4_review.xlsx"
TEXT_COL = "Description"

NUMBER = "<NUMBER>"
EMAIL = "<EMAIL>"

I = re.IGNORECASE
M = re.MULTILINE

# ---------------------------------------------------------------- patterns
IMAGE_RE = re.compile(r"!\[[^\]\n]*\]\([^)\n]*\)")

HEADER_RE = re.compile(
    r"^\s*(from|sent|date|to|cc|bcc|subject|reply-to)\s*:.*$", I)
DIVIDER_RE = re.compile(
    r"^\s*(-{2,}\s*(forwarded|original)\s+message\s*-{2,}"
    r"|begin forwarded message\s*:?"
    r"|on\s.{5,150}\swrote\s*:)\s*$", I)

GREETING_LINE_RE = re.compile(
    r"^\s*(dear\b.{0,40}|hi\b.{0,20}|hello\b.{0,20}|sir\s*/?\s*madam"
    r"|good\s+(morning|afternoon|evening|day)\b.{0,20})[\s,.:!]*$", I)
GREETING_PREFIX_RE = re.compile(
    r"^\s*dear\s+(sir\s*(/|or)\s*madam|sirs?|madam|team|leco)\s*[,.:!]\s*", I)
SIGNOFF_RE = re.compile(
    r"^\s*(thanks(\s+(and|&)\s+regards)?|thank\s+you(\s+very\s+much)?|thanking\s+you"
    r"|(best|kind|warm)?\s*regards|yours\s+(faithfully|sincerely|truly)"
    r"|sincerely|br|sent from my .{1,40})[\s,.!]*$", I)

FILLER_RE = re.compile(
    r"(kindly\s+|please\s+|pls\s+)?(find|see|refer|check)(\s+the)?(\s+(below|above))?"
    r"\s+(attached|attachments?)(\s+(herewith|here with))?"
    r"(\s+(files?|images?|photos?|pictures?|documents?|screenshots?|bills?|letters?))?"
    r"(\s+(herewith|here with))?\s*[.!]*", I)

EMAIL_RE = re.compile(r"<?[\w.+-]+@[\w-]+(\.[\w-]+)+>?")

# A plain number is tagged only if it has at least this many digits.
# Phone and account numbers have 9-10; meter readings and unit counts have fewer.
MIN_DIGITS = 7
# A letters+digits reference code is tagged if it has at least this many digits
# (so "T12" or "CEB2024" stay, "OTDL26081702" is tagged).
MIN_CODE_DIGITS = 5

# Tried in this order at each position. "keep_" groups are left unchanged.
NUMBER_RE = re.compile(r"""
    (?P<keep_price>                                  # Rs. 12,500.00   LKR 2500/=
        (?<![A-Za-z])(?:rs|lkr|\u0dbb\u0dd4)\.?\s*\d[\d,]*(?:\.\d+)?(?:\s*/[=-])?
      | (?<![\w.])\d[\d,]*(?:\.\d+)?\s*(?:/=|/-|rupees\b|rs\b|lkr\b)
      | (?<![\w.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d])   # 1,250,000.00
    )
  | (?P<keep_date>                                   # 2026/07/20  20-07-2026
        (?<![\w/.-])\d{1,4}(?P<sep>[./-])\d{1,2}(?P=sep)\d{1,4}(?![\w/-])
      | (?<![\w/.-])[0-3]?\d[ ][01]?\d[ ](?:19|20)\d{2}(?![\w/-])      # 20 07 2026
      | (?<![\w/.-])(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])(?![\w/-])
    )
  | (?P<keep_decimal>(?<![\w.])\d+\.\d{1,2}(?![\d]))      # 230.5   1500000.00
  | (?P<code>                                        # OTDL26081702  OT/DL/26/07/20/08
        (?<![A-Za-z0-9])[A-Za-z]{2,6}(?:[/-][A-Za-z]{1,6})*[/-]?\d{2,}(?:[/-]\d+)*
        (?![A-Za-z0-9*])
    )
  | (?P<masked>                                      # 07******57   ******1234
        (?<![\d*])\+?(?=[\d*]*\d)(?=[\d*]*\*{2})[\d*]{4,}(?![\d*])
    )
  | (?P<plain>                                       # 0771234567  0206 964 406
        (?<![\w/.])\+?\d{2,}(?:[ -]\d{2,})*(?![\w/])
    )
""", re.VERBOSE | re.IGNORECASE)


def _tag(m: re.Match) -> str:
    kind, text = m.lastgroup, m.group(0)
    digits = sum(ch.isdigit() for ch in text)
    if kind == "masked":
        return NUMBER
    if kind == "code" and digits >= MIN_CODE_DIGITS:
        return NUMBER
    if kind == "plain" and digits >= MIN_DIGITS:
        return NUMBER
    return text


def tag_numbers(text: str) -> str:
    return NUMBER_RE.sub(_tag, text)


# Reference codes on a line of their own:  OTDL26081702   OT/DL/26/07/20/08
CODE_LINE_RE = re.compile(
    r"^\s*[A-Za-z]{2,6}(?:[/-][A-Za-z]{1,6})*[/-]?\d{2,}(?:[/-]\d+)*\s*$")

NUMBER_ONLY_LINE_RE = re.compile(r"^[\s,;:/()-]*(<NUMBER>[\s,;:/()-]*)+$")
NAME_LINE_RE = re.compile(r"^\s*((?:[A-Z]\.\s*)+|[A-Z][A-Za-z'.-]*\s*){1,4}$")


# ---------------------------------------------------------------- helpers
def looks_like_email(text: str) -> bool:
    lines = text.splitlines()
    headers = sum(1 for ln in lines if HEADER_RE.match(ln))
    return headers >= 2 or any(DIVIDER_RE.match(ln) for ln in lines)


def clean_lines(lines, email: bool):
    """Clean one message: returns the list of lines that are kept."""
    out = []
    for ln in lines:
        if email and (HEADER_RE.match(ln) or ln.lstrip().startswith(">")):
            continue
        if CODE_LINE_RE.match(ln) or GREETING_LINE_RE.match(ln):
            continue
        out.append(GREETING_PREFIX_RE.sub("", ln))

    # sign-off: drop it together with the short signature lines that follow
    for i, ln in enumerate(out):
        if SIGNOFF_RE.match(ln):
            tail = [t for t in out[i + 1:] if t.strip()]
            if email or (len(tail) <= 4 and all(len(t) <= 60 for t in tail)):
                out = out[:i]
            break
    return out


def drop_trailing_signature(lines, email: bool):
    """Remove trailing lines that are only a number, or (emails only) a name."""
    while lines and not lines[-1].strip():
        lines.pop()
    removed_name = 0
    while len(lines) > 1:
        last = lines[-1]
        if not last.strip() or NUMBER_ONLY_LINE_RE.match(last):
            lines.pop()
        elif email and removed_name < 2 and NAME_LINE_RE.match(last):
            lines.pop()
            removed_name += 1
        else:
            break
    return lines


def clean_text(text: str):
    """Returns (cleaned text, needs_review)."""
    original = text
    text = IMAGE_RE.sub(" ", text)
    email = looks_like_email(text)

    # split into messages (newest first) and keep the first one with a body
    messages, current = [], []
    for ln in text.splitlines():
        if email and DIVIDER_RE.match(ln):
            messages.append(current)
            current = []
        else:
            current.append(ln)
    messages.append(current)

    kept, had_signoff = [], False
    for msg in messages:
        had_signoff = had_signoff or any(SIGNOFF_RE.match(ln) for ln in msg)
        kept = clean_lines(msg, email)
        if any(ln.strip() for ln in kept):
            break

    text = "\n".join(kept)
    text = FILLER_RE.sub(" ", text)
    text = EMAIL_RE.sub(EMAIL, text)
    text = tag_numbers(text)

    lines = drop_trailing_signature(text.splitlines(), email)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in lines]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

    needs_review = (email or had_signoff) and text != original.strip()
    return text, needs_review


# ---------------------------------------------------------------- main
def main() -> None:
    apply = "apply" in sys.argv
    if not apply and REVIEW_FILE.exists() and "rebuild" not in sys.argv:
        sys.exit(f"{REVIEW_FILE.name} already exists - not overwriting your edits.\n"
                 "Run with 'apply' to use it, or 'rebuild' to start again.")

    df = pd.read_excel(INPUT_FILE)
    before = df[TEXT_COL].fillna("").astype(str)
    result = before.apply(clean_text)
    cleaned = result.str[0]
    review_mask = result.str[1]

    if apply:
        review = pd.read_excel(REVIEW_FILE, keep_default_na=False, dtype=str)
        edits = 0
        for row, new in zip(review["row"], review["cleaned"]):
            idx = int(float(row)) - 2
            new = new.replace("_x000D_", "").replace("\r\n", "\n").strip()
            if idx in cleaned.index and cleaned[idx] != new:
                cleaned[idx] = new
                edits += 1
        print(f"Edits applied from review  : {edits}")
    else:
        review = pd.DataFrame({
            "row": before.index[review_mask] + 2,
            "before": before[review_mask],
            "cleaned": cleaned[review_mask],
        })
        review.to_excel(REVIEW_FILE, index=False)

    empty = cleaned.str.strip() == ""
    out = df[~empty].copy()
    out[TEXT_COL] = cleaned[~empty]
    out.to_excel(OUTPUT_FILE, index=False)
    df[empty].to_excel(EMPTIED_FILE, index=False)

    print(f"Input rows                 : {len(df)}")
    print(f"Rows changed               : {(cleaned != before).sum()}")
    print(f"  with {NUMBER:<9}           : {cleaned.str.contains(NUMBER).sum()}")
    print(f"  with {EMAIL:<9}           : {cleaned.str.contains(EMAIL).sum()}")
    print(f"Rows left empty (removed)  : {empty.sum()}")
    print(f"Rows written               : {len(out)}")
    if not apply:
        print(f"Rows in review file        : {review_mask.sum()}")
        print(f"\nCheck {REVIEW_FILE.name}. If you edit the 'cleaned' column, run:")
        print("    python leco_clean_step4.py apply")


if __name__ == "__main__":
    main()