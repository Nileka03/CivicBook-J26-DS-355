"""
CivicBook Component 4 - build the multi-topic complaint dataset  (VERSION 2)

Changes from version 1 (build_dataset.py) - all to fix weaknesses found in testing:
  A. Language confound: reviewed NLLB translations (translations.xlsx) are added -
     English LECO complaints in Sinhala and Sinhala MC complaints in English.
     A translation always stays in the SAME split as its original, and only
     train / validation complaints are ever translated (the script checks this).
  B. Joins without connector words now also use a comma only ("... එනවා, light ...")
     or nothing at all, not only a full stop. Sinhala connector words can follow a
     comma too.
  C. Hand-written test set (handwritten_test.xlsx, optional): complaints written by
     the team, never used for training - the honest test of the language confound.
  D. Each split has its own random generator, so adding training data later does
     not change the validation or test records.

Reads (same folder):
    leco_final.xlsx       columns: Category, Description   (4 LECO departments)
    mc_engineering.xlsx   one complaint column             (MC Engineering)
    mc_health.xlsx        one complaint column             (MC Health)
    translations.xlsx     optional - output of translate_nllb.ipynb, after your review
    handwritten_test.xlsx optional - from handwritten_test_template.xlsx

Writes (folder "dataset_v2"):
    train.jsonl, validation.jsonl, test.jsonl   one record per line - for training
    handwritten_test.jsonl                      only if handwritten_test.xlsx exists
    dataset_view.xlsx                           the same records as readable sheets
    source_split.xlsx                           which complaint went to which split
    dataset_stats.txt                           counts for the methodology section

How it works
  1. Every real complaint is split 80/10/10 into train / validation / test,
     separately per department (stratified). This is done FIRST, so a real
     complaint used in a test record never appears in a train record (no leakage).
  2. Inside each split, records are built only from that split's complaints:
        single     one real complaint                      (1 sub-complaint)
        two_same   2 departments of the same institution   (2 sub-complaints)
        two_cross  1 LECO department + 1 MC department     (2 sub-complaints)
        three      3 different departments                 (3 sub-complaints)
     Sub-complaints are joined with Sinhala / English / Singlish connectors,
     or with only a full stop (so the model cannot rely on connector words).
     Some connectors are TEST-ONLY: they never appear in train or validation,
     so test results on them show whether the model generalises to connector
     words it has never seen (records are marked "unseen_connector": true).
  3. Tokens and BIO tags are produced: first token of each sub-complaint = B,
     the rest = I, connector tokens = O.
  4. Every record is checked (tokens, tags, spans, labels, leakage). The script
     stops with an error if any check fails.

Run:
    pip install pandas openpyxl
    python build_dataset.py
"""

import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

# =============================================================== settings
SEED = 42

LECO_FILE = Path("leco_final.xlsx")
LECO_TEXT_COL = "Description"
LECO_LABEL_COL = "Category"
MC_FILES = {                       # department name -> file (first column = complaint)
    "Engineering": Path("mc_engineering.xlsx"),
    "Health": Path("mc_health.xlsx"),
}
OUT_DIR = Path("dataset_v2")
TRANSLATIONS_FILE = Path("translations.xlsx")        # optional
HANDWRITTEN_FILE = Path("handwritten_test.xlsx")     # optional

SPLIT_RATIO = {"train": 0.8, "validation": 0.1, "test": 0.1}

# Single-topic records: at most this many real complaints per department
# (before the split) - stops LECO's 3,000+ bill complaints swamping MC.
# None = use every complaint.
SINGLE_CAP_PER_DEPARTMENT = 500

# Multi-topic records to build per split.
MULTI_COUNTS = {
    "train":      {"two_same": 2000, "two_cross": 2000, "three": 1500},
    "validation": {"two_same": 250,  "two_cross": 250,  "three": 190},
    "test":       {"two_same": 250,  "two_cross": 250,  "three": 190},
}

# Length limits (words). XLM-R reads at most 512 sub-word pieces, and a Sinhala
# word is often 2-3 pieces, so whole records are kept under ~150 words.
MAX_SEGMENT_WORDS = 60      # a complaint longer than this is used only as a single
MAX_TOTAL_WORDS = 150       # no record (single or multi) longer than this

# Share of joins that use no connector word, only punctuation (or nothing).
NO_WORD_CONNECTOR_SHARE = 0.25
# How those joins look: full stop / comma only / nothing at all
NO_WORD_STYLES = {"stop": 0.4, "comma": 0.4, "none": 0.2}
# Sinhala / Singlish connector words: share that follow a comma instead of a full stop
SI_COMMA_LEAD_SHARE = 0.3
# ======================================================================

INSTITUTION = {"LECO": "LECO", "MC": "MC"}
SINHALA_RE = re.compile(r"[඀-෿]")
SENTENCE_END = (".", "!", "?", "।", "෴")

# ---------------------------------------------------------------- connectors
# Each connector is the text placed BETWEEN two sub-complaints. All its tokens are O.
CONNECTORS = {
    "si": ["ඒ වගේම", "ඒ වාගේම", "එමෙන්ම", "තවද", "ඊට අමතරව", "ඒ හැර",
           "ඒත් එක්කම", "තවත් දෙයක් තමයි", "අනෙක් කාරණය නම්",
           "තව ප්‍රශ්නයක් තියෙනවා.", "තවත් ගැටලුවක් තියෙනවා."],
    "en": ["Also", "And also", "In addition", "Furthermore", "Moreover",
           "Apart from that", "Besides that", "As well as that",
           "Another issue:", "There is another problem too."],
    "singlish": ["ewagema", "e wagema", "thawa", "ita amatharawa", "anith eka",
                 "thawa prashnayak thiyenawa.", "e witharak nemei"],
}
EN_LOWER_AFTER_COMMA = ["and", "and also", "also", "as well as that"]

# TEST-ONLY connectors: used only when building test records, never in train or
# validation. Keep them different from every connector above (the script checks).
TEST_ONLY_CONNECTORS = {
    "si": ["ඒ අතරම", "මීට අමතරව", "එසේම", "ඒ අතරේ", "තවත් කරුණක් තමයි",
           "තවත් පැමිණිල්ලක් තියෙනවා."],
    "en": ["On top of that", "Additionally", "What is more", "Along with that",
           "One more problem:", "I also have another complaint."],
    "singlish": ["mee athare", "ekka thawa", "thawa ekak", "thawa deyak thiyenawa."],
}
# In test records, this share of connector joins uses a test-only connector.
TEST_ONLY_SHARE = 0.5


# ---------------------------------------------------------------- tokeniser
# Splits on whitespace, then separates punctuation at the start/end of a word.
# Sinhala letters, vowel signs and zero-width joiners stay inside the word.
# <NUMBER> / <EMAIL> stay whole. The same function is used everywhere.
PEEL = set(".,!?;:\"'()[]{}…“”‘’|෴")


def tokenize(text: str):
    tokens = []
    for chunk in text.split():
        lead, trail = [], []
        while chunk and chunk[0] in PEEL:
            lead.append(chunk[0])
            chunk = chunk[1:]
        while chunk and chunk[-1] in PEEL:
            trail.insert(0, chunk[-1])
            chunk = chunk[:-1]
        tokens.extend(lead)
        if chunk:
            tokens.append(chunk)
        tokens.extend(trail)
    return tokens


def clean(text) -> str:
    """NFC-normalise and collapse whitespace (line breaks become spaces)."""
    text = unicodedata.normalize("NFC", "" if pd.isna(text) else str(text))
    return " ".join(text.split())


def compare_key(text: str) -> str:
    """Sinhala-aware key used only to detect the same complaint twice."""
    text = "".join(ch for ch in text if ch not in "​‌‍﻿").lower()
    return "".join(ch for ch in text
                   if ch.isalnum() or unicodedata.category(ch).startswith("M"))


def language(text: str) -> str:
    return "si" if SINHALA_RE.search(text) else "en"


# ---------------------------------------------------------------- load sources
def load_sources():
    rows = []
    leco = pd.read_excel(LECO_FILE)
    for c in (LECO_TEXT_COL, LECO_LABEL_COL):
        if c not in leco.columns:
            sys.exit(f"{LECO_FILE}: column '{c}' not found. Columns: {list(leco.columns)}")
    for i, (t, d) in enumerate(zip(leco[LECO_TEXT_COL], leco[LECO_LABEL_COL])):
        rows.append({"source_id": f"LECO-{i + 2}", "institution": "LECO",
                     "department": clean(d), "text": clean(t)})
    for dept, path in MC_FILES.items():
        mc = pd.read_excel(path)
        col = mc.columns[0]
        for i, t in enumerate(mc[col]):
            rows.append({"source_id": f"MC-{dept[:3].upper()}-{i + 2}", "institution": "MC",
                         "department": dept, "text": clean(t)})

    src = pd.DataFrame(rows)
    src["translated"] = False
    src["original_source_id"] = ""
    src["n_words"] = src["text"].apply(lambda t: len(tokenize(t)))
    src["key"] = src["text"].apply(compare_key)
    src["language"] = src["text"].apply(language)

    dropped = {}
    empty = (src["key"] == "") | (src["department"] == "")
    dropped["empty text or department"] = int(empty.sum())
    src = src[~empty]
    dup = src.duplicated("key", keep="first")       # same complaint in two files/departments
    dropped["duplicate across files"] = int(dup.sum())
    src = src[~dup]
    return src.reset_index(drop=True), dropped


# ---------------------------------------------------------------- split
def stable_order(key: str) -> str:
    return hashlib.md5(f"{SEED}|{key}".encode("utf-8")).hexdigest()


def split_sources(src):
    """v2: inside each department, complaints are ordered by a hash of their text and
    cut 10% test / 10% validation / 80% train. A complaint keeps its split even when
    other complaints are added, removed or relabelled (e.g. after LECO's answer), so
    translations made earlier stay valid and the test set stays stable."""
    src = src.copy()
    src["split"] = ""
    for _, idx in src.groupby(["institution", "department"]).groups.items():
        idx = sorted(idx, key=lambda i: stable_order(src.at[i, "key"]))
        n = len(idx)
        n_test = max(1, round(n * SPLIT_RATIO["test"]))
        n_val = max(1, round(n * SPLIT_RATIO["validation"]))
        src.loc[idx[:n_test], "split"] = "test"
        src.loc[idx[n_test:n_test + n_val], "split"] = "validation"
        src.loc[idx[n_test + n_val:], "split"] = "train"
    return src


# ---------------------------------------------------------------- translations (v2)
REVIEW_OK = {"good", "understandable"}


def add_translations(src):
    """Adds reviewed translations as extra source rows in the SAME split as the original.
    A row is used if review != Wrong and (no automatic flag, or review is Good/Understandable)."""
    info = {"file": "not found - no translations added"}
    if not TRANSLATIONS_FILE.exists():
        return src, info
    tr = pd.read_excel(TRANSLATIONS_FILE, sheet_name="translations").fillna("")
    need = {"source_id", "translated_text", "auto_flag", "review", "target_language"}
    if need - set(tr.columns):
        sys.exit(f"{TRANSLATIONS_FILE}: missing columns {need - set(tr.columns)}")

    review = tr["review"].astype(str).str.strip().str.lower()
    flagged = tr["auto_flag"].astype(str).str.strip() != ""
    wrong = review == "wrong"
    keep = ~wrong & (~flagged | review.isin(REVIEW_OK))
    reviewed = review[review.isin(REVIEW_OK | {"wrong"})]
    info = {"file": str(TRANSLATIONS_FILE), "rows in file": len(tr),
            "removed - marked Wrong": int(wrong.sum()),
            "removed - automatic flag, not reviewed OK": int((flagged & ~wrong & ~review.isin(REVIEW_OK)).sum()),
            "reviewed": len(reviewed),
            "review results": dict(Counter(reviewed.str.capitalize()))}
    tr = tr[keep]

    if "original_text" not in tr.columns:
        sys.exit(f"{TRANSLATIONS_FILE}: column 'original_text' is needed")
    # match on the complaint TEXT, not the row number, so the file still works after
    # the LECO/MC files are corrected (rows removed, labels changed)
    by_key = {r["key"]: r for r in src.to_dict("records")}
    rows, skipped = [], Counter()
    for t in tr.itertuples():
        orig = by_key.get(compare_key(clean(t.original_text)))
        if orig is None:
            skipped["original no longer in the source files"] += 1
            continue
        sid = orig["source_id"]
        if orig["split"] not in ("train", "validation"):     # never translate test complaints
            skipped["original is now in the test split"] += 1
            continue
        text = clean(t.translated_text)
        lang = language(text)
        if lang != t.target_language or lang == orig["language"]:
            skipped["translation is in the wrong language"] += 1
            continue
        rows.append({"source_id": f"{sid}-T", "institution": orig["institution"],
                     "department": orig["department"], "text": text, "translated": True,
                     "original_source_id": sid, "n_words": len(tokenize(text)),
                     "key": compare_key(text), "language": lang, "split": orig["split"]})
    for reason, n in skipped.items():
        info[f"removed - {reason}"] = n
    if sum(skipped.values()) > 0.2 * max(len(tr), 1):
        print(f"\nWARNING: {sum(skipped.values())} of {len(tr)} translations could not be matched "
              "to the current source files. If the LECO/MC files changed, run the translation "
              "notebook again on the new source_split.xlsx.")

    add = pd.DataFrame(rows)
    if add.empty:
        return src, info
    dup = add["key"].isin(set(src["key"])) | add.duplicated("key") | (add["key"] == "")
    info["removed - same text as another complaint"] = int(dup.sum())
    add = add[~dup]
    info["added"] = len(add)
    return pd.concat([src, add], ignore_index=True), info


# ---------------------------------------------------------------- record building
def make_connector(prev_text, next_text, rng, split):
    """Returns (connector_text, glue, word, unseen, strip_prev).
    glue=True means no space before it; word = connector word used ('' if none);
    unseen = True if the word is a test-only connector;
    strip_prev = True means remove the full stop at the end of the previous part
    (used for comma-only joins: "... බිල වැඩියි, පාර කැඩිලා ...")."""
    ends_sentence = prev_text.endswith(SENTENCE_END)
    if rng.random() < NO_WORD_CONNECTOR_SHARE:
        style = rng.choices(list(NO_WORD_STYLES), list(NO_WORD_STYLES.values()))[0]
        if style == "comma":
            return ",", True, "", False, ends_sentence
        if style == "none" or ends_sentence:          # just a space between the parts
            return "", False, "", False, False
        return ".", True, "", False, False

    a, b = language(prev_text), language(next_text)
    if a == b == "si":
        lang = rng.choices(["si", "singlish"], [0.75, 0.25])[0]
    elif a == b == "en":
        lang = rng.choices(["en", "singlish"], [0.85, 0.15])[0]
    else:
        lang = rng.choice(["si", "en", "singlish"])

    unseen = split == "test" and rng.random() < TEST_ONLY_SHARE
    word = rng.choice((TEST_ONLY_CONNECTORS if unseen else CONNECTORS)[lang])
    sentence_like = word.endswith(".") or word.endswith(":")

    if ends_sentence:
        lead, glue = "", False
    elif sentence_like:
        lead, glue = ".", True
    elif lang != "en":                                 # "..., ඒ වගේම ..." as well as ". ඒ වගේම"
        lead, glue = (",", True) if rng.random() < SI_COMMA_LEAD_SHARE else (".", True)
    else:
        lead, glue = rng.choice([(".", True), (",", True)])

    lowered = False
    if lang == "en":
        if lead == "," and not sentence_like and not unseen:
            word = rng.choice(EN_LOWER_AFTER_COMMA)      # ", and ..."  ", also ..."
            lowered = True
        elif lead == "," and not sentence_like:          # test-only word after a comma
            word = word[0].lower() + word[1:]            # ", on top of that ..."
            lowered = True
        elif not sentence_like:
            word = word[0].upper() + word[1:]
    # optional comma after the connector word ("Also, ...", "ඒ වගේම, ...")
    trail = "" if sentence_like or lowered or rng.random() < 0.5 else ","
    text = f"{lead} {word}{trail}".strip() if lead else f"{word}{trail}"
    return text, glue, word, unseen, False


def fixed_connector(conn):
    """Connector typed by hand in the hand-written test set (may be empty)."""
    conn = clean(conn)
    glue = conn[:1] in {",", ".", ";", ":", "!", "?"}
    word = conn.strip(",.;:!? ")
    known = {w.lower().strip(",.;:!? ") for ws in CONNECTORS.values() for w in ws} | \
            set(EN_LOWER_AFTER_COMMA)
    unseen = bool(word) and word.lower() not in known      # never seen in training
    return conn, glue, word, unseen, False


def build_record(parts, rng, split, connectors=None):
    """parts = list of source rows (dicts). Returns (raw, tokens, tags, subs, conn_info).
    connectors = hand-typed connector texts (hand-written test set); else random."""
    pieces = []             # (text, is_sub_index or None, glue)
    words, unseen_any = [], False
    for i, p in enumerate(parts):
        if i > 0:
            if connectors is not None:
                conn, glue, word, unseen, strip_prev = fixed_connector(connectors[i - 1])
            else:
                conn, glue, word, unseen, strip_prev = make_connector(
                    parts[i - 1]["text"], p["text"], rng, split)
            if strip_prev:                              # "... වැඩියි." -> "... වැඩියි" before a comma
                text, sub_i, g = pieces[-1]
                stripped = text.rstrip("".join(SENTENCE_END)).rstrip()
                if stripped:
                    pieces[-1] = (stripped, sub_i, g)
            words.append(word)
            unseen_any = unseen_any or unseen
            if conn:
                pieces.append((conn, None, glue))
        pieces.append((p["text"], i, False))

    raw, tokens, tags = "", [], []
    subs = [None] * len(parts)
    for text, sub_i, glue in pieces:
        if raw and not glue:
            raw += " "
        start_char = len(raw)
        raw += text
        toks = tokenize(text)
        start_tok = len(tokens)
        tokens.extend(toks)
        if sub_i is None:
            tags.extend(["O"] * len(toks))
        else:
            tags.extend(["B"] + ["I"] * (len(toks) - 1))
            p = parts[sub_i]
            subs[sub_i] = {
                "text": text, "institution": p["institution"], "department": p["department"],
                "token_start": start_tok, "token_end": len(tokens),
                "char_start": start_char, "char_end": len(raw),
                "language": p["language"], "source_id": p["source_id"],
                "translated": bool(p.get("translated", False)),
            }
    return raw, tokens, tags, subs, (words, unseen_any)


# ---------------------------------------------------------------- generation
def generate(src):
    depts = sorted({(r.institution, r.department) for r in src.itertuples()})
    leco_depts = [d for d in depts if d[0] == "LECO"]
    mc_depts = [d for d in depts if d[0] == "MC"]
    combos = {
        "two_same_by_inst": {"LECO": list(combinations(leco_depts, 2)),
                             "MC": list(combinations(mc_depts, 2))},
        "two_cross": [(a, b) for a in leco_depts for b in mc_depts],
        "three": [c for c in combinations(depts, 3)
                  if len({d[0] for d in c}) >= 1],
    }

    records, seen_text = {}, set()
    usage = Counter()
    for split in SPLIT_RATIO:
        rng = random.Random(f"{SEED}-{split}")      # v2: own generator per split
        part = src[src["split"] == split]
        pool = {d: part[(part.institution == d[0]) & (part.department == d[1])]
                .to_dict("records") for d in depts}
        multi_pool = {d: [r for r in rows if r["n_words"] <= MAX_SEGMENT_WORDS]
                      for d, rows in pool.items()}
        out = []

        # --- single-topic
        for d in depts:
            rows = [r for r in pool[d] if r["n_words"] <= MAX_TOTAL_WORDS]
            if SINGLE_CAP_PER_DEPARTMENT:
                cap = round(SINGLE_CAP_PER_DEPARTMENT * SPLIT_RATIO[split])
                rows = rng.sample(rows, min(cap, len(rows)))
            for r in rows:
                raw, toks, tags, subs, conn = build_record([r], rng, split)
                seen_text.add(raw)
                usage[r["source_id"]] += 1
                out.append(("single", raw, toks, tags, subs, conn))

        # --- multi-topic
        def pick(d):
            rows = multi_pool[d]
            if not rows:
                return None
            low = min(usage[r["source_id"]] for r in rows)
            return rng.choice([r for r in rows if usage[r["source_id"]] == low])

        for kind, n in MULTI_COUNTS[split].items():
            made, tries = 0, 0
            while made < n and tries < n * 50:
                tries += 1
                if kind == "two_same":
                    inst = rng.choice([i for i, c in combos["two_same_by_inst"].items() if c])
                    combo = rng.choice(combos["two_same_by_inst"][inst])
                else:
                    combo = rng.choice(combos[kind])
                parts = [pick(d) for d in combo]
                if any(p is None for p in parts):
                    continue
                if sum(p["n_words"] for p in parts) > MAX_TOTAL_WORDS - 10:
                    continue
                rng.shuffle(parts)
                raw, toks, tags, subs, conn = build_record(parts, rng, split)
                if raw in seen_text:
                    continue
                seen_text.add(raw)
                for p in parts:
                    usage[p["source_id"]] += 1
                out.append((kind, raw, toks, tags, subs, conn))
                made += 1
            if made < n:
                print(f"  WARNING {split}/{kind}: built {made} of {n}")

        rng.shuffle(out)
        prefix = {"train": "TR", "validation": "VA", "test": "TE"}[split]
        records[split] = []
        for i, (kind, raw, toks, tags, subs, (cwords, unseen)) in enumerate(out, 1):
            records[split].append({
                "complaint_id": f"{prefix}-{i:06d}",
                "split": split,
                "complaint_type": kind,
                "num_sub_complaints": len(subs),
                "raw_complaint_text": raw,
                "tokens": toks,
                "bio_tags": tags,
                "sub_complaints": subs,
                "target_institutions": [s["institution"] for s in subs],
                "target_departments": [s["department"] for s in subs],
                "connector_words": cwords,
                "unseen_connector": unseen,
                "has_translated_part": any(s["translated"] for s in subs),
                "language_swap": any(is_swap(s) for s in subs),
                "is_fake": False,
            })
    return records, usage


def is_swap(sub):
    """LECO in Sinhala or MC in English - the case the language confound gets wrong."""
    return (sub["institution"], sub["language"]) in {("LECO", "si"), ("MC", "en")}


# ---------------------------------------------------------------- hand-written test set (v2)
def build_handwritten(src):
    """handwritten_test.xlsx: one complaint per row, written by the team (never trained on).
    Columns: part_1, department_1, connector_1, part_2, department_2, connector_2,
             part_3, department_3  (part_2 / part_3 may be empty)."""
    if not HANDWRITTEN_FILE.exists():
        return [], []
    hw = pd.read_excel(HANDWRITTEN_FILE).fillna("")
    dept_inst = dict(zip(src["department"], src["institution"]))
    records, rows, errors = [], [], []
    for n, r in enumerate(hw.to_dict("records"), start=2):          # Excel row number
        parts, conns = [], []
        for k in (1, 2, 3):
            text = clean(r.get(f"part_{k}", ""))
            dept = clean(r.get(f"department_{k}", ""))
            if not text and not dept:
                continue
            if not text or dept not in dept_inst:
                errors.append(f"row {n}: part_{k} needs text and one of these departments: "
                              f"{sorted(dept_inst)} (got '{dept}')")
                continue
            if parts:
                conns.append(r.get(f"connector_{k - 1}", ""))
            sid = f"HW-{n}-{k}"
            parts.append({"source_id": sid, "institution": dept_inst[dept], "department": dept,
                          "text": text, "language": language(text), "translated": False})
            rows.append({**parts[-1], "split": "handwritten_test", "original_source_id": "",
                         "n_words": len(tokenize(text)), "key": compare_key(text)})
        if not parts:
            continue
        raw, toks, tags, subs, (cwords, unseen) = build_record(parts, None, "handwritten_test", conns)
        insts = {s["institution"] for s in subs}
        kind = ("single" if len(subs) == 1 else "three" if len(subs) == 3
                else "two_cross" if len(insts) == 2 else "two_same")
        records.append({
            "complaint_id": f"HW-{n:04d}", "split": "handwritten_test", "complaint_type": kind,
            "num_sub_complaints": len(subs), "raw_complaint_text": raw, "tokens": toks,
            "bio_tags": tags, "sub_complaints": subs,
            "target_institutions": [s["institution"] for s in subs],
            "target_departments": [s["department"] for s in subs],
            "connector_words": cwords, "unseen_connector": unseen,
            "has_translated_part": False, "language_swap": any(is_swap(s) for s in subs),
            "is_fake": False,
        })
    if errors:
        print(f"\n{len(errors)} problems in {HANDWRITTEN_FILE}:")
        for e in errors[:30]:
            print("  ", e)
        sys.exit(1)
    return records, rows


# ---------------------------------------------------------------- checks
def check(records, src):
    errors = []
    test_only = {w.lower() for ws in TEST_ONLY_CONNECTORS.values() for w in ws}
    normal = {w.lower() for ws in CONNECTORS.values() for w in ws} | \
             {w.lower() for w in EN_LOWER_AFTER_COMMA}
    overlap = test_only & normal
    if overlap:
        errors.append(f"test-only connectors also in the normal list: {overlap}")
    dept_inst = {(r.department): r.institution for r in src.itertuples()}
    key_split = dict(zip(src["source_id"], src["split"]))
    all_text = Counter()
    for split, recs in records.items():
        for r in recs:
            cid, toks, tags = r["complaint_id"], r["tokens"], r["bio_tags"]
            all_text[r["raw_complaint_text"]] += 1
            if toks != tokenize(r["raw_complaint_text"]):
                errors.append(f"{cid}: tokens do not match the text")
            if len(toks) != len(tags):
                errors.append(f"{cid}: {len(toks)} tokens but {len(tags)} tags")
            if set(tags) - {"B", "I", "O"}:
                errors.append(f"{cid}: unknown tag")
            for i, t in enumerate(tags):
                if t == "I" and (i == 0 or tags[i - 1] == "O"):
                    errors.append(f"{cid}: I tag at {i} not inside a span")
            if tags.count("B") != len(r["sub_complaints"]):
                errors.append(f"{cid}: B count != number of sub-complaints")
            covered = set()
            for s in r["sub_complaints"]:
                a, b = s["token_start"], s["token_end"]
                covered.update(range(a, b))
                if toks[a:b] != tokenize(s["text"]):
                    errors.append(f"{cid}: span tokens != sub-complaint text")
                if tags[a:b] != ["B"] + ["I"] * (b - a - 1):
                    errors.append(f"{cid}: span tags wrong")
                if r["raw_complaint_text"][s["char_start"]:s["char_end"]] != s["text"]:
                    errors.append(f"{cid}: char span wrong")
                if dept_inst.get(s["department"]) != s["institution"]:
                    errors.append(f"{cid}: department/institution mismatch")
                if key_split.get(s["source_id"]) != split:
                    errors.append(f"{cid}: LEAKAGE - source {s['source_id']} is not in {split}")
            for i, t in enumerate(tags):
                if i not in covered and t != "O":
                    errors.append(f"{cid}: connector token not O")
            used = {w.lower() for w in r["connector_words"] if w}
            if split in ("train", "validation") and used & test_only:
                errors.append(f"{cid}: LEAKAGE - test-only connector {used & test_only} in {split}")
            if split in SPLIT_RATIO and r["unseen_connector"] != bool(used & test_only):
                errors.append(f"{cid}: unseen_connector flag wrong")
            if split == "test" and r["has_translated_part"]:
                errors.append(f"{cid}: LEAKAGE - translated complaint in the test split")
            if r["complaint_type"] == "two_cross" and len(set(r["target_institutions"])) != 2:
                errors.append(f"{cid}: two_cross but not two institutions")
            if r["complaint_type"] == "two_same" and len(set(r["target_institutions"])) != 1:
                errors.append(f"{cid}: two_same but two institutions")
            if r["complaint_type"] != "single" and \
                    len(set(r["target_departments"])) != r["num_sub_complaints"]:
                errors.append(f"{cid}: repeated department")
    dups = [t for t, n in all_text.items() if n > 1]
    if dups:
        errors.append(f"{len(dups)} record texts appear more than once")
    return errors


# ---------------------------------------------------------------- output
def write_outputs(records, src, usage, dropped, tr_info):
    OUT_DIR.mkdir(exist_ok=True)
    for split, recs in records.items():
        with open(OUT_DIR / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with pd.ExcelWriter(OUT_DIR / "dataset_view.xlsx", engine="openpyxl") as xl:
        for split, recs in records.items():
            rows = []
            for r in recs:
                row = {"complaint_id": r["complaint_id"], "complaint_type": r["complaint_type"],
                       "raw_complaint_text": r["raw_complaint_text"]}
                for k in range(3):
                    s = r["sub_complaints"][k] if k < len(r["sub_complaints"]) else None
                    row[f"sub_complaint_{k + 1}"] = s["text"] if s else ""
                    row[f"institution_{k + 1}"] = s["institution"] if s else ""
                    row[f"department_{k + 1}"] = s["department"] if s else ""
                    row[f"language_{k + 1}"] = s["language"] if s else ""
                    row[f"translated_{k + 1}"] = s["translated"] if s else ""
                row["tokens"] = " | ".join(r["tokens"])
                row["bio_tags"] = " ".join(r["bio_tags"])
                row["connector_words"] = " | ".join(w for w in r["connector_words"] if w)
                row["unseen_connector"] = r["unseen_connector"]
                row["language_swap"] = r["language_swap"]
                row["is_fake"] = r["is_fake"]
                rows.append(row)
            pd.DataFrame(rows).to_excel(xl, sheet_name=split[:31], index=False)

    s = src.copy()
    s["times_used"] = s["source_id"].map(usage).fillna(0).astype(int)
    s.drop(columns=["key"]).to_excel(OUT_DIR / "source_split.xlsx", index=False)

    main_recs = {k: records[k] for k in SPLIT_RATIO}
    real = s[~s.translated]
    lines = ["== Dataset version 2",
             "  split: per department, by a hash of the complaint text (stable), 80/10/10",
             "\n== Sources (real complaints)"]
    for k, v in dropped.items():
        lines.append(f"  removed - {k}: {v}")
    t = real.groupby(["institution", "department", "split"]).size().unstack(fill_value=0)
    lines.append(t.to_string())

    lines.append("\n== Translations (NLLB-200)")
    for k, v in tr_info.items():
        lines.append(f"  {k}: {v}")
    tr = s[s.translated]
    if len(tr):
        lines.append(tr.groupby(["institution", "department", "split"]).size()
                     .unstack(fill_value=0).to_string())
    lines.append("  language of all source complaints (real + translated), train split:")
    lang = s[s.split == "train"].groupby(["institution", "language"]).size().unstack(fill_value=0)
    lines.append("  " + lang.to_string().replace("\n", "\n  "))

    lines.append(f"\n  multi-topic eligible (<= {MAX_SEGMENT_WORDS} words): "
                 f"{(s.n_words <= MAX_SEGMENT_WORDS).sum()} of {len(s)}")
    lines.append("\n== Records per split and type")
    tab = pd.DataFrame([(r["split"], r["complaint_type"]) for recs in main_recs.values()
                        for r in recs], columns=["split", "type"])
    lines.append(tab.groupby(["type", "split"]).size().unstack(fill_value=0)
                 [list(SPLIT_RATIO)].to_string())
    total = len(tab)
    for sp in SPLIT_RATIO:
        n = (tab.split == sp).sum()
        lines.append(f"  {sp:<11}: {n:>6}  ({100 * n / total:.1f}%)")
    lines.append("\n== Records with a translated part / with a language-swapped part")
    for sp, recs in main_recs.items():
        lines.append(f"  {sp:<11}: translated {sum(r['has_translated_part'] for r in recs):>5}"
                     f"   language swap {sum(r['language_swap'] for r in recs):>5}   of {len(recs)}")
    lines.append("\n== Sub-complaints per department (train + validation + test)")
    dc = Counter((sc["institution"], sc["department"]) for recs in main_recs.values()
                 for r in recs for sc in r["sub_complaints"])
    for (i, d), n in sorted(dc.items()):
        lines.append(f"  {i:<5} {d:<22}: {n}")
    te = [r for r in records["test"] if r["complaint_type"] != "single"]
    n_un = sum(r["unseen_connector"] for r in te)
    lines.append(f"\n== Test multi-topic records with a test-only (unseen) connector: "
                 f"{n_un} of {len(te)}")
    lines.append("\n== Joins between sub-complaints (all splits)")
    jc = Counter()
    for recs in main_recs.values():
        for r in recs:
            spans = sorted((sc["token_end"], sc["token_start"]) for sc in r["sub_complaints"])
            for (end_a, _), (_, start_b) in zip(spans, spans[1:]):
                between = r["tokens"][end_a:start_b]
                words = [w for w in between if any(ch.isalnum() for ch in w)]
                jc["connector word" if words else
                   "comma only" if "," in between else
                   "full stop only" if between else "nothing (space)"] += 1
    for k, n in jc.most_common():
        lines.append(f"  {k:<16}: {n}")
    lines.append("\n== Languages inside multi-topic records")
    lc = Counter("+".join(sorted({sc["language"] for sc in r["sub_complaints"]}))
                 for recs in main_recs.values() for r in recs if r["complaint_type"] != "single")
    for k, n in lc.items():
        lines.append(f"  {k:<6}: {n}")
    used = s[s.times_used > 0]
    lines.append("\n== Re-use of source complaints in records")
    for (i, d), g in used.groupby(["institution", "department"]):
        lines.append(f"  {i:<5} {d:<22}: mean {g.times_used.mean():.1f}, max {g.times_used.max()}")
    lens = [len(r["tokens"]) for recs in main_recs.values() for r in recs]
    lines.append(f"\n== Record length (words): median {pd.Series(lens).median():.0f}, "
                 f"max {max(lens)}")

    hw = records.get("handwritten_test", [])
    lines.append("\n== Hand-written test set")
    if not hw:
        lines.append(f"  {HANDWRITTEN_FILE} not found - not built")
    else:
        lines.append(f"  complaints: {len(hw)}   with a language-swapped part: "
                     f"{sum(r['language_swap'] for r in hw)}")
        lines.append("  by type: " + ", ".join(f"{k} {v}" for k, v in
                                               Counter(r["complaint_type"] for r in hw).items()))
        sc = Counter((x["institution"], x["language"]) for r in hw for x in r["sub_complaints"])
        lines.append("  parts by institution/language: " +
                     ", ".join(f"{i}-{l} {n}" for (i, l), n in sorted(sc.items())))
    (OUT_DIR / "dataset_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    src, dropped = load_sources()
    src = split_sources(src)
    src, tr_info = add_translations(src)
    records, usage = generate(src)

    hw_records, hw_rows = build_handwritten(src)
    check_src = src
    if hw_records:
        hw = pd.DataFrame(hw_rows)
        copied = hw[hw["key"].isin(set(src["key"]))]
        if len(copied):
            print(f"\n{len(copied)} hand-written parts are the same as a real complaint "
                  f"(rows {sorted({s.split('-')[1] for s in copied.source_id})}). "
                  "Hand-written complaints must be new text - change them.")
            sys.exit(1)
        records["handwritten_test"] = hw_records
        check_src = pd.concat([src, hw], ignore_index=True)

    errors = check(records, check_src)
    if errors:
        print(f"\n{len(errors)} CHECK FAILURES - dataset not written. First ones:")
        for e in errors[:20]:
            print("  ", e)
        sys.exit(1)
    write_outputs(records, src, usage, dropped, tr_info)
    print(f"\nAll checks passed. Files written to '{OUT_DIR}/'.")


if __name__ == "__main__":
    main()
