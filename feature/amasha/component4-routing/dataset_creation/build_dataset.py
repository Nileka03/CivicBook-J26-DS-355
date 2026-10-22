"""
CivicBook Component 4 - build the multi-topic complaint dataset

Reads (same folder):
    leco_final.xlsx       columns: Category, Description   (4 LECO departments)
    mc_engineering.xlsx   one complaint column             (MC Engineering)
    mc_health.xlsx        one complaint column             (MC Health)

Writes (folder "dataset"):
    train.jsonl, validation.jsonl, test.jsonl   one record per line - for training
    dataset_view.xlsx                           the same records as readable sheets
    source_split.xlsx                           which real complaint went to which split
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
OUT_DIR = Path("dataset")

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

# Share of joins that use no connector word, only punctuation.
NO_WORD_CONNECTOR_SHARE = 0.15
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
def split_sources(src, rng):
    src = src.copy()
    src["split"] = ""
    for _, idx in src.groupby(["institution", "department"]).groups.items():
        idx = list(idx)
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, round(n * SPLIT_RATIO["test"]))
        n_val = max(1, round(n * SPLIT_RATIO["validation"]))
        src.loc[idx[:n_test], "split"] = "test"
        src.loc[idx[n_test:n_test + n_val], "split"] = "validation"
        src.loc[idx[n_test + n_val:], "split"] = "train"
    return src


# ---------------------------------------------------------------- record building
def make_connector(prev_text, next_text, rng, split):
    """Returns (connector_text, glue, word, unseen).
    glue=True means no space before it; word = connector word used ('' if none);
    unseen = True if the word is a test-only connector."""
    ends_sentence = prev_text.endswith(SENTENCE_END)
    if rng.random() < NO_WORD_CONNECTOR_SHARE:
        return (("", False) if ends_sentence else (".", True)) + ("", False)

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
    elif sentence_like or lang != "en":
        lead, glue = ".", True
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
    return text, glue, word, unseen


def build_record(parts, rng, split):
    """parts = list of source rows (dicts). Returns (raw, tokens, tags, subs, conn_info)."""
    pieces = []             # (text, is_sub_index or None, glue)
    words, unseen_any = [], False
    for i, p in enumerate(parts):
        if i > 0:
            conn, glue, word, unseen = make_connector(parts[i - 1]["text"], p["text"],
                                                      rng, split)
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
            }
    return raw, tokens, tags, subs, (words, unseen_any)


# ---------------------------------------------------------------- generation
def generate(src, rng):
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
                "is_fake": False,
            })
    return records, usage


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
            if split != "test" and used & test_only:
                errors.append(f"{cid}: LEAKAGE - test-only connector {used & test_only} in {split}")
            if r["unseen_connector"] != bool(used & test_only):
                errors.append(f"{cid}: unseen_connector flag wrong")
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
def write_outputs(records, src, usage, dropped):
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
                row["tokens"] = " | ".join(r["tokens"])
                row["bio_tags"] = " ".join(r["bio_tags"])
                row["connector_words"] = " | ".join(w for w in r["connector_words"] if w)
                row["unseen_connector"] = r["unseen_connector"]
                row["is_fake"] = r["is_fake"]
                rows.append(row)
            pd.DataFrame(rows).to_excel(xl, sheet_name=split, index=False)

    s = src.copy()
    s["times_used"] = s["source_id"].map(usage).fillna(0).astype(int)
    s.drop(columns=["key"]).to_excel(OUT_DIR / "source_split.xlsx", index=False)

    lines = ["== Sources (real complaints)"]
    for k, v in dropped.items():
        lines.append(f"  removed - {k}: {v}")
    t = s.groupby(["institution", "department", "split"]).size().unstack(fill_value=0)
    lines.append(t.to_string())
    lines.append(f"  multi-topic eligible (<= {MAX_SEGMENT_WORDS} words): "
                 f"{(s.n_words <= MAX_SEGMENT_WORDS).sum()} of {len(s)}")
    lines.append("\n== Records per split and type")
    tab = pd.DataFrame([(r["split"], r["complaint_type"]) for recs in records.values()
                        for r in recs], columns=["split", "type"])
    lines.append(tab.groupby(["type", "split"]).size().unstack(fill_value=0)
                 [list(SPLIT_RATIO)].to_string())
    total = len(tab)
    for sp in SPLIT_RATIO:
        n = (tab.split == sp).sum()
        lines.append(f"  {sp:<11}: {n:>6}  ({100 * n / total:.1f}%)")
    lines.append("\n== Sub-complaints per department (all splits)")
    dc = Counter((sc["institution"], sc["department"]) for recs in records.values()
                 for r in recs for sc in r["sub_complaints"])
    for (i, d), n in sorted(dc.items()):
        lines.append(f"  {i:<5} {d:<22}: {n}")
    te = [r for r in records["test"] if r["complaint_type"] != "single"]
    n_un = sum(r["unseen_connector"] for r in te)
    lines.append(f"\n== Test multi-topic records with a test-only (unseen) connector: "
                 f"{n_un} of {len(te)}")
    lines.append("\n== Languages inside multi-topic records")
    lc = Counter("+".join(sorted({sc["language"] for sc in r["sub_complaints"]}))
                 for recs in records.values() for r in recs if r["complaint_type"] != "single")
    for k, n in lc.items():
        lines.append(f"  {k:<6}: {n}")
    used = s[s.times_used > 0]
    lines.append("\n== Re-use of real complaints in multi-topic records")
    for (i, d), g in used.groupby(["institution", "department"]):
        lines.append(f"  {i:<5} {d:<22}: mean {g.times_used.mean():.1f}, max {g.times_used.max()}")
    lens = [len(r["tokens"]) for recs in records.values() for r in recs]
    lines.append(f"\n== Record length (words): median {pd.Series(lens).median():.0f}, "
                 f"max {max(lens)}")
    (OUT_DIR / "dataset_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    rng = random.Random(SEED)
    src, dropped = load_sources()
    src = split_sources(src, rng)
    records, usage = generate(src, rng)
    errors = check(records, src)
    if errors:
        print(f"\n{len(errors)} CHECK FAILURES - dataset not written. First ones:")
        for e in errors[:20]:
            print("  ", e)
        sys.exit(1)
    write_outputs(records, src, usage, dropped)
    print(f"\nAll checks passed. Files written to '{OUT_DIR}/'.")


if __name__ == "__main__":
    main()