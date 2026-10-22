"""Build data/past_complaints.jsonl (the RAG memory) from a CSV or text files.

Examples
  python scripts/build_memory.py --csv routing_records.csv --text-col complaint --id-col id
  python scripts/build_memory.py --txt-dir transcriptions/        (one .txt per complaint)

Anonymise first: phone numbers and NIC numbers are masked automatically,
but remove names yourself. Then call POST /api/v1/admin/reload-memory
(or restart the server).
"""
import argparse
import csv
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "past_complaints.jsonl"
PHONE = re.compile(r"(\+94|0)\s?\d{2}\s?\d{3}\s?\d{4}")
NIC = re.compile(r"\b(\d{9}[VvXx]|\d{12})\b")


def mask(t: str) -> str:
    return NIC.sub("<NIC>", PHONE.sub("<PHONE>", t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--id-col")
    ap.add_argument("--extra-cols", default="", help="comma list copied as metadata, e.g. institution,division")
    ap.add_argument("--txt-dir")
    ap.add_argument("--append", action="store_true")
    a = ap.parse_args()
    rows = []
    if a.csv:
        with open(a.csv, encoding="utf-8-sig", newline="") as f:
            for i, r in enumerate(csv.DictReader(f)):
                text = (r.get(a.text_col) or "").strip()
                if not text:
                    continue
                d = {"id": str(r[a.id_col]) if a.id_col else f"csv{i}", "text": mask(text)}
                for c in filter(None, a.extra_cols.split(",")):
                    d[c] = r.get(c)
                rows.append(d)
    if a.txt_dir:
        for p in sorted(Path(a.txt_dir).glob("*.txt")):
            text = p.read_text(encoding="utf-8").strip()
            if text:
                rows.append({"id": p.stem, "text": mask(text)})
    mode = "a" if a.append else "w"
    with open(OUT, mode, encoding="utf-8") as f:
        for d in rows:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} complaints to {OUT}")


if __name__ == "__main__":
    main()
