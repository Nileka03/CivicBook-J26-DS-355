"""Evaluation + ablation for PP1: CER/WER before and after correction, and
how often the correction breaks words the OCR had right (correct-to-wrong).

Prepare a folder of test complaints (NOT in data/past_complaints.jsonl):

    eval_set/
      c001_p1.jpg   c001_p2.jpg   c001.gt.txt   c001.desc.txt (optional)
      c002_p1.png                 c002.gt.txt

  * pages of one complaint share the prefix before "_p<number>"
  * <id>.gt.txt  = your typed ground truth (any layout; brackets/page headers are ignored)
  * <id>.desc.txt = what the citizen typed in the form (optional)

Run (from backend/):
    python scripts/evaluate.py eval_set --configs full,no_verifier,no_rag,blind_rag

Writes eval_results.csv (per complaint) and prints a summary table you can paste into slides.
Every config calls the LLM, so expect len(configs) x complaints API calls.
"""
import argparse
import csv
import difflib
import re
import sys
import uuid
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rapidfuzz.distance import Levenshtein  # noqa: E402

from app import storage  # noqa: E402
from app.config import settings  # noqa: E402
from app.pipeline.orchestrator import process_complaint  # noqa: E402
from app.schemas import ComplaintPackage, Handwritten, Location  # noqa: E402

CONFIGS = {
    "full": dict(CORRECTION_MODE="full", VERIFIER_ENABLED=True, RETRIEVAL_TOP_K=20),
    "no_verifier": dict(CORRECTION_MODE="full", VERIFIER_ENABLED=False, RETRIEVAL_TOP_K=20),
    "no_rag": dict(CORRECTION_MODE="full", VERIFIER_ENABLED=True, RETRIEVAL_TOP_K=0),
    "no_rag_no_verifier": dict(CORRECTION_MODE="full", VERIFIER_ENABLED=False, RETRIEVAL_TOP_K=0),
    "blind_rag": dict(CORRECTION_MODE="blind_rag", VERIFIER_ENABLED=True, RETRIEVAL_TOP_K=20),
}
TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".pdf": "application/pdf"}


def norm(text: str) -> str:
    text = re.sub(r"^--- Page \d+ of \d+ ---$", " ", text, flags=re.M)
    text = text.replace("[", " ").replace("]", " ")
    return " ".join(text.split())


def cer(ref: str, hyp: str) -> float:
    return Levenshtein.distance(ref, hyp) / max(1, len(ref))


def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    return Levenshtein.distance(r, h) / max(1, len(r))


def correct_positions(ref_words, hyp_words) -> set[int]:
    """Indices of reference words that the hypothesis reproduces exactly."""
    ok = set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ref_words, b=hyp_words, autojunk=False).get_opcodes():
        if tag == "equal":
            ok.update(range(i1, i2))
    return ok


def load_samples(folder: Path):
    groups = defaultdict(list)
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in TYPES:
            sid = re.sub(r"_p\d+$", "", p.stem)
            groups[sid].append(p)
    for sid, pages in groups.items():
        gt = folder / f"{sid}.gt.txt"
        if not gt.exists():
            print(f"skip {sid}: no {gt.name}")
            continue
        desc = folder / f"{sid}.desc.txt"
        yield sid, pages, gt.read_text(encoding="utf-8"), desc.read_text(encoding="utf-8") if desc.exists() else None


def run_one(pages, description):
    files = [storage.save_file(p.read_bytes(), p.name, TYPES[p.suffix.lower()]) for p in pages]
    pkg = ComplaintPackage(
        complaint_id=f"EVAL-{uuid.uuid4().hex[:10]}", submitted_at=datetime.now(timezone.utc),
        complaint_date=date.today(), location=Location(mode="typed", address="evaluation"),
        description=description, handwritten=Handwritten(files=files),
    )
    storage.save_package(pkg)
    process_complaint(pkg.complaint_id)
    return storage.get_package(pkg.complaint_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--configs", default="full,no_verifier")
    ap.add_argument("--out", default="eval_results.csv")
    a = ap.parse_args()
    configs = [c.strip() for c in a.configs.split(",")]
    settings.STORAGE_DIR = Path(settings.STORAGE_DIR) / "eval"

    rows = []
    for sid, pages, gt, desc in load_samples(a.folder):
        ref = norm(gt)
        raw_done = False
        for cfg in configs:
            for k, v in CONFIGS[cfg].items():
                setattr(settings, k, v)
            pkg = run_one(pages, desc)
            hw = pkg.handwritten
            if pkg.status != "done":
                print(f"{sid} [{cfg}] failed: {pkg.error}")
                continue
            raw, fin = norm(hw.raw_transcript), norm(hw.final_transcript)
            ref_w = ref.split()
            ok_raw = correct_positions(ref_w, raw.split())
            ok_fin = correct_positions(ref_w, fin.split())
            c2w = len(ok_raw - ok_fin) / max(1, len(ok_raw))
            w2c = len(ok_fin - ok_raw) / max(1, len(ref_w) - len(ok_raw))
            if not raw_done:
                rows.append(dict(sample=sid, config="ocr_only", cer=cer(ref, raw), wer=wer(ref, raw),
                                 c2w=0.0, w2c=0.0, accepted=0, rejected=0, decision="-"))
                raw_done = True
            rows.append(dict(sample=sid, config=cfg, cer=cer(ref, fin), wer=wer(ref, fin), c2w=c2w, w2c=w2c,
                             accepted=sum(e.accepted for e in hw.edits),
                             rejected=sum(not e.accepted for e in hw.edits), decision=hw.verifier_decision))
            print(f"{sid} [{cfg}] CER raw {cer(ref, raw):.3f} -> {cer(ref, fin):.3f}  C->W {c2w:.3f}  {hw.verifier_decision}")

    if not rows:
        print("no samples evaluated")
        return
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n| Config | CER | WER | Correct→wrong | Wrong→correct | Edits kept | Edits reverted |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for cfg in ["ocr_only"] + configs:
        rs = [r for r in rows if r["config"] == cfg]
        if not rs:
            continue
        m = lambda k: sum(r[k] for r in rs) / len(rs)  # noqa: E731
        print(f"| {cfg} | {m('cer'):.3f} | {m('wer'):.3f} | {m('c2w'):.3f} | {m('w2c'):.3f} | "
              f"{sum(r['accepted'] for r in rs)} | {sum(r['rejected'] for r in rs)} |")
    print(f"\nper-complaint rows written to {a.out}")


if __name__ == "__main__":
    main()
