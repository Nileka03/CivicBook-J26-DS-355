"""Runs Component 2 for one complaint (all its handwritten pages together).

intake -> CRNN (segment_page -> line crops -> text + character confidences)
       -> separators -> layout
       -> retrieve past complaints (char TF-IDF, top 20)
       -> c2_correction.correct_page over every line of every page in one call
          (mark uncertain -> LLM edit list -> verifier -> re-check)
       -> corrected lines mapped back to their page and line item -> package
"""
import logging
import re
import time
from dataclasses import dataclass

import cv2

from ..config import settings
from ..schemas import ComplaintPackage, EditOut, PageTranscript, RetrievedContext, StoredFile, WordOut
from .. import storage
from . import c2_correction as C
from .intake import is_blank, load_pages, prepare_page
from .layout import Word, build_layout, detect_separators, render
from .llm_client import LLMUnavailable, get_client
from .ocr_engines import OCRLine, PageOCR, recognize_with_fallback
from .retriever import get_retriever, load_lexicon

log = logging.getLogger(__name__)

C2_ID = re.compile(r"l(\d+)w(\d+)")  # c2_correction word ids: line index across the complaint, word index


def _page_header(n: int, total: int) -> str:
    return f"--- Page {n} of {total} ---"


def _join_pages(texts: list[str]) -> str:
    if len(texts) == 1:
        return texts[0]
    return "\n\n".join(f"{_page_header(i + 1, len(texts))}\n{t}" for i, t in enumerate(texts))


@dataclass
class _Page:
    file: StoredFile
    source_page: int
    ocr: PageOCR
    derived: StoredFile
    items: list[Word]  # one per OCR line item, in reading order
    h_seps: list[int]
    v_seps: list[int]


def process_complaint(complaint_id: str) -> None:
    pkg = storage.get_package(complaint_id)
    if pkg is None:
        return
    pkg.status = "processing"
    storage.save_package(pkg)
    try:
        if pkg.handwritten and pkg.handwritten.files:
            _process_handwritten(pkg)
        pkg.status = "done"
    except Exception as exc:  # never leave a complaint stuck in "processing"
        log.exception("Processing failed for %s", complaint_id)
        pkg.status = "failed"
        pkg.error = str(exc)
    storage.save_package(pkg)


def _is_uncertain(line: OCRLine) -> bool:
    return any(c < settings.CORRECTION_TAU for _, c in C.words_with_conf(C.norm(line.text), line.char_confs))


def _correct(lines: list[OCRLine], refs: list[str], description: str | None) -> tuple[dict, bool]:
    """Runs the configured correction once for the whole complaint. Returns (c2 result, LLM used)."""
    pairs = [(l.text, l.char_confs) for l in lines]
    blind = settings.CORRECTION_MODE == "blind_rag"
    client = None
    if blind or any(_is_uncertain(l) for l in lines):  # otherwise correct_page returns NO_CHANGES itself
        try:
            client = get_client()
        except LLMUnavailable as exc:
            return {"final": "\n".join(l.text for l in lines), "decision": "FALLBACK_RAW",
                    "error": str(exc), "edits": [], "rounds": 0}, False
    if blind:
        return C.correct_blind(pairs, client, settings.LLM_MODEL, refs), True
    lexicon = load_lexicon(settings.LEXICON_PATH)
    out = C.correct_page(pairs, client, settings.LLM_MODEL, refs, lexicon, tau=settings.CORRECTION_TAU,
                         use_verifier=settings.VERIFIER_ENABLED, description=description or "",
                         sleep=settings.LLM_ROUND_PAUSE_S)
    return out, client is not None


def _final_lines(lines: list[OCRLine], out: dict, reasons: list[str]) -> list[str]:
    """One final text per line item. Lines without an accepted edit keep the OCR text exactly."""
    raw = [l.text for l in lines]
    if out["decision"] == "FALLBACK_RAW":
        return raw
    fin = out["final"].split("\n")
    if out["decision"] == "BLIND":
        if len(fin) != len(raw):  # the rewrite is free to change the line count
            reasons.append(f"blind_rag: LLM returned {len(fin)} lines for {len(raw)}")
            if len(fin) < len(raw):
                fin += [""] * (len(raw) - len(fin))
            else:
                fin = fin[:len(raw) - 1] + [" ".join(fin[len(raw) - 1:])]
        return fin
    edited = {int(m.group(1)) for e in out["edits"] if e["ok"] and (m := C2_ID.fullmatch(e["id"]))}
    return [fin[i] if i in edited else t for i, t in enumerate(raw)]


def _process_handwritten(pkg: ComplaintPackage) -> None:
    hw = pkg.handwritten
    timings: dict[str, int] = {}
    t0 = time.perf_counter()

    # ---------- 1. pages: load, OCR line items, separators ----------------------
    pages: list[_Page] = []
    warnings = []
    engine_used = None
    page_no = 0
    for f in hw.files:
        data = storage.file_path(f.file_id).read_bytes()
        for sp, gray in enumerate(load_pages(data, f.content_type), start=1):
            page_no += 1
            if is_blank(gray):
                ocr, h_seps, v_seps = PageOCR(*prepare_page(gray), []), [], []
            else:
                ocr, engine_used, warn = recognize_with_fallback(gray)
                h_seps, v_seps = detect_separators(ocr.image)
                if warn:
                    warnings.append(warn)
            ok, buf = cv2.imencode(".png", ocr.image)
            derived = storage.save_file(buf.tobytes(), f"page_{page_no}.png", "image/png")
            items = [
                Word(f"p{page_no}l{i}", page_no, l.text,
                     round(sum(l.char_confs) / len(l.char_confs), 3) if l.char_confs else 0.0,
                     l.x0, l.y0, l.x1, l.y1, uncertain=_is_uncertain(l))
                for i, l in enumerate(ocr.lines)
            ]
            pages.append(_Page(f, sp, ocr, derived, items, h_seps, v_seps))
    layouts = [build_layout(p.items, p.h_seps, p.v_seps, pkg.layout_hint) for p in pages]
    timings["ocr_and_layout"] = int((time.perf_counter() - t0) * 1000)

    # every line of the complaint, page by page, in reading order
    lines = [l for p in pages for l in p.ocr.lines]
    items = [w for p in pages for w in p.items]

    # ---------- 2. retrieval + correction (once for the whole complaint) ------
    reasons: list[str] = []
    references: list[dict] = []
    if not lines:
        out = {"decision": "REVIEW", "edits": [], "rounds": 0}
        reasons.append("no text recognised")
        final_by_id = {}
    else:
        t1 = time.perf_counter()
        if settings.RETRIEVAL_TOP_K > 0:
            query = " ".join([pkg.description or ""] + [l.text for l in lines])
            references = get_retriever().search(query, settings.RETRIEVAL_TOP_K)
        timings["retrieval"] = int((time.perf_counter() - t1) * 1000)

        t2 = time.perf_counter()
        out, llm_used = _correct(lines, [r["text"] for r in references], pkg.description)
        hw.llm_model = settings.LLM_MODEL if llm_used else None
        timings["llm_and_verifier"] = int((time.perf_counter() - t2) * 1000)
        final_by_id = {w.id: t for w, t in zip(items, _final_lines(lines, out, reasons))}

    decision = out["decision"]
    if decision == "FALLBACK_RAW":
        reasons.append(f"LLM correction unavailable: {out.get('error', 'unknown error')}")
    elif decision == "BLIND":  # comparison mode: an unverified rewrite always goes to a human
        decision = "REVIEW"
        reasons.append("blind_rag correction (comparison mode): LLM rewrite not verified")

    def final_text(w: Word) -> str:
        return final_by_id.get(w.id, w.text)

    raw_pages = [render(lay) for lay in layouts]
    final_pages = [render(lay, final_text) for lay in layouts]

    edits = []
    for e in out.get("edits", []):  # correct_blind has no edit list
        m = C2_ID.fullmatch(e["id"])
        li, wi = (int(m.group(1)), int(m.group(2))) if m else (-1, -1)
        w = items[li] if 0 <= li < len(items) else None
        accepted, why = e["ok"], list(e["reasons"])
        if out["decision"] == "FALLBACK_RAW" and accepted:  # a later round failed and the raw text was returned
            accepted, why = False, why + ["not_applied_llm_failed"]
        edits.append(EditOut(word_id=f"{w.id}w{wi}" if w else e["id"], page_no=w.page_no if w else 0,
                             from_text=e["frm"], to_text=e["to"], evidence=str(e.get("evidence", "")),
                             accepted=accepted, reasons=why, round=e["round"]))

    # ---------- 3. assemble the package -------------------------------------
    pages_out = []
    for p, lay, raw, fin in zip(pages, layouts, raw_pages, final_pages):
        confs = [c for l in p.ocr.lines for c in l.char_confs]
        page_n = len(pages_out) + 1
        pages_out.append(PageTranscript(
            page_no=page_n, source_file_id=p.file.file_id, source_page=p.source_page,
            image_uri=p.derived.uri, width=p.ocr.image.shape[1], height=p.ocr.image.shape[0],
            skew_deg=round(p.ocr.skew_deg, 2),
            layout=lay.kind, separators={"horizontal": lay.h_seps, "vertical": lay.v_seps},
            raw_text=raw, final_text=fin,
            mean_confidence=round(sum(confs) / len(confs), 3) if confs else None,
            n_words=len(p.items), n_uncertain=sum(w.uncertain for w in p.items),
            words=[WordOut(id=w.id, text=w.text, final_text=final_text(w), confidence=w.conf,
                           uncertain=w.uncertain, bbox=[w.x0, w.y0, w.x1, w.y1]) for w in p.items],
        ))
    hw.pages = pages_out
    hw.raw_transcript = _join_pages(raw_pages)
    hw.final_transcript = _join_pages(final_pages)
    hw.ocr_engine = engine_used
    hw.verifier_decision = decision
    hw.needs_human_review = decision in ("REVIEW", "FALLBACK_RAW") or bool(reasons)
    hw.review_reasons = reasons + warnings
    hw.edits = edits
    hw.llm_rounds = out.get("rounds", 1 if out["decision"] == "BLIND" else 0)
    hw.retrieved_context = [RetrievedContext(**r) for r in references]
    timings["total"] = int((time.perf_counter() - t0) * 1000)
    hw.timings_ms = timings
