"""CivicBook - Component 2: handwritten complaint intake + document OCR service.

Run:  uvicorn app.main:app --reload      (from the backend/ folder)
Docs: http://127.0.0.1:8000/docs
"""
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from typing import Literal

import httpx
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import storage
from .config import settings
from .pipeline.ocr_engines import get_engine
from .pipeline.orchestrator import process_complaint
from .pipeline.retriever import get_retriever, reload_retriever
from .schemas import ComplaintPackage, ComplaintSummary, Handwritten, Location

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("component2")


@asynccontextmanager
async def lifespan(_app):
    n = len(get_retriever().docs)  # builds the TF-IDF index once
    try:
        get_engine(settings.OCR_ENGINE)  # load the CRNN weights now, not on the first complaint
    except Exception as exc:
        log.error("OCR engine '%s' could not be loaded: %s", settings.OCR_ENGINE, exc)
    log.info("OCR engine=%s fallback=%s | correction=%s | LLM=%s (key %s) | past complaints=%d",
             settings.OCR_ENGINE, settings.OCR_FALLBACK_ENGINE, settings.CORRECTION_MODE, settings.LLM_MODEL,
             "set" if settings.GEMINI_API_KEY else "MISSING", n)
    yield


app = FastAPI(
    lifespan=lifespan,
    title="CivicBook - Component 2: Handwritten Complaint OCR",
    version="1.0.0",
    description="Accepts one complaint (form fields + evidence images + handwritten pages), "
                "transcribes the handwritten pages with the trained CRNN and layout, corrects low-confidence words "
                "with retrieval-grounded LLM edits, verifies every edit, and returns one ComplaintPackage.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()],
    allow_methods=["*"], allow_headers=["*"],
)

IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
HANDWRITTEN_TYPES = IMAGE_TYPES | {"application/pdf"}
API = "/api/v1"


async def _read_upload(f: UploadFile, allowed: set[str], label: str) -> tuple[bytes, str]:
    ctype = (f.content_type or "").lower()
    if ctype == "image/jpg":
        ctype = "image/jpeg"
    if ctype not in allowed:
        raise HTTPException(415, f"{label} '{f.filename}': type {ctype or 'unknown'} not allowed "
                                 f"({', '.join(sorted(allowed))})")
    data = await f.read()
    if not data:
        raise HTTPException(400, f"{label} '{f.filename}' is empty")
    if len(data) > settings.MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(413, f"{label} '{f.filename}' is larger than {settings.MAX_FILE_MB} MB")
    return data, ctype


@app.get(f"{API}/health")
def health():
    return {
        "status": "ok",
        "ocr_engine": settings.OCR_ENGINE,
        "ocr_fallback": settings.OCR_FALLBACK_ENGINE,
        "correction_mode": settings.CORRECTION_MODE,
        "llm_model": settings.LLM_MODEL,
        "llm_key_present": bool(settings.GEMINI_API_KEY),
        "past_complaints": len(get_retriever().docs),
    }


@app.post(f"{API}/complaints", status_code=202, response_model=ComplaintSummary)
async def submit_complaint(
    background: BackgroundTasks,
    complaint_date: date | None = Form(None),
    location_mode: Literal["gps", "typed"] = Form(...),
    latitude: float | None = Form(None),
    longitude: float | None = Form(None),
    accuracy_m: float | None = Form(None),
    address: str | None = Form(None),
    description: str | None = Form(None),
    layout_hint: Literal["auto", "letter", "table"] = Form("auto"),
    evidence_images: list[UploadFile] = File(default=[]),
    handwritten_pages: list[UploadFile] = File(default=[]),
):
    """Submit ONE complaint. Several handwritten files (or a multi-page PDF) are
    treated as pages of the same complaint, in upload order."""
    try:
        location = Location(mode=location_mode, latitude=latitude, longitude=longitude,
                            accuracy_m=accuracy_m, address=(address or "").strip() or None)
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    evidence_images = [f for f in evidence_images if f.filename]
    handwritten_pages = [f for f in handwritten_pages if f.filename]
    description = (description or "").strip() or None
    if not description and not handwritten_pages:
        raise HTTPException(422, "A complaint needs a typed description or at least one handwritten page")
    if len(handwritten_pages) > settings.MAX_HANDWRITTEN_FILES:
        raise HTTPException(422, f"At most {settings.MAX_HANDWRITTEN_FILES} handwritten files")
    if len(evidence_images) > settings.MAX_EVIDENCE_IMAGES:
        raise HTTPException(422, f"At most {settings.MAX_EVIDENCE_IMAGES} evidence images")

    # validate everything before storing anything
    ev_data = [(await _read_upload(f, IMAGE_TYPES, "Evidence image"), f.filename) for f in evidence_images]
    hw_data = [(await _read_upload(f, HANDWRITTEN_TYPES, "Handwritten page"), f.filename)
               for f in handwritten_pages]

    if hw_data:
        from .pipeline.intake import load_pages
        total = 0
        for (data, ctype), name in hw_data:
            try:
                total += len(load_pages(data, ctype))
            except ValueError as exc:
                raise HTTPException(422, f"'{name}': {exc}")
        if total > settings.MAX_HANDWRITTEN_PAGES:
            raise HTTPException(422, f"{total} pages uploaded; the limit is {settings.MAX_HANDWRITTEN_PAGES}")

    pkg = ComplaintPackage(
        complaint_id=f"C2-{uuid.uuid4().hex[:12].upper()}",
        submitted_at=datetime.now(timezone.utc),
        complaint_date=complaint_date or date.today(),
        location=location,
        description=description,
        layout_hint=layout_hint,
        evidence_images=[storage.save_file(d, n, c) for (d, c), n in ev_data],
        handwritten=Handwritten(files=[storage.save_file(d, n, c) for (d, c), n in hw_data]) if hw_data else None,
    )
    if not hw_data:
        pkg.status = "done"  # nothing for Component 2 to transcribe
    storage.save_package(pkg)
    if hw_data:
        background.add_task(process_complaint, pkg.complaint_id)
    return _summary(pkg)


def _summary(pkg: ComplaintPackage) -> ComplaintSummary:
    return ComplaintSummary(
        complaint_id=pkg.complaint_id, status=pkg.status, submitted_at=pkg.submitted_at,
        complaint_date=pkg.complaint_date,
        n_pages=len(pkg.handwritten.pages) if pkg.handwritten else 0,
        decision=pkg.handwritten.verifier_decision if pkg.handwritten else None,
    )


@app.get(f"{API}/complaints", response_model=list[ComplaintSummary])
def list_complaints(limit: int = 50):
    return [_summary(p) for p in storage.list_packages(limit)]


@app.get(f"{API}/complaints/{{complaint_id}}", response_model=ComplaintPackage)
def get_complaint(complaint_id: str):
    pkg = storage.get_package(complaint_id)
    if pkg is None:
        raise HTTPException(404, "Complaint not found")
    return pkg


@app.post(f"{API}/complaints/{{complaint_id}}/reprocess", status_code=202, response_model=ComplaintSummary)
def reprocess(complaint_id: str, background: BackgroundTasks):
    pkg = storage.get_package(complaint_id)
    if pkg is None:
        raise HTTPException(404, "Complaint not found")
    if not pkg.handwritten:
        raise HTTPException(409, "Complaint has no handwritten pages")
    pkg.status, pkg.error = "queued", None
    storage.save_package(pkg)
    background.add_task(process_complaint, complaint_id)
    return _summary(pkg)


@app.post(f"{API}/complaints/{{complaint_id}}/forward")
async def forward(complaint_id: str):
    """Send the finished package to Component 3 (fake detection)."""
    pkg = storage.get_package(complaint_id)
    if pkg is None:
        raise HTTPException(404, "Complaint not found")
    if pkg.status != "done":
        raise HTTPException(409, f"Complaint is '{pkg.status}', not ready to forward")
    payload = pkg.model_dump(mode="json", exclude={"forwarded"})
    if settings.COMPONENT3_URL:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(settings.COMPONENT3_URL, json=payload)
            result = {"target": settings.COMPONENT3_URL, "status_code": r.status_code,
                      "response": _safe_json(r)}
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Component 3 unreachable: {exc}")
    else:
        result = {"target": "mock", "status_code": 200, "response": mock_component3(payload)}
    result["at"] = datetime.now(timezone.utc).isoformat()
    pkg.forwarded.append(result)
    storage.save_package(pkg)
    return result


def _safe_json(r: httpx.Response):
    try:
        return r.json()
    except ValueError:
        return r.text[:500]


@app.post(f"{API}/mock/component3")
def mock_component3(payload: dict):
    """Stand-in for Component 3 until its API exists: validates and acknowledges."""
    ComplaintPackage.model_validate(payload)
    hw = payload.get("handwritten") or {}
    log.info("MOCK C3 received %s (decision=%s)", payload["complaint_id"], hw.get("verifier_decision"))
    return {"received": True, "complaint_id": payload["complaint_id"],
            "text_chars": len((payload.get("description") or "") + hw.get("final_transcript", ""))}


@app.get(f"{API}/files/{{file_id}}")
def get_file(file_id: str):
    path = storage.file_path(file_id)
    if path is None:
        raise HTTPException(404, "File not found")
    return FileResponse(path)


@app.post(f"{API}/admin/reload-memory")
def reload_memory():
    """Reload data/past_complaints.jsonl after adding records."""
    return {"past_complaints": reload_retriever()}
