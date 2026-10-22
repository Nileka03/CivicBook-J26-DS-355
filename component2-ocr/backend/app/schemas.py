"""Data contracts.

ComplaintPackage is the single object this component produces. Component 3
(fake detection) and Component 4 (routing) consume it, so change it only after
agreeing with the team, and bump SCHEMA_VERSION when you do.
"""
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "1.0"

Decision = Literal[
    "ACCEPT",          # every LLM edit passed the verifier
    "PARTIAL",         # some edits reverted by the verifier, the rest kept
    "NO_CHANGES",      # OCR was confident everywhere, nothing to correct
    "REVIEW",          # too uncertain / too many rejected edits -> human review
    "FALLBACK_RAW",    # LLM unavailable -> raw OCR text passed on
    "NOT_PROCESSED",   # no handwritten pages / still running / failed
]


class Location(BaseModel):
    mode: Literal["gps", "typed"]
    latitude: float | None = None
    longitude: float | None = None
    accuracy_m: float | None = None
    address: str | None = None

    @model_validator(mode="after")
    def _check(self):
        if self.mode == "gps" and (self.latitude is None or self.longitude is None):
            raise ValueError("GPS location needs latitude and longitude")
        if self.mode == "typed" and not (self.address and self.address.strip()):
            raise ValueError("Typed location needs an address")
        return self


class StoredFile(BaseModel):
    file_id: str
    filename: str
    content_type: str
    size_bytes: int
    uri: str  # served by GET /api/v1/files/{file_id}; object storage path later


class WordOut(BaseModel):
    id: str
    text: str  # what the OCR read
    final_text: str  # after accepted corrections
    confidence: float
    uncertain: bool
    bbox: list[int]  # x0, y0, x1, y1 in the page image


class PageTranscript(BaseModel):
    page_no: int  # 1-based across the whole complaint
    source_file_id: str
    source_page: int  # page inside that file (PDFs have several)
    image_uri: str  # deskewed page image the boxes refer to
    width: int
    height: int
    skew_deg: float
    layout: Literal["letter", "sectioned", "table", "empty"]
    separators: dict[str, list[int]]  # {"horizontal": [y..], "vertical": [x..]}
    raw_text: str
    final_text: str
    mean_confidence: float | None
    n_words: int
    n_uncertain: int
    words: list[WordOut]


class EditOut(BaseModel):
    word_id: str
    page_no: int
    from_text: str
    to_text: str
    evidence: str
    accepted: bool
    reasons: list[str] = []
    round: int


class RetrievedContext(BaseModel):
    id: str
    text: str
    score: float


class Handwritten(BaseModel):
    files: list[StoredFile]
    pages: list[PageTranscript] = []
    raw_transcript: str = ""
    final_transcript: str = ""
    ocr_engine: str | None = None
    llm_model: str | None = None
    verifier_decision: Decision = "NOT_PROCESSED"
    needs_human_review: bool = False
    review_reasons: list[str] = []
    edits: list[EditOut] = []
    llm_rounds: int = 0
    retrieved_context: list[RetrievedContext] = []
    timings_ms: dict[str, int] = {}


class ComplaintPackage(BaseModel):
    schema_version: str = SCHEMA_VERSION
    source_component: str = "C2-document-ocr"
    complaint_id: str
    status: Literal["queued", "processing", "done", "failed"] = "queued"
    error: str | None = None
    submitted_at: datetime
    complaint_date: date
    location: Location
    description: str | None = None
    layout_hint: Literal["auto", "letter", "table"] = "auto"
    evidence_images: list[StoredFile] = []
    handwritten: Handwritten | None = None
    forwarded: list[dict] = Field(default_factory=list)


class ComplaintSummary(BaseModel):
    complaint_id: str
    status: str
    submitted_at: datetime
    complaint_date: date
    n_pages: int
    decision: str | None
