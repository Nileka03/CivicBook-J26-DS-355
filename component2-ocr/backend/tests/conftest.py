import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_DIR", tmp_path / "storage")
    monkeypatch.setattr(settings, "PAST_COMPLAINTS_PATH", tmp_path / "past.jsonl")
    monkeypatch.setattr(settings, "LEXICON_PATH", tmp_path / "lexicon.txt")
    # the same settings whatever backend/.env says
    monkeypatch.setattr(settings, "OCR_ENGINE", "crnn")
    monkeypatch.setattr(settings, "OCR_FALLBACK_ENGINE", None)
    monkeypatch.setattr(settings, "CORRECTION_MODE", "full")
    monkeypatch.setattr(settings, "CORRECTION_TAU", 0.8)
    monkeypatch.setattr(settings, "VERIFIER_ENABLED", True)
    monkeypatch.setattr(settings, "RETRIEVAL_TOP_K", 20)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", None)
    from app.pipeline import c2_correction, llm_client, ocr_engines, retriever
    monkeypatch.setattr(c2_correction.time, "sleep", lambda s: None)  # rate-limit pauses
    retriever._retriever = None
    llm_client._client = None
    ocr_engines._cache.clear()
    yield
    ocr_engines._cache.clear()


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def inked_page(w=800, h=600) -> np.ndarray:
    """White page with a few dark blobs so it is not treated as blank."""
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    for y in (100, 200, 300):           # short "words", not page-wide bars
        for x in (100, 250, 400):
            d.rectangle([x, y, x + 100, y + 20], fill=0)
    return np.array(img)


def written_page(n_lines=2, w=800, h=600, columns=((60, 700),), rules=(), row_rules=()) -> np.ndarray:
    """Page with n_lines rows of pen-stroke "letters" per column; segment_page finds one line per row.
    rules / row_rules = x / y positions of drawn column / row lines (a table)."""
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    for x in rules:
        d.line([(x, 20), (x, h - 20)], fill=0, width=3)
    for y in row_rules:
        d.line([(20, y), (w - 20, y)], fill=0, width=3)
    for x0, x1 in columns:
        for i in range(n_lines):
            y = 80 + i * 90
            for x in range(x0, x1, 14):
                d.line([(x, y), (x + 4, y + 28)], fill=20, width=3)
    return np.array(img)
