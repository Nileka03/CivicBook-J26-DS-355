"""Central configuration. Values come from environment variables or backend/.env."""
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    # --- OCR ---------------------------------------------------------------
    # "crnn" = our trained CNN + BiLSTM + CTC line recogniser (models/crnn_best.pt, CPU)
    # "google" = Google Cloud Vision (needs GOOGLE_APPLICATION_CREDENTIALS)
    # "tesseract" = offline baseline (needs Tesseract + sin.traineddata installed)
    OCR_ENGINE: str = "crnn"
    OCR_FALLBACK_ENGINE: str | None = None  # empty = fail instead of silently using another engine
    CRNN_MODEL_PATH: Path = BACKEND_DIR / "models" / "crnn_best.pt"
    CRNN_CHARSET_PATH: Path = BACKEND_DIR / "models" / "charset.json"
    TESSERACT_LANG: str = "sin+eng"
    TESSERACT_CMD: str | None = None  # e.g. C:\Program Files\Tesseract-OCR\tesseract.exe
    MAX_IMAGE_SIDE: int = 2400  # pages are downscaled to this before OCR
    PDF_DPI: int = 200  # the CRNN training pages were rendered at 200 DPI

    # --- LLM correction (app/pipeline/c2_correction.py) -------------------------
    GEMINI_API_KEY: str | None = None
    LLM_MODEL: str = "gemini-2.5-flash"
    LLM_TIMEOUT_S: int = 60
    LLM_ROUND_PAUSE_S: float = 1  # pause after each correct_page LLM round (rate limit)
    # "full" = uncertainty-scoped edit list + verifier (correct_page)
    # "blind_rag" = LLM rewrites every line with the retrieved references (correct_blind), comparison only
    CORRECTION_MODE: Literal["full", "blind_rag"] = "full"
    CORRECTION_TAU: float = 0.8  # a word is uncertain if its lowest CRNN character confidence is below this
    VERIFIER_ENABLED: bool = True  # False = accept every in-scope edit (ablation only)

    # --- Retrieval (past complaints memory) ----------------------------------
    PAST_COMPLAINTS_PATH: Path = BACKEND_DIR / "data" / "past_complaints.jsonl"
    LEXICON_PATH: Path = BACKEND_DIR / "data" / "lexicon_si.txt"
    RETRIEVAL_TOP_K: int = 20

    # --- Storage / limits ---------------------------------------------------
    STORAGE_DIR: Path = BACKEND_DIR / "storage"
    MAX_FILE_MB: int = 15
    MAX_HANDWRITTEN_FILES: int = 10
    MAX_HANDWRITTEN_PAGES: int = 20
    MAX_EVIDENCE_IMAGES: int = 10

    # --- Integration ---------------------------------------------------------
    # Where /forward sends the package. Empty = this service's own mock endpoint.
    COMPONENT3_URL: str | None = None
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"


settings = Settings()
