"""OCR engines behind one interface.

Every engine turns one page into a PageOCR: the image its boxes refer to, the
skew that was removed, and a list of OCRLine items (text, one confidence per
character, box). Layout, uncertainty marking and correction only depend on that.

  crnn       our CNN + BiLSTM + CTC model. The page is cut into lines with
             segment_lines.segment_page (the exact cutting used for the training
             data) and every line crop is recognised on its own.
  google     Cloud Vision; every word becomes one item.
  tesseract  offline baseline; every word becomes one item.
"""
import json
import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn

from ..config import settings
from .intake import downscale, prepare_page
from .segment_lines import binarize as seg_binarize
from .segment_lines import segment_page

log = logging.getLogger(__name__)


@dataclass
class OCRWord:
    text: str
    conf: float
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass
class OCRLine:
    text: str
    char_confs: list[float]  # one per character of text (spaces included)
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass
class PageOCR:
    image: np.ndarray  # deskewed page; boxes and separators refer to it
    skew_deg: float
    lines: list[OCRLine]


class OCREngine:
    name = "base"

    def recognize(self, gray: np.ndarray) -> list[OCRWord]:  # pragma: no cover
        raise NotImplementedError

    def read_page(self, gray: np.ndarray) -> PageOCR:
        """Word engines: deskew, read words, one item per word (its confidence for every character)."""
        img, angle = prepare_page(gray)
        items = [OCRLine(w.text, [w.conf] * len(w.text), w.x0, w.y0, w.x1, w.y1)
                 for w in self.recognize(img) if w.text]
        return PageOCR(img, angle, items)


# ---------------------------------------------------------------------------
# CRNN (notebook 01). Architecture, preprocessing and decoding must match training.
# ---------------------------------------------------------------------------
CRNN_H, CRNN_MIN_W, CRNN_MAX_W = 64, 16, 1024


class CRNN(nn.Module):
    def __init__(self, n):
        super().__init__()
        def blk(i, o, pool):
            return [nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(True)] + ([nn.MaxPool2d(pool)] if pool else [])
        self.cnn = nn.Sequential(*blk(1, 64, (2, 2)), *blk(64, 128, (2, 2)), *blk(128, 256, None),
                                 *blk(256, 256, (2, 1)), *blk(256, 512, None), *blk(512, 512, (2, 1)), nn.Dropout2d(0.2))
        self.rnn = nn.LSTM(512 * 4, 256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(512, n)

    def forward(self, x):
        f = self.cnn(x); b, c, h, w = f.shape
        return self.fc(self.rnn(f.permute(0, 3, 1, 2).reshape(b, w, c * h))[0])


def crnn_input(crop: np.ndarray) -> torch.Tensor:
    """Grayscale line crop -> (1, 1, 64, W) tensor, ink = 1, paper = 0."""
    h, w = crop.shape
    img = cv2.resize(crop, (min(CRNN_MAX_W, max(CRNN_MIN_W, int(w * CRNN_H / h))), CRNN_H))
    return torch.from_numpy(1 - img.astype(np.float32) / 255)[None, None]


def greedy_decode(logits: torch.Tensor, charset: list[str]) -> tuple[str, list[float]]:
    """(T, classes) -> text and one confidence per output character. Class 0 = CTC blank."""
    conf, best = logits.softmax(-1).max(-1); out, confs, prev = [], [], 0
    for k, c in zip(best.tolist(), conf.tolist()):
        if k != prev and k != 0:
            out.append(charset[k - 1]); confs.append(round(c, 3))
        prev = k
    return "".join(out), confs


def segment_skew(gray: np.ndarray) -> float:
    """The angle segment_lines.deskew rotates by (it only returns the image), for the package."""
    h, w = gray.shape
    segs = cv2.HoughLinesP(seg_binarize(gray), 1, np.pi / 720, 150, minLineLength=int(.2 * w), maxLineGap=25)
    if segs is None:
        return 0.0
    a = [math.degrees(math.atan2(y2 - y1, x2 - x1)) for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4)]
    a = [x for x in a if abs(x) < 10]
    return float(np.median(a)) if a else 0.0


class CRNNEngine(OCREngine):
    name = "crnn"

    def __init__(self, model_path=None, charset_path=None):
        model_path = model_path or settings.CRNN_MODEL_PATH
        charset_path = charset_path or settings.CRNN_CHARSET_PATH
        with open(charset_path, encoding="utf-8") as f:
            self.charset: list[str] = json.load(f)
        self.model = CRNN(len(self.charset) + 1)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        self.model.eval()
        log.info("CRNN loaded from %s (%d classes)", model_path, len(self.charset) + 1)

    def recognize_line(self, crop: np.ndarray) -> tuple[str, list[float]]:
        with torch.inference_mode():
            return greedy_decode(self.model(crnn_input(crop))[0], self.charset)

    def read_page(self, gray: np.ndarray) -> PageOCR:
        gray = downscale(gray)  # no-op for 200 DPI pages, like the training pages
        g, boxes = segment_page(gray)
        items = []
        for _, (x0, y0, x1, y1) in boxes:  # reading order: column, then line
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            text, confs = self.recognize_line(g[y0:y1, x0:x1])
            if text.strip():
                items.append(OCRLine(text, confs, x0, y0, x1, y1))
        return PageOCR(g, segment_skew(gray), items)


class GoogleVisionEngine(OCREngine):
    """Google Cloud Vision DOCUMENT_TEXT_DETECTION (handwriting capable).

    Auth: set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON key.
    """
    name = "google-vision"

    def __init__(self):
        from google.cloud import vision  # imported lazily: optional dependency
        self._vision = vision
        self._client = vision.ImageAnnotatorClient()

    def recognize(self, gray: np.ndarray) -> list[OCRWord]:
        ok, buf = cv2.imencode(".png", gray)
        image = self._vision.Image(content=buf.tobytes())
        resp = self._client.document_text_detection(
            image=image, image_context={"language_hints": ["si", "en"]}
        )
        if resp.error.message:
            raise RuntimeError(f"Cloud Vision error: {resp.error.message}")
        words: list[OCRWord] = []
        for page in resp.full_text_annotation.pages:
            for block in page.blocks:
                for para in block.paragraphs:
                    for w in para.words:
                        text = "".join(s.text for s in w.symbols).strip()
                        if not text:
                            continue
                        xs = [v.x for v in w.bounding_box.vertices]
                        ys = [v.y for v in w.bounding_box.vertices]
                        words.append(OCRWord(text, float(w.confidence),
                                             min(xs), min(ys), max(xs), max(ys)))
        return words


class TesseractEngine(OCREngine):
    """Offline engine. Weak on handwriting, but lets the whole pipeline run
    without internet or a Google account (development, demos, tests)."""
    name = "tesseract"

    def __init__(self):
        import pytesseract
        if settings.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
        self._tess = pytesseract

    def recognize(self, gray: np.ndarray) -> list[OCRWord]:
        d = self._tess.image_to_data(
            gray, lang=settings.TESSERACT_LANG, config="--psm 11",
            output_type=self._tess.Output.DICT,
        )
        words = []
        for i, text in enumerate(d["text"]):
            text = (text or "").strip()
            conf = float(d["conf"][i])
            if not text or conf < 0:
                continue
            x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            words.append(OCRWord(text, conf / 100.0, x, y, x + w, y + h))
        return words


_ENGINES = {"crnn": CRNNEngine, "google": GoogleVisionEngine, "tesseract": TesseractEngine}
_cache: dict[str, OCREngine] = {}


def get_engine(name: str) -> OCREngine:
    if name not in _ENGINES:
        raise ValueError(f"Unknown OCR engine '{name}'")
    if name not in _cache:
        _cache[name] = _ENGINES[name]()
    return _cache[name]


def recognize_with_fallback(gray: np.ndarray) -> tuple[PageOCR, str, str | None]:
    """Try the configured engine, then the fallback. Returns (page, engine_name, warning)."""
    try:
        eng = get_engine(settings.OCR_ENGINE)
        return eng.read_page(gray), eng.name, None
    except Exception as exc:
        log.warning("Primary OCR engine failed: %s", exc)
        if not settings.OCR_FALLBACK_ENGINE or settings.OCR_FALLBACK_ENGINE == settings.OCR_ENGINE:
            raise
        eng = get_engine(settings.OCR_FALLBACK_ENGINE)
        return eng.read_page(gray), eng.name, f"primary OCR failed ({exc}); used {eng.name}"
