"""Turn uploaded files into clean, upright grayscale page images.

One complaint may contain several files and a PDF may contain several pages;
every page becomes one numpy array (uint8, grayscale) in upload order.
"""
import io
import math

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageOps

from ..config import settings


def load_pages(data: bytes, content_type: str) -> list[np.ndarray]:
    """Return grayscale pages. Raises ValueError for unreadable files."""
    if content_type == "application/pdf":
        try:
            doc = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:  # corrupt or not a PDF
            raise ValueError(f"Could not open PDF: {exc}") from exc
        pages = []
        for page in doc:
            pix = page.get_pixmap(dpi=settings.PDF_DPI, colorspace=fitz.csGRAY)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
            pages.append(arr.copy())
        if not pages:
            raise ValueError("PDF has no pages")
        return pages
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)  # phone photos carry their rotation in EXIF
        return [np.array(img.convert("L"))]
    except Exception as exc:
        raise ValueError(f"Could not read image: {exc}") from exc


def downscale(gray: np.ndarray, max_side: int | None = None) -> np.ndarray:
    max_side = max_side or settings.MAX_IMAGE_SIDE
    h, w = gray.shape
    s = max_side / max(h, w)
    if s >= 1:
        return gray
    return cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Ink = 255, paper = 0. Adaptive so shadows and uneven light do not matter."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 12
    )


def estimate_skew(gray: np.ndarray, max_angle: float = 10.0) -> float:
    """Skew angle in degrees from long near-horizontal strokes (ruling, separators, text baselines).

    Uses the length-weighted median angle of Hough segments, so a few stray
    diagonal strokes cannot dominate.
    """
    b = binarize(gray)
    h, w = gray.shape
    segs = cv2.HoughLinesP(b, 1, np.pi / 720, threshold=150,
                           minLineLength=int(0.2 * w), maxLineGap=25)
    if segs is None:
        return 0.0
    angles, weights = [], []
    for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4):
        a = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(a) < max_angle:
            angles.append(a)
            weights.append(math.hypot(x2 - x1, y2 - y1))
    if not angles:
        return 0.0
    order = np.argsort(angles)
    cum = np.cumsum(np.asarray(weights)[order])
    return float(np.asarray(angles)[order][np.searchsorted(cum, cum[-1] / 2)])


def rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.1:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)


def prepare_page(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Downscale + deskew. OCR and layout both run on the returned image,
    so word boxes and separator positions share one coordinate system."""
    small = downscale(gray)
    angle = estimate_skew(small)
    return rotate(small, angle), angle


def is_blank(gray: np.ndarray) -> bool:
    return (binarize(gray) > 0).mean() < 0.002
