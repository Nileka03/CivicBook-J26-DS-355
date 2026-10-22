"""Layout-aware transcript.

Rules (agreed format):
  * table   -> one output line per table row, each cell in brackets:
               [2023/04/29 02/2023] [01.] [name, address, phone] [complaint] [action]
  * letter  -> plain text, line by line, in reading order
  * sectioned (letter split by drawn separator lines) -> each section in brackets,
               keeping its own line breaks:
               [Section one line 1
               Section one line 2]
               [Section two ...]

The page is first cut into horizontal bands by detected horizontal separators,
and into columns by detected vertical separators. Words are placed by their
centre point. Inside a cell/section, words are grouped into text lines by
vertical position and ordered left to right.

Separator detection is deliberately conservative on ruled paper: printed
ruling lines are evenly spaced, thin and comparatively light, so a horizontal
line only counts as a separator when it is clearly darker AND thicker than the
page's typical ruling line.
"""
from dataclasses import dataclass, field
from statistics import median
from typing import Callable

import cv2
import numpy as np

from .intake import binarize


@dataclass
class Word:
    id: str
    page_no: int
    text: str
    conf: float
    x0: int
    y0: int
    x1: int
    y1: int
    uncertain: bool = False

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def h(self) -> int:
        return max(1, self.y1 - self.y0)


Line = list[Word]
Cell = list[Line]


@dataclass
class PageLayout:
    kind: str  # letter | sectioned | table | empty
    bands: list[list[Cell]] = field(default_factory=list)  # bands -> cells -> lines -> words
    h_seps: list[int] = field(default_factory=list)
    v_seps: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# separator detection
# ---------------------------------------------------------------------------
def _content_x_range(b: np.ndarray) -> tuple[int, int]:
    xs = np.where(b > 0)[1]
    if xs.size == 0:
        return 0, b.shape[1]
    x0, x1 = np.percentile(xs, [2, 98]).astype(int)
    return int(x0), int(max(x1, x0 + 1))


def _content_y_range(b: np.ndarray) -> tuple[int, int]:
    ys = np.where(b > 0)[0]
    if ys.size == 0:
        return 0, b.shape[0]
    y0, y1 = np.percentile(ys, [1, 99]).astype(int)
    return int(y0), int(max(y1, y0 + 1))


def _horizontal_candidates(gray, b, x0, x1, min_cov=0.25):
    seg = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (41, 1)))
    cov = (seg[:, x0:x1] > 0).mean(1)
    h = gray.shape[0]
    out, y = [], 0
    while y < h:
        if cov[y] > min_cov:
            s = y
            while y < h and cov[y] > min_cov * 0.4:
                y += 1
            m = seg[s:y, x0:x1] > 0
            col = m.sum(0)
            out.append({
                "y": (s + y) // 2,
                "cov": float(cov[s:y].max()),
                "dark": float(gray[s:y, x0:x1][m].mean()) if m.any() else 255.0,
                "thick": float(np.median(col[col > 0])) if (col > 0).any() else 0.0,
            })
        y += 1
    return out


def _vertical_candidates(gray, b, y0, y1, min_cov=0.5):
    seg = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 41)))
    cov = (seg[y0:y1, :] > 0).mean(0)
    w = gray.shape[1]
    out, x = [], 0
    while x < w:
        if cov[x] > min_cov:
            s = x
            while x < w and cov[x] > min_cov * 0.4:
                x += 1
            out.append({"x": (s + x) // 2, "cov": float(cov[s:x].max())})
        x += 1
    return out


def detect_separators(gray: np.ndarray) -> tuple[list[int], list[int]]:
    """Return (horizontal y positions, vertical x positions) of drawn separators."""
    b = binarize(gray)
    h, w = gray.shape
    x0, x1 = _content_x_range(b)
    y0, y1 = _content_y_range(b)

    # -- horizontal -------------------------------------------------------
    max_thick = max(8, 0.008 * h)  # a pen line, not a filled bar or a block of text
    cands = [c for c in _horizontal_candidates(gray, b, x0, x1)
             if 0.07 * h < c["y"] < 0.93 * h  # page/crop borders are not separators
             and c["thick"] <= max_thick]
    if len(cands) >= 4:
        # ruled paper: keep only lines clearly darker AND thicker than the ruling
        med_dark = median(c["dark"] for c in cands)
        med_thick = median(c["thick"] for c in cands)
        h_seps = [c["y"] for c in cands
                  if c["dark"] <= med_dark - 25 and c["thick"] >= med_thick + 1]
    else:
        h_seps = [c["y"] for c in cands if c["cov"] >= 0.4]

    # -- vertical -----------------------------------------------------------
    v_seps = [c["x"] for c in _vertical_candidates(gray, b, y0, y1)
              if 0.03 * w < c["x"] < 0.97 * w]
    return sorted(h_seps), sorted(v_seps)


# ---------------------------------------------------------------------------
# grouping words
# ---------------------------------------------------------------------------
def group_lines(words: list[Word]) -> list[Line]:
    """Cluster words into text lines by vertical centre, then order left to right."""
    if not words:
        return []
    tol = 0.5 * median(w.h for w in words)
    lines: list[dict] = []
    for w in sorted(words, key=lambda w: w.cy):
        best = None
        for ln in lines:
            if abs(w.cy - ln["cy"]) <= tol and (best is None or abs(w.cy - ln["cy"]) < abs(w.cy - best["cy"])):
                best = ln
        if best is None:
            lines.append({"cy": w.cy, "words": [w]})
        else:
            best["words"].append(w)
            best["cy"] = sum(x.cy for x in best["words"]) / len(best["words"])
    lines.sort(key=lambda ln: ln["cy"])
    return [sorted(ln["words"], key=lambda w: w.x0) for ln in lines]


def _bucket(value: float, cuts: list[int]) -> int:
    i = 0
    while i < len(cuts) and value >= cuts[i]:
        i += 1
    return i


def build_layout(words: list[Word], h_seps: list[int], v_seps: list[int],
                 hint: str = "auto") -> PageLayout:
    if not words:
        return PageLayout("empty", [], h_seps, v_seps)
    if hint == "letter":
        h_seps, v_seps = [], []
    # a line through the middle of a word is a strike-through/underline, not a separator
    h_seps = [y for y in h_seps
              if not any(w.y0 + 0.3 * w.h < y < w.y1 - 0.3 * w.h for w in words)]
    v_seps = [x for x in v_seps
              if not any(w.x0 + 0.3 * (w.x1 - w.x0) < x < w.x1 - 0.3 * (w.x1 - w.x0) for w in words)]

    n_bands, n_cols = len(h_seps) + 1, len(v_seps) + 1
    grid: list[list[list[Word]]] = [[[] for _ in range(n_cols)] for _ in range(n_bands)]
    for w in words:
        grid[_bucket(w.cy, h_seps)][_bucket(w.cx, v_seps)].append(w)

    # drop columns that never contain text (e.g. a printed margin line at the edge)
    keep_cols = [c for c in range(n_cols) if any(grid[b][c] for b in range(n_bands))]
    # drop bands with no text at all
    keep_bands = [b for b in range(n_bands) if any(grid[b][c] for c in keep_cols)]
    bands = [[group_lines(grid[b][c]) for c in keep_cols] for b in keep_bands]

    if len(keep_cols) > 1 or hint == "table":
        kind = "table"
    elif len(bands) > 1:
        kind = "sectioned"
    else:
        kind = "letter"
    return PageLayout(kind, bands, h_seps, v_seps)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
TextOf = Callable[[Word], str]


def _line_text(line: Line, text_of: TextOf) -> str:
    return " ".join(t for t in (text_of(w) for w in line) if t)


def render(layout: PageLayout, text_of: TextOf | None = None) -> str:
    text_of = text_of or (lambda w: w.text)
    if layout.kind == "empty":
        return ""
    if layout.kind == "table":
        rows = []
        for band in layout.bands:
            cells = [" ".join(_line_text(ln, text_of) for ln in cell) for cell in band]
            rows.append(" ".join(f"[{c}]" for c in cells))
        return "\n".join(rows)
    if layout.kind == "sectioned":
        return "\n".join(
            "[" + "\n".join(_line_text(ln, text_of) for ln in band[0]) + "]"
            for band in layout.bands
        )
    return "\n".join(_line_text(ln, text_of) for ln in layout.bands[0][0])
