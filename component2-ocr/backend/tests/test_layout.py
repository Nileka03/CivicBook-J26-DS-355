import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.pipeline.layout import Word, build_layout, detect_separators, render


def W(i, text, x0, y0, x1=None, y1=None, conf=0.9):
    return Word(f"p1w{i}", 1, text, conf, x0, y0, x1 or x0 + 60, y1 or y0 + 30)


def test_letter_is_line_by_line():
    words = [W(0, "dear", 10, 10), W(1, "sir", 100, 12), W(2, "drain", 10, 80)]
    lay = build_layout(words, [], [])
    assert lay.kind == "letter"
    assert render(lay) == "dear sir\ndrain"


def test_table_row_uses_brackets_per_cell():
    words = [W(0, "2023/04/29", 10, 10), W(1, "01.", 210, 10),
             W(2, "lamp", 410, 10), W(3, "broken", 480, 10), W(4, "fixed", 610, 10)]
    lay = build_layout(words, [], [200, 400, 600])
    assert lay.kind == "table"
    assert render(lay) == "[2023/04/29] [01.] [lamp broken] [fixed]"


def test_multiline_cell_joined_in_one_bracket():
    words = [W(0, "name", 10, 10), W(1, "address", 10, 60), W(2, "complaint", 300, 10)]
    lay = build_layout(words, [], [250])
    assert render(lay) == "[name address] [complaint]"


def test_empty_table_cell_kept_as_empty_brackets():
    words = [W(0, "a", 10, 10), W(1, "c", 410, 10), W(2, "b", 210, 150)]
    lay = build_layout(words, [100], [200, 400])
    assert render(lay) == "[a] [] [c]\n[] [b] []"


def test_sectioned_letter_brackets_each_section():
    words = [W(0, "To", 10, 10), W(1, "Chairman", 10, 60), W(2, "Garbage", 10, 200)]
    lay = build_layout(words, [150], [])
    assert lay.kind == "sectioned"
    assert render(lay) == "[To\nChairman]\n[Garbage]"


def test_column_without_text_is_dropped():
    words = [W(0, "only", 300, 10)]
    lay = build_layout(words, [], [50])  # margin line at the left edge
    assert lay.kind == "letter"
    assert render(lay) == "only"


def test_letter_hint_ignores_separators():
    words = [W(0, "a", 10, 10), W(1, "b", 300, 10)]
    assert render(build_layout(words, [], [200], hint="letter")) == "a b"


def test_empty_page():
    assert render(build_layout([], [], [])) == ""


def test_detects_table_lines_and_ignores_light_ruling():
    img = Image.new("L", (1000, 800), 255)
    d = ImageDraw.Draw(img)
    for y in range(80, 760, 60):          # light, thin ruled-paper lines
        d.line([(40, y), (960, y)], fill=170, width=1)
    d.line([(40, 380), (960, 380)], fill=20, width=4)   # separator drawn over a ruling line
    for x in (300, 600):                 # drawn column lines
        d.line([(x, 40), (x, 760)], fill=20, width=3)
    for y in range(80, 700, 60):         # some "text" between the ruling lines
        d.rectangle([60, y + 15, 250, y + 40], fill=40)
    h, v = detect_separators(np.array(img))
    assert len(h) == 1 and abs(h[0] - 380) < 6
    assert [abs(a - b) < 6 for a, b in zip(v, (300, 600))] == [True, True]


def _font():
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf",
                 "/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(path, 34)
        except OSError:
            continue
    return None


def test_rendered_table_end_to_end_with_tesseract():
    """Real OCR on a printed (not handwritten) table: checks geometry + format."""
    import shutil
    import pytest
    from app.config import settings
    from app.pipeline.ocr_engines import TesseractEngine
    font = _font()
    if font is None or not (settings.TESSERACT_CMD or shutil.which("tesseract")):
        pytest.skip("needs Tesseract and a TrueType font")
    img = Image.new("L", (1400, 260), 255)
    d = ImageDraw.Draw(img)
    for x in (380, 760):
        d.line([(x, 10), (x, 250)], fill=0, width=3)
    d.text((30, 60), "2023/05/02", font=font, fill=0)
    d.text((410, 60), "street lamp", font=font, fill=0)
    d.text((410, 130), "not working", font=font, fill=0)
    d.text((790, 60), "repaired", font=font, fill=0)
    gray = np.array(img)
    h, v = detect_separators(gray)
    eng = TesseractEngine()
    import app.config as cfg
    old = cfg.settings.TESSERACT_LANG
    cfg.settings.TESSERACT_LANG = "eng"
    try:
        ocr = eng.recognize(gray)
    finally:
        cfg.settings.TESSERACT_LANG = old
    words = [Word(f"p1w{i}", 1, o.text, o.conf, o.x0, o.y0, o.x1, o.y1) for i, o in enumerate(ocr)]
    out = render(build_layout(words, h, v))
    assert out == "[2023/05/02] [street lamp not working] [repaired]"
