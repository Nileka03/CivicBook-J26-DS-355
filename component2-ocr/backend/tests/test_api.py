import json
from types import SimpleNamespace

import fitz
import numpy as np
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.pipeline import llm_client, ocr_engines, retriever
from app.pipeline.ocr_engines import CRNNEngine
from conftest import inked_page, png_bytes, written_page

client = TestClient(app)
LOC = {"location_mode": "typed", "address": "57/2, Malabe Road, Kaduwela"}


def line(text, unsure=()):
    """(text, char confidences): 0.95 everywhere, 0.3 on the characters of the words in `unsure`."""
    confs, words = [], text.split(" ")
    for i, w in enumerate(words):
        confs += [0.3 if w in unsure else 0.95] * len(w) + ([0.95] if i < len(words) - 1 else [])
    return text, confs


class FakeCRNN(CRNNEngine):
    """Real segmentation and cropping; the network is replaced by a fixed list of lines, in call order."""
    def __init__(self, lines, default=("Please help", [0.95] * 11)):
        self.lines, self.default, self.crops = list(lines), default, []

    def recognize_line(self, crop):
        self.crops.append(crop.shape)
        return self.lines.pop(0) if self.lines else self.default


class FakeGemini:
    """Stands in for google-genai: client.models.generate_content(...).text, one reply per call."""
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []
        self.models = self

    def generate_content(self, model, contents, config):
        self.prompts.append(contents)
        r = self.replies.pop(0) if self.replies else {"edits": []}
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(text=json.dumps(r, ensure_ascii=False))


def use(crnn=None, llm=None):
    if crnn is not None:
        ocr_engines._cache["crnn"] = crnn
    if llm is not None:
        llm_client.set_client(llm)


PAGE1 = [line("Garbage not colected", unsure=["colected"]), line("near Malsbe", unsure=["Malsbe"])]
PAGE2 = [line("Please help"), line("Malabe")]


def submit(files, **form):
    data = {**LOC, **form}
    return client.post("/api/v1/complaints", data=data, files=files)


def hw(name, arr):
    return ("handwritten_pages", (name, png_bytes(arr), "image/png"))


def package(r):
    return client.get(f"/api/v1/complaints/{r.json()['complaint_id']}").json()


def test_health():
    h = client.get("/api/v1/health").json()
    assert h["status"] == "ok" and h["ocr_engine"] == "crnn" and h["correction_mode"] == "full"


def test_location_required():
    r = client.post("/api/v1/complaints", data={"location_mode": "typed", "description": "x"})
    assert r.status_code == 422


def test_gps_location_needs_coordinates():
    r = client.post("/api/v1/complaints", data={"location_mode": "gps", "description": "x"})
    assert r.status_code == 422


def test_needs_description_or_pages():
    assert submit([]).status_code == 422


def test_description_only_complaint_is_done_immediately():
    r = submit([("evidence_images", ("e.png", png_bytes(inked_page()), "image/png"))],
               description="Street lamp broken", complaint_date="2026-10-20")
    assert r.status_code == 202
    pkg = package(r)
    assert pkg["status"] == "done" and pkg["handwritten"] is None
    assert pkg["complaint_date"] == "2026-10-20"
    assert len(pkg["evidence_images"]) == 1
    assert client.get(pkg["evidence_images"][0]["uri"]).status_code == 200


def test_wrong_type_and_empty_file_rejected():
    assert submit([("handwritten_pages", ("a.txt", b"hi", "text/plain"))]).status_code == 415
    assert submit([("handwritten_pages", ("a.png", b"", "image/png"))]).status_code == 400
    assert submit([("handwritten_pages", ("a.pdf", b"not a pdf", "application/pdf"))]).status_code == 422


def test_two_pages_one_complaint_with_verified_correction():
    crnn = FakeCRNN(PAGE1 + PAGE2)
    llm = FakeGemini([{"edits": [
        {"id": "l1w1", "to": "Malabe", "evidence": "complaint"},      # supported by page 2
        {"id": "l0w2", "to": "collected", "evidence": "description"},  # supported by description
        {"id": "l0w0", "to": "Rubbish", "evidence": "spelling"},       # confident word -> R1
    ]}])
    use(crnn, llm)
    r = submit([hw("p1.png", written_page(2)), hw("p2.png", written_page(2))],
               description="garbage not collected for two weeks")
    pkg = package(r)
    h = pkg["handwritten"]
    assert pkg["status"] == "done", pkg["error"]
    assert h["ocr_engine"] == "crnn" and len(crnn.crops) == 4 and all(s[0] > 15 for s in crnn.crops)
    assert len(h["pages"]) == 2
    # the whole complaint (both pages, in line order) went to the LLM in one prompt
    assert "«l0w0" not in llm.prompts[0]  # confident words are not offered for editing
    assert "Garbage not «l0w2:colected»\nnear «l1w1:Malsbe»\nPlease help\nMalabe" in llm.prompts[0]
    assert "garbage not collected for two weeks" in llm.prompts[0]
    assert h["final_transcript"] == ("--- Page 1 of 2 ---\nGarbage not collected\nnear Malabe\n\n"
                                     "--- Page 2 of 2 ---\nPlease help\nMalabe")
    assert h["raw_transcript"].startswith("--- Page 1 of 2 ---\nGarbage not colected\nnear Malsbe")
    assert h["verifier_decision"] == "PARTIAL"
    # 1 of 3 edits rejected (> 30 %) -> one re-check round with feedback
    assert h["llm_rounds"] == 2 and "FEEDBACK" in llm.prompts[1] and "Rubbish" in llm.prompts[1]
    by_id = {e["word_id"]: e for e in h["edits"]}
    assert by_id["p1l0w0"]["accepted"] is False and "R1_not_uncertain" in by_id["p1l0w0"]["reasons"]
    assert by_id["p1l0w2"]["accepted"] and by_id["p1l1w1"]["to_text"] == "Malabe"
    assert all(e["page_no"] == 1 for e in h["edits"])
    # line items keep their box, OCR text, corrected text and confidence
    w = h["pages"][0]["words"][1]
    assert w["id"] == "p1l1" and w["text"] == "near Malsbe" and w["final_text"] == "near Malabe"
    assert w["uncertain"] and w["bbox"][3] - w["bbox"][1] > 15 and w["confidence"] < 0.95
    assert h["pages"][1]["n_uncertain"] == 0
    # forward to the mock Component 3
    f = client.post(f"/api/v1/complaints/{pkg['complaint_id']}/forward").json()
    assert f["status_code"] == 200 and f["response"]["received"] is True


def test_retrieval_and_lexicon_feed_the_prompt():
    docs = [{"id": "a1", "text": "garbage not collected near Malabe junction"},
            {"id": "b2", "text": "street lamp broken in Kaduwela"}]
    settings.PAST_COMPLAINTS_PATH.write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")
    settings.LEXICON_PATH.write_text("Malabe\nකාණුව\n", encoding="utf-8")
    retriever._retriever = None
    llm = FakeGemini([{"edits": [{"id": "l0w2", "to": "collected", "evidence": "reference"}]}])
    use(FakeCRNN([PAGE1[0]]), llm)
    h = package(submit([hw("p1.png", written_page(1))]))["handwritten"]
    assert "- garbage not collected near Malabe junction" in llm.prompts[0]
    assert "LEXICON: Malabe, කාණුව" in llm.prompts[0]
    assert h["retrieved_context"][0]["id"] == "a1"
    assert h["verifier_decision"] == "ACCEPT" and h["final_transcript"] == "Garbage not collected"


def test_table_page_rendered_as_cells():
    crnn = FakeCRNN([line("2023/04/29"), line("01."), line("lamp broken"), line("fixed")])
    use(crnn)
    page = written_page(2, columns=((60, 350), (430, 740)), rules=(390,), row_rules=(140,))
    p = package(submit([hw("t.png", page)]))["handwritten"]["pages"][0]
    assert p["layout"] == "table"
    # segment order is column by column; the layout puts each line item into its cell
    assert p["raw_text"] == "[2023/04/29] [lamp broken]\n[01.] [fixed]"
    assert len(crnn.crops) == 4  # the ruling is removed before cutting lines


def test_llm_error_falls_back_to_raw():
    use(FakeCRNN(PAGE1[:1]), FakeGemini([RuntimeError("quota")] * 4))
    h = package(submit([hw("p1.png", written_page(1))]))["handwritten"]
    assert h["verifier_decision"] == "FALLBACK_RAW" and h["needs_human_review"]
    assert h["final_transcript"] == h["raw_transcript"] == "Garbage not colected"
    assert any("quota" in r for r in h["review_reasons"])


def test_missing_api_key_falls_back_to_raw():
    use(FakeCRNN(PAGE1[:1]))
    h = package(submit([hw("p1.png", written_page(1))]))["handwritten"]
    assert h["verifier_decision"] == "FALLBACK_RAW"
    assert h["final_transcript"] == "Garbage not colected" and h["llm_model"] is None


def test_confident_ocr_needs_no_llm():
    llm = FakeGemini([])
    use(FakeCRNN(PAGE2), llm)
    h = package(submit([hw("p1.png", written_page(2))]))["handwritten"]
    assert h["verifier_decision"] == "NO_CHANGES" and llm.prompts == []
    assert h["final_transcript"] == "Please help\nMalabe"


def test_blind_rag_mode_for_comparison(monkeypatch):
    monkeypatch.setattr(settings, "CORRECTION_MODE", "blind_rag")
    llm = FakeGemini([{"lines": ["Garbage not collected", "near Malabe"]}])
    use(FakeCRNN(PAGE1), llm)
    h = package(submit([hw("p1.png", written_page(2))]))["handwritten"]
    assert "OCR TEXT:\nGarbage not colected\nnear Malsbe" in llm.prompts[0]
    assert h["final_transcript"] == "Garbage not collected\nnear Malabe"
    assert h["verifier_decision"] == "REVIEW" and h["edits"] == []
    assert any("blind_rag" in r for r in h["review_reasons"])


def test_blank_page_goes_to_review():
    use(FakeCRNN([]))
    pkg = package(submit([hw("blank.png", np.full((600, 800), 255, np.uint8))]))
    assert pkg["status"] == "done"
    assert pkg["handwritten"]["verifier_decision"] == "REVIEW"
    assert pkg["handwritten"]["pages"][0]["layout"] == "empty"


def test_pdf_pages_count_as_pages_of_one_complaint():
    use(FakeCRNN([]))  # every line reads "Please help", confidently
    doc = fitz.open()
    for _ in range(2):
        p = doc.new_page()
        for x in range(40, 550, 5):  # a row of pen strokes
            p.draw_line(fitz.Point(x, 60), fitz.Point(x + 2, 72), width=2)
    pdf = doc.tobytes()
    r = submit([("handwritten_pages", ("letter.pdf", pdf, "application/pdf")),
                hw("p3.png", written_page(1))])
    h = package(r)["handwritten"]
    assert [p["page_no"] for p in h["pages"]] == [1, 2, 3]
    assert [p["source_page"] for p in h["pages"]] == [1, 2, 1]
    assert all(p["n_words"] == 1 for p in h["pages"])
    assert h["verifier_decision"] == "NO_CHANGES"


def test_unknown_complaint_404():
    assert client.get("/api/v1/complaints/nope").status_code == 404
