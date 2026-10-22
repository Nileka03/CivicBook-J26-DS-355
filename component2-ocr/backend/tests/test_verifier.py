"""The verifier gate of c2_correction (rules R1-R4) and its decision values, with a fake LLM."""
from app.pipeline import c2_correction as C
from test_api import FakeGemini, line

LINES = [line("කසළ කාණව", unsure=["කාණව"]), line("0771234567", unsure=["0771234567"]),
         line("Malabe Malsbe", unsure=["Malsbe"])]
REFS = ["කාණුව අවහිර වී ඇත"]
LEX = ["කාණුව"]


def run(edits, **kw):
    llm = FakeGemini([{"edits": edits}] + kw.pop("more", []))
    return C.correct_page(LINES, llm, "fake", REFS, LEX, tau=0.8, **kw), llm


def reasons(out, wid):
    return next(e["reasons"] for e in out["edits"] if e["id"] == wid)


def test_supported_small_fix_accepted():
    out, _ = run([{"id": "l0w1", "to": "කාණුව", "evidence": "lexicon"}])
    assert out["decision"] == "ACCEPT" and out["final"].split("\n")[0] == "කසළ කාණුව"


def test_other_page_word_is_evidence():
    out, _ = run([{"id": "l2w1", "to": "Malabe", "evidence": "complaint"}])
    assert out["decision"] == "ACCEPT"


def test_confident_word_cannot_be_edited():
    out, _ = run([{"id": "l0w0", "to": "කසල"}])
    assert "R1_not_uncertain" in reasons(out, "l0w0")


def test_unknown_id_rejected():
    out, _ = run([{"id": "l9w9", "to": "x"}])
    assert reasons(out, "l9w9") == ["R1_unknown_id"]


def test_unsupported_word_rejected():
    out, _ = run([{"id": "l2w1", "to": "Mabole"}])
    assert "R3_unsupported" in reasons(out, "l2w1")


def test_wholesale_replacement_rejected():
    out, _ = run([{"id": "l0w1", "to": "අවහිර"}])
    assert "R2_too_different" in reasons(out, "l0w1")


def test_digits_cannot_change_without_evidence():
    out, _ = run([{"id": "l1w0", "to": "0771234568"}])
    assert "R4_digits_changed" in reasons(out, "l1w0")


def test_rejections_trigger_a_recheck_round_then_review():
    out, llm = run([{"id": "l2w1", "to": "Mabole"}], more=[{"edits": [{"id": "l2w1", "to": "Mabole"}]}])
    assert out["rounds"] == 3 and "FEEDBACK" in llm.prompts[1]  # 3rd round: LLM drops the edit
    assert out["decision"] == "REVIEW" and out["final"] == out["raw"]


def test_no_uncertain_words_means_no_llm_call():
    out = C.correct_page([line("Please help")], None, "fake", [], [], tau=0.8)
    assert out["decision"] == "NO_CHANGES"
