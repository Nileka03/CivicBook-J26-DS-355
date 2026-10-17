"""Checks the confidence-gate logic against the six cases of the pipeline diagram.
Run:  python -m pytest tests   (or:  python tests/test_gates.py)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline import department_gate, institution_gate, tags_to_spans, tokenize  # noqa: E402

T = 0.5


def test_institution_cases():
    both = {"LECO": 0.9, "MC": 0.7}
    one = {"LECO": 0.9, "MC": 0.2}
    none = {"LECO": 0.3, "MC": 0.4}
    expect = [  # scores, is_fake, case, institution, continue, needs_verification
        (both, False, 1, "LECO", True, None),
        (both, True, 2, "LECO", False, None),
        (one, False, 3, "LECO", True, None),
        (one, True, 4, "LECO", False, None),
        (none, False, 5, "MC", False, "institution"),
        (none, True, 6, "MC", False, "institution"),
    ]
    for scores, fake, case, inst, cont, nv in expect:
        g = institution_gate(scores, T, fake)
        assert (g["case"], g["institution"], g["continue_to_department"], g["needs_verification"]) == \
               (case, inst, cont, nv), (scores, fake, g)


def test_department_gate():
    assert department_gate({"A": 0.8, "B": 0.6}, T) == {
        **department_gate({"A": 0.8, "B": 0.6}, T), "department": "A", "needs_verification": None}
    g = department_gate({"A": 0.2, "B": 0.4}, T)
    assert g["department"] == "B" and g["needs_verification"] == "department"


def test_spans_and_tokens():
    assert tags_to_spans(["B", "I", "O", "O", "B", "I"]) == [(0, 2), (4, 6)]
    assert tags_to_spans(["I", "I", "B"]) == [(0, 2), (2, 3)]
    toks, offs = tokenize("බිල වැඩියි. Also, (call) me")
    assert toks == ["බිල", "වැඩියි", ".", "Also", ",", "(", "call", ")", "me"]
    text = "බිල වැඩියි. Also, (call) me"
    assert all(text[a:b] == t for t, (a, b) in zip(toks, offs))


if __name__ == "__main__":
    test_institution_cases(); test_department_gate(); test_spans_and_tokens()
    print("All gate tests passed.")
