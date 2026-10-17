"""
Component 4 pipeline: pre-processing -> tokens -> BIO tags -> sub-complaints ->
Gate 1 (institution x is_fake) -> Gate 2 (department).

The gate logic follows the Component 4 internal pipeline diagram (cases 1-6).
"""

import re
import time
import unicodedata
import uuid

import torch

PEEL = set(".,!?;:\"'()[]{}…“”‘’|෴")

# ------------------------------------------------------------------ pre-processing
EMAIL_RE = re.compile(r"<?[\w.+-]+@[\w-]+(\.[\w-]+)+>?")
MASKED_RE = re.compile(r"(?<![\d*])\+?(?=[\d*]*\d)(?=[\d*]*\*{2})[\d*]{4,}(?![\d*])")
PLAIN_NUMBER_RE = re.compile(r"(?<![\w/.])\+?\d{2,}(?:[ -]\d{2,})*(?![\w/])")


def normalise(text: str) -> str:
    """Unicode NFC and single spaces (line breaks become spaces), as in training."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def mask_identifiers(text: str) -> str:
    """Emails -> <EMAIL>; masked numbers and numbers with 7+ digits -> <NUMBER> (as in training)."""
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = MASKED_RE.sub("<NUMBER>", text)
    return PLAIN_NUMBER_RE.sub(
        lambda m: "<NUMBER>" if sum(c.isdigit() for c in m.group()) >= 7 else m.group(), text)


def tokenize(text: str):
    """Same word tokeniser as the dataset. Returns tokens and their (start, end) in text."""
    tokens, offsets = [], []
    for m in re.finditer(r"\S+", text):
        s, e = m.start(), m.end()
        lead, trail = [], []
        while s < e and text[s] in PEEL:
            lead.append((s, s + 1)); s += 1
        while e > s and text[e - 1] in PEEL:
            trail.insert(0, (e - 1, e)); e -= 1
        spans = lead + ([(s, e)] if s < e else []) + trail
        for a, b in spans:
            tokens.append(text[a:b]); offsets.append((a, b))
    return tokens, offsets


def tags_to_spans(tags):
    """B starts a sub-complaint, I continues it; an I after O also starts one."""
    spans, start = [], None
    for i, t in enumerate(tags):
        if t == "B":
            if start is not None:
                spans.append((start, i))
            start = i
        elif t == "I":
            if start is None:
                start = i
        else:
            if start is not None:
                spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(tags)))
    return spans


# ------------------------------------------------------------------ gates (pure functions)
CASE_TEXT = {
    1: "All institution scores ≥ t, not fake → route to the higher-scoring institution and continue to the department head.",
    2: "All institution scores ≥ t, is_fake → route to the higher-scoring institution and STOP; the officer checks real/fake.",
    3: "One institution score ≥ t, not fake → route to that institution and continue to the department head.",
    4: "One institution score ≥ t, is_fake → route to that institution and STOP; the officer checks real/fake.",
    5: "No institution score ≥ t, not fake → route to the higher-scoring institution anyway, flag needs_verification (institution) and STOP.",
    6: "No institution score ≥ t, is_fake → route to the higher-scoring institution anyway, flag needs_verification and is_fake, and STOP.",
}


def institution_gate(scores: dict, t: float, is_fake: bool):
    passing = [k for k, v in scores.items() if v >= t]
    best = max(scores, key=scores.get)
    if len(passing) == len(scores):
        condition = "all"
    elif passing:
        condition = "some"
    else:
        condition = "none"
    chosen = max(passing, key=scores.get) if passing else best
    case = {("all", False): 1, ("all", True): 2, ("some", False): 3,
            ("some", True): 4, ("none", False): 5, ("none", True): 6}[(condition, is_fake)]
    return {"institution": chosen, "case": case, "condition": condition,
            "continue_to_department": case in (1, 3),
            "needs_verification": "institution" if case in (5, 6) else None,
            "explanation": CASE_TEXT[case]}


def department_gate(scores: dict, t: float):
    passing = [k for k, v in scores.items() if v >= t]
    best = max(scores, key=scores.get)
    chosen = max(passing, key=scores.get) if passing else best
    flagged = not passing
    return {"department": chosen, "needs_verification": "department" if flagged else None,
            "explanation": ("No department score ≥ t → route to the highest-scoring department "
                            "anyway and flag needs_verification (department)." if flagged else
                            "Route to the highest-scoring department among those ≥ t.")}


# ------------------------------------------------------------------ engine
class RoutingEngine:
    def __init__(self, model, tokenizer, cfg):
        self.model, self.tokenizer, self.cfg = model, tokenizer, cfg
        self.institutions = cfg["institutions"]
        self.departments = cfg["departments"]
        self.tags = cfg["tags"]
        self.max_len = cfg.get("max_len", 512)
        self.t_inst = float(cfg.get("threshold_institution", 0.5))
        self.t_dept = float(cfg.get("threshold_department", 0.5))

    @torch.no_grad()
    def _run_model(self, tokens):
        enc = self.tokenizer(tokens, is_split_into_words=True, truncation=True,
                             max_length=self.max_len, return_tensors="pt")
        word_ids = enc.word_ids(0)
        hidden = self.model.encoder(input_ids=enc["input_ids"],
                                    attention_mask=enc["attention_mask"]).last_hidden_state[0]
        bio = self.model.bio_head(hidden).argmax(-1).tolist()
        first, last = {}, {}
        for pos, w in enumerate(word_ids):
            if w is None:
                continue
            first.setdefault(w, pos)
            last[w] = pos
        tags = ["O"] * len(tokens)
        for w, pos in first.items():
            tags[w] = self.tags[bio[pos]]
        truncated = len(first) < len(tokens)
        return hidden, tags, first, last, truncated

    @torch.no_grad()
    def _scores(self, hidden, first, last, span):
        ws = [w for w in range(span[0], span[1]) if w in first]
        if not ws:
            return None
        rep = hidden[first[ws[0]]: last[ws[-1]] + 1].mean(0, keepdim=True)
        inst = torch.sigmoid(self.model.inst_head(rep))[0].tolist()
        dept = {i: torch.sigmoid(self.model.dept_heads[i](rep))[0].tolist() for i in self.institutions}
        return ({i: round(inst[k], 4) for k, i in enumerate(self.institutions)},
                {i: {d: round(dept[i][k], 4) for k, d in enumerate(self.departments[i])}
                 for i in self.institutions})

    def route(self, complaint_text: str, is_fake: bool, complaint_id: str | None = None,
              t_inst: float | None = None, t_dept: float | None = None):
        t0 = time.perf_counter()
        t_inst = self.t_inst if t_inst is None else t_inst
        t_dept = self.t_dept if t_dept is None else t_dept
        complaint_id = complaint_id or f"CB-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

        clean = mask_identifiers(normalise(complaint_text))
        tokens, offsets = tokenize(clean)
        if not tokens:
            raise ValueError("complaint_text is empty")

        hidden, tags, first, last, truncated = self._run_model(tokens)
        spans = [s for s in tags_to_spans(tags)
                 if any(tok not in PEEL for tok in tokens[s[0]:s[1]])]
        fallback = not spans
        if fallback:                       # model found no sub-complaint: treat the whole text as one
            spans = [(0, len(tokens))]

        subs = []
        for n, (a, b) in enumerate(spans, 1):
            scored = self._scores(hidden, first, last, (a, b))
            if scored is None:
                continue
            inst_scores, all_dept = scored
            g1 = institution_gate(inst_scores, t_inst, is_fake)
            sub = {
                "sub_complaint_id": f"{complaint_id}-{n}",
                "sub_complaint_text": clean[offsets[a][0]:offsets[b - 1][1]],
                "token_start": a, "token_end": b,
                "char_start": offsets[a][0], "char_end": offsets[b - 1][1],
                "institution": g1["institution"],
                "institution_scores": inst_scores,
                "gate1_case": g1["case"],
                "gate1_explanation": g1["explanation"],
                "department": None, "department_scores": None, "gate2_explanation": None,
                "needs_verification": g1["needs_verification"],
                "is_fake": bool(is_fake),
            }
            if g1["continue_to_department"]:
                d_scores = all_dept[g1["institution"]]
                g2 = department_gate(d_scores, t_dept)
                sub.update(department=g2["department"], department_scores=d_scores,
                           needs_verification=g2["needs_verification"],
                           gate2_explanation=g2["explanation"],
                           status=("routed to department (flagged)" if g2["needs_verification"]
                                   else "routed to department"))
            else:
                sub["status"] = "stopped at institution"
            subs.append(sub)

        return {
            "complaint_id": complaint_id,
            "raw_complaint_text": complaint_text,
            "processed_text": clean,
            "is_fake": bool(is_fake),
            "num_sub_complaints": len(subs),
            "sub_complaints": subs,
            "target_institutions": [s["institution"] for s in subs],
            "target_departments": [s["department"] for s in subs],
            "tokens": tokens,
            "bio_tags": tags,
            "thresholds": {"institution": t_inst, "department": t_dept},
            "segmentation_fallback": fallback,
            "truncated": truncated,
            "processing_ms": round((time.perf_counter() - t0) * 1000, 1),
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
