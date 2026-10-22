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
SENTENCE_END = {".", "!", "?", "෴", "।"}
COMMA = {",", ";", "،"}
# a comma followed by one of these continues the same issue ("a fence, which is dangerous")
CONTINUATION_WORDS = [
    "which", "who", "whom", "whose", "where", "when", "that", "because", "since", "so", "therefore",
    "hence", "thus", "as", "causing", "making", "due to", "even", "especially", "including", "but still",
    "ඒ නිසා", "එම නිසා", "එබැවින්", "ඒ හේතුවෙන්", "එහි", "එය", "ඒක", "ඒකෙන්", "ඒ නිසාම",
    "e nisa", "ekai", "eka",
]
# connector words at the start of a new part are left out of it
LEAD_CONNECTORS = [
    "ඒ වගේම", "ඒ වාගේම", "එමෙන්ම", "තවද", "ඊට අමතරව", "ඒ හැර", "ඒත් එක්කම", "සහ", "හා", "නමුත්", "ඒත්",
    "also", "and also", "and", "in addition", "furthermore", "moreover", "apart from that", "besides that",
    "as well as that", "but", "though", "although", "ewagema", "e wagema", "thawa", "ita amatharawa",
]
_norm = lambda w: unicodedata.normalize("NFC", w).lower().split()
CONTINUATION_PATTERNS = sorted({tuple(_norm(w)) for w in CONTINUATION_WORDS}, key=len, reverse=True)
LEAD_PATTERNS = sorted({tuple(_norm(w)) for w in LEAD_CONNECTORS}, key=len, reverse=True)


def _starts_with(low, k, patterns):
    return next((p for p in patterns if tuple(low[k:k + len(p)]) == p), None)

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
        # design rule: two issues of the same department are ONE sub-complaint, so neighbouring
        # parts with the same institution and department are merged after prediction.
        # "postprocess" (chosen on validation in the notebook, saved in routing_config.json):
        #   none | merge_same_department | merge_same_department_no_connector (default)
        #   | sentence_split_<conf>  (e.g. sentence_split_0.7): split a part at a full stop when the
        #     sentences on both sides are confidently about a different institution/department
        #   | sentence_comma_split_<stop conf>_<comma conf>  (e.g. sentence_comma_split_0.7_0.85):
        #     the same at full stops, and also at commas with the (stricter) comma confidence
        self.postprocess = cfg.get("postprocess", "merge_same_department_no_connector")

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

    @staticmethod
    def _label(scored):
        inst_s, dept_s = scored
        inst = max(inst_s, key=inst_s.get)
        return inst, max(dept_s[inst], key=dept_s[inst].get)

    @staticmethod
    def _confidence(scored):
        inst_s, dept_s = scored
        inst = max(inst_s, key=inst_s.get)
        return min(inst_s[inst], max(dept_s[inst].values()))

    def _sentence_split(self, hidden, first, last, spans, tokens, stop_conf, comma_conf=None):
        """Inside each part, at every full stop (and comma, if comma_conf is set): if the text before
        and after is CONFIDENTLY about a different institution/department, split there; otherwise
        keep it together. Returns (spans, number of full-stop splits, number of comma splits)."""
        low = [unicodedata.normalize("NFC", t).lower() for t in tokens]
        words = lambda a, b: sum(any(ch.isalnum() for ch in t) for t in tokens[a:b])
        out, n_stop, n_comma = [], 0, 0
        for a, b in spans:
            cuts = []                                    # (position, kind, confidence, min words)
            for k in range(a, b - 1):
                if tokens[k] in SENTENCE_END:
                    cuts.append((k + 1, "stop", stop_conf, 3))
                elif (comma_conf is not None and tokens[k] in COMMA
                      and not _starts_with(low, k + 1, CONTINUATION_PATTERNS)):
                    cuts.append((k + 1, "comma", comma_conf, 4))
            if not cuts:
                out.append((a, b)); continue
            ends = [c[0] for c in cuts[1:]] + [b]
            cur = (a, cuts[0][0]); cur_sc = self._scores(hidden, first, last, cur)
            for (s0, kind, conf, min_words), e0 in zip(cuts, ends):
                s1 = s0
                while s1 < e0 and low[s1] in PEEL:
                    s1 += 1
                hit = _starts_with(low, s1, LEAD_PATTERNS)
                if hit:                                  # "..., and they ..." -> "they ..."
                    s1 += len(hit)
                    while s1 < e0 and low[s1] in PEEL:
                        s1 += 1
                pc = (s1, e0)
                sc = self._scores(hidden, first, last, pc) if s1 < e0 else None
                if (cur_sc is not None and sc is not None
                        and words(*cur) >= min_words and words(*pc) >= min_words
                        and self._label(cur_sc) != self._label(sc)
                        and self._confidence(cur_sc) >= conf and self._confidence(sc) >= conf):
                    out.append(cur)
                    n_stop += kind == "stop"; n_comma += kind == "comma"
                    cur, cur_sc = pc, sc
                else:
                    cur = (cur[0], e0); cur_sc = self._scores(hidden, first, last, cur)
            out.append(cur)
        return out, n_stop, n_comma

    def _merge_same_department(self, hidden, first, last, spans, tokens=None, only_without_connector=False):
        """Neighbouring parts with the same top institution + department -> one part (re-scored).
        only_without_connector: never merge across a connector WORD (සහ, and, ...)."""
        out = []                                   # [(span, scores)]
        for sp in spans:
            sc = self._scores(hidden, first, last, sp)
            gap_ok = True
            if out and only_without_connector and tokens is not None:
                gap_ok = all(t in PEEL for t in tokens[out[-1][0][1]:sp[0]])
            if (out and gap_ok and sc is not None and out[-1][1] is not None
                    and self._label(out[-1][1]) == self._label(sc)):
                new = (out[-1][0][0], sp[1])
                out[-1] = (new, self._scores(hidden, first, last, new))
            else:
                out.append((sp, sc))
        return [s for s, _ in out], len(spans) - len(out)

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
        model_spans = len(spans)
        merged = split_at_stop = split_at_comma = 0
        if self.postprocess.startswith("sentence_comma_split_"):
            stop_conf, comma_conf = map(float, self.postprocess.split("_")[-2:])
            spans, split_at_stop, split_at_comma = self._sentence_split(
                hidden, first, last, spans, tokens, stop_conf, comma_conf)
        elif self.postprocess.startswith("sentence_split_"):
            conf = float(self.postprocess.rsplit("_", 1)[-1])
            spans, split_at_stop, _ = self._sentence_split(hidden, first, last, spans, tokens, conf)
        elif self.postprocess != "none" and len(spans) > 1:
            spans, merged = self._merge_same_department(
                hidden, first, last, spans, tokens,
                only_without_connector=self.postprocess == "merge_same_department_no_connector")

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
            "parts_found_by_model": model_spans,
            "merged_same_department": merged,
            "split_at_full_stop": split_at_stop,
            "split_at_comma": split_at_comma,
            "truncated": truncated,
            "processing_ms": round((time.perf_counter() - t0) * 1000, 1),
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
