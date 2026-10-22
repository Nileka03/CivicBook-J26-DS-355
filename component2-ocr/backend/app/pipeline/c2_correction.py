"""Component 2 - uncertainty-scoped, retrieval-grounded LLM correction with a verifier gate."""
import re, json, time, unicodedata
from rapidfuzz.distance import Levenshtein

SIN = re.compile(r'[\u0D80-\u0DFF]'); DIG = re.compile(r'\d')
ID_RE = re.compile(r'l\d+w\d+')                # word ids look like l3w5 (line 3, word 5)
PUNCT = '"\'.,;:!?()[]{}<>|/\\-–—_*#~`“”‘’'
norm = lambda t: unicodedata.normalize('NFC', str(t))
tokens = lambda t: [w.strip(PUNCT) for w in norm(t).split() if w.strip(PUNCT)]
# consonant skeleton: drop vowel signs, virama (්), ං ඃ and the joiner -> the consonants that identify the word.
# Handwriting OCR mostly confuses the small vowel strokes, so R2 compares skeletons.
VOWEL_MARKS = re.compile(r'[\u0DCA-\u0DDF\u0DF2\u0DF3\u0D82\u0D83\u200d]')
skeleton = lambda s: VOWEL_MARKS.sub('', s)

# ---------- 1. uncertainty marking from CRNN character confidences ----------
def words_with_conf(text, confs):
    out, i = [], 0
    for w in text.split(' '):
        c = confs[i:i + len(w)]; i += len(w) + 1
        if w: out.append((w, min(c) if c else 0.0))
    return out

def build_page(lines, tau):
    """lines = [(ocr_text, char_confidences), ...] -> [[{id,text,conf,unsure}, ...], ...]"""
    return [[dict(id=f'l{li}w{wi}', text=w, conf=c, unsure=c < tau)
             for wi, (w, c) in enumerate(words_with_conf(norm(t), conf))]
            for li, (t, conf) in enumerate(lines)]

def render(page, edits=None, marked=False):
    edits = edits or {}
    return '\n'.join(' '.join(
        f"«{w['id']}:{w['text']}»" if marked and w['unsure'] and w['id'] not in edits else edits.get(w['id'], w['text'])
        for w in line) for line in page)

# ---------- 2. prompts ----------
PROMPT_SCOPED = """You fix OCR errors in ONE handwritten civic complaint from Sri Lanka (Sinhala, English or mixed).
Words the OCR was unsure about look like «ID:word» (for example «l2w5:පදුලි» has ID l2w5). Everything else is reliable.
Line breaks only show layout.

RULES
1. Only propose replacements for «ID:word» items. Never touch anything else.
2. Propose a fix only if supported by: a REFERENCE complaint, the LEXICON, the same word elsewhere in this
   complaint, the DESCRIPTION, or the obvious spelling of a common Sinhala/English word.
3. The replacement must look like the OCR word (a few wrong or missing letters). Do not translate,
   summarise or add information. If unsure, leave it out.
4. Never change digits, dates, numbers or codes unless the exact value appears in the evidence.
5. Use the ID exactly as written (e.g. "l2w5"), without « » or the word.
6. Return JSON only: {{"edits": [{{"id": "l2w5", "to": "...", "evidence": "reference|lexicon|complaint|description|spelling"}}]}}

DESCRIPTION (typed by the citizen): {description}

REFERENCE (similar past complaints):
{refs}

LEXICON: {lexicon}

COMPLAINT (OCR):
{text}
{feedback}"""

PROMPT_BLIND = """Correct the OCR errors in this handwritten Sri Lankan civic complaint (Sinhala/English).
Keep the same number of lines. Return JSON only: {{"lines": ["...", "..."]}}
{refs}
OCR TEXT:
{text}"""

def call_llm(client, model, prompt, retries=4):
    for a in range(retries):
        try:
            r = client.models.generate_content(model=model, contents=prompt,
                    config={'temperature': 0, 'response_mime_type': 'application/json'})
            return json.loads(re.search(r'\{.*\}', r.text or '', re.S).group(0))
        except Exception as e:
            if a == retries - 1: raise
            time.sleep(10 * (a + 1))          # rate limit / temporary error -> wait and retry

# ---------- 3. verifier (rules R1-R4) ----------
def check(w, to, vocab, use_verifier=True, max_ratio=0.5):
    if w is None: return ['R1_unknown_id']
    r = []
    if not w['unsure']: r.append('R1_not_uncertain')
    if not to or len(to.split()) > 2: r.append('R1_bad_replacement')
    if use_verifier and not r:
        a, b = skeleton(w['text']), skeleton(to)                                 # R2 on consonant skeletons
        if Levenshtein.distance(a, b) > max(1, max_ratio * max(len(a), len(b))):
            r.append('R2_too_different')
        tk = tokens(to)
        if not tk or any(t not in vocab for t in tk): r.append('R3_unsupported')
        if DIG.findall(w['text']) != DIG.findall(to) and to not in vocab: r.append('R4_digits_changed')
    return r

# ---------- 4. the full loop: mark -> LLM edit list -> verify -> re-check ----------
def correct_page(lines, client, model, refs, lexicon, tau=0.8, use_verifier=True, max_rounds=3,
                 description='', sleep=4):
    page = build_page(lines, tau)
    by_id = {w['id']: w for line in page for w in line}
    raw = render(page)
    base = dict(raw=raw, refs=refs, edits=[], prompts=[], rounds=0)
    if not any(w['unsure'] for w in by_id.values()):
        return {**base, 'final': raw, 'decision': 'NO_CHANGES'}
    vocab = (set(lexicon) | {w['text'] for w in by_id.values() if not w['unsure']}
             | {t for d in refs for t in tokens(d)} | set(tokens(description)))
    lex_txt = ', '.join(lexicon) if lexicon else '(none)'
    refs_txt = '\n'.join(f'- {d}' for d in refs) or '(none)'
    accepted, log, prompts, feedback, rnd = {}, [], [], '', 0
    for rnd in range(1, max_rounds + 1):
        prompt = PROMPT_SCOPED.format(description=description or '(none)', refs=refs_txt, lexicon=lex_txt,
                                      text=render(page, accepted, marked=True),
                                      feedback=f'\nFEEDBACK ON YOUR PREVIOUS EDITS (fix or drop):\n{feedback}' if feedback else '')
        prompts.append(prompt)
        try:
            out = call_llm(client, model, prompt); time.sleep(sleep)
        except Exception as e:
            return {**base, 'final': raw, 'edits': log, 'prompts': prompts, 'rounds': rnd,
                    'decision': 'FALLBACK_RAW', 'error': str(e)}
        rej, n = [], 0
        for e in out.get('edits', []):
            m = ID_RE.search(str(e.get('id', ''))); eid = m.group(0) if m else str(e.get('id', ''))
            to = norm(e.get('to', '')).strip(); w = by_id.get(eid)
            if eid in accepted or (w and to == w['text']): continue
            reasons = check(w, to, vocab, use_verifier); n += 1
            log.append(dict(id=eid, frm=w['text'] if w else '', to=to, evidence=e.get('evidence', ''),
                            ok=not reasons, reasons=reasons, round=rnd))
            if reasons: rej.append(log[-1])
            else: accepted[eid] = to
        if not use_verifier or not rej or len(rej) / max(1, n) <= 0.3: break
        feedback = '\n'.join(f"- {r['id']}: '{r['frm']}' -> '{r['to']}' rejected ({', '.join(r['reasons'])})" for r in rej)
    nrej = sum(not l['ok'] for l in log)
    decision = ('ACCEPT' if accepted and not nrej else 'PARTIAL' if accepted else 'REVIEW' if nrej else 'NO_CHANGES')
    return {**base, 'final': render(page, accepted), 'edits': log, 'prompts': prompts, 'rounds': rnd, 'decision': decision}

# ---------- 5. ablation baseline: LLM rewrites everything (no marks, no verifier) ----------
def correct_blind(lines, client, model, refs=()):
    raw = '\n'.join(norm(t) for t, _ in lines)
    refs_txt = ('REFERENCE (similar past complaints):\n' + '\n'.join(f'- {d}' for d in refs) + '\n') if refs else ''
    try:
        out = call_llm(client, model, PROMPT_BLIND.format(refs=refs_txt, text=raw))
        return dict(raw=raw, final='\n'.join(norm(x) for x in out.get('lines', [])), decision='BLIND')
    except Exception as e:
        return dict(raw=raw, final=raw, decision='FALLBACK_RAW', error=str(e))
