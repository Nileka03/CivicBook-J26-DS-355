"""Past-complaints memory (the "R" in RAG).

Character n-gram TF-IDF (2-4 chars) is used instead of word embeddings because
the query is noisy OCR text: a word with two wrong letters still shares most
of its character n-grams with the correct word, while a whole-word embedding
often does not. It also runs offline with no model download.
On the validation pages, char TF-IDF top-20 made 0.297 of the wrong OCR words
fixable vs 0.036 for dense multilingual-e5 (notebook 02), so it is the only retriever.
"""
import json
import logging
import unicodedata
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from ..config import settings

log = logging.getLogger(__name__)


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


class ComplaintRetriever:
    def __init__(self, docs: list[dict]):
        self.docs = [d for d in docs if d.get("text", "").strip()]
        self._vec = None
        if self.docs:
            self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
            self._mat = self._vec.fit_transform([normalize(d["text"]) for d in self.docs])

    @classmethod
    def from_jsonl(cls, path: Path) -> "ComplaintRetriever":
        docs = []
        if path.exists():
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    d.setdefault("id", f"doc{i}")
                    docs.append(d)
                except json.JSONDecodeError:
                    log.warning("Skipping bad JSON line %d in %s", i + 1, path)
        log.info("Loaded %d past complaints from %s", len(docs), path)
        return cls(docs)

    def search(self, query: str, k: int | None = None, exclude_ids: set[str] = frozenset()):
        k = k or settings.RETRIEVAL_TOP_K
        if not self._vec or not query.strip():
            return []
        sims = linear_kernel(self._vec.transform([normalize(query)]), self._mat).ravel()
        out = []
        for i in sims.argsort()[::-1]:
            d = self.docs[i]
            if d["id"] in exclude_ids or sims[i] <= 0:
                continue
            out.append({"id": str(d["id"]), "text": d["text"], "score": round(float(sims[i]), 4)})
            if len(out) >= k:
                break
        return out


def load_lexicon(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [normalize(l.strip()) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]


_retriever: ComplaintRetriever | None = None


def get_retriever() -> ComplaintRetriever:
    global _retriever
    if _retriever is None:
        _retriever = ComplaintRetriever.from_jsonl(settings.PAST_COMPLAINTS_PATH)
    return _retriever


def reload_retriever() -> int:
    global _retriever
    _retriever = ComplaintRetriever.from_jsonl(settings.PAST_COMPLAINTS_PATH)
    return len(_retriever.docs)
