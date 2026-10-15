"""Small in-memory BM25 index with positive, log1p-smoothed IDF."""

from collections import Counter, defaultdict
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import replace
import math
import re

from visionevidence.schemas import Evidence

TOKENIZER = "ascii-alphanumeric-lower-v1"


def tokenize(text: str) -> list[str]:
    """Lowercase ASCII words/numbers; punctuation separates tokens. English only."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return re.findall(r"[a-z0-9]+", text.lower())


def validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("top-k must be a positive integer")


class BM25Index:
    """Build once, query repeatedly. No qrels or truth labels enter this index."""

    def __init__(self, documents: Iterable[Evidence], *, k1: float = 1.5, b: float = 0.75):
        for name, value in (("k1", k1), ("b", b)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self.k1 = float(k1)
        self.b = float(b)
        self._documents: list[Evidence] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._lengths: list[int] = []
        ids: set[str] = set()
        for original in documents:
            # Snapshot mutable schema objects so caller changes cannot stale the index.
            document = replace(deepcopy(original), score=None)
            if document.document_id in ids:
                raise ValueError(f"duplicate evidence_id: {document.document_id}")
            ids.add(document.document_id)
            terms = Counter(tokenize(document.text))
            position = len(self._documents)
            self._documents.append(document)
            self._lengths.append(sum(terms.values()))
            for term, frequency in terms.items():
                self._postings[term].append((position, frequency))
        if not self._documents:
            raise ValueError("evidence corpus must not be empty")
        self.document_ids = frozenset(ids)
        self.average_length = sum(self._lengths) / len(self._documents)

    def search(self, query: str, k: int = 3) -> list[Evidence]:
        """Return only matching documents, ordered by descending score then ID."""
        validate_k(k)
        terms = sorted(set(tokenize(query)))
        if not terms:
            raise ValueError("query must contain at least one English letter or digit")
        scores: dict[int, float] = defaultdict(float)
        for term in terms:
            postings = self._postings.get(term, [])
            if not postings:
                continue
            idf = math.log1p((len(self._documents) - len(postings) + 0.5) / (len(postings) + 0.5))
            for position, frequency in postings:
                normalization = 1 - self.b + self.b * self._lengths[position] / self.average_length
                scores[position] += idf * (frequency * (self.k1 + 1)) / (frequency + self.k1 * normalization)
        ranked = sorted(scores, key=lambda position: (-scores[position], self._documents[position].document_id))
        return [replace(deepcopy(self._documents[pos]), score=scores[pos]) for pos in ranked[:k]]
