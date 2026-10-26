"""Tiny knowledge base: markdown files split by `##` sections, searched with BM25.

A handful of short policy pages does not need embeddings (DeepSeek has no embedding API anyway).
At production scale the same interface fronts hybrid search (BM25 + embeddings) with a reranker.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import snowballstemmer
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_STOP = {
    "и", "в", "во", "на", "с", "со", "по", "к", "у", "о", "об", "из", "за", "для", "до", "от", "не", "ли",
    "а", "но", "или", "что", "как", "это", "то", "же", "бы", "мне", "меня", "я", "мы", "вы", "он", "она",
    "the", "a", "an", "of", "to", "in", "is", "and", "or", "for", "on", "my", "i",
}  # fmt: skip


@dataclass(frozen=True)
class KBChunk:
    source: str
    title: str
    section: str
    text: str


class KnowledgeBase:
    def __init__(self, directory: Path, min_score: float = 0.5) -> None:
        self._ru = snowballstemmer.stemmer("russian")
        self._en = snowballstemmer.stemmer("english")
        self._min_score = min_score
        self.chunks = [chunk for path in sorted(directory.glob("*.md")) for chunk in self._split(path)]
        if not self.chunks:
            raise ValueError(f"knowledge base is empty: {directory}")
        # The section heading is repeated: it is the most precise description of what the chunk answers.
        self._bm25 = BM25Okapi([self._tokens(f"{c.title} {c.section} {c.section} {c.text}") for c in self.chunks])

    def _tokens(self, text: str) -> list[str]:
        words = [w for w in _TOKEN.findall(text.lower().replace("ё", "е")) if w not in _STOP]
        return [self._ru.stemWord(w) if re.search("[а-я]", w) else self._en.stemWord(w) for w in words]

    @staticmethod
    def _split(path: Path) -> list[KBChunk]:
        title, section, lines, chunks = path.stem, "", [], []

        def flush() -> None:
            body = " ".join(line.strip() for line in lines if line.strip())
            if section and body:
                chunks.append(KBChunk(path.name, title, section, body))

        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("## "):
                flush()
                section, lines = line[3:].strip(), []
            else:
                lines.append(line)
        flush()
        return chunks

    def search(self, query: str, k: int = 3) -> list[tuple[KBChunk, float]]:
        tokens = self._tokens(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip(self.chunks, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [(chunk, float(score)) for chunk, score in ranked[:k] if score >= self._min_score]
