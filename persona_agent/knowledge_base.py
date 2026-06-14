import json
from dataclasses import dataclass
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config


@dataclass
class KBArticle:
    id: str
    title: str
    tags: List[str]
    content: str

    def to_search_text(self) -> str:
        return f"{self.title} {' '.join(self.tags)} {self.content}"


class KnowledgeBase:
    def __init__(self, path: str = config.KB_PATH):
        with open(path, "r", encoding="utf-8") as f:
            self.articles = [KBArticle(**item) for item in json.load(f)]

        texts = [a.to_search_text() for a in self.articles]
        self._vec = TfidfVectorizer(stop_words="english")
        self._mat = self._vec.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = config.KB_TOP_K) -> List[Tuple[KBArticle, float]]:
        qvec = self._vec.transform([query])
        sims = cosine_similarity(qvec, self._mat)[0]
        top_idx = sims.argsort()[::-1][:top_k]
        return [(self.articles[i], float(sims[i])) for i in top_idx]

    def best_match(self, query: str) -> Tuple[KBArticle, float]:
        return self.retrieve(query, top_k=1)[0]
