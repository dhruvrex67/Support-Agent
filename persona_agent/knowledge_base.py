import json
from dataclasses import dataclass
from typing import List, Tuple
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config

logger = logging.getLogger(__name__)


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
        self.articles = []
        self._vec = None
        self._mat = None
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    logger.warning("KB file does not contain a list, treating as empty")
                    return
                
                self.articles = [KBArticle(**item) for item in data]
                
                if not self.articles:
                    logger.warning("KB file is empty")
                    return
                
                texts = [a.to_search_text() for a in self.articles]
                self._vec = TfidfVectorizer(stop_words="english", min_df=1)
                self._mat = self._vec.fit_transform(texts)
                logger.info(f"Knowledge base loaded with {len(self.articles)} articles")
                
        except FileNotFoundError:
            logger.error(f"KB file not found at {path}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in KB file: {e}")
        except KeyError as e:
            logger.error(f"Missing required field in KB article: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading KB: {e}")

    def retrieve(self, query: str, top_k: int = config.KB_TOP_K) -> List[Tuple[KBArticle, float]]:
        """Retrieve top-k articles matching the query."""
        if not self.articles or self._vec is None or self._mat is None:
            logger.warning("KB is empty or not initialized, returning fallback")
            return [(self._get_fallback_article(), 0.0)]
        
        try:
            qvec = self._vec.transform([query])
            sims = cosine_similarity(qvec, self._mat)[0]
            top_idx = sims.argsort()[::-1][:top_k]
            results = [(self.articles[i], float(sims[i])) for i in top_idx]
            return results
        except Exception as e:
            logger.error(f"Error during KB retrieval: {e}")
            return [(self._get_fallback_article(), 0.0)]

    def best_match(self, query: str) -> Tuple[KBArticle, float]:
        """Return the single best matching article."""
        results = self.retrieve(query, top_k=1)
        if results:
            return results[0]
        return (self._get_fallback_article(), 0.0)

    @staticmethod
    def _get_fallback_article() -> KBArticle:
        """Return a fallback article when KB is empty."""
        return KBArticle(
            id="fallback",
            title="General Help",
            tags=["help", "support"],
            content="I don't have a specific answer in my knowledge base. A human agent will be better equipped to help you."
        )

    @classmethod
    def _create_empty(cls):
        """Create an empty KB instance for fallback."""
        kb = cls.__new__(cls)
        kb.articles = []
        kb._vec = None
        kb._mat = None
        logger.info("Created empty KB instance")
        return kb
