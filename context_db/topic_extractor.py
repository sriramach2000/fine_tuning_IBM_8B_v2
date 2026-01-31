"""Extract topics and code entity references from context text."""

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExtractedTopic:
    name: str
    description: str = ""


@dataclass
class ExtractedCodeEntity:
    file_path: str
    entity_type: str  # "file", "function", "class", "module"
    name: str


# Regex patterns for code entity extraction
FILE_PATH_PATTERN = re.compile(
    r'(?:^|[\s\'"`(])([a-zA-Z0-9_./\\-]+\.(?:py|c|h|cpp|hpp|yaml|yml|json|toml|cfg|sh|md))\b'
)
FUNCTION_PATTERN = re.compile(r'\bdef\s+(\w+)\s*\(')
CLASS_PATTERN = re.compile(r'\bclass\s+(\w+)[\s(:]')


class TopicExtractor:
    """Extracts topics and code references from context text.

    Uses KeyBERT for keyword-based topic extraction and regex for code entities.
    Falls back to simple TF-IDF extraction if KeyBERT is unavailable.
    """

    def __init__(self, use_keybert: bool = True):
        self._keybert_model = None
        self._use_keybert = use_keybert
        if use_keybert:
            try:
                from keybert import KeyBERT
                self._keybert_model = KeyBERT()
            except ImportError:
                self._use_keybert = False

    def extract_topics(
        self, text: str, top_n: int = 5, min_length: int = 2
    ) -> List[ExtractedTopic]:
        """Extract key topics from text."""
        if self._use_keybert and self._keybert_model is not None:
            return self._extract_keybert(text, top_n)
        return self._extract_simple(text, top_n, min_length)

    def _extract_keybert(self, text: str, top_n: int) -> List[ExtractedTopic]:
        keywords = self._keybert_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=top_n,
            use_mmr=True,
            diversity=0.5,
        )
        return [
            ExtractedTopic(name=kw, description=f"Score: {score:.3f}")
            for kw, score in keywords
        ]

    def _extract_simple(
        self, text: str, top_n: int, min_length: int
    ) -> List[ExtractedTopic]:
        """Fallback: frequency-based keyword extraction."""
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "during",
            "before", "after", "above", "below", "between", "out", "off", "over",
            "under", "again", "further", "then", "once", "and", "but", "or", "nor",
            "not", "so", "yet", "both", "either", "neither", "each", "every", "all",
            "any", "few", "more", "most", "other", "some", "such", "no", "only",
            "own", "same", "than", "too", "very", "just", "because", "if", "when",
            "while", "how", "what", "which", "who", "whom", "this", "that", "these",
            "those", "it", "its", "i", "me", "my", "we", "our", "you", "your",
            "he", "him", "his", "she", "her", "they", "them", "their",
        }
        words = re.findall(r'\b[a-zA-Z_]\w*\b', text.lower())
        words = [w for w in words if len(w) >= min_length and w not in stop_words]
        freq: dict = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [
            ExtractedTopic(name=w, description=f"Frequency: {c}")
            for w, c in sorted_words[:top_n]
        ]

    def extract_code_entities(self, text: str) -> List[ExtractedCodeEntity]:
        """Extract file paths, function names, and class names from text."""
        entities: List[ExtractedCodeEntity] = []
        seen = set()

        for match in FILE_PATH_PATTERN.finditer(text):
            path = match.group(1)
            if path not in seen:
                seen.add(path)
                name = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                entities.append(ExtractedCodeEntity(
                    file_path=path, entity_type="file", name=name,
                ))

        for match in FUNCTION_PATTERN.finditer(text):
            fname = match.group(1)
            key = f"func:{fname}"
            if key not in seen:
                seen.add(key)
                entities.append(ExtractedCodeEntity(
                    file_path="", entity_type="function", name=fname,
                ))

        for match in CLASS_PATTERN.finditer(text):
            cname = match.group(1)
            key = f"class:{cname}"
            if key not in seen:
                seen.add(key)
                entities.append(ExtractedCodeEntity(
                    file_path="", entity_type="class", name=cname,
                ))

        return entities
