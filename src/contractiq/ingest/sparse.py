import json
import math
import re
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself", "yourselves", "shall", "may", "upon", "pursuant"]
)
_K1 = 1.5
_B = 0.75
_MAX_TERMS = 300
_SUFFIXES = ("ations", "ation", "ments", "ment", "ings", "ing", "edly", "ness", "ed", "es", "ly", "s")


def _stem(token: str) -> str:
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    return [
        _stem(tok)
        for tok in _TOKEN_RE.findall(text.lower())
        if len(tok) >= 2 and tok not in _STOPWORDS
    ]


class BM25SparseEncoder:
    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: list[float] = []
        self.avgdl: float = 1.0

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def fit(self, corpus_tokens: list[list[str]]) -> None:
        n_docs = len(corpus_tokens)
        df: dict[str, int] = {}
        total_len = 0
        for tokens in corpus_tokens:
            total_len += len(tokens)
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        self.vocab = {term: idx for idx, term in enumerate(sorted(df))}
        n = max(n_docs, 1)
        self.idf = [
            math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5)) for term in sorted(df)
        ]
        self.avgdl = max(total_len / n, 1.0)

    def encode(self, text: str) -> dict:
        tokens = tokenize(text)
        tf: dict[int, int] = {}
        for tok in tokens:
            idx = self.vocab.get(tok)
            if idx is not None:
                tf[idx] = tf.get(idx, 0) + 1
        if not tf:
            return {"indices": [], "values": []}
        scored = []
        dl = max(len(tokens), 1)
        norm = _K1 * (1 - _B + _B * dl / self.avgdl)
        for idx, freq in tf.items():
            score = self.idf[idx] * freq * (_K1 + 1) / (freq + norm)
            scored.append((idx, score))
        scored.sort(key=lambda pair: -pair[1])
        scored = scored[:_MAX_TERMS]
        max_score = max(score for _, score in scored)
        scaled = [(idx, score / max_score) for idx, score in sorted(scored)]
        return {
            "indices": [idx for idx, _ in scaled],
            "values": [round(score, 6) for _, score in scaled],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "k1": _K1,
            "b": _B,
            "avgdl": self.avgdl,
            "vocab": self.vocab,
            "idf": self.idf,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25SparseEncoder":
        payload = json.loads(path.read_text(encoding="utf-8"))
        encoder = cls()
        encoder.vocab = payload["vocab"]
        encoder.idf = payload["idf"]
        encoder.avgdl = payload["avgdl"]
        return encoder
