"""Embedding functions. The provider + model are part of the collection name so vectors never mix."""

from __future__ import annotations

import hashlib
import itertools
import math
import re

from chromadb import Documents, EmbeddingFunction, Embeddings

from app.core.config import Settings


class HashingEmbeddingFunction(EmbeddingFunction[Documents]):
    """Deterministic feature-hashing embeddings (unigrams + bigrams). For tests and offline demos only."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        features = tokens + [f"{a}_{b}" for a, b in itertools.pairwise(tokens)]
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def name() -> str:
        return "hashing"

    def get_config(self) -> dict:
        return {"dim": self.dim}

    @staticmethod
    def build_from_config(config: dict) -> HashingEmbeddingFunction:
        return HashingEmbeddingFunction(dim=config.get("dim", 512))


def embedding_identity(settings: Settings) -> str:
    if settings.embedding_provider == "openai":
        return f"openai-{settings.openai_embedding_model}"
    if settings.embedding_provider == "hash":
        return "hash-512"
    return "local-all-minilm-l6-v2"


def build_embedding_function(settings: Settings) -> EmbeddingFunction:
    if settings.embedding_provider == "openai":
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

        key = settings.secret("openai_api_key")
        if not key:
            raise RuntimeError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY")
        # The key is held in memory only; Chroma persists just the env var *name* in collection config.
        return OpenAIEmbeddingFunction(
            api_key=key, api_key_env_var="OPENAI_API_KEY", model_name=settings.openai_embedding_model
        )
    if settings.embedding_provider == "hash":
        return HashingEmbeddingFunction()
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    return DefaultEmbeddingFunction()
