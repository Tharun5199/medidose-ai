"""Shared embedding model + Chroma vector store.

WHY a separate module: ingestion (writes) and the retrieval tool (reads) MUST use
the exact same embedding model and distance metric. If they drifted apart, query
vectors would live in a different space from document vectors and search would
silently return garbage. Building both from one function makes that impossible.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load all-MiniLM-L6-v2 once per process (~90 MB, runs on CPU).

    WHY this model: free, local (no API key, no data leaves the machine), fast on
    CPU, 384-dim vectors, and a strong general-purpose semantic-search baseline.
    normalize_embeddings=True gives unit-length vectors, so cosine similarity
    equals the dot product and scores are comparable across queries.
    """
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore() -> Chroma:
    """Open (or create) the persistent Chroma collection on disk.

    WHY cosine: sentence-transformers models are trained with cosine similarity,
    so it's the metric their vectors are meant to be compared with
    (Chroma's default is squared L2).
    """
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
