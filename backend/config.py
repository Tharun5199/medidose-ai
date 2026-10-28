"""Central configuration for MediDose AI.

WHY a single config module:
- Every tunable (model names, chunk sizes, paths) lives in one place and is read
  from environment variables, so we can change behaviour without editing code
  (12-factor app style) and keep secrets like GROQ_API_KEY out of git.
- Paths are resolved relative to the project root, so scripts work no matter
  which directory you launch them from.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _path(env_name: str, default: str) -> Path:
    """Read a path from the environment; relative paths are anchored at the project root."""
    p = Path(os.getenv(env_name, default))
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass(frozen=True)
class Settings:
    # LLM
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    # For reasoning models (gpt-oss, qwen3): "low" keeps hidden reasoning tokens, and so
    # latency and free-tier token usage, small. Leave empty for non-reasoning models.
    groq_reasoning_effort: str = os.getenv("GROQ_REASONING_EFFORT", "low")

    # Embeddings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Data + vector store
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    drug_list_path: Path = PROJECT_ROOT / "data" / "drug_list.json"
    chroma_dir: Path = _path("CHROMA_DIR", "data/chroma")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "drug_labels")

    # Chunking / retrieval
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("TOP_K", "4"))

    # openFDA
    openfda_api_key: str = os.getenv("OPENFDA_API_KEY", "")

    # API
    # If set, POST /ingest requires header "X-Admin-Token: <value>" (rebuilding is expensive).
    admin_token: str = os.getenv("ADMIN_TOKEN", "")
    # Comma-separated origins allowed to call the API from a browser (Vite dev server by default).
    cors_origins: tuple[str, ...] = tuple(
        o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip())


settings = Settings()
