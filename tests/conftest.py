"""Shared fixtures.

Design choice: NO test calls the Groq API. LLM behaviour is tested with a scripted
fake model, so the suite is free, fast, deterministic and runs offline. Live-LLM
quality is measured separately by the evaluation in eval/ (Phase 7).
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from backend.config import settings


class FakeLLM:
    """Replays scripted AIMessages and records the messages it was sent."""

    def __init__(self, script: list[AIMessage]):
        self.script = list(script)
        self.calls: list[list] = []
        self.bound_kwargs: list[dict] = []

    def bind_tools(self, tools, **kwargs):
        self.bound_kwargs.append(kwargs)
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))  # copy: the agent keeps appending to its list
        return self.script.pop(0)


def tool_call(call_id: str, **args) -> dict:
    return {"id": call_id, "name": "search_drug_info", "args": args}


@pytest.fixture
def fake_llm():
    return FakeLLM


@pytest.fixture(scope="session")
def index_ready() -> bool:
    try:
        from backend.vectorstore import get_vectorstore
        return get_vectorstore()._collection.count() > 0
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _skip_without_index(request, index_ready):
    if request.node.get_closest_marker("index") and not index_ready:
        pytest.skip(f"vector index not built in {settings.chroma_dir}; run `python -m backend.ingest`")
