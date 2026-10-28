"""Retrieval tool tests. Alias resolution is pure; the rest uses the real Chroma index."""
from __future__ import annotations

import pytest

from backend.tools import (MAX_K, SECTIONS, format_for_llm, known_drugs, resolve_drug_name, retrieve,
                           search_drug_info)


@pytest.mark.parametrize("name, expected", [
    ("Paracetamol", "acetaminophen"),
    ("TYLENOL", "acetaminophen"),
    ("salbutamol", "albuterol"),
    ("metformin hydrochloride", "metformin"),   # salt form
    ("Lasix", "furosemide"),                    # brand
    ("ozempic", None),                          # not in the knowledge base
    (None, None),
])
def test_resolve_drug_name(name, expected):
    assert resolve_drug_name(name) == expected


def test_known_drugs_has_all_30():
    assert len(known_drugs()) == 30


def test_unknown_drug_returns_note_and_no_results():
    """Must NOT fall back to an unfiltered search (that would answer with another drug's text)."""
    out = retrieve("starting dose", drug_name="semaglutide")
    assert out["results"] == []
    assert "not in the verified knowledge base" in out["note"]


@pytest.mark.index
def test_metadata_filters_are_applied():
    out = retrieve("drugs that increase bleeding risk", drug_name="warfarin", section="interactions", k=4)
    assert out["results"]
    assert all(r["drug_name"] == "warfarin" and r["section"] == "interactions" for r in out["results"])


@pytest.mark.index
def test_alias_is_resolved_before_filtering():
    out = retrieve("maximum daily dose", drug_name="paracetamol", section="dosage", k=2)
    assert out["filters"]["drug_name"] == "acetaminophen"
    assert {r["drug_name"] for r in out["results"]} == {"acetaminophen"}


@pytest.mark.index
def test_semantic_search_finds_right_drug_without_filters():
    out = retrieve("warfarin interactions that increase bleeding", k=3)
    assert out["results"][0]["drug_name"] == "warfarin"


@pytest.mark.index
def test_off_topic_query_returns_nothing():
    out = retrieve("how do I bake sourdough bread")
    assert out["results"] == []
    assert "No sufficiently relevant passages" in out["note"]


@pytest.mark.index
def test_drug_and_section_filters_bypass_score_cutoff():
    """Regression (Phase 7, q11): terse 'boxed warning' scored ~0.14 and was cut off -> false refusal."""
    out = retrieve("boxed warning", drug_name="ciprofloxacin", section="boxed_warning", k=2)
    assert len(out["results"]) == 2


@pytest.mark.index
def test_invalid_section_is_ignored_with_note():
    out = retrieve("common adverse effects", drug_name="atorvastatin", section="adverse_effects", k=2)
    assert out["filters"]["section"] is None
    assert "Unknown section" in out["note"]
    assert out["results"]


@pytest.mark.index
def test_k_is_clamped():
    out = retrieve("dose", drug_name="warfarin", section="dosage", k=50)
    assert out["filters"]["k"] == MAX_K
    assert len(out["results"]) <= MAX_K


@pytest.mark.index
def test_tool_returns_text_for_llm_and_structured_artifact():
    msg = search_drug_info.invoke({"type": "tool_call", "id": "t1", "name": "search_drug_info",
                                   "args": {"query": "bleeding risk", "drug_name": "warfarin", "k": 2}})
    assert msg.content.startswith("[1] drug=warfarin")
    assert msg.artifact["results"][0]["source_url"].startswith("https://dailymed.nlm.nih.gov/")


def test_section_enum_in_tool_schema():
    """The LLM sees the allowed section values as an enum, so it can't invent one."""
    schema = search_drug_info.tool_call_schema.model_json_schema()
    enum = schema["properties"]["section"]["anyOf"][0]["enum"]
    assert set(enum) == set(SECTIONS)


def test_format_for_llm_skips_repeated_passages():
    payload = {"note": None, "results": [
        {"rank": 1, "drug_name": "x", "section": "dosage", "source_url": "u", "score": 0.5, "text": "FULL TEXT"},
        {"rank": 1, "repeat": True, "drug_name": "x", "section": "dosage", "source_url": "u", "score": 0.5,
         "text": "FULL TEXT"},
    ]}
    out = format_for_llm(payload)
    assert out.count("FULL TEXT") == 1
    assert "same passage as [1]" in out
