"""Phase 3 — The retrieval tool the agent can choose to call.

Try it standalone from the project root:
    python -m backend.tools "maximum daily dose" --drug paracetamol --section dosage
    python -m backend.tools "drugs that increase bleeding risk" --drug warfarin
    python -m backend.tools "how do I bake bread"

Design notes (the WHY):
- The tool's NAME, DOCSTRING and ARGUMENT DESCRIPTIONS are sent to the LLM as a JSON
  schema. They are effectively part of the prompt: they are how the agent decides
  *whether* to call the tool and *how* to fill in drug_name / section.
- `section` is a Literal (an enum in the schema), so the LLM can only pick a value
  that actually exists in our metadata. A free-text field would invite made-up values
  like "adverse_effects" that silently match nothing.
- Name normalisation happens HERE, not in the LLM: "paracetamol", "Tylenol" and
  "acetaminophen" all map to the canonical `drug_name` stored in Chroma. Exact
  metadata filters are unforgiving, so we resolve names in code.
- Unknown drugs return an explicit "not in the knowledge base" message instead of
  falling back to an unfiltered search. Otherwise a question about a drug we don't
  have would quietly retrieve chunks about a *different* drug, the classic way a
  RAG system hallucinates with confidence.
- A minimum relevance score drops weak matches, so "nothing relevant" really comes
  back as nothing. That is the agent's signal to refine the query or say
  "I don't have verified information on that".
- response_format="content_and_artifact": the LLM receives compact numbered text
  to cite ([1], [2] ...), while the API receives the same results as structured
  dicts (the artifact) to render the Sources panel. No parsing of LLM output needed.
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from typing import Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from backend.config import settings

Section = Literal[
    "indications", "dosage", "contraindications", "warnings", "boxed_warning",
    "side_effects", "interactions", "special_populations", "overdose", "pharmacology",
]
SECTIONS: tuple[str, ...] = Section.__args__

# Results scoring below this cosine relevance are treated as "not relevant".
# Calibrated in Phase 3 (see README): on-topic hits score ~0.45-0.8, unrelated
# questions ~0.1-0.25 even against their closest chunk.
MIN_RELEVANCE = 0.30
# Hard cap per search: Groq's free tier allows ~8,000 tokens/minute, and every
# passage (~150 tokens) is re-sent to the LLM on each step of the agent loop.
MAX_K = 5


@lru_cache(maxsize=1)
def _alias_map() -> dict[str, str]:
    """{'paracetamol': 'acetaminophen', 'tylenol': 'acetaminophen', 'acetaminophen': ...}"""
    drugs = json.loads(settings.drug_list_path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for d in drugs:
        for name in [d["name"], *d.get("aliases", [])]:
            mapping[name.lower()] = d["name"]
    return mapping


def known_drugs() -> list[str]:
    return sorted(set(_alias_map().values()))


def resolve_drug_name(name: str | None) -> str | None:
    """Map a user/LLM-supplied drug name to the canonical name in the index, or None.

    Handles case, aliases/brands, and salt forms ("metformin hydrochloride").
    """
    if not name:
        return None
    key = name.strip().lower()
    aliases = _alias_map()
    if key in aliases:
        return aliases[key]
    for word in key.replace("-", " ").split():  # "metformin hydrochloride" -> "metformin"
        if word in aliases:
            return aliases[word]
    return None


def retrieve(query: str, drug_name: str | None = None, section: str | None = None,
             k: int | None = None, min_relevance: float = MIN_RELEVANCE) -> dict:
    """Plain-Python retrieval (used by the tool, the eval script and tests).

    Returns {"results": [...], "note": str|None, "filters": {...}}.
    """
    from backend.vectorstore import get_vectorstore  # heavy import, lazily

    k = max(1, min(k or settings.top_k, MAX_K))
    note = None

    canonical = resolve_drug_name(drug_name)
    if drug_name and canonical is None:
        return {"results": [], "filters": {"drug_name": drug_name, "section": section},
                "note": (f"'{drug_name}' is not in the verified knowledge base. "
                         f"Available drugs: {', '.join(known_drugs())}.")}

    if section and section not in SECTIONS:
        note = f"Unknown section '{section}' ignored. Valid sections: {', '.join(SECTIONS)}."
        section = None

    conditions = [{"drug_name": canonical}] if canonical else []
    if section:
        conditions.append({"section": section})
    where = (conditions[0] if len(conditions) == 1
             else {"$and": conditions} if conditions else None)

    # With BOTH a drug and a section filter, the metadata already guarantees the passages
    # are on-topic, so the score cutoff is skipped. WHY (found in Phase 7 evaluation): a
    # terse query like "boxed warning" scores only 0.10-0.14 even against the right
    # section, which overlaps nonsense queries (0.04-0.15), so no threshold can separate
    # them. The agent then refused a question it had the data for.
    if canonical and section:
        min_relevance = min(min_relevance, -1.0)

    hits = get_vectorstore().similarity_search_with_relevance_scores(query, k=k, filter=where)
    results = [
        {
            "rank": i,
            "score": round(score, 3),
            "drug_name": doc.metadata.get("drug_name"),
            "section": doc.metadata.get("section"),
            "source_url": doc.metadata.get("source_url"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "text": doc.page_content,
        }
        for i, (doc, score) in enumerate(hits, start=1)
        if score >= min_relevance
    ]
    if not results:
        note = ((note + " ") if note else "") + (
            "No sufficiently relevant passages found. Try rephrasing the query, "
            "removing the section filter, or checking the drug name.")
    return {"results": results, "note": note,
            "filters": {"drug_name": canonical, "section": section, "k": k}}


def format_for_llm(payload: dict) -> str:
    """Compact, numbered text the LLM reads and cites as [1], [2] ..."""
    parts = []
    if payload["note"]:
        parts.append(f"NOTE: {payload['note']}")
    for r in payload["results"]:
        if r.get("repeat"):  # already shown earlier in this turn: don't spend tokens twice
            parts.append(f"[{r['rank']}] (same passage as [{r['rank']}] shown earlier)")
            continue
        parts.append(f"[{r['rank']}] drug={r['drug_name']} | section={r['section']} | "
                     f"source={r['source_url']} | relevance={r['score']}\n{r['text']}")
    return "\n\n".join(parts) if parts else "No results."


class SearchDrugInfoInput(BaseModel):
    query: str = Field(description=(
        "A focused natural-language search query, e.g. 'maximum daily dose in adults' or "
        "'dose adjustment in renal impairment'. Include patient context (age group, "
        "kidney/liver function) when relevant."))
    drug_name: Optional[str] = Field(default=None, description=(
        "Generic or brand name of ONE drug to restrict the search to, e.g. 'warfarin', "
        "'paracetamol'. Leave empty to search across all drugs."))
    section: Optional[Section] = Field(default=None, description=(
        "Restrict to one label section. dosage = doses/administration; "
        "special_populations = renal/hepatic impairment, pediatric, geriatric, pregnancy; "
        "side_effects = adverse reactions; interactions = drug-drug interactions. "
        "Leave empty if unsure."))
    # No `le=` constraint on purpose: a model asking for k=10 should be clamped in
    # retrieve(), not rejected with a validation error that costs another LLM round-trip.
    k: int = Field(default=settings.top_k,
                   description=f"Number of passages to return (default {settings.top_k}, max {MAX_K}).")


@tool("search_drug_info", args_schema=SearchDrugInfoInput, response_format="content_and_artifact")
def search_drug_info(query: str, drug_name: Optional[str] = None,
                     section: Optional[str] = None, k: int = settings.top_k):
    """Search verified FDA drug-label passages (indications, dosage, contraindications,
    warnings, side effects, interactions, special populations, overdose, pharmacology)
    for the drugs in the MediDose knowledge base. Use this for ANY question about a
    drug's medical facts. Returns numbered passages with source URLs to cite."""
    payload = retrieve(query, drug_name=drug_name, section=section, k=k)
    return format_for_llm(payload), payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the drug index directly (no LLM).")
    parser.add_argument("query")
    parser.add_argument("--drug", default=None)
    parser.add_argument("--section", default=None)
    parser.add_argument("-k", type=int, default=settings.top_k)
    args = parser.parse_args()

    # Invoke through LangChain exactly as the agent will, using a ToolCall so we get
    # the artifact back as well as the text content.
    msg = search_drug_info.invoke({
        "type": "tool_call", "id": "cli", "name": "search_drug_info",
        "args": {"query": args.query, "drug_name": args.drug, "section": args.section, "k": args.k},
    })
    print(f"filters: {msg.artifact['filters']}\n")
    print(msg.content)
