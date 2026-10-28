"""Phase 4 — The agent: an LLM that DECIDES whether (and how often) to retrieve.

Try it from the project root (needs GROQ_API_KEY in .env):
    python -m backend.agent --demo
    python -m backend.agent "What is the dose of gabapentin?" --renal
    python -m backend.agent "Paracetamol dose for a child" --age-group child --weight 20

The loop (a "ReAct-style" tool-calling loop):

    user msg ──► LLM (with tool schema) ──► tool_calls?  ── no ──► final answer
                     ▲                          │ yes
                     └── ToolMessage(results) ◄─┘  search_drug_info(...)

WHY a hand-written loop instead of a prebuilt agent class (e.g. LangChain's
create_agent): it is ~80 lines, every step is visible and logged, and it gives
us control we need for a medical app:
  - a hard cap on searches (bounded latency and Groq free-tier usage),
  - global citation numbering across multiple searches, so "[3]" in the answer
    maps to exactly one entry in the Sources list,
  - graceful recovery when the model sends invalid tool arguments,
  - deterministic post-processing (disclaimer, citation checks) in code.
"""
from __future__ import annotations

import argparse
import logging
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from backend.config import settings
from backend.prompts import DISCLAIMER, NO_INFO_SENTENCE, SYSTEM_PROMPT, build_patient_context
from backend.tools import format_for_llm, search_drug_info

log = logging.getLogger("medidose.agent")

MAX_SEARCHES = 3      # retrieval calls allowed per question (initial + 2 refinements)
MAX_HISTORY_TURNS = 6 # previous messages kept for follow-up questions


@dataclass
class AgentResult:
    answer: str
    sources: list[dict] = field(default_factory=list)
    retrieved: bool = False
    tool_calls: list[dict] = field(default_factory=list)
    # Numbers in the answer that appear in no retrieved passage (see check_numbers).
    unverified_numbers: list[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=1)
def get_llm():
    """Groq-hosted chat model. temperature=0: we want repeatable, conservative answers."""
    from langchain_groq import ChatGroq

    if not settings.groq_api_key or settings.groq_api_key.startswith("your_"):
        raise RuntimeError("GROQ_API_KEY is not set. Add it to .env (see .env.example).")
    extra = {"reasoning_effort": settings.groq_reasoning_effort} if settings.groq_reasoning_effort else {}
    # max_retries: the Groq client backs off and retries on 429s, which are routine on the
    # free tier (8,000 tokens/minute), instead of failing the request outright.
    return ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key,
                    temperature=0, max_retries=5, **extra)


class AgentError(RuntimeError):
    """Raised when the LLM provider fails (e.g. rate limit). The API maps it to HTTP 503."""


# gpt-oss models were trained to cite as 【1†L1-L4】; normalise any such marker to [1].
_NATIVE_CITATION = re.compile(r"【(\d+)(?:†[^】]*)?】")
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")  # [1] or [1, 3]


def normalise_citations(answer: str) -> tuple[str, set[int]]:
    """Return the answer with citations in [n] form, plus the set of cited numbers."""
    answer = _NATIVE_CITATION.sub(r"[\1]", answer)
    answer = re.sub(r"(\[\d+\])(?:\s*\1)+", r"\1", answer)  # "[3][3]" -> "[3]"
    cited = {int(n) for group in _CITATION.findall(answer) for n in group.split(",")}
    return answer, cited


def _numbers(text: str) -> list[str]:
    # Join digit groups like "1,500" / "3 750" (incl. narrow no-break spaces) before extracting.
    text = re.sub(r"(?<=\d)[,\s  ](?=\d{3}(?!\d))", "", text)
    return re.findall(r"\d+(?:\.\d+)?", text)


def check_numbers(answer: str, sources: list[dict], weight_kg: float | None = None) -> list[str]:
    """Deterministic grounding check: every number in the answer must come from a passage.

    WHY: in testing, the model wrote a full renal-dosing table from memory because the
    label's table is an image the text never contained, and still cited [1]. A prompt
    cannot fully prevent that; a cheap check in code can at least DETECT it. Doses are
    numbers, so numbers are the highest-risk facts in this domain.

    Allowed: numbers present in any retrieved passage, the patient's weight, and
    weight x a passage number (transparent mg/kg arithmetic the prompt permits).
    """
    allowed = {float(n) for s in sources for n in _numbers(s["text"])}
    if weight_kg:
        allowed |= {float(weight_kg)} | {round(a * weight_kg, 2) for a in allowed}
    body = _CITATION.sub("", answer)  # citation markers are not claims
    return sorted({n for n in _numbers(body) if float(n) not in allowed}, key=float)


def _history_to_messages(history: list[dict] | None) -> list[BaseMessage]:
    """Only the text of earlier turns is kept, not old tool results.

    WHY: old passages would bloat the prompt; if a follow-up needs facts, the
    agent simply searches again, so every answer is grounded in fresh retrieval.
    """
    msgs: list[BaseMessage] = []
    for turn in (history or [])[-MAX_HISTORY_TURNS:]:
        if turn.get("role") == "user":
            msgs.append(HumanMessage(turn.get("content", "")))
        elif turn.get("role") == "assistant":
            msgs.append(AIMessage(turn.get("content", "")))
    return msgs


class _SourceRegistry:
    """Assigns one global number per retrieved chunk across all searches in a turn."""

    def __init__(self) -> None:
        self.sources: list[dict] = []
        self._ids: dict[str, int] = {}

    def register(self, payload: dict) -> dict:
        renumbered = []
        for r in payload["results"]:
            repeat = r["chunk_id"] in self._ids
            if not repeat:
                self._ids[r["chunk_id"]] = len(self.sources) + 1
                self.sources.append({**r, "id": self._ids[r["chunk_id"]]})
            renumbered.append({**r, "rank": self._ids[r["chunk_id"]], "repeat": repeat})
        return {**payload, "results": renumbered}


def _run_tool(call: dict, registry: _SourceRegistry) -> tuple[ToolMessage, dict]:
    """Execute one tool call; never raise, so the model can see the error and retry."""
    try:
        msg = search_drug_info.invoke({**call, "type": "tool_call"})
        payload = registry.register(msg.artifact)
        content = format_for_llm(payload)
    except Exception as exc:  # e.g. pydantic validation error for a bad `section` value
        payload = {"results": [], "note": f"Tool error: {exc}", "filters": call.get("args", {})}
        content = f"Tool error: {exc}. Fix the arguments and try again."
    return ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]), payload


def run_agent(message: str, history: list[dict] | None = None,
              patient: dict | None = None, llm=None) -> AgentResult:
    llm = llm or get_llm()
    llm_with_tools = llm.bind_tools([search_drug_info])
    # Once the search budget is spent we still send the tool schema but forbid calling it.
    # (Dropping the tools entirely while the history contains tool calls made gpt-oss
    # return an empty answer in testing.)
    llm_answer_only = llm.bind_tools([search_drug_info], tool_choice="none")

    messages: list[BaseMessage] = [
        SystemMessage(SYSTEM_PROMPT),
        *_history_to_messages(history),
        HumanMessage(message + build_patient_context(patient)),
    ]
    registry = _SourceRegistry()
    trace: list[dict] = []

    log.info("USER: %s%s", message, f"  | patient={patient}" if patient else "")
    def invoke_llm(model, msgs) -> AIMessage:
        try:
            return model.invoke(msgs)
        except Exception as exc:  # rate limit after retries, network, provider outage
            log.error("LLM call failed: %s", exc)
            raise AgentError(f"The language model is unavailable ({type(exc).__name__}). "
                             "Please try again in a minute.") from exc

    while True:
        budget_left = len(trace) < MAX_SEARCHES
        ai = invoke_llm(llm_with_tools if budget_left else llm_answer_only, messages)
        messages.append(ai)

        if not ai.tool_calls:
            if not trace:
                log.info("[DECISION] NO RETRIEVAL - answering directly")
            break

        for call in ai.tool_calls:
            if len(trace) >= MAX_SEARCHES:  # model asked for more searches than allowed in one go
                messages.append(ToolMessage("Search limit reached. Answer with the passages you have.",
                                            tool_call_id=call["id"], name=call["name"]))
                continue
            kind = "RETRIEVE" if not trace else "RE-RETRIEVE (refined)"
            log.info("[DECISION] %s -> search_drug_info(%s)", kind,
                     ", ".join(f"{k}={v!r}" for k, v in call["args"].items()))
            tool_msg, payload = _run_tool(call, registry)
            messages.append(tool_msg)
            scores = [r["score"] for r in payload["results"]]
            log.info("    -> %d passages%s%s", len(scores),
                     f", top relevance {max(scores):.2f}" if scores else "",
                     f" | note: {payload['note'][:80]}" if payload.get("note") else "")
            trace.append({**call["args"], "n_results": len(scores), "note": payload.get("note")})

    answer = (ai.content if isinstance(ai.content, str) else str(ai.content)).strip()
    if not answer:
        # Models occasionally return an empty message after tool use; ask once more explicitly.
        log.warning("Empty answer from model; asking it to answer from the passages.")
        messages.append(HumanMessage("Now write the final answer using only the passages above, "
                                     f"with [n] citations, or reply exactly: {NO_INFO_SENTENCE}"))
        ai = invoke_llm(llm_answer_only, messages)
        answer = (ai.content if isinstance(ai.content, str) else str(ai.content)).strip() or NO_INFO_SENTENCE

    sources = [{k: v for k, v in s.items() if k not in ("rank", "repeat")} for s in registry.sources]

    # Post-check citations in code: mark which sources were cited and flag any [n]
    # that points at nothing (a fabricated citation is a hallucination signal).
    answer, cited = normalise_citations(answer)
    for s in sources:
        s["cited"] = s["id"] in cited
    bogus = cited - {s["id"] for s in sources}
    if bogus:
        log.warning("Answer cites non-existent sources: %s", sorted(bogus))
    if trace and not sources and NO_INFO_SENTENCE not in answer:
        log.warning("Retrieval found nothing but the model answered anyway; check grounding.")
    unverified = check_numbers(answer, sources, (patient or {}).get("weight_kg")) if trace else []
    if unverified:
        log.warning("UNVERIFIED NUMBERS (not found in any retrieved passage): %s", unverified)

    log.info("ANSWER: %d chars, %d searches, %d sources (%d cited)",
             len(answer), len(trace), len(sources), sum(s["cited"] for s in sources))
    return AgentResult(answer=answer, sources=sources, retrieved=bool(trace), tool_calls=trace,
                       unverified_numbers=unverified)


DEMO = [
    ("Hi! What can you help me with?", None),
    ("What are the common side effects of atorvastatin?", None),
    ("What is the dose of gabapentin?", {"age_group": "adult", "renal_impairment": True}),
    ("What is the paracetamol dose for a child?", {"age_group": "child", "weight_kg": 20}),
    ("Can I take ibuprofen together with lisinopril?", None),
    ("What is the recommended dose of semaglutide?", None),
]


def _print_result(question: str, patient: dict | None, result: AgentResult) -> None:
    print("\n" + "=" * 90)
    print(f"Q: {question}" + (f"   patient={patient}" if patient else ""))
    print(f"retrieved={result.retrieved}  searches={len(result.tool_calls)}")
    print("-" * 90)
    print(result.answer)
    if result.sources:
        print("\nSources:")
        for s in result.sources:
            print(f"  [{s['id']}]{'*' if s['cited'] else ' '} {s['drug_name']}/{s['section']} "
                  f"(rel {s['score']:.2f})  {s['source_url']}")
    if result.unverified_numbers:
        print(f"\nWARNING: numbers not found in any source: {', '.join(result.unverified_numbers)}")
    print(f"\n{result.disclaimer}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "sentence_transformers", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Ask MediDose AI a question.")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--demo", action="store_true", help="run the scripted demo questions")
    parser.add_argument("--age-group", choices=["neonate", "infant", "child", "adolescent", "adult", "elderly"])
    parser.add_argument("--weight", type=float, help="weight in kg")
    parser.add_argument("--renal", action="store_true", help="renal impairment")
    parser.add_argument("--hepatic", action="store_true", help="hepatic impairment")
    args = parser.parse_args()

    if args.demo:
        runs = DEMO
    elif args.question:
        patient = {"age_group": args.age_group, "weight_kg": args.weight,
                   "renal_impairment": args.renal, "hepatic_impairment": args.hepatic}
        runs = [(args.question, patient if any(patient.values()) else None)]
    else:
        parser.error("give a question or --demo")

    for q, p in runs:
        try:
            _print_result(q, p, run_agent(q, patient=p))
        except AgentError as exc:
            print(f"\nQ: {q}\nERROR: {exc}")
