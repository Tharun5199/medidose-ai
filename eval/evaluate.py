"""Phase 7 — Evaluation.

Run from the project root:
    python -m eval.evaluate --retrieval-only   # no LLM: fast, free, deterministic
    python -m eval.evaluate                    # full: retrieval + agent + LLM-judge faithfulness
    python -m eval.evaluate --report           # re-print the last saved run (no API calls)
    python -m eval.evaluate --only q04 q06     # a subset

Three layers are measured separately, so a bad answer can be traced to its cause
(did retrieval miss, did the agent search badly, or did the LLM ignore good passages?):

1. RETRIEVAL (no LLM): the raw question is embedded and searched
   a) unfiltered, as a plain RAG system would, and
   b) filtered by the expected drug, as the agent does when it fills drug_name.
   Metrics: drug hit@k, drug+section hit@k, MRR, fact recall@k (are the expected
   facts present in the retrieved text at all?).
2. AGENT: routing accuracy (retrieved when it should / didn't when it shouldn't),
   whether its own searches hit the expected section, answer fact recall,
   correct refusal for out-of-scope drugs, citation validity and coverage,
   numbers not found in sources.
3. FAITHFULNESS (LLM-as-judge): a separate model splits the answer into claims and
   labels each SUPPORTED / UNSUPPORTED by the retrieved passages.
   Faithfulness = supported claims / all claims.

WHY a different, smaller judge model (gpt-oss-20b): a model grading its own output
tends to be lenient (self-preference bias); a separate model also draws on a
separate Groq free-tier rate-limit bucket.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import time
import unicodedata
from datetime import datetime
from pathlib import Path

from backend.config import PROJECT_ROOT, settings
from backend.ingest import parse_markdown
from backend.prompts import NO_INFO_SENTENCE

EVAL_DIR = PROJECT_ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"
K = settings.top_k
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "openai/gpt-oss-20b")
REFUSAL_RE = re.compile(r"don.t have verified information|not in (the|our) (verified )?knowledge base", re.I)


def load_questions() -> list[dict]:
    return json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))


def normalise(text: str) -> str:
    """Canonical form for fact matching.

    WHY (a scorer bug found in the first full run): gpt-oss writes "4000<U+202F>mg" (narrow
    no-break space), "life<U+2011>threatening" (non-breaking hyphen) and "15 mg /kg", so correct
    answers scored 0%. NFKC folds the Unicode variants; then spacing around "/" is removed.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)  # Unicode hyphens/dashes/minus -> "-"
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"\s*/\s*", "/", text)


def fact_hits(text: str, facts: list[list[str]]) -> list[bool]:
    """Each fact is a list of acceptable wordings; a fact counts if any wording appears."""
    low = normalise(text)
    return [any(normalise(v) in low for v in variants) for variants in facts]


def ratio(hits: list[bool]) -> float | None:
    return sum(hits) / len(hits) if hits else None


# ---------------------------------------------------------------- gold-set check

def validate_gold(questions: list[dict]) -> list[str]:
    """Every expected fact must exist in the drug's label. Otherwise the gold set is wrong."""
    problems = []
    for q in questions:
        if not q.get("expected_drug"):
            continue
        _, sections = parse_markdown((settings.raw_dir / f"{q['expected_drug']}.md").read_text(encoding="utf-8"))
        label = " ".join(sections.values())
        for variants, ok in zip(q["expected_facts"], fact_hits(label, q["expected_facts"])):
            if not ok:
                problems.append(f"{q['id']}: none of {variants} appears in the {q['expected_drug']} label "
                                "(acceptable only if the fact is computed, e.g. mg/kg x weight)")
    return problems


# ---------------------------------------------------------------- layer 1: retrieval

def eval_retrieval(q: dict) -> dict:
    from backend.tools import retrieve

    out = {}
    for mode, drug in (("unfiltered", None), ("drug_filtered", q["expected_drug"])):
        results = retrieve(q["question"], drug_name=drug, k=K, min_relevance=-1)["results"]
        match_ranks = [r["rank"] for r in results
                       if r["drug_name"] == q["expected_drug"] and r["section"] in q["expected_sections"]]
        out[mode] = {
            "drug_hit": any(r["drug_name"] == q["expected_drug"] for r in results),
            "section_hit": bool(match_ranks),
            "rr": 1 / match_ranks[0] if match_ranks else 0.0,
            "fact_recall": ratio(fact_hits(" ".join(r["text"] for r in results), q["expected_facts"])),
            "top": [f"{r['drug_name']}/{r['section']}" for r in results],
        }
    return out


# ---------------------------------------------------------------- layer 3: judge

JUDGE_PROMPT = """You are a strict fact-checker for a medical education app.
Split the ANSWER into its individual factual claims (ignore greetings, headings and statements
that information is unavailable). For each claim decide whether it is SUPPORTED by the PASSAGES:
supported means the passages state it or it follows by simple arithmetic from numbers in the
passages and the patient data in the question. Anything else, including facts that are true in
general but absent from the passages, is UNSUPPORTED.

Return JSON only: {{"claims": [{{"claim": "...", "supported": true, "reason": "..."}}]}}

QUESTION: {question}

PASSAGES:
{passages}

ANSWER:
{answer}"""


def judge_faithfulness(question: str, answer: str, sources: list[dict]) -> dict:
    from langchain_groq import ChatGroq

    judge = ChatGroq(model=JUDGE_MODEL, api_key=settings.groq_api_key, temperature=0, max_retries=6,
                     reasoning_effort="low").bind(response_format={"type": "json_object"})
    passages = "\n\n".join(f"[{s['id']}] {s['text']}" for s in sources)
    msg = judge.invoke(JUDGE_PROMPT.format(question=question, passages=passages, answer=answer))
    try:
        claims = json.loads(msg.content)["claims"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"faithfulness": None, "claims": [], "error": msg.content[:200]}
    supported = [c for c in claims if c.get("supported")]
    return {"faithfulness": len(supported) / len(claims) if claims else None,
            "claims": claims, "n_claims": len(claims)}


# ---------------------------------------------------------------- layer 2: agent

def citation_coverage(answer: str) -> float | None:
    """Share of factual-looking lines/sentences that carry at least one [n] citation."""
    units = [u for u in re.split(r"(?<=[.!?])\s+|\n", answer)
             if len(re.sub(r"[^A-Za-z]", "", u)) > 25 and not u.strip().startswith(("#", "|", "**") )]
    return ratio([bool(re.search(r"\[\d+\]", u)) for u in units])


def eval_agent(q: dict, judge: bool) -> dict:
    from backend.agent import run_agent

    t0 = time.perf_counter()
    res = run_agent(q["question"], patient=q.get("patient"))
    expect_retrieval = q.get("expect_retrieval", True)
    out = {
        "seconds": round(time.perf_counter() - t0, 1),
        "answer": res.answer,
        "retrieved": res.retrieved,
        "routing_ok": res.retrieved == expect_retrieval,
        "searches": [f"{t.get('drug_name')}/{t.get('section')}: {t.get('query')}" for t in res.tool_calls],
        "n_sources": len(res.sources),
        "unverified_numbers": res.unverified_numbers,
    }
    if q.get("expect_refusal"):
        out["refusal_ok"] = bool(REFUSAL_RE.search(res.answer))
    if q.get("expected_drug"):
        out["agent_section_hit"] = any(s["drug_name"] == q["expected_drug"] and s["section"] in q["expected_sections"]
                                       for s in res.sources)
        out["answer_fact_recall"] = ratio(fact_hits(res.answer, q["expected_facts"]))
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", res.answer)}
        out["citations_valid"] = cited <= {s["id"] for s in res.sources}
        out["citation_coverage"] = citation_coverage(res.answer)
        if judge and res.sources and NO_INFO_SENTENCE not in res.answer:
            out["judge"] = judge_faithfulness(q["question"], res.answer, res.sources)
    return out


# ---------------------------------------------------------------- report

def mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if vals else None


def pct(v) -> str:
    return "  -  " if v is None else f"{v:5.0%}"


def yn(v) -> str:
    return " - " if v is None else ("yes" if v else "NO ")


def print_report(run: dict) -> None:
    rows = run["results"]
    drug_rows = [r for r in rows if r["expected_drug"]]

    print(f"\nMediDose AI evaluation | {run['timestamp']} | agent={run['agent_model']} "
          f"judge={run.get('judge_model')} | k={run['k']}")

    print("\n=== 1. Retrieval (no LLM) ===")
    print(f"{'id':<4} {'category':<20} {'unfiltered: drug sect  facts':<30} {'drug-filtered: sect  facts':<28}")
    for r in drug_rows:
        u, f = r["retrieval"]["unfiltered"], r["retrieval"]["drug_filtered"]
        print(f"{r['id']:<4} {r['category']:<20} {yn(u['drug_hit']):>13} {yn(u['section_hit']):>4} {pct(u['fact_recall']):>6}"
              f"   {yn(f['section_hit']):>17} {pct(f['fact_recall']):>6}")
    for mode in ("unfiltered", "drug_filtered"):
        rs = [r["retrieval"][mode] for r in drug_rows]
        print(f"  {mode:<14} drug hit@{K} {pct(mean(x['drug_hit'] for x in rs))} | "
              f"drug+section hit@{K} {pct(mean(x['section_hit'] for x in rs))} | "
              f"MRR {mean(x['rr'] for x in rs):.2f} | fact recall@{K} {pct(mean(x['fact_recall'] for x in rs))}")

    if not any("agent" in r for r in rows):
        return

    print("\n=== 2-3. Agent + faithfulness ===")
    print(f"{'id':<4} {'category':<20} {'route':>5} {'srch':>4} {'hit':>4} {'facts':>6} {'cite%':>6} "
          f"{'valid':>5} {'faith':>6} {'claims':>6} {'unver':>5} {'secs':>5}")
    for r in rows:
        a = r.get("agent")
        if not a:
            continue
        j = a.get("judge") or {}
        extra = f"  refusal {yn(a['refusal_ok'])}" if "refusal_ok" in a else ""
        print(f"{r['id']:<4} {r['category']:<20} {yn(a['routing_ok']):>5} {len(a['searches']):>4} "
              f"{yn(a.get('agent_section_hit')):>4} {pct(a.get('answer_fact_recall')):>6} "
              f"{pct(a.get('citation_coverage')):>6} {yn(a.get('citations_valid')):>5} "
              f"{pct(j.get('faithfulness')):>6} {j.get('n_claims', '-'):>6} {len(a['unverified_numbers']):>5} "
              f"{a['seconds']:>5}{extra}")

    agents = [r["agent"] for r in rows if r.get("agent")]
    drug_agents = [r["agent"] for r in drug_rows if r.get("agent")]
    faith = [a["judge"]["faithfulness"] for a in drug_agents if a.get("judge")]
    claims = [c for a in drug_agents if a.get("judge") for c in a["judge"]["claims"]]
    unsupported = [c for c in claims if not c.get("supported")]

    print("\nSummary")
    print(f"  Routing accuracy (retrieve vs. not)   {pct(mean(a['routing_ok'] for a in agents))}  ({len(agents)} questions)")
    print(f"  Agent retrieval hit (drug+section)    {pct(mean(a.get('agent_section_hit') for a in drug_agents))}")
    print(f"  Answer fact recall                    {pct(mean(a.get('answer_fact_recall') for a in drug_agents))}")
    print(f"  Out-of-scope refusal                  {pct(mean(a['refusal_ok'] for a in agents if 'refusal_ok' in a))}")
    print(f"  Citations valid (no fake [n])         {pct(mean(a.get('citations_valid') for a in drug_agents))}")
    print(f"  Citation coverage (sentences cited)   {pct(mean(a.get('citation_coverage') for a in drug_agents))}")
    print(f"  Faithfulness (LLM judge, mean)        {pct(mean(faith))}  over {len(claims)} claims, "
          f"{len(unsupported)} unsupported")
    print(f"  Answers with unverified numbers       {sum(bool(a['unverified_numbers']) for a in drug_agents)}/{len(drug_agents)}")
    print(f"  Mean latency                          {mean(a['seconds'] for a in agents):.1f}s (incl. free-tier rate-limit waits)")
    if unsupported:
        print("\nUnsupported claims flagged by the judge:")
        for r in drug_rows:
            for c in ((r.get("agent") or {}).get("judge") or {}).get("claims", []):
                if not c.get("supported"):
                    print(f"  {r['id']}: {c.get('claim', '')[:110]}  <- {c.get('reason', '')[:80]}")


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MediDose AI.")
    parser.add_argument("--retrieval-only", action="store_true", help="skip the LLM (agent + judge)")
    parser.add_argument("--no-judge", action="store_true", help="run the agent but skip the LLM judge")
    parser.add_argument("--report", action="store_true", help="print the last saved run and exit")
    parser.add_argument("--only", nargs="*", help="question ids to run, e.g. q04 q06")
    args = parser.parse_args()

    latest = RESULTS_DIR / "latest.json"
    if args.report:
        # Re-score saved answers with the CURRENT scoring code: scorer fixes need no new LLM calls.
        run = json.loads(latest.read_text(encoding="utf-8"))
        gold = {q["id"]: q for q in load_questions()}
        for r in run["results"]:
            a, q = r.get("agent"), gold.get(r["id"])
            if a and q and q.get("expected_drug"):
                a["answer_fact_recall"] = ratio(fact_hits(a["answer"], q["expected_facts"]))
                a["citation_coverage"] = citation_coverage(a["answer"])
        print_report(run)
        return

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    questions = [q for q in load_questions() if not args.only or q["id"] in args.only]

    problems = validate_gold(questions)
    print(f"Gold set: {len(questions)} questions; {len(problems)} fact(s) not literally in the labels")
    for p in problems:
        print("  note:", p)

    results = []
    for i, q in enumerate(questions, 1):
        row = {k: q.get(k) for k in ("id", "category", "question", "expected_drug", "expected_sections")}
        if q.get("expected_drug"):
            row["retrieval"] = eval_retrieval(q)
        if not args.retrieval_only:
            print(f"[{i}/{len(questions)}] {q['id']} {q['question'][:70]}", flush=True)
            try:
                row["agent"] = eval_agent(q, judge=not args.no_judge)
            except Exception as exc:  # keep going: one rate-limit failure shouldn't lose the run
                print(f"  failed: {exc}")
        results.append(row)

    run = {"timestamp": datetime.now().isoformat(timespec="seconds"), "k": K,
           "agent_model": settings.groq_model,
           "judge_model": None if args.retrieval_only or args.no_judge else JUDGE_MODEL,
           "results": results}
    RESULTS_DIR.mkdir(exist_ok=True)
    if not args.retrieval_only:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        for path in (RESULTS_DIR / f"run-{stamp}.json", latest):
            path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    print_report(run)


if __name__ == "__main__":
    main()
