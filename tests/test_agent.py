"""Agent loop tests with a scripted fake LLM (no network, deterministic)."""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from backend.agent import MAX_SEARCHES, check_numbers, normalise_citations, run_agent
from backend.prompts import DISCLAIMER, NO_INFO_SENTENCE, build_patient_context
from tests.conftest import tool_call


def test_chit_chat_answers_without_retrieval(fake_llm):
    llm = fake_llm([AIMessage("Hello! Ask me about a drug.")])
    res = run_agent("hi", llm=llm)
    assert res.retrieved is False
    assert res.sources == [] and res.tool_calls == []
    assert res.disclaimer == DISCLAIMER  # added by code, never by the LLM


@pytest.mark.index
def test_drug_question_retrieves_and_marks_cited_sources(fake_llm):
    llm = fake_llm([
        AIMessage("", tool_calls=[tool_call("a", query="bleeding risk", drug_name="warfarin",
                                            section="interactions", k=2)]),
        AIMessage("Some drugs increase bleeding risk [1]."),
    ])
    res = run_agent("warfarin interactions?", llm=llm)
    assert res.retrieved is True
    assert [s["id"] for s in res.sources] == [1, 2]
    assert [s["cited"] for s in res.sources] == [True, False]
    # The tool result really was fed back to the model as a ToolMessage.
    assert isinstance(llm.calls[1][-1], ToolMessage)


@pytest.mark.index
def test_invalid_tool_args_are_returned_to_model_and_it_recovers(fake_llm):
    llm = fake_llm([
        AIMessage("", tool_calls=[tool_call("a", query="side effects", drug_name="atorvastatin",
                                            section="adverse_effects")]),
        AIMessage("", tool_calls=[tool_call("b", query="side effects", drug_name="atorvastatin",
                                            section="side_effects", k=1)]),
        AIMessage("Myalgia [1]."),
    ])
    res = run_agent("atorvastatin side effects", llm=llm)
    error_msg = [m for m in llm.calls[1] if isinstance(m, ToolMessage)][0]
    assert "Tool error" in error_msg.content
    assert len(res.sources) == 1


@pytest.mark.index
def test_citation_numbers_continue_across_searches(fake_llm):
    llm = fake_llm([
        AIMessage("", tool_calls=[tool_call("a", query="dose", drug_name="warfarin", section="dosage", k=2)]),
        AIMessage("", tool_calls=[tool_call("b", query="bleeding", drug_name="warfarin",
                                            section="interactions", k=2)]),
        AIMessage("Dose [1]. Bleeding [3]."),
    ])
    res = run_agent("warfarin", llm=llm)
    assert [s["id"] for s in res.sources] == [1, 2, 3, 4]
    second_result = [m for m in llm.calls[2] if isinstance(m, ToolMessage)][-1].content
    assert second_result.startswith("[3]")


@pytest.mark.index
def test_search_budget_is_capped(fake_llm):
    llm = fake_llm([
        AIMessage("", tool_calls=[tool_call(str(i), query="dose", drug_name="warfarin", k=1) for i in range(5)]),
        AIMessage("Done [1]."),
    ])
    res = run_agent("warfarin dose", llm=llm)
    assert len(res.tool_calls) == MAX_SEARCHES
    limit_msgs = [m for m in llm.calls[1] if isinstance(m, ToolMessage) and "limit reached" in m.content]
    assert len(limit_msgs) == 5 - MAX_SEARCHES
    # After the budget is spent the model is called with tool_choice="none".
    assert {"tool_choice": "none"} in llm.bound_kwargs


def test_empty_answer_falls_back_to_no_info(fake_llm):
    llm = fake_llm([AIMessage(""), AIMessage("")])
    res = run_agent("something", llm=llm)
    assert res.answer == NO_INFO_SENTENCE


def test_patient_context_reaches_the_prompt(fake_llm):
    llm = fake_llm([AIMessage("ok")])
    run_agent("gabapentin dose?", patient={"age_group": "adult", "weight_kg": 70, "renal_impairment": True},
              llm=llm)
    user_msg = llm.calls[0][-1].content
    assert "PATIENT CONTEXT" in user_msg and "70 kg" in user_msg and "Renal" in user_msg


def test_patient_context_empty_when_nothing_set():
    assert build_patient_context(None) == ""
    assert build_patient_context({"age_group": None, "renal_impairment": False}) == ""


def test_history_is_passed_as_text_turns(fake_llm):
    llm = fake_llm([AIMessage("ok")])
    run_agent("and its side effects?", history=[{"role": "user", "content": "tell me about metformin"},
                                                {"role": "assistant", "content": "Metformin is..."}], llm=llm)
    assert [type(m).__name__ for m in llm.calls[0]] == ["SystemMessage", "HumanMessage", "AIMessage", "HumanMessage"]


# ---------------- pure post-processing helpers

def test_normalise_gpt_oss_native_citations():
    answer, cited = normalise_citations("Dose 15 mg/kg【1†L1-L4】【2†L3】. Max【1†L5】【1†L5】.")
    assert answer == "Dose 15 mg/kg[1][2]. Max[1]."
    assert cited == {1, 2}


def test_citation_lists_are_parsed():
    _, cited = normalise_citations("See [1, 3] and [4].")
    assert cited == {1, 3, 4}


SOURCES = [{"text": "Children: 15 mg/kg every 6 hours to a maximum of 75 mg/kg per day. Max 3750 mg."}]


def test_check_numbers_accepts_source_numbers_and_weight_arithmetic():
    answer = "15 mg/kg x 20 kg = 300 mg [1]; max 75 x 20 = 1,500 mg; never above 3 750 mg [2]."
    assert check_numbers(answer, SOURCES, weight_kg=20) == []


def test_check_numbers_flags_invented_values():
    """Regression (Phase 4): the model wrote a gabapentin CrCl table from memory."""
    answer = "If CrCl is 30-59 give 400 mg every 48 hours [1]."
    assert check_numbers(answer, SOURCES) == ["30", "48", "59", "400"]
