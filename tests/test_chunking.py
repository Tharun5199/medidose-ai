"""Chunking / ingestion tests: pure functions, no index or model needed."""
from __future__ import annotations

from backend.ingest import (MIN_CHUNK_CHARS, chunk_document, load_and_chunk,
                            merge_small_pieces, parse_markdown)

SAMPLE = """# Testdrug
drug_name: testdrug
aliases: td, testo
generic_name: TESTDRUG HYDROCHLORIDE
product_type: HUMAN PRESCRIPTION DRUG
source_url: https://dailymed.example/testdrug

## dosage

2 DOSAGE AND ADMINISTRATION 2.1 Adults Take 10 mg once daily. {filler_a} 2.2 Renal Impairment Reduce the dose to 5 mg when CrCl is below 30 mL/min. {filler_b}

## side_effects

6 ADVERSE REACTIONS The most common adverse reactions are headache and nausea.
"""


def make_md(n_sentences: int = 25) -> str:
    filler = " ".join(f"Sentence number {i} about dosing details." for i in range(n_sentences))
    return SAMPLE.format(filler_a=filler, filler_b=filler)


def test_parse_markdown_reads_header_and_sections():
    meta, sections = parse_markdown(make_md())
    assert meta["drug_name"] == "testdrug"
    assert meta["source_url"] == "https://dailymed.example/testdrug"
    assert list(sections) == ["dosage", "side_effects"]
    assert sections["side_effects"].startswith("6 ADVERSE REACTIONS")


def test_chunks_never_mix_sections_and_carry_metadata():
    meta, sections = parse_markdown(make_md())
    docs = chunk_document(meta, sections, chunk_size=300, chunk_overlap=50)
    assert len(docs) > 2
    for d in docs:
        section = d.metadata["section"]
        assert d.page_content.startswith(f"[testdrug | {section}]\n")
        assert d.metadata["drug_name"] == "testdrug"
        assert d.metadata["source_url"] == "https://dailymed.example/testdrug"
        # Content from one section must not leak into another section's chunks.
        if section == "dosage":
            assert "ADVERSE REACTIONS" not in d.page_content
        else:
            assert "DOSAGE" not in d.page_content


def test_chunk_size_respected_and_metadata_types_are_chroma_safe():
    meta, sections = parse_markdown(make_md())
    docs = chunk_document(meta, sections, chunk_size=300, chunk_overlap=50)
    for d in docs:
        body = d.page_content.split("\n", 1)[1]
        # A merged heading fragment may push a chunk past chunk_size by < MIN_CHUNK_CHARS.
        assert len(body) <= 300 + MIN_CHUNK_CHARS
        assert all(isinstance(v, (str, int, float, bool)) for v in d.metadata.values())


def test_no_heading_only_chunks():
    """Regression: '2 DOSAGE AND ADMINISTRATION' used to become a chunk on its own."""
    meta, sections = parse_markdown(make_md(40))
    docs = chunk_document(meta, sections, chunk_size=300, chunk_overlap=50)
    bodies = [d.page_content.split("\n", 1)[1] for d in docs]
    assert all(len(b) >= MIN_CHUNK_CHARS for b in bodies)
    assert bodies[0].startswith("2 DOSAGE AND ADMINISTRATION 2.1 Adults")


def test_merge_small_pieces():
    assert merge_small_pieces(["HEADING", "a" * 60, "b" * 60]) == ["HEADING " + "a" * 60, "b" * 60]
    assert merge_small_pieces(["a" * 60, "( 1.13 )"]) == ["a" * 60 + " ( 1.13 )"]  # trailing -> previous
    assert merge_small_pieces(["tiny"]) == ["tiny"]  # never drop content


def test_overlap_between_consecutive_chunks():
    meta, sections = parse_markdown(make_md(40))
    docs = [d for d in chunk_document(meta, sections, chunk_size=300, chunk_overlap=80)
            if d.metadata["section"] == "dosage"]
    first, second = (d.page_content.split("\n", 1)[1] for d in docs[:2])
    tail_words = first.split()[-4:]
    assert " ".join(tail_words) in second, "consecutive chunks should share overlapping text"


def test_splits_at_label_subsection_numbers():
    """A long section should break right before '2.2 Renal Impairment', keeping it together."""
    meta, sections = parse_markdown(make_md(25))
    docs = chunk_document(meta, sections, chunk_size=800, chunk_overlap=0)
    assert any(d.page_content.split("\n", 1)[1].startswith("2.2 Renal Impairment") for d in docs)


def test_no_regex_separator_leaks_into_text():
    """Regression: keep_separator=False made LangChain join pieces with the literal pattern '\\s'."""
    meta, sections = parse_markdown(make_md(40))
    docs = chunk_document(meta, sections, chunk_size=200, chunk_overlap=20)
    assert not any("\\s" in d.page_content or "(?=" in d.page_content for d in docs)


def test_chunk_ids_are_unique_and_deterministic():
    meta, sections = parse_markdown(make_md())
    ids1 = [d.metadata["chunk_id"] for d in chunk_document(meta, sections, 300, 50)]
    ids2 = [d.metadata["chunk_id"] for d in chunk_document(meta, sections, 300, 50)]
    assert ids1 == ids2
    assert len(ids1) == len(set(ids1))
    assert ids1[0] == "testdrug:dosage:0"


def test_load_and_chunk_drops_exact_duplicates(tmp_path):
    """Regression: repeated label text filled 2 of the top-3 results before deduplication."""
    md = make_md()
    (tmp_path / "a.md").write_text(md, encoding="utf-8")
    _, single = load_and_chunk(tmp_path)
    (tmp_path / "b.md").write_text(md, encoding="utf-8")  # identical content -> identical chunks
    n_files, docs = load_and_chunk(tmp_path)
    assert n_files == 2
    assert len(docs) == len(single), "identical chunks must be indexed once"
