"""Phase 2 — Ingestion: load -> chunk -> embed -> index in ChromaDB with metadata.

Run from the project root:
    python -m backend.ingest            # rebuild the index from data/raw/*.md
    python -m backend.ingest --keep     # upsert into the existing index instead of rebuilding

Pipeline:
  1. LOAD     each data/raw/<drug>.md (written in Phase 1) and parse its metadata header.
  2. SPLIT #1 on the '## <section>' headings -> one block per clinical section.
              WHY first: guarantees a chunk never straddles two sections, so each chunk
              gets exactly one honest `section` tag for metadata filtering.
  3. SPLIT #2 each section with a recursive splitter (CHUNK_SIZE chars, CHUNK_OVERLAP
              overlap), preferring label subsection boundaries like "2.3 Renal Impairment".
  4. ENRICH   prefix every chunk with "[drug | section]" and attach metadata
              (drug_name, section, source_url, ...).
  5. EMBED + INDEX into a persistent Chroma collection with deterministic IDs.
"""
from __future__ import annotations

import argparse
import statistics
import time
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import settings

# Separators are tried in order: the splitter only falls back to the next one if a
# piece is still bigger than chunk_size. So we cut at the most meaningful
# boundary available. openFDA flattens labels into long paragraphs, but keeps
# the numbered subsection markers ("2.1 General Dosing", "8.6 Renal Impairment"),
# so we split just before those first, then at paragraphs, lines, sentences, words.
SEPARATORS = [
    r"\n\n",
    r"\s(?=\d{1,2}\.\d{1,2}\s+[A-Z])",  # before a numbered subsection like "8.6 Renal..."
    r"\n",
    r"(?<=[.;])\s",                      # sentence boundary
    r"\s",
    r"",
]

# all-MiniLM-L6-v2 truncates input after 256 word-piece tokens: anything beyond
# that is silently NOT embedded. We measure how many chunks exceed it.
EMBED_MAX_TOKENS = 256

# Fragments shorter than this are merged into the following chunk. WHY (found by a
# Phase 8 unit test): splitting before "2.1 ..." leaves bare headings such as
# "12 CLINICAL PHARMACOLOGY" or "( 1.13 )" as chunks of their own. They hold no facts
# but can still occupy top-k slots. Merged, the heading stays with the content it introduces.
MIN_CHUNK_CHARS = 50


def parse_markdown(text: str) -> tuple[dict, dict[str, str]]:
    """Split a Phase-1 .md file into (metadata header, {section_name: section_text})."""
    header, _, body = text.partition("\n## ")
    meta: dict[str, str] = {}
    for line in header.splitlines():
        key, sep, value = line.partition(": ")
        if sep and not line.startswith("#"):
            meta[key.strip()] = value.strip()

    sections: dict[str, str] = {}
    for block in ("## " + body).split("\n## ") if body else []:
        name, _, content = block.removeprefix("## ").partition("\n")
        if content.strip():
            sections[name.strip()] = content.strip()
    return meta, sections


def make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        separators=SEPARATORS,
        is_separator_regex=True,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Must be True with regex separators: with False, LangChain re-joins pieces
        # using the separator *pattern* itself (a literal "\s" would leak into chunks).
        keep_separator=True,
        strip_whitespace=True,
    )


def merge_small_pieces(pieces: list[str], min_chars: int = MIN_CHUNK_CHARS) -> list[str]:
    """Attach fragments shorter than min_chars to the next piece (or the previous one if last)."""
    merged: list[str] = []
    carry = ""
    for piece in pieces:
        piece = f"{carry} {piece}" if carry else piece
        carry = ""
        if len(piece) < min_chars:
            carry = piece
        else:
            merged.append(piece)
    if carry:
        if merged:
            merged[-1] = f"{merged[-1]} {carry}"
        else:
            merged.append(carry)  # a section that is only a short fragment is kept as-is
    return merged


def chunk_document(meta: dict, sections: dict[str, str],
                   chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[Document]:
    """Turn one drug's sections into LangChain Documents with metadata."""
    splitter = make_splitter(chunk_size or settings.chunk_size,
                             settings.chunk_overlap if chunk_overlap is None else chunk_overlap)
    drug = meta["drug_name"]
    docs: list[Document] = []
    for section, text in sections.items():
        for i, piece in enumerate(merge_small_pieces(splitter.split_text(text))):
            # WHY the prefix: a chunk from deep inside a label often never names the
            # drug ("Reduce the dose in patients with CrCl < 30 mL/min"). Putting
            # drug + section in the text gets them into the embedding, so the query
            # "metformin renal dosing" can match it, and the LLM always knows
            # which drug a chunk is about.
            content = f"[{drug} | {section}]\n{piece}"
            docs.append(Document(
                page_content=content,
                # Chroma metadata values must be str/int/float/bool, so lists are joined.
                metadata={
                    "drug_name": drug,
                    "section": section,
                    "source_url": meta.get("source_url", ""),
                    "generic_name": meta.get("generic_name", ""),
                    "product_type": meta.get("product_type", ""),
                    "aliases": meta.get("aliases", ""),
                    "chunk_index": i,
                    # Deterministic ID: re-running ingestion overwrites instead of duplicating.
                    "chunk_id": f"{drug}:{section}:{i}",
                },
            ))
    return docs


def load_and_chunk(raw_dir: Path | None = None) -> tuple[int, list[Document]]:
    files = sorted((raw_dir or settings.raw_dir).glob("*.md"))
    docs: list[Document] = []
    seen: set[str] = set()
    for path in files:
        meta, sections = parse_markdown(path.read_text(encoding="utf-8"))
        for doc in chunk_document(meta, sections):
            # Safety net: drop exact duplicate chunks (labels repeat boilerplate),
            # otherwise identical text can occupy several of the k retrieval slots.
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                docs.append(doc)
    return len(files), docs


def compute_stats(n_files: int, docs: list[Document], seconds: float) -> dict:
    from transformers import AutoTokenizer  # installed with sentence-transformers

    tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
    sizes = [len(d.page_content) for d in docs]
    tokens = [len(tokenizer.encode(d.page_content)) for d in docs]
    over = sum(t > EMBED_MAX_TOKENS for t in tokens)
    return {
        "documents": n_files,
        "chunks": len(docs),
        "avg_chunk_chars": round(statistics.mean(sizes)),
        "median_chunk_chars": round(statistics.median(sizes)),
        "min_chunk_chars": min(sizes),
        "max_chunk_chars": max(sizes),
        "avg_chunk_tokens": round(statistics.mean(tokens)),
        "max_chunk_tokens": max(tokens),
        "chunks_over_token_limit": over,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "chunks_per_section": dict(Counter(d.metadata["section"] for d in docs).most_common()),
        "seconds": round(seconds, 1),
    }


def print_stats(st: dict) -> None:
    print("\n=== Ingestion stats ===")
    print(f"Documents (drugs)     : {st['documents']}")
    print(f"Chunks                : {st['chunks']}  (avg {st['chunks'] / max(st['documents'], 1):.1f} per drug)")
    print(f"Chunk size (chars)    : avg {st['avg_chunk_chars']}, median {st['median_chunk_chars']}, "
          f"min {st['min_chunk_chars']}, max {st['max_chunk_chars']}")
    print(f"Chunk size (tokens)   : avg {st['avg_chunk_tokens']}, max {st['max_chunk_tokens']}")
    print(f"Over {EMBED_MAX_TOKENS}-token limit  : {st['chunks_over_token_limit']} chunks "
          f"({st['chunks_over_token_limit'] / st['chunks']:.1%}) <- these get truncated by the embedder")
    print(f"Chunk size / overlap  : {st['chunk_size']} / {st['chunk_overlap']} chars")
    print("Chunks per section    : " + ", ".join(f"{s}={c}" for s, c in st["chunks_per_section"].items()))
    print(f"Index location        : {settings.chroma_dir} (collection '{settings.chroma_collection}')")
    print(f"Time                  : {st['seconds']}s")


def build_index(reset: bool = True, batch_size: int = 256, verbose: bool = True) -> dict:
    """Rebuild (or upsert into) the index. Returns the stats dict (used by POST /ingest)."""
    from backend.vectorstore import get_vectorstore  # heavy import, only when indexing

    t0 = time.perf_counter()
    n_files, docs = load_and_chunk()
    if not docs:
        raise FileNotFoundError(f"No .md files in {settings.raw_dir}. Run: python -m data.download_openfda")
    if verbose:
        print(f"Loaded {n_files} drug documents -> {len(docs)} chunks. Embedding + indexing...")

    store = get_vectorstore()
    if reset:
        # A full rebuild keeps the index consistent with data/raw (e.g. if a drug was
        # removed, or chunk size changed and old chunk IDs would otherwise linger).
        store.delete_collection()
        store = get_vectorstore()

    for start in range(0, len(docs), batch_size):
        batch = docs[start:start + batch_size]
        store.add_documents(batch, ids=[d.metadata["chunk_id"] for d in batch])
        if verbose:
            print(f"  indexed {min(start + batch_size, len(docs))}/{len(docs)}", end="\r")

    return compute_stats(n_files, docs, time.perf_counter() - t0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the MediDose AI vector index.")
    parser.add_argument("--keep", action="store_true", help="upsert into the existing index instead of rebuilding")
    try:
        print_stats(build_index(reset=not parser.parse_args().keep))
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
