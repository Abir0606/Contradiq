from contractiq.ingest.chunker import chunk_contract

SAMPLE = """MASTER SERVICES AGREEMENT

This Agreement is entered into as of May 8, 2014.

ARTICLE 1 DEFINITIONS

1.1 "Confidential Information" means all non-public information disclosed by either party.

1.2 "Services" means the services described in Exhibit A.

ARTICLE 2 TERM AND TERMINATION

2.1 Term. This Agreement shall commence on the Effective Date and continue for three (3) years.

2.2 Termination for Convenience. Either party may terminate this Agreement upon sixty (60) days prior written notice.
"""


def test_chunks_have_breadcrumbs() -> None:
    chunks = chunk_contract(SAMPLE)
    assert len(chunks) >= 3
    breadcrumbs = {c.section_breadcrumb for c in chunks}
    assert any("DEFINITIONS" in b for b in breadcrumbs)
    assert any("TERM AND TERMINATION" in b for b in breadcrumbs)


def test_no_chunk_exceeds_limit() -> None:
    big_text = SAMPLE * 200
    chunks = chunk_contract(big_text, chunk_size_tokens=800, chunk_overlap_tokens=100)
    import tiktoken

    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    assert all(len(enc.encode(c.text)) <= 800 for c in chunks)


def test_preamble_labeled() -> None:
    chunks = chunk_contract(SAMPLE)
    assert any("Preamble" in c.section_breadcrumb for c in chunks)
