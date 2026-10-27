import pytest

from app.rag.pdf import PageText, PDFError, chunk_pages, extract_pages
from app.rag.store import DocumentStore
from scripts.make_samples import build_pdf
from tests.helpers import SAMPLES

SAMPLE_PDF = (SAMPLES / "gnn_primer.pdf").read_bytes()


def test_extract_pages_from_sample():
    pages, title = extract_pages(SAMPLE_PDF)
    assert title == "A Short Primer on Graph Neural Networks"
    assert [p.page for p in pages] == [1, 2, 3]
    assert "Over-smoothing" in pages[2].text and "graph attention network" in pages[1].text


@pytest.mark.parametrize(
    "data,message",
    [
        (b"", "empty"),
        (b"hello world, not a pdf", "not a PDF"),
        (b"%PDF-1.4\n garbage garbage", "could not be parsed|no pages|No extractable"),
    ],
)
def test_extract_rejects_bad_input(data, message):
    with pytest.raises(PDFError, match=message):
        extract_pages(data)


def test_image_only_pdf_has_no_text():
    blank = build_pdf([("", [])], "blank").replace(b"() Tj", b"")
    with pytest.raises(PDFError, match="image-only"):
        extract_pages(blank)


def test_chunking_respects_pages_and_overlap():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_pages([PageText(1, text), PageText(2, "short page")], chunk_size=300, overlap=60)
    assert chunks[-1].page == 2 and chunks[-1].text == "short page"
    page1 = [c for c in chunks if c.page == 1]
    assert all(len(c.text) <= 300 for c in page1)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # consecutive chunks overlap
    tail = page1[0].text.split()[-3:]
    assert all(w in page1[1].text.split() for w in tail)
    joined = set(" ".join(c.text for c in page1).split())
    assert joined == set(text.split())


def test_ingest_query_and_duplicate(store: DocumentStore):
    first = store.ingest(SAMPLE_PDF, "../../gnn primer.pdf")
    assert not first.duplicate and first.pages == 3 and first.chunks >= 3
    assert first.filename == "gnn_primer.pdf"
    assert len(first.sha256) == 64 and first.document_id == first.sha256[:16]
    second = store.ingest(SAMPLE_PDF, "copy.pdf")
    assert second.duplicate and second.document_id == first.document_id and second.chunks == first.chunks

    result = store.query("What are the known limitations of message passing such as over-smoothing?")
    assert result.evidence_available
    top = result.passages[0]
    assert top.page == 3 and top.citation == "[gnn_primer.pdf, p. 3]" and "Over-smoothing" in top.text
    assert result.documents_indexed == 1
    docs = store.list_documents()
    assert len(docs) == 1 and docs[0]["chunks"] == first.chunks


def test_query_without_documents_or_evidence(store: DocumentStore):
    empty = store.query("anything at all")
    assert not empty.evidence_available and "no documents" in empty.message
    store.ingest(SAMPLE_PDF, "gnn_primer.pdf")
    unrelated = store.query("zebra croissant volcano saxophone", max_distance=0.9)
    assert not unrelated.evidence_available and unrelated.passages == []
    assert "Evidence unavailable" in unrelated.message


def test_query_rejects_empty_question(store):
    with pytest.raises(ValueError):
        store.query("   ")
