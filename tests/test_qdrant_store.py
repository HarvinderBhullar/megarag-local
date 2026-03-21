import pytest


def test_upsert_and_search():
    try:
        from megarag.vectorstore.qdrant_store import QdrantStore
    except Exception:
        pytest.skip("Qdrant not reachable")

    store = QdrantStore()
    fake_embeddings = [[0.1] * 128 for _ in range(10)]

    store.upsert_page(
        doc_name="test.pdf",
        page_index=0,
        img_path=__import__("pathlib").Path("data/pages/test.pdf_page0000.png"),
        embeddings=fake_embeddings,
    )

    hits = store.search(fake_embeddings, top_k=1)
    assert len(hits) >= 1
    assert hits[0].payload["doc_name"] == "test.pdf"
