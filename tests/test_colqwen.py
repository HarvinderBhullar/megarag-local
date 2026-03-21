import pytest
from pathlib import Path
from PIL import Image


def test_embed_query_shape():
    try:
        from megarag.embedding.colqwen import ColQwenEmbedder
    except Exception:
        pytest.skip("ColQwen not available in this env")

    embedder = ColQwenEmbedder()
    vecs = embedder.embed_query("What is the main finding?")
    assert isinstance(vecs, list)
    assert len(vecs) > 0
    assert len(vecs[0]) == 128


def test_embed_page_shape(tmp_path):
    try:
        from megarag.embedding.colqwen import ColQwenEmbedder
    except Exception:
        pytest.skip("ColQwen not available in this env")

    img = Image.new("RGB", (400, 600), color=(255, 255, 255))
    img_path = tmp_path / "page0000.png"
    img.save(img_path)

    embedder = ColQwenEmbedder()
    vecs = embedder.embed_page(img_path)
    assert isinstance(vecs, list)
    assert len(vecs) > 0
    assert len(vecs[0]) == 128
