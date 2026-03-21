from pathlib import Path
import pytest
from megarag.ingestion.pdf_processor import pdf_to_images


def test_pdf_to_images_returns_paths(tmp_path):
    # Requires a real PDF — skip in CI if not available
    sample = Path("tests/fixtures/sample.pdf")
    if not sample.exists():
        pytest.skip("No sample PDF fixture found")

    paths = pdf_to_images(sample, tmp_path / "pages")
    assert len(paths) > 0
    assert all(p.suffix == ".png" for p in paths)
    assert all(p.exists() for p in paths)
