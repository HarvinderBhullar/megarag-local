"""Convert each PDF page to a PNG image using PyMuPDF."""
from pathlib import Path
import fitz  # pymupdf


def pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = 150) -> list[Path]:
    """
    Render every page of *pdf_path* as a PNG and save under *output_dir*.
    Returns list of saved image paths in page order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    paths: list[Path] = []
    stem = pdf_path.stem

    for i, page in enumerate(doc):
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_path = output_dir / f"{stem}_page{i:04d}.png"
        pix.save(str(img_path))
        paths.append(img_path)

    doc.close()
    return paths
