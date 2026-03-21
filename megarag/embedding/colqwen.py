"""
Load ColQwen once (singleton) and expose embed_page / embed_query.
Uses MPS on Apple Silicon, falls back to CPU if unavailable.
"""
from pathlib import Path
from functools import lru_cache

import torch
import numpy as np
from PIL import Image
from colpali_engine.models import ColQwen2, ColQwen2Processor

from config.settings import get_settings


@lru_cache(maxsize=1)
def _load_model() -> tuple[ColQwen2, ColQwen2Processor]:
    cfg = get_settings()
    device = cfg.colqwen_device

    # Graceful fallback if MPS not available
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    model = ColQwen2.from_pretrained(
        cfg.colqwen_model,
        torch_dtype=torch.float16,
        device_map=device,
    ).eval()

    processor = ColQwen2Processor.from_pretrained(cfg.colqwen_model)
    return model, processor


class ColQwenEmbedder:
    def __init__(self):
        self.model, self.processor = _load_model()

    def embed_page(self, img_path: Path) -> list[list[float]]:
        """
        Embed a single page image.
        Returns a list of patch vectors (shape: [n_patches, 128]).
        """
        img = Image.open(img_path).convert("RGB")
        inputs = self.processor.process_images([img]).to(self.model.device)

        with torch.no_grad():
            embeddings = self.model(**inputs)  # (1, n_patches, 128)

        vecs = embeddings[0].cpu().float().numpy()  # (n_patches, 128)
        return vecs.tolist()

    def embed_query(self, query: str) -> list[list[float]]:
        """
        Embed a text query.
        Returns a list of token vectors (shape: [n_tokens, 128]).
        """
        inputs = self.processor.process_queries([query]).to(self.model.device)

        with torch.no_grad():
            embeddings = self.model(**inputs)  # (1, n_tokens, 128)

        vecs = embeddings[0].cpu().float().numpy()
        return vecs.tolist()

    def embed_text_mean(self, text: str) -> np.ndarray:
        """
        Embed text and return a single mean-pooled vector (shape: [128]).

        Useful for fast cosine-similarity comparisons between entity
        descriptions and page text during KG subgraph retrieval.
        The mean pool collapses the multi-vector (n_tokens, 128) output
        into one representative vector while preserving directional
        semantics well enough for top-k retrieval.
        """
        vecs = np.array(self.embed_query(text), dtype=np.float32)  # (n_tokens, 128)
        mean_vec = vecs.mean(axis=0)                                # (128,)
        norm = np.linalg.norm(mean_vec)
        return mean_vec / norm if norm > 0 else mean_vec
