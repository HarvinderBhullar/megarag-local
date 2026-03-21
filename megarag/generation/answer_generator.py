"""
2-stage answer generation:
  Stage 1 — draft answer from KG subgraph context (text)
  Stage 2 — refine answer with retrieved page images (vision)
             Only runs when OLLAMA_VISION_MODEL is set in config.
"""
import base64
import logging
from pathlib import Path

from openai import OpenAI
from config.settings import get_settings

logger = logging.getLogger(__name__)


def _encode_image(img_path: Path) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _stage1_draft(client: OpenAI, model: str, question: str, subgraph: dict) -> str:
    """Draft an answer from the KG subgraph (text only)."""
    kg_text = (
        f"Entities: {subgraph.get('entities', [])}\n"
        f"Relations: {subgraph.get('relations', [])}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful research assistant. "
                    "Use the provided knowledge graph context to draft an answer. "
                    "Be concise. Flag anything uncertain."
                ),
            },
            {
                "role": "user",
                "content": f"Knowledge graph context:\n{kg_text}\n\nQuestion: {question}",
            },
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return resp.choices[0].message.content or ""


def _stage2_refine(
    client: OpenAI,
    model: str,
    question: str,
    draft: str,
    img_paths: list[Path],
) -> str:
    """Refine the draft using retrieved page images (vision)."""
    image_content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{_encode_image(p)}",
                "detail": "high",
            },
        }
        for p in img_paths[:5]
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful research assistant with vision capabilities. "
                    "You will be given a draft answer and the original document pages. "
                    "Refine the answer using visual evidence from the pages. "
                    "Cite page content where relevant."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Question: {question}"},
                    {"type": "text", "text": f"Draft answer:\n{draft}"},
                    {"type": "text", "text": "Relevant document pages:"},
                    *image_content,
                    {"type": "text", "text": "Please provide a refined, grounded answer."},
                ],
            },
        ],
        temperature=0.2,
        max_tokens=1200,
    )
    return resp.choices[0].message.content or ""


def generate_answer(question: str, subgraph: dict, img_paths: list[Path]) -> dict:
    """
    Full 2-stage generation.
    Returns dict with 'draft', 'answer', and 'sources'.

    Stage 2 (vision refinement) runs only when OLLAMA_VISION_MODEL is set,
    e.g. OLLAMA_VISION_MODEL=llama3.2-vision in your .env.
    """
    cfg = get_settings()
    client = OpenAI(base_url=cfg.ollama_base_url, api_key="ollama")

    # Stage 1: draft from KG text
    draft = _stage1_draft(client, cfg.ollama_model, question, subgraph)

    # Stage 2: refine with page images if a vision model is configured
    answer = draft
    if img_paths and cfg.ollama_vision_model:
        try:
            answer = _stage2_refine(client, cfg.ollama_vision_model, question, draft, img_paths)
        except Exception as exc:
            logger.warning("[answer] Stage 2 vision refinement failed, using draft: %s", exc)
    elif img_paths:
        logger.info(
            "[answer] Stage 2 skipped — set OLLAMA_VISION_MODEL=llama3.2-vision "
            "in your .env to enable visual refinement"
        )

    # Return relative URLs so the browser can fetch them via the /pages/ static mount
    page_urls = [f"/pages/{Path(p).name}" for p in img_paths]

    return {
        "draft": draft,
        "answer": answer,
        "sources": page_urls,
    }
