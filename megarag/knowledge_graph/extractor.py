"""Extract entities and relations from markdown using a local Ollama LLM."""
import json
import logging
import re
import time
from typing import Optional
import numpy as np
import ray
from openai import OpenAI
from config.settings import get_settings

logger = logging.getLogger(__name__)


def _s(value) -> str:
    """Safe strip — returns '' when value is None (LLM may emit null fields)."""
    return (value or "").strip()


_SYSTEM = """You are an information extraction assistant for a multimodal knowledge graph.
Given a page of a document in markdown format, extract all named entities and the
relations between them — following the MegaRAG schema.

Respond ONLY with a JSON object — no markdown fences, no preamble — using this schema:
{
  "entities": [
    {
      "name": "exact name of the entity",
      "type": "one of: PERSON | ORG | CONCEPT | LOCATION | EVENT | PRODUCT | METHOD | DATASET | METRIC | OTHER",
      "description": "one sentence describing the entity and its role in the document"
    }
  ],
  "relations": [
    {
      "source": "entity name",
      "relation": "concise label ≤5 words",
      "target": "entity name",
      "description": "one sentence explaining this relationship",
      "keywords": "comma-separated keywords that characterise this relation"
    }
  ]
}

Rules:
- Extract entities explicitly mentioned on this page.
- Include method names, datasets, metrics and key concepts as entities.
- Relations may connect entities on this page OR reference entities from the
  "Known entities from previous pages" list provided below — this is how you
  build cross-page connections and keep the graph fully connected.
- Keep descriptions factual and grounded in the page text."""

_CONTEXT_HEADER = "\n\n---\nKnown entities from previous pages (you may reference these in relations):\n"

# Token budget — large enough for dense pages
_MAX_TOKENS = 4096


def _build_client(cfg):
    """Return an OpenAI-compatible client for the local Ollama server."""
    logger.info("[kg] using Ollama backend (url=%s, model=%s)", cfg.ollama_base_url, cfg.ollama_model)
    return OpenAI(
        base_url=cfg.ollama_base_url,
        api_key="ollama",  # required by the client but ignored by Ollama
    ), cfg.ollama_model


def _recover_partial_json(raw: str) -> dict:
    """Best-effort recovery from a truncated LLM JSON response.

    When the model hits the token limit mid-stream the closing braces/brackets
    are missing, causing json.loads to fail.  We scan for complete top-level
    JSON objects inside the ``entities`` and ``relations`` arrays and return
    whatever is parseable.
    """
    result: dict = {}

    # Match complete {...} objects (no nested objects, which is fine for our
    # flat entity/relation schema).
    obj_pattern = re.compile(r'\{[^{}]+\}', re.DOTALL)

    def _extract_objects(key: str) -> list[dict]:
        match = re.search(rf'"{key}"\s*:\s*\[', raw)
        if not match:
            return []
        array_start = match.end()
        objects = []
        for m in obj_pattern.finditer(raw, array_start):
            try:
                objects.append(json.loads(m.group()))
            except json.JSONDecodeError:
                pass
        return objects

    entities = _extract_objects("entities") or _extract_objects("new_entities")
    relations = _extract_objects("relations") or _extract_objects("new_relations")

    if entities:
        result["entities"] = entities
        result["new_entities"] = entities
    if relations:
        result["relations"] = relations
        result["new_relations"] = relations

    return result

_REFINE_SYSTEM = """You are a knowledge graph refinement assistant.

You are given:
1. A page of a document in markdown format.
2. A partial knowledge graph subgraph (entities + relations already known for this page).

Your task is to find ONLY what is MISSING:
- New entities mentioned on this page not yet in the subgraph.
- Implicit or cross-modal relations between entities that the initial pass missed
  (e.g. a figure that illustrates a concept, a table that supports a claim).

Respond ONLY with a JSON object — no markdown fences, no preamble:
{
  "new_entities": [
    {"name": "...", "type": "PERSON|ORG|CONCEPT|LOCATION|EVENT|PRODUCT|METHOD|DATASET|METRIC|OTHER", "description": "..."}
  ],
  "new_relations": [
    {"source": "...", "relation": "concise label ≤5 words", "target": "...", "description": "...", "keywords": "..."}
  ]
}

Rules:
- Do NOT repeat entities or relations already in the subgraph.
- Focus on implicit and cross-page connections (e.g. "illustrates", "supports", "compared with").
- Both source and target of every relation MUST be entity names (from new_entities OR the subgraph).
- If nothing is missing, return {"new_entities": [], "new_relations": []}."""


_MAX_CONTEXT_ENTITIES = 60   # cap to avoid overflowing the context window


def _call_llm(
    client,
    model: str,
    text: str,
    context_entities: list[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Send one page/chunk to the LLM and return (entities, relations)."""
    user_content = text
    if context_entities:
        recent = context_entities[-_MAX_CONTEXT_ENTITIES:]
        user_content += _CONTEXT_HEADER + ", ".join(recent)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        max_tokens=_MAX_TOKENS,
    )
    raw = response.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = _recover_partial_json(raw)
        if data:
            logger.warning(
                "[kg] JSON truncated (hit token limit) — recovered %d entities, "
                "%d relations via partial parse",
                len(data.get("entities", [])),
                len(data.get("relations", [])),
            )
        else:
            logger.warning(
                "[kg] JSON parse failed with no recovery — raw response (first 500 chars): %s",
                raw[:500],
            )
    return data.get("entities", []), data.get("relations", [])


_MAX_SUBGRAPH_ENTITIES = 30   # entities shown to the refinement LLM per page
_MAX_SUBGRAPH_RELATIONS = 20  # relations shown to the refinement LLM per page

EntityIndex = dict[str, np.ndarray]   # entity name_lower → mean-pooled vector


def _build_entity_index(entities: list[dict], embedder) -> EntityIndex:
    """Embed every entity's ``name + description`` and return a lookup index."""
    index: EntityIndex = {}
    for e in entities:
        text = f"{e['name']}: {e.get('description', '')}".strip()
        try:
            index[_s(e["name"]).lower()] = embedder.embed_text_mean(text)
        except Exception:
            pass
    return index


def _retrieve_relevant_entities(
    page_text: str,
    entity_index: EntityIndex,
    all_entities: list[dict],
    embedder,
    top_k: int = _MAX_SUBGRAPH_ENTITIES,
) -> list[dict]:
    """Return the top-k entities most semantically similar to *page_text*."""
    if not entity_index:
        return all_entities[:top_k]

    try:
        page_vec = embedder.embed_text_mean(page_text[:4000])
    except Exception:
        return all_entities[:top_k]

    names = list(entity_index.keys())
    matrix = np.stack([entity_index[n] for n in names])
    scores = matrix @ page_vec

    top_indices = np.argsort(scores)[::-1][:top_k]
    top_names = {names[i] for i in top_indices}

    return [e for e in all_entities if _s(e["name"]).lower() in top_names]


def _refine_page(
    client,
    model: str,
    page_text: str,
    page_entities: list[dict],
    page_relations: list[dict],
    entity_index: Optional[EntityIndex] = None,
    embedder=None,
    all_entities: Optional[list[dict]] = None,
) -> tuple[list[dict], list[dict]]:
    """Second-pass refinement for a single page (MegaRAG Stage 2)."""
    if entity_index and embedder and all_entities:
        relevant_entities = _retrieve_relevant_entities(
            page_text, entity_index, all_entities, embedder
        )
    else:
        relevant_entities = page_entities[:_MAX_SUBGRAPH_ENTITIES]

    ent_lines = "\n".join(
        f'  - {e["name"]} ({e.get("type","OTHER")}): {e.get("description","")}'
        for e in relevant_entities
    )
    rel_lines = "\n".join(
        f'  - {r["source"]} --[{r["relation"]}]--> {r["target"]}'
        for r in page_relations[:_MAX_SUBGRAPH_RELATIONS]
    )
    subgraph_block = (
        f"Known entities:\n{ent_lines or '  (none)'}\n\n"
        f"Known relations:\n{rel_lines or '  (none)'}"
    )
    user_content = f"{page_text}\n\n---\nCurrent subgraph for this page:\n{subgraph_block}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _REFINE_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        max_tokens=_MAX_TOKENS,
    )
    raw = response.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = _recover_partial_json(raw)
        if data:
            logger.warning(
                "[kg:refine] JSON truncated — recovered %d entities, %d relations",
                len(data.get("new_entities", [])),
                len(data.get("new_relations", [])),
            )
        else:
            logger.warning(
                "[kg:refine] JSON parse failed — raw response (first 500 chars): %s",
                raw[:500],
            )
    return data.get("new_entities", []), data.get("new_relations", [])


_CONNECT_SYSTEM = """You are a knowledge graph connectivity assistant.

You are given:
1. ISOLATED entities — nodes that currently have no edges in the graph.
2. HUB entities — the most connected nodes in the graph (most likely the main topics).

For each isolated entity suggest exactly ONE relation connecting it to one of the hub entities.
Only suggest a relation that is factually plausible — these entities come from the same document,
so there should always be at least a thematic connection.

Respond ONLY with a JSON array — no markdown fences, no preamble:
[
  {
    "source": "isolated entity name",
    "relation": "concise label ≤5 words",
    "target": "hub entity name",
    "description": "one sentence explaining the connection",
    "keywords": "comma-separated keywords"
  }
]

If an isolated entity genuinely has no meaningful relation to any hub entity, omit it."""

_CONNECT_BATCH = 15   # isolated nodes per LLM call


def _link_isolated_nodes(
    client,
    model: str,
    isolated: list[dict],
    hubs: list[dict],
    source_doc: str,
) -> list[dict]:
    """Stage 3 connectivity pass — link every isolated node to a hub."""
    if not isolated or not hubs:
        return []

    hub_lines = "\n".join(
        f'  - {h["name"]} ({h.get("type", "OTHER")}): {h.get("description", "")}'
        for h in hubs[:20]
    )
    new_relations: list[dict] = []

    for batch_start in range(0, len(isolated), _CONNECT_BATCH):
        batch = isolated[batch_start: batch_start + _CONNECT_BATCH]
        isolated_lines = "\n".join(
            f'  - {e["name"]} ({e.get("type", "OTHER")}): {e.get("description", "")}'
            for e in batch
        )
        user_content = (
            f"ISOLATED entities:\n{isolated_lines}\n\n"
            f"HUB entities:\n{hub_lines}"
        )
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _CONNECT_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
                max_tokens=_MAX_TOKENS,
            )
            raw = response.choices[0].message.content or "[]"
            try:
                items = json.loads(raw)
                if not isinstance(items, list):
                    items = []
            except json.JSONDecodeError:
                recovered = _recover_partial_json(raw)
                items = recovered.get("relations", [])
            for r in items:
                if r.get("source") and r.get("target") and r.get("relation"):
                    new_relations.append(dict(r, source_doc=source_doc))
        except Exception as exc:
            logger.warning("[kg:connect] LLM call failed for batch: %s", exc)

    return new_relations


@ray.remote
def _refine_page_remote(
    page_text: str,
    page_entities: list[dict],
    page_relations: list[dict],
    relevant_entities: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Ray remote: Stage 2 refinement for a single page (LLM call only).
    relevant_entities is pre-computed by the caller so ColQwen is not needed here.
    """
    cfg = get_settings()
    client, model = _build_client(cfg)
    # Build subgraph block directly from pre-computed relevant_entities
    ent_lines = "\n".join(
        f'  - {e["name"]} ({e.get("type", "OTHER")}): {e.get("description", "")}'
        for e in relevant_entities
    )
    rel_lines = "\n".join(
        f'  - {r["source"]} --[{r["relation"]}]--> {r["target"]}'
        for r in page_relations[:_MAX_SUBGRAPH_RELATIONS]
    )
    subgraph_block = (
        f"Known entities:\n{ent_lines or '  (none)'}\n\n"
        f"Known relations:\n{rel_lines or '  (none)'}"
    )
    user_content = f"{page_text}\n\n---\nCurrent subgraph for this page:\n{subgraph_block}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _REFINE_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        max_tokens=_MAX_TOKENS,
    )
    raw = response.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = _recover_partial_json(raw)
    return data.get("new_entities", []), data.get("new_relations", [])


@ray.remote
def _connect_batch_remote(
    batch: list[dict],
    hub_lines: str,
    source_doc: str,
) -> list[dict]:
    """
    Ray remote: Stage 3 connectivity LLM call for one batch of isolated nodes.
    """
    cfg = get_settings()
    client, model = _build_client(cfg)
    isolated_lines = "\n".join(
        f'  - {e["name"]} ({e.get("type", "OTHER")}): {e.get("description", "")}'
        for e in batch
    )
    user_content = f"ISOLATED entities:\n{isolated_lines}\n\nHUB entities:\n{hub_lines}"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _CONNECT_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            max_tokens=_MAX_TOKENS,
        )
        raw = response.choices[0].message.content or "[]"
        try:
            items = json.loads(raw)
            if not isinstance(items, list):
                items = []
        except json.JSONDecodeError:
            items = _recover_partial_json(raw).get("relations", [])
        return [
            dict(r, source_doc=source_doc)
            for r in items
            if r.get("source") and r.get("target") and r.get("relation")
        ]
    except Exception as exc:
        logger.warning("[kg:connect] remote batch failed: %s", exc)
        return []


def extract_entities_relations(
    pages: list[str], source: str = ""
) -> tuple[list[dict], list[dict]]:
    """
    Extract entities and relations from a list of per-page markdown strings.

    Each page is sent to the LLM individually (matching the MegaRAG paper's
    page-level extraction approach).  Results across pages are deduplicated by
    entity name and (source, relation, target) for relations.
    """
    cfg = get_settings()
    client, model = _build_client(cfg)

    total_pages = len(pages)
    logger.info("[kg] ── starting extraction: %d page(s) from '%s'", total_pages, source)
    extraction_start = time.perf_counter()

    seen_entity_names: set[str] = set()
    seen_relations: set[tuple] = set()
    all_entities: list[dict] = []
    all_relations: list[dict] = []

    for i, page_text in enumerate(pages):
        context = [e["name"] for e in all_entities]

        page_start = time.perf_counter()
        page_entities, page_relations = _call_llm(client, model, page_text, context)
        page_elapsed = time.perf_counter() - page_start

        new_entities = 0
        for e in page_entities:
            name_key = _s(e.get("name")).lower()
            if name_key and name_key not in seen_entity_names:
                seen_entity_names.add(name_key)
                all_entities.append(dict(e, source=source))
                new_entities += 1

        new_relations = 0
        for r in page_relations:
            key = (_s(r.get("source")).lower(),
                   _s(r.get("relation")).lower(),
                   _s(r.get("target")).lower())
            if key not in seen_relations:
                seen_relations.add(key)
                all_relations.append(dict(r, source_doc=source))
                new_relations += 1

        logger.info(
            "[kg] page %d/%d — %.2fs | +%d entities (+%d new), +%d relations (+%d new) | "
            "running totals: %d entities, %d relations",
            i + 1, total_pages, page_elapsed,
            len(page_entities), new_entities,
            len(page_relations), new_relations,
            len(all_entities), len(all_relations),
        )

    total_elapsed = time.perf_counter() - extraction_start
    avg_per_page = total_elapsed / total_pages if total_pages else 0
    logger.info(
        "[kg] ── stage 1 complete: %d entities, %d relations | "
        "total %.2fs (avg %.2fs/page)",
        len(all_entities), len(all_relations), total_elapsed, avg_per_page,
    )

    # ── Stage 2: Refinement ──────────────────────────────────────────────────
    logger.info("[kg] ── starting stage 2 refinement: %d page(s)", total_pages)
    refine_start = time.perf_counter()
    refine_entities = 0
    refine_relations = 0

    entity_index: EntityIndex = {}
    embedder = None
    try:
        from megarag.embedding.colqwen import ColQwenEmbedder
        embedder = ColQwenEmbedder()
        entity_index = _build_entity_index(all_entities, embedder)
        logger.info(
            "[kg] ── entity embedding index built: %d/%d entities indexed",
            len(entity_index), len(all_entities),
        )
    except Exception as exc:
        logger.warning(
            "[kg] ColQwen not available — falling back to global top-%d subgraph: %s",
            _MAX_SUBGRAPH_ENTITIES, exc,
        )

    # Pre-compute relevant_entities per page (uses ColQwen locally — one pass)
    # so Ray workers only need to do the LLM call with no GPU dependency.
    page_relevant_entities = [
        _retrieve_relevant_entities(page_text, entity_index, all_entities, embedder)
        for page_text in pages
    ]

    # Submit all pages to Ray in parallel
    futures = [
        _refine_page_remote.remote(
            page_text,
            all_entities[:_MAX_SUBGRAPH_ENTITIES],
            all_relations[:_MAX_SUBGRAPH_RELATIONS],
            page_relevant_entities[i],
        )
        for i, page_text in enumerate(pages)
    ]
    refine_results = ray.get(futures)

    for i, (new_ents, new_rels) in enumerate(refine_results):
        added_e = 0
        for e in new_ents:
            name_key = _s(e.get("name")).lower()
            if name_key and name_key not in seen_entity_names:
                seen_entity_names.add(name_key)
                all_entities.append(dict(e, source=source))
                added_e += 1

        added_r = 0
        for r in new_rels:
            key = (_s(r.get("source")).lower(),
                   _s(r.get("relation")).lower(),
                   _s(r.get("target")).lower())
            if key not in seen_relations:
                seen_relations.add(key)
                all_relations.append(dict(r, source_doc=source))
                added_r += 1

        refine_entities += added_e
        refine_relations += added_r
        logger.info(
            "[kg:refine] page %d/%d — parallel | +%d entities, +%d relations",
            i + 1, total_pages, added_e, added_r,
        )

    logger.info(
        "[kg] ── stage 2 complete: +%d entities, +%d relations | %.2fs",
        refine_entities, refine_relations, time.perf_counter() - refine_start,
    )

    # ── Stage 3: Connectivity pass ───────────────────────────────────────────
    from collections import Counter
    name_degree: Counter = Counter()
    for r in all_relations:
        name_degree[_s(r.get("source")).lower()] += 1
        name_degree[_s(r.get("target")).lower()] += 1

    isolated = [
        e for e in all_entities
        if name_degree[_s(e["name"]).lower()] == 0
    ]

    if isolated:
        logger.info(
            "[kg] ── stage 3 connectivity pass: %d/%d nodes are isolated",
            len(isolated), len(all_entities),
        )
        connect_start = time.perf_counter()

        hub_names = {name for name, _ in name_degree.most_common(20)}
        hubs = [e for e in all_entities if _s(e["name"]).lower() in hub_names]

        hub_lines = "\n".join(
            f'  - {h["name"]} ({h.get("type", "OTHER")}): {h.get("description", "")}'
            for h in hubs[:20]
        )
        # Submit all connectivity batches to Ray in parallel
        batch_futures = [
            _connect_batch_remote.remote(
                isolated[start: start + _CONNECT_BATCH],
                hub_lines,
                source,
            )
            for start in range(0, len(isolated), _CONNECT_BATCH)
        ]
        batch_results = ray.get(batch_futures)
        connect_rels = [r for batch in batch_results for r in batch]

        added_connect = 0
        for r in connect_rels:
            key = (_s(r.get("source")).lower(),
                   _s(r.get("relation")).lower(),
                   _s(r.get("target")).lower())
            if key not in seen_relations:
                seen_relations.add(key)
                all_relations.append(r)
                added_connect += 1

        logger.info(
            "[kg] ── stage 3 complete: linked %d/%d isolated nodes | %.2fs",
            added_connect, len(isolated), time.perf_counter() - connect_start,
        )
    else:
        logger.info("[kg] ── stage 3 skipped: no isolated nodes found")

    logger.info(
        "[kg] ── final totals: %d entities, %d relations | overall %.2fs",
        len(all_entities), len(all_relations),
        time.perf_counter() - extraction_start,
    )
    return all_entities, all_relations
