"""Retrieve relevant KG subgraph using keyword-matched entities."""
from config.settings import get_settings
from megarag.knowledge_graph.store import KGStore


def retrieve_subgraph(
    keywords: list[str],
    schema: str | None = None,
) -> dict:
    """
    Search for entities matching *keywords*, then pull their relations.

    If *schema* is given, search only that document's DuckDB schema.
    Otherwise fan-out across all per-document schemas (prefix 'doc_') and merge.

    Returns dict with 'entities' and 'relations' lists.
    Returns empty result if no documents have been ingested yet.
    """
    cfg = get_settings()

    if not cfg.kg_db_path.exists():
        return {"entities": [], "relations": []}

    if schema:
        schemas = [schema]
    else:
        schemas = KGStore.list_doc_schemas(cfg.kg_db_path)

    if not schemas:
        return {"entities": [], "relations": []}

    all_entities: list[dict] = []
    all_relations: list[dict] = []

    for s in schemas:
        try:
            kg = KGStore(cfg.kg_db_path, schema=s, read_only=True)
        except FileNotFoundError:
            continue
        entities = kg.search_entities(keywords)
        entity_names = [e["name"] for e in entities]
        relations = kg.get_subgraph(entity_names)
        all_entities.extend(entities)
        all_relations.extend(relations)

    return {"entities": all_entities, "relations": all_relations}
