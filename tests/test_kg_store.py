from pathlib import Path
from megarag.knowledge_graph.store import KGStore


def test_upsert_and_search(tmp_path):
    db = KGStore(tmp_path / "test.db")

    entities = [
        {"name": "Alice", "type": "PERSON", "source": "doc1.pdf"},
        {"name": "Acme Corp", "type": "ORG", "source": "doc1.pdf"},
    ]
    relations = [
        {"source": "Alice", "relation": "works at", "target": "Acme Corp", "source_doc": "doc1.pdf"},
    ]

    db.upsert_entities(entities)
    db.upsert_relations(relations)

    results = db.search_entities(["Alice"])
    assert len(results) == 1
    assert results[0]["name"] == "Alice"

    subgraph = db.get_subgraph(["Alice"])
    assert len(subgraph) == 1
    assert subgraph[0]["relation"] == "works at"
