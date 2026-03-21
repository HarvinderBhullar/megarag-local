from fastapi import APIRouter

from megarag.api.schemas import QueryRequest, QueryResponse
from megarag.retrieval.keyword_parser import parse_keywords
from megarag.retrieval.visual_retriever import retrieve_pages
from megarag.retrieval.kg_retriever import retrieve_subgraph
from megarag.generation.answer_generator import generate_answer

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    # Derive store identifiers from doc_id if a scoped query was requested.
    collection_name: str | None = None
    schema: str | None = None
    if req.doc_id:
        collection_name = f"megarag_{req.doc_id}"
        schema = f"doc_{req.doc_id}"

    # 1. Parse keywords
    kw = parse_keywords(req.question)
    all_keywords = kw.get("low_level", []) + kw.get("high_level", [])

    # 2. Retrieve in parallel tracks.
    expanded_query = req.question
    if all_keywords:
        extra = " ".join(k for k in all_keywords if k.lower() not in req.question.lower())
        if extra:
            expanded_query = f"{req.question} {extra}"

    pages = retrieve_pages(expanded_query, top_k=req.top_k, collection_name=collection_name)
    subgraph = retrieve_subgraph(all_keywords, schema=schema)

    # 3. Generate answer
    img_paths = [p["img_path"] for p in pages]
    result = generate_answer(req.question, subgraph, img_paths)

    return QueryResponse(question=req.question, **result)
