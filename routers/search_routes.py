from fastapi import APIRouter
from schemas.search_models import SearchQuery
from functional.search_engine import search_services

router = APIRouter()

@router.post("/search/")
async def search(query_data: SearchQuery):
    results = search_services(query_data.query, query_data.top_k)
    return {"query": query_data.query, "results": results}
